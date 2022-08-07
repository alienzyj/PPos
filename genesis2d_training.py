import argparse
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_utils import painting, nonlinear_transformation, local_pixel_shuffling
from my_dataset import GenesisDataset2D
from models import Genesis2D


def get_genesis2D(args, writer):
    model = Genesis2D(args.layers, args.heights, args.atrous_rates)
    writer.add_graph(model, torch.empty([1, 14] + args.resize_shape, dtype=torch.float))
    model = model.to(args.rank)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    best_loss = 1e3
    epochs = 0
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min, epochs - 1)
    if args.resume:
        checkpoint = torch.load(args.path, torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs = checkpoint['epochs']
        best_loss = checkpoint['best_iou']

    return model, optimizer, best_loss, epochs, criterion, scheduler


def train(args, model, optimizer, writer, scheduler, best_loss, training_loader, validation_loader, epochs, criterion):
    training_loss = 0
    start_time = time.time()

    for epoch in range(epochs, args.max_epochs):
        for i, data in enumerate(training_loader):
            sample = data[0].to(args.rank, non_blocking=True)
            mask = data[1].to(args.rank, non_blocking=True)
            out = model(sample)
            loss = criterion(out, mask)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            training_loss = training_loss + loss.item()

            if (i + 1) % args.training_steps == 0:
                training_loss = training_loss / args.training_steps

                writer.add_scalar('training_loss', training_loss, epoch * len(training_loader) + i)

                print(
                    '[{:0>5d} {:0>5d}] training loss: {:.5f}'.format(epoch + 1, i + 1, training_loss))
                training_loss = 0

            if (i + 1) % args.validation_steps == 0:
                best_loss = validation(args, model, optimizer, writer, best_loss, validation_loader, epoch, i,
                                       start_time, criterion, epoch * len(training_loader) + i)
                start_time = time.time()

        scheduler.step()
        training_loss = 0
    print('Finish training!')


@torch.no_grad()
def validation(args, model, optimizer, writer, best_loss, validation_loader, epoch, i, start_time, criterion, steps):
    model.eval()
    validation_loss = 0

    for validation_data in validation_loader:
        x = validation_data[0].to(args.rank, non_blocking=True)
        mask = validation_data[1].to(args.rank, non_blocking=True)
        out = model(x)
        validation_loss = validation_loss + criterion(out, mask).item()
    validation_loss = validation_loss / len(validation_loader)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_loss': best_loss,
                    'epochs': epoch + 1}, args.path)
        writer.add_scalars('best_metrics', {'loss': validation_loss}, steps)
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_data', param, epoch)

    writer.add_scalar('validation_loss', validation_loss, steps)

    print(
        '[{:0>5d} {:0>5d}] validation loss: {:.5f}\tbest loss: {:.5f}'.format(epoch + 1, i + 1, validation_loss,
                                                                              best_loss))
    print('process time: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))
    print('-' * 50)

    model.train()
    return best_loss


def run():
    parser = argparse.ArgumentParser(description='Genesis2D Training')
    parser.add_argument('--training_dir', default='dst/train', help='directory of training dataset')
    parser.add_argument('--validation_dir', default='dst/validation', help='directory of validation dataset')
    parser.add_argument('--batch_size', type=int, default=7, help='batch-size of dataset')
    parser.add_argument('--training_steps', type=int, default=240, help='steps to show training information')
    parser.add_argument('--validation_steps', type=int, default=970, help='steps to show validation information')
    parser.add_argument('--project', default='genesis2d', help='name of the project')
    parser.add_argument('--path', default='genesis2d.pth', help='path of model weights')
    parser.add_argument('--name', default='1', help='name of the current run')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume this run')
    parser.add_argument('--max_epochs', type=int, default=300, help='the number of maximum training epochs')
    parser.add_argument('--rank', type=int, default=0, help='which gpu to use')
    parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD solver')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--gamma', type=int, default=2, help='hyper-parameter of focal loss for unet')
    parser.add_argument('--layers', type=int, nargs=4, default=(3, 4, 6, 3), help='number of bottlenecks in encoder')
    parser.add_argument('--heights', type=int, nargs=4, default=(2, 3, 4, 3, 2),
                        help='number of bottlenecks in decoder')
    parser.add_argument('--atrous_rates', type=int, nargs=3, default=(5, 9, 17),
                        help='the dilation rates of convolution in ASPP')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[256, 512],
                        help='patch size of training samples')
    parser.add_argument('--notes', default='', help='a longer description of the run')
    parser.add_argument('--eta_min', type=float, default=0, help='minimum learning rate')
    parser.add_argument("--nonlinear_rate", type=float, default=0, help="the probability of nonlinear")
    parser.add_argument("--paint_rate", type=float, default=0.9, help="the probability of paint")
    parser.add_argument("--outpaint_rate", type=float, default=0.8, help="the probability of outpaint")
    parser.add_argument("--local_rate", type=float, default=0.5, help="the probability of local")
    parser.add_argument("--flip_rate", type=float, default=0.4, help="the probability of flip")

    args = parser.parse_args()
    args.path = f'weights/genesis2d/genesis2d{args.name}.pth'
    args.validation_steps = len(os.listdir(f'{args.training_dir}/samples'))

    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.rank}"

    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(f'runs/genesis2d/{args.name}', args.notes)

    writer.add_hparams(
        {'max_epochs': args.max_epochs, 'batch_size': args.batch_size, 'gamma': args.gamma,
         'num_workers': args.num_workers, 'rank': args.rank, 'learning_rate': args.learning_rate,
         'momentum': args.momentum, 'weight_decay': args.weight_decay, 'resize_shape': torch.tensor(args.resize_shape),
         'atrous_rates': torch.tensor(args.atrous_rates), 'heights': torch.tensor(args.heights),
         'layers': torch.tensor(args.layers), 'eta_min': args.eta_min, "nonlinear_rate": args.nonlinear_rate,
         "paint_rate": args.paint_rate, "outpaint_rate": args.outpaint_rate, "local_rate": args.local_rate,
         "flip_rate": args.flip_rate}, {})


    args.training_steps = args.training_steps // args.batch_size
    args.validation_steps = args.validation_steps // args.batch_size

    print('*' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(args.name)
    print(f'max_epochs: {args.max_epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'num_workers: {args.num_workers}')
    print(f'gamma: {args.gamma}')
    print(f'rank: {args.rank}')
    print(f'learning_rate: {args.learning_rate}')
    print(f'momentum: {args.momentum}')
    print(f'layers: {args.layers}')
    print(f'heights: {args.heights}')
    print(f'atrous_rates: {args.atrous_rates}')
    print(f'weight_decay: {args.weight_decay}')
    print(f'resize_shape: {args.resize_shape}')
    print(f'eta_min: {args.eta_min}')
    print(f"nonlinear_rate: {args.nonlinear_rate}")
    print(f"paint_rate: {args.paint_rate}")
    print(f"outpaint_rate: {args.outpaint_rate}")
    print(f"local_rate: {args.local_rate}")
    print(f"flip_rate: {args.flip_rate}")
    print(f"max_epochs: {args.max_epochs}")
    print('*' * 50)

    args.rank = 0

    model, optimizer, best_iou, epochs, criterion, scheduler = get_genesis2D(args, writer)

    transform = transforms.Compose([lambda x: local_pixel_shuffling(x, args.local_rate, "2D"),
                                    lambda x: nonlinear_transformation(x, args.nonlinear_rate),
                                    lambda x: painting(x, args.paint_rate, args.outpaint_rate, "2D")])
    training_set = GenesisDataset2D(args.training_dir, args.resize_shape, transform, args.flip_rate)
    validation_set = GenesisDataset2D(args.validation_dir, args.resize_shape, transform, args.flip_rate)
    training_loader = DataLoader(training_set, args.batch_size, True, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch_size, False, num_workers=args.num_workers,
                                   pin_memory=True)

    train(args, model, optimizer, writer, scheduler, best_iou, training_loader, validation_loader, epochs, criterion)
    writer.close()


if __name__ == '__main__':
    run()

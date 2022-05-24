import argparse
import collections
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import PPos2D
from my_dataset import PPos2DDataset
from my_utils import normalize, elastic_transform, random_crop_resize, random_flip, random_rotate


def get_ppos2d(args):
    model = PPos2D(args.layers, args.atrous_rates, args.num_classes)
    model = model.to(args.rank)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    best_loss = 1e3
    epochs = 0

    if args.resume:
        checkpoint = torch.load(args.path, torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs = checkpoint['epochs']
        best_loss = checkpoint['best_loss']
    elif args.pretrain > 0:
        checkpoint = torch.load(f'weights/ppos2d/ppos2d{args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
    elif args.pretrain < 0:
        checkpoint = torch.load(f'weights/genesis2d/genesis2d{-args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))["model"]
        encoder_weights = collections.OrderedDict()
        for key, value in checkpoint.items():
            if "encoder" in key:
                encoder_weights[key] = value
        model.load_state_dict(encoder_weights, strict=False)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min, epochs - 1)

    return model, optimizer, best_loss, epochs, nn.MSELoss().to(args.rank), scheduler


def train(args, model, optimizer, writer, scheduler, best_loss, training_loader, validation_loader, epochs, criterion):
    training_loss = 0
    start_time = time.time()

    for epoch in range(epochs, args.max_epochs):
        for i, data in enumerate(training_loader):
            sample = data[0].to(args.rank, non_blocking=True)
            targets = data[1].to(args.rank, non_blocking=True)
            outputs = model(sample)
            loss = criterion(outputs, targets)
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
        y = validation_data[1].to(args.rank, non_blocking=True)
        outputs = model(x)
        validation_loss = validation_loss + criterion(outputs, y).item()
    validation_loss = validation_loss / len(validation_loader)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_loss': best_loss,
                    'epochs': epoch + 1}, args.path)
        writer.add_scalars('best_metrics', {'loss': validation_loss}, steps)

    writer.add_scalar('validation_loss', validation_loss, steps)

    print(
        '[{:0>5d} {:0>5d}] validation loss: {:.5f}\tbest loss: {:.5f}'.format(epoch + 1, i + 1, validation_loss,
                                                                              best_loss))
    print('process time: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))
    print('-' * 50)

    model.train()
    return best_loss


def run():
    parser = argparse.ArgumentParser(description='PPos2D Training')
    parser.add_argument('--training_dir', default='dst/train', help='directory of training dataset')
    parser.add_argument('--validation_dir', default='dst/validation',
                        help='directory of validation dataset')
    parser.add_argument('--batch_size', type=int, default=13, help='batch-size of dataset')
    parser.add_argument('--training_steps', type=int, default=240, help='steps to show training information')
    parser.add_argument('--validation_steps', type=int, default=970, help='steps to show validation information')
    parser.add_argument('--project', default='ppos2d', help='name of the project')
    parser.add_argument('--path', default='ppos2d.pth', help='path of model weights')
    parser.add_argument('--name', default='1', help='name of the current run')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume this run')
    parser.add_argument('--pretrain', type=int, default=0, help='whether this run is pretrained')
    parser.add_argument('--max_epochs', type=int, default=800, help='the number of maximum training epochs')
    parser.add_argument('--rank', type=int, default=0, help='which gpu to use')
    parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD solver')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--layers', type=int, nargs=4, default=(3, 4, 6, 3), help='number of bottlenecks in encoder')
    parser.add_argument('--atrous_rates', type=int, nargs=3, default=(5, 9, 17),
                        help='the dilation rates of convolution in ASPP')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[256, 512],
                        help='patch size of training samples')
    parser.add_argument('--notes', default='', help='a longer description of the run')
    parser.add_argument('--eta_min', type=float, default=0, help='minimum learning rate')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument("--elastic_alpha", type=float, nargs=2, default=(100., 350.),
                        help="alpha coefficient for elastic transformation")
    parser.add_argument("--elastic_sigma", type=float, nargs=2, default=(14., 17.),
                        help="sigma coefficient for elastic transformation")
    parser.add_argument("--degree", type=float, default=10, help="random rotation degree")
    parser.add_argument("--scale", type=float, nargs=2, default=(0.8, 1), help="random scale factor")
    parser.add_argument("--flip_axis", type=int, nargs="+", default=-1, help="axis along which to flip")
    parser.add_argument("--flip_rate", type=float, default=0.5, help="probability to flip")

    args = parser.parse_args()
    args.path = f'weights/ppos2d/ppos2d{args.name}.pth'
    args.validation_steps = len(os.listdir(f'{args.training_dir}/samples'))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)

    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(f'runs/ppos2d/{args.name}', args.notes)

    writer.add_hparams(
        {'max_epochs': args.max_epochs, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
         'rank': args.rank, 'learning_rate': args.learning_rate, 'momentum': args.momentum,
         'weight_decay': args.weight_decay, 'resize_shape': torch.tensor(args.resize_shape),
         'atrous_rates': torch.tensor(args.atrous_rates), 'layers': torch.tensor(args.layers), 'eta_min': args.eta_min,
         'pretrain': args.pretrain, "num_classes": args.num_classes, "elastic_alpha": torch.tensor(args.elastic_alpha),
         "elastic_sigma": torch.tensor(args.elastic_sigma), "degree": args.degree, "scale": torch.tensor(args.scale),
         "flip_axis": args.flip_axis, "flip_rate": args.flip_rate}, {})

    args.training_steps = args.training_steps // args.batch_size
    args.validation_steps = args.validation_steps // args.batch_size

    profiles = f'''
    {time.strftime('%Y-%m-%d %H:%M:%S')}
    {args.name}
    max_epochs: {args.max_epochs}
    batch_size: {args.batch_size}
    num_workers: {args.num_workers}
    rank: {args.rank}
    learning_rate: {args.learning_rate}
    momentum: {args.momentum}
    layers: {args.layers}
    atrous_rates: {args.atrous_rates}
    weight_decay: {args.weight_decay}
    resize_shape: {args.resize_shape}
    eta_min: {args.eta_min}
    pretrain: {args.pretrain}
    num_classes: {args.num_classes}
    elastic_alpha: {args.elastic_alpha}
    elastic_sigma: {args.elastic_sigma}
    degree: {args.degree}
    scale: {args.scale}
    flip_axis: {args.flip_axis}
    flip_rate: {args.flip_rate}
    '''

    print('*' * 50)
    print(profiles)
    print('*' * 50)

    writer.add_text("profile", profiles)

    args.rank = 0

    model, optimizer, best_loss, epochs, criterion, scheduler = get_ppos2d(args)

    training_transform = transforms.Compose([random_rotate(args.degree, (-1, -2), 'nearest', False),
                                             random_crop_resize(args.resize_shape, "nearest", args.scale),
                                             random_flip(args.flip_axis, args.flip_rate), normalize])
    validation_transform = transforms.Compose([random_crop_resize(args.resize_shape, "nearest"), normalize])
    training_set = PPos2DDataset(args.training_dir, args.resize_shape, args.num_classes, training_transform)
    validation_set = PPos2DDataset(args.validation_dir, args.resize_shape, args.num_classes, validation_transform)
    training_loader = DataLoader(training_set, args.batch_size, True, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch_size, False, num_workers=args.num_workers,
                                   pin_memory=True)

    train(args, model, optimizer, writer, scheduler, best_loss, training_loader, validation_loader, epochs, criterion)
    writer.close()


if __name__ == '__main__':
    run()

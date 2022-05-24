import argparse
import datetime
import os
import time
import collections

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_utils import InverseWeightingFocalLoss, normalize, my_iou, TanimotoLoss
from my_dataset import UNetDataset
from models import UNet


def get_unet(args):
    model = UNet(args.layers, args.heights, args.atrous_rates)
    model = model.to(args.rank)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    best_iou = 0
    epochs = 0
    focal_loss = InverseWeightingFocalLoss(args.gamma).to(args.rank)
    tanimoto_loss = TanimotoLoss(args.gamma).to(args.rank)

    def criterion(x, y):
        return tanimoto_loss(x, y) + focal_loss(x, y)

    iou = my_iou(args.threshold, True)
    
    if args.resume:
        checkpoint = torch.load(args.path, torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs = checkpoint['epochs']
        best_iou = checkpoint['best_iou']
    elif args.pretrain > 0:
        checkpoint = torch.load(f'weights/unet/unet{args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
    elif args.pretrain < 0:
        # checkpoint = torch.load(f'weights/genesis2d/genesis2d{-args.pretrain}.pth',
        #                         torch.device(f'cuda:{args.rank}'))["model"]
        # checkpoint = torch.load(f'weights/ppos2d/ppos2d{-args.pretrain}.pth',
        #                         torch.device(f'cuda:{args.rank}'))["model"]
        checkpoint = torch.load(f'weights/pcl2d/pcl2d{-args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))["model"]
        # checkpoint = torch.load(f'weights/gcl2d/gcl2d{-args.pretrain}.pth',
        #                         torch.device(f'cuda:{args.rank}'))["model"]
        encoder_weights = collections.OrderedDict()
        for key, value in checkpoint.items():
            if "encoder" in key:
                encoder_weights[key] = value
        model.load_state_dict(encoder_weights, strict=False)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min, epochs - 1)

    return model, optimizer, best_iou, epochs, criterion, iou, scheduler


def train(args, model, optimizer, writer, scheduler, best_iou, training_loader, validation_loader, epochs, iou,
          unet_criterion):
    training_loss = 0
    training_iou = 0
    start_time = time.time()

    for epoch in range(epochs, args.max_epochs):
        for i, data in enumerate(training_loader):
            sample = data[0].to(args.rank, non_blocking=True)
            mask = data[1].to(args.rank, non_blocking=True)
            mask2 = data[2].to(args.rank, non_blocking=True)
            mask3 = data[3].to(args.rank, non_blocking=True)
            out1, out2, out3, out4, out5, out6 = model(sample)
            loss = 0.1 * unet_criterion(out1, mask3) + 0.1 * unet_criterion(out2, mask3) + 0.1 * unet_criterion(out3,
                                                                                                                mask3) + 0.1 * unet_criterion(
                out4, mask3) + 0.1 * unet_criterion(out5, mask2) + 0.5 * unet_criterion(out6, mask)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            training_loss = training_loss + loss.item()
            training_iou = training_iou + iou(out6, mask)

            if (i + 1) % args.training_steps == 0:
                training_loss = training_loss / args.training_steps
                training_iou = training_iou / args.training_steps

                writer.add_scalar('training_loss', training_loss, epoch * len(training_loader) + i)
                writer.add_scalar('training_iou', training_iou, epoch * len(training_loader) + i)

                print(
                    '[{:0>5d} {:0>5d}] training loss: {:.5f}\ttraining iou: {:.5f}'.format(epoch + 1, i + 1,
                                                                                           training_loss, training_iou))
                training_loss = 0
                training_iou = 0

            if (i + 1) % args.validation_steps == 0:
                best_iou = validation(args, model, optimizer, writer, best_iou, validation_loader, epoch, i,
                                      start_time, iou, unet_criterion, epoch * len(training_loader) + i)
                start_time = time.time()

        scheduler.step()
        training_loss = 0
        training_iou = 0
    print('Finish training!')


@torch.no_grad()
def validation(args, model, optimizer, writer, best_iou, validation_loader, epoch, i, start_time, iou,
               unet_criterion, steps):
    model.eval()
    validation_loss = 0
    validation_iou = 0

    for validation_data in validation_loader:
        x = validation_data[0].to(args.rank, non_blocking=True)
        mask = validation_data[1].to(args.rank, non_blocking=True)
        mask2 = validation_data[2].to(args.rank, non_blocking=True)
        mask3 = validation_data[3].to(args.rank, non_blocking=True)
        out1, out2, out3, out4, out5, out6 = model(x)
        validation_loss = validation_loss + (
                0.1 * unet_criterion(out1, mask3) + 0.1 * unet_criterion(out2, mask3) + 0.1 * unet_criterion(out3,
                                                                                                             mask3) + 0.1 * unet_criterion(
            out4, mask3) + 0.1 * unet_criterion(out5, mask2) + 0.5 * unet_criterion(out6, mask)).item()
        validation_iou = validation_iou + iou(out6, mask)
    validation_loss = validation_loss / len(validation_loader)
    validation_iou = validation_iou / len(validation_loader)

    if validation_iou > best_iou:
        best_iou = validation_iou
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_iou': best_iou,
                    'epochs': epoch + 1}, args.path)
        writer.add_scalars('best_metrics', {'iou': validation_iou}, steps)

    writer.add_scalar('validation_loss', validation_loss, steps)
    writer.add_scalar('validation_iou', validation_iou, steps)

    print(
        '[{:0>5d} {:0>5d}] validation loss: {:.5f}\tvalidation iou: {:.5f}\tbest iou: {:.5f}'.format(
            epoch + 1, i + 1, validation_loss, validation_iou, best_iou))
    print('process time: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))
    print('-' * 50)

    model.train()
    return best_iou


def run():
    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--training_dir', default='slices/train50', help='directory of training dataset')
    parser.add_argument('--validation_dir', default='slices/validation', help='directory of validation dataset')
    parser.add_argument('--batch_size', type=int, default=9, help='batch-size of dataset')
    parser.add_argument('--training_steps', type=int, default=1000, help='steps to show training information')
    parser.add_argument('--validation_steps', type=int, default=13000, help='steps to show validation information')
    parser.add_argument('--project', default='unet', help='name of the project')
    parser.add_argument('--path', default='unet.pth', help='path of model weights')
    parser.add_argument('--name', default='1', help='name of the current run')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume this run')
    parser.add_argument('--pretrain', type=int, default=0, help='whether this run is pretrained')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold that determines whether a pixel belongs to nodule or not')
    parser.add_argument('--max_epochs', type=int, default=100, help='the number of maximum training epochs')
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
    parser.add_argument('--eta_min', type=float, default=0.0001, help='minimum learning rate')

    args = parser.parse_args()
    args.path = f'weights/unet/unet{args.name}.pth'
    args.validation_steps = len(os.listdir(f'{args.training_dir}/samples'))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)

    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(f'runs/unet/{args.name}', args.notes)

    writer.add_hparams(
        {'threshold': args.threshold, 'max_epochs': args.max_epochs, 'batch_size': args.batch_size, 'gamma': args.gamma,
         'num_workers': args.num_workers, 'rank': args.rank, 'learning_rate': args.learning_rate,
         'momentum': args.momentum, 'weight_decay': args.weight_decay, 'resize_shape': torch.tensor(args.resize_shape),
         'atrous_rates': torch.tensor(args.atrous_rates), 'heights': torch.tensor(args.heights),
         'layers': torch.tensor(args.layers), 'eta_min': args.eta_min, 'pretrain': args.pretrain, "training_dir": args.training_dir}, {})

    args.training_steps = args.training_steps // args.batch_size
    args.validation_steps = args.validation_steps // args.batch_size

    profiles = f'''
    {time.strftime('%Y-%m-%d %H:%M:%S')}
    {args.name}
    threshold: {args.threshold}
    max_epochs: {args.max_epochs}
    batch_size: {args.batch_size}
    num_workers: {args.num_workers}
    gamma: {args.gamma}
    rank: {args.rank}
    learning_rate: {args.learning_rate}
    momentum: {args.momentum}
    layers: {args.layers}
    heights: {args.heights}
    atrous_rates: {args.atrous_rates}
    weight_decay: {args.weight_decay}
    resize_shape: {args.resize_shape}
    eta_min: {args.eta_min}
    pretrain: {args.pretrain}
    training_dir: {args.training_dir}
    '''

    print('*' * 50)
    print(profiles)
    print('*' * 50)

    writer.add_text("profile", profiles)

    args.rank = 0

    model, optimizer, best_iou, epochs, criterion, iou, scheduler = get_unet(args)
    # wandb.watch(model, criterion)

    transform = transforms.Compose([normalize])
    training_set = UNetDataset(args.training_dir, args.resize_shape, True, transform)
    validation_set = UNetDataset(args.validation_dir, args.resize_shape, False, transform)
    training_loader = DataLoader(training_set, args.batch_size, True, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch_size, False, num_workers=args.num_workers,
                                   pin_memory=True)

    train(args, model, optimizer, writer, scheduler, best_iou, training_loader, validation_loader, epochs, iou,
          criterion)
    writer.close()


if __name__ == '__main__':
    run()

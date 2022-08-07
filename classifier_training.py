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

from models import Classifier
from my_dataset import ClassifierDataset
from my_utils import FocalLoss, normalize, random_rotate, random_flip


def get_classifier(args):
    model = Classifier(args.blocks).to(args.rank)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    best_f1 = 0
    epochs = 0
    if args.resume:
        checkpoint = torch.load(args.path, torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_f1 = checkpoint['best_f1']
        epochs = checkpoint['epochs']
    elif args.pretrain > 0:
        checkpoint = torch.load(f'weights/classifier/classifier{args.pretrain}.pth', torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
    elif args.pretrain < 0:
        checkpoint = torch.load(f"weights/ppos_classifier/ppos_classifier{-args.pretrain}.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min, epochs - 1)
    criterion = FocalLoss(args.alpha, args.gamma).to(args.rank)

    return model, optimizer, best_f1, epochs, criterion, scheduler


def train(args, model, optimizer, scheduler, writer, best_f1, training_loader, validation_loader, epochs, criterion):
    training_loss = 0
    tp = 0
    fp = 0
    fn = 0
    acc = 0
    start_time = time.time()

    for epoch in range(epochs, args.max_epochs):
        for i, data in enumerate(training_loader):
            sample, label = data[0].to(args.rank, non_blocking=True), data[1].to(args.rank, non_blocking=True)

            pred = model(sample)
            loss = criterion(pred, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            training_loss = training_loss + loss.item()
            pred = pred > args.threshold
            tp = tp + torch.sum(pred * label)
            fp = fp + torch.sum(pred * (1 - label))
            fn = fn + torch.sum((~pred) * label)
            acc = acc + torch.sum(pred == label).item() / sample.size(0)

            if (i + 1) % args.training_steps == 0:
                training_loss = training_loss / args.training_steps
                tp = tp.item()
                fp = fp.item()
                fn = fn.item()
                if tp + fp == 0:
                    precision = 1
                else:
                    precision = tp / (tp + fp)
                if tp + fn == 0:
                    recall = 1
                else:
                    recall = tp / (tp + fn)
                if precision + recall != 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                acc = acc / args.training_steps

                writer.add_scalar('training_loss', training_loss, epoch * len(training_loader) + i)
                writer.add_scalar('training_precision', precision, epoch * len(training_loader) + i)
                writer.add_scalar('training_recall', recall, epoch * len(training_loader) + i)
                writer.add_scalar('training_f1', f1, epoch * len(training_loader) + i)
                writer.add_scalar('training_acc', acc, epoch * len(training_loader) + i)

                print('[{:d}, {:0>5d}] training loss: {:.5f}'.format(epoch + 1, i + 1, training_loss))
                print('[{:d}, {:0>5d}] training precision: {:.5f}'.format(epoch + 1, i + 1, precision))
                print('[{:d}, {:0>5d}] training recall: {:.5f}'.format(epoch + 1, i + 1, recall))
                print('[{:d}, {:0>5d}] training f1: {:.5f}'.format(epoch + 1, i + 1, f1))
                print('[{:d}, {:0>5d}] training acc: {:.5f}'.format(epoch + 1, i + 1, acc))

                training_loss = 0
                tp = 0
                fp = 0
                fn = 0
                acc = 0

            if (i + 1) % args.validation_steps == 0:
                best_f1 = validation(args, model, optimizer, writer, best_f1, validation_loader, epoch, i, start_time,
                                     criterion, epoch * len(training_loader) + i)
                start_time = time.time()
        training_loss = 0
        acc = 0
        scheduler.step()
    print('Finish training!')


@torch.no_grad()
def validation(args, model, optimizer, writer, best_f1, validation_loader, epoch, i, start_time, criterion, steps):
    model.eval()
    validation_loss = 0
    tp = 0
    fp = 0
    fn = 0
    acc = 0

    for validation_data in validation_loader:
        x = validation_data[0].to(args.rank, non_blocking=True)
        y = validation_data[1].to(args.rank, non_blocking=True)
        p = model(x)
        validation_loss = validation_loss + criterion(p, y).item()
        p = p > args.threshold
        tp = tp + torch.sum(p * y)
        fp = fp + torch.sum(p * (1 - y))
        fn = fn + torch.sum((~p) * y)
        acc = acc + torch.sum(p == y).item() / x.size(0)

    tp = tp.item()
    fp = fp.item()
    fn = fn.item()
    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1
    else:
        recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    acc = acc / len(validation_loader)
    validation_loss = validation_loss / len(validation_loader)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_f1': best_f1,
             'precision': precision, 'recall': recall, 'acc': acc, 'epochs': epoch + 1}, args.path)
        writer.add_scalars('best_metrics', {'f1': f1, 'precision': precision, 'recall': recall, 'acc': acc}, steps)

    writer.add_scalar('validation_loss', validation_loss, steps)
    writer.add_scalar('validation_precision', precision, steps)
    writer.add_scalar('validation_recall', recall, steps)
    writer.add_scalar('validation_f1', f1, steps)
    writer.add_scalar('validation_acc', acc, steps)

    print('[{:d}, {:0>5d}] validation loss: {:.5f}'.format(epoch + 1, i + 1, validation_loss))
    print('[{:d}, {:0>5d}] validation precision: {:.5f}'.format(epoch + 1, i + 1, precision))
    print('[{:d}, {:0>5d}] validation recall: {:.5f}'.format(epoch + 1, i + 1, recall))
    print('[{:d}, {:0>5d}] validation f1: {:.5f}'.format(epoch + 1, i + 1, f1))
    print('[{:d}, {:0>5d}] validation acc: {:.5f}'.format(epoch + 1, i + 1, acc))
    print('[{:d}, {:0>5d}] best f1: {:.5f}'.format(epoch + 1, i + 1, best_f1))
    print('process time: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))
    print('-' * 50)

    model.train()

    return best_f1


def run():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--training_dir', default='cube_classifier/train', help='directory of training dataset')
    parser.add_argument('--validation_dir', default='cube_classifier/validation',
                        help='directory of validation dataset')
    parser.add_argument('--batch_size', type=int, default=18, help='batch-size of dataset')
    parser.add_argument('--training_steps', type=int, default=240, help='steps to show training information')
    parser.add_argument('--validation_steps', type=int, default=2000, help='steps to show validation information')
    parser.add_argument('--project', default='classifier', help='name of the project')
    parser.add_argument('--path', default='classifier.pth', help='path of model weights')
    parser.add_argument('--name', default='1', help='name of the current run')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume this run')
    parser.add_argument('--pretrain', type=int, default=0, help='whether this run is pretrained')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold that determines whether a sample has a nodule or not')
    parser.add_argument('--max_epochs', type=int, default=600, help='the number of maximum training epochs')
    parser.add_argument('--rank', type=int, choices=[0, 1, 2, 3], default=1, help='which gpu to use')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyper-parameter of focal loss')
    parser.add_argument('--gamma', type=int, default=2, help='hyper-parameter of focal loss')
    parser.add_argument('--num_workers', type=int, default=24, help='how many subprocesses to use for data loading')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD solver')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--resize_shape', default=[48, 96, 96], type=int, nargs=3,
                        help='resize shape of classifier input')
    parser.add_argument('--blocks', type=int, nargs=4, default=(3, 4, 23, 3),
                        help='number of bottlenecks in classifier')
    parser.add_argument('--notes', default='', help='a longer description of the run')
    parser.add_argument('--eta_min', type=float, default=0, help='minimum learning rate')

    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    args.path = f'weights/classifier/classifier{args.name}.pth'
    args.validation_steps = len(os.listdir(f'{args.training_dir}/samples'))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)

    writer = SummaryWriter(f'runs/classifier/{args.name}', args.notes)

    hparams = {'threshold': args.threshold, 'max_epochs': args.max_epochs, 'batch_size': args.batch_size,
               'alpha': args.alpha, 'gamma': args.gamma, 'num_workers': args.num_workers, 'rank': args.rank,
               'learning_rate': args.learning_rate, 'momentum': args.momentum, 'weight_decay': args.weight_decay,
               'blocks': torch.tensor(args.blocks), 'resize_shape': torch.tensor(args.resize_shape),
               'eta_min': args.eta_min, 'pretrain': args.pretrain}
    writer.add_hparams(hparams, {})


    args.training_steps = args.training_steps // args.batch_size
    args.validation_steps = args.validation_steps // args.batch_size

    profiles = f'''
    {time.strftime('%Y-%m-%d %H:%M:%S')}
    {args.name}
    threshold: {args.threshold}
    max_epochs: {args.max_epochs}
    batch_size: {args.batch_size}
    alpha: {args.alpha}
    gamma: {args.gamma}
    num_workers: {args.num_workers}
    rank: {args.rank}
    learning_rate: {args.learning_rate}
    momentum: {args.momentum}
    weight_decay: {args.weight_decay}
    blocks: {args.blocks}
    resize_shape: {args.resize_shape}
    eta_min: {args.eta_min}
    pretrain: {args.pretrain}
    '''

    print('*' * 50)
    print(profiles)
    print('*' * 50)

    writer.add_text("profile", profiles)
    args.rank = 0

    model, optimizer, best_f1, epochs, criterion, scheduler = get_classifier(args)

    training_transform = transforms.Compose(
        [normalize,
         random_rotate(5, (-1, -2), 'nearest', False),
         random_flip(-1, 0.5)])
    validation_transform = transforms.Compose([normalize])
    training_set = ClassifierDataset(args.training_dir, args.resize_shape, True, training_transform)
    validation_set = ClassifierDataset(args.validation_dir, args.resize_shape, False, validation_transform)
    training_loader = DataLoader(training_set, args.batch_size, True, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch_size, False, num_workers=args.num_workers,
                                   pin_memory=True)

    train(args, model, optimizer, scheduler, writer, best_f1, training_loader, validation_loader, epochs, criterion)
    writer.close()


if __name__ == "__main__":
    run()

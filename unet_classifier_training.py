import argparse
import datetime
import os
import collections
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_utils import FocalLoss, normalize, my_iou, TanimotoLoss, InverseWeightingFocalLoss
from my_dataset import UNetClassifierDataset
from models import UNetClassifier


def get_unet_classifier(args):
    model = UNetClassifier(args.encoder_heights, args.unet_heights, args.classifier_blocks, args.group_num,
                           args.axial_atrous_rates, args.depth_atrous_rates, args.drop_rate)
    model = model.to(args.rank)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    best_iou = 0
    epochs = 0
    iou = my_iou(args.unet_threshold, True)
    focal_loss = InverseWeightingFocalLoss(args.unet_gamma).to(args.rank)
    tanimoto_loss = TanimotoLoss(args.unet_gamma).to(args.rank)

    def unet_criterion(x, y):
        return tanimoto_loss(x, y) + focal_loss(x, y)

    if args.resume:
        checkpoint = torch.load(args.path, torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs = checkpoint['epochs']
        best_iou = checkpoint['best_iou']
    elif args.pretrain > 0:
        checkpoint = torch.load(f'weights/unet_classifier/unet_classifier{args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))
        model.load_state_dict(checkpoint['model'])
    elif args.pretrain < 0:
        # checkpoint = torch.load(f'weights/ppos_unet_classifier/ppos_unet_classifier{-args.pretrain}.pth',
        #                         torch.device(f'cuda:{args.rank}'))["model"]
        checkpoint = torch.load(f'weights/genesis_unet_classifier/genesis_unet_classifier{-args.pretrain}.pth',
                                torch.device(f'cuda:{args.rank}'))["model"]
        encoder_weights = collections.OrderedDict()
        for key, value in checkpoint.items():
            if "encoder" in key:
                encoder_weights[key] = value
        model.load_state_dict(encoder_weights, strict=False)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=args.eta_min,
                                                     last_epoch=epochs - 1)

    return model, optimizer, scheduler, unet_criterion, FocalLoss(args.classifier_alpha, args.classifier_gamma).to(
        args.rank), iou, best_iou, epochs


def train(args, model, optimizer, writer, scheduler, best_iou, training_loader, validation_loader, epochs, iou,
          unet_criterion, classifier_criterion):
    training_loss = 0
    unet_loss = 0
    classifier_loss = 0
    training_iou = 0
    training_acc = 0
    tp = 0
    fp = 0
    fn = 0
    start_time = time.time()

    for epoch in range(epochs, args.max_epochs):
        for i, data in enumerate(training_loader):
            sample = data[0].to(args.rank, non_blocking=True)
            mask = data[1].to(args.rank, non_blocking=True)
            mask2 = data[2].to(args.rank, non_blocking=True)
            mask3 = data[3].to(args.rank, non_blocking=True)
            label = data[4].to(args.rank, non_blocking=True)
            out1, out2, out3, out4, out5, l = model(sample)
            loss1 = 0.1 * unet_criterion(out1, mask3) + 0.1 * unet_criterion(out2, mask3) + 0.1 * unet_criterion(out3,
                                                                                                                 mask3) + 0.1 * unet_criterion(
                out4, mask2) + 0.6 * unet_criterion(out5, mask)
            loss2 = args.loss_weight * classifier_criterion(l, label)
            loss = loss1 + loss2
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            training_loss = training_loss + loss.item()
            unet_loss = unet_loss + loss1.item()
            classifier_loss = classifier_loss + loss2.item()
            training_iou = training_iou + iou(out5, mask)
            l = l > args.classifier_threshold
            tp = tp + torch.sum(l * label)
            fp = fp + torch.sum(l * (1 - label))
            fn = fn + torch.sum((~l) * label)
            training_acc = training_acc + torch.sum(l == label).item() / sample.size(0)

            if (i + 1) % args.training_steps == 0:
                training_loss = training_loss / args.training_steps
                unet_loss = unet_loss / args.training_steps
                classifier_loss = classifier_loss / args.training_steps
                training_iou = training_iou / args.training_steps
                training_acc = training_acc / args.training_steps
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

                writer.add_scalar('unet_loss', unet_loss, epoch * len(training_loader) + i)
                writer.add_scalar('classifier_loss', classifier_loss, epoch * len(training_loader) + i)
                writer.add_scalar('training_loss', training_loss, epoch * len(training_loader) + i)
                writer.add_scalar('training_iou', training_iou, epoch * len(training_loader) + i)
                writer.add_scalar('training_precision', precision, epoch * len(training_loader) + i)
                writer.add_scalar('training_recall', recall, epoch * len(training_loader) + i)
                writer.add_scalar('training_f1', f1, epoch * len(training_loader) + i)
                writer.add_scalar('training_acc', training_acc, epoch * len(training_loader) + i)

                print(
                    '[{:0>5d} {:0>5d}] training loss: {:.5f}\t unet loss: {:.5f}\t classifier loss: {:.5f}\t training iou: {:.5f}\t training acc: {:.5f}'.format(
                        epoch + 1, i + 1, training_loss, unet_loss, classifier_loss, training_iou, training_acc))
                print(
                    f'[{epoch + 1:0>5d} {i + 1:0>5d}] training precision: {precision:.5f}\t training recall: {recall:.5f}\t training f1: {f1:.5f}')
                training_loss = 0
                unet_loss = 0
                classifier_loss = 0
                training_iou = 0
                training_acc = 0
                tp = 0
                fp = 0
                fn = 0

            if (i + 1) % args.validation_steps == 0:
                best_iou = validation(args, model, optimizer, writer, best_iou, validation_loader, epoch, i,
                                      start_time, iou, unet_criterion, classifier_criterion,
                                      epoch * len(training_loader) + i)
                start_time = time.time()
        scheduler.step()
        training_loss = 0
        unet_loss = 0
        classifier_loss = 0
        training_iou = 0
        training_acc = 0
    print('Finish training!')


@torch.no_grad()
def validation(args, model, optimizer, writer, best_iou, validation_loader, epoch, i, start_time, iou,
               unet_criterion, classifier_criterion, steps):
    model.eval()
    validation_loss = 0
    validation_iou = 0
    validation_acc = 0
    tp = 0
    fp = 0
    fn = 0

    for validation_data in validation_loader:
        x = validation_data[0].to(args.rank, non_blocking=True)
        mask = validation_data[1].to(args.rank, non_blocking=True)
        mask2 = validation_data[2].to(args.rank, non_blocking=True)
        mask3 = validation_data[3].to(args.rank, non_blocking=True)
        label = validation_data[4].to(args.rank, non_blocking=True)
        out1, out2, out3, out4, out5, l = model(x)
        validation_loss = validation_loss + (
                0.1 * unet_criterion(out1, mask3) + 0.1 * unet_criterion(out2, mask3) + 0.1 * unet_criterion(
            out3, mask3) + 0.1 * unet_criterion(out4, mask2) + 0.6 * unet_criterion(out5,
                                                                                    mask)).item() + args.loss_weight * classifier_criterion(
            l, label).item()
        validation_iou = validation_iou + iou(out5, mask)
        l = l > args.classifier_threshold
        tp = tp + torch.sum(l * label)
        fp = fp + torch.sum(l * (1 - label))
        fn = fn + torch.sum((~l) * label)
        validation_acc = validation_acc + torch.sum(l == label).item() / x.size(0)
    validation_loss = validation_loss / len(validation_loader)
    validation_iou = validation_iou / len(validation_loader)
    validation_acc = validation_acc / len(validation_loader)
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

    if validation_iou > best_iou:
        best_iou = validation_iou
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_iou': best_iou,
                    'acc': validation_acc, 'f1': f1, 'precision': precision, 'recall': recall,
                    'epochs': epoch + 1}, args.path)
        writer.add_scalars('best_metrics', {'f1': f1, 'precision': precision, 'recall': recall, 'acc': validation_acc,
                                            'iou': validation_iou}, steps)

    writer.add_scalar('validation_loss', validation_loss, steps)
    writer.add_scalar('validation_iou', validation_iou, steps)
    writer.add_scalar('validation_precision', precision, steps)
    writer.add_scalar('validation_recall', recall, steps)
    writer.add_scalar('validation_f1', f1, steps)
    writer.add_scalar('validation_acc', validation_acc, steps)

    print(
        '[{:0>5d} {:0>5d}] validation loss: {:.5f}\t validation iou: {:.5f}\t validation acc: {:.5f}\t best iou: {:.5f}'.format(
            epoch + 1, i + 1, validation_loss, validation_iou, validation_acc, best_iou))
    print(
        f'[{epoch + 1:0>5d} {i + 1:0>5d}] validation precision: {precision:.5f}\t validation recall: {recall:.5f}\t validation f1: {f1:.5f}')
    print('process time: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))
    print('-' * 50)

    model.train()
    return best_iou


def run():
    parser = argparse.ArgumentParser(description='UNetClassifier Training')
    parser.add_argument('--training_dir', default='cube_unet/train', help='directory of training dataset')
    parser.add_argument('--validation_dir', default='cube_unet/validation', help='directory of validation dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch-size of dataset')
    parser.add_argument('--training_steps', type=int, default=240, help='steps to show training information')
    parser.add_argument('--validation_steps', type=int, default=970, help='steps to show validation information')
    parser.add_argument('--project', default='unet_classifier', help='name of the project')
    parser.add_argument('--path', default='unet_classifier.pth', help='path of model weights')
    parser.add_argument('--name', default='1', help='name of the current run')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume this run')
    parser.add_argument('--pretrain', type=int, default=0, help='whether this run is pretrained')
    parser.add_argument('--unet_threshold', type=float, default=0.5,
                        help='threshold that determines whether a pixel belongs to nodule or not')
    parser.add_argument('--classifier_threshold', type=float, default=0.5,
                        help='threshold that determines whether a sample has a nodule or not')
    parser.add_argument('--max_epochs', type=int, default=100, help='the number of maximum training epochs')
    parser.add_argument('--rank', type=int, default=0, help='which gpu to use')
    parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
    parser.add_argument('--group_num', type=int, default=8, help='number of groups in group normalization')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD solver')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--unet_gamma', type=int, default=2, help='hyper-parameter of focal loss for unet')
    parser.add_argument('--classifier_alpha', type=float, default=612 / 989,
                        help='hyper-parameter of focal loss for classifier')
    parser.add_argument('--classifier_gamma', type=int, default=2, help='hyper-parameter of focal loss for classifier')
    parser.add_argument('--classifier_blocks', type=int, nargs=4, default=(2, 2, 2, 2),
                        help='number of bottlenecks in classifier')
    parser.add_argument('--encoder_heights', type=int, nargs=4, default=(5, 4, 4, 4),
                        help='number of bottlenecks in encoder')
    parser.add_argument('--unet_heights', type=int, nargs=4, default=(3, 3, 4, 4),
                        help='number of bottlenecks in decoder of unet')
    parser.add_argument('--axial_atrous_rates', type=int, nargs=3, default=(5, 9, 17),
                        help='the dilation rates of convolution in ASPP')
    parser.add_argument('--depth_atrous_rates', type=int, nargs=3, default=(3, 5, 9),
                        help='the dilation rates of convolution in ASPP')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate of aspp')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 128],
                        help='patch size of training samples')
    parser.add_argument('--notes', default='', help='a longer description of the run')
    parser.add_argument('--loss_weight', type=float, default=1, help='the scale factor of the loss of classifier')
    parser.add_argument('--eta_min', type=float, default=0.0001, help='minimum learning rate')

    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    args.path = f'weights/unet_classifier/unet_classifier{args.name}.pth'
    args.validation_steps = len(os.listdir(f'{args.training_dir}/samples'))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)

    writer = SummaryWriter(f'runs/unet_classifier/{args.name}', args.notes)

    hparams = {'unet_threshold': args.unet_threshold, 'max_epochs': args.max_epochs, 'batch_size': args.batch_size,
               'unet_gamma': args.unet_gamma, 'classifier_alpha': args.classifier_alpha,
               'classifier_gamma': args.classifier_gamma, 'num_workers': args.num_workers, 'rank': args.rank,
               'learning_rate': args.learning_rate, 'momentum': args.momentum, 'weight_decay': args.weight_decay,
               'classifier_threshold': args.classifier_threshold, 'group_num': args.group_num,
               'patch_size': torch.tensor(args.patch_size), 'drop_rate': args.drop_rate,
               'depth_atrous_rates': torch.tensor(args.depth_atrous_rates),
               'axial_atrous_rates': torch.tensor(args.axial_atrous_rates), 'loss_weight': args.loss_weight,
               'unet_heights': torch.tensor(args.unet_heights), 'encoder_heights': torch.tensor(args.encoder_heights),
               'classifier_blocks': torch.tensor(args.classifier_blocks), 'eta_min': args.eta_min,
               'pretrain': args.pretrain, "training_dir": args.training_dir}

    writer.add_hparams(hparams, {})

    args.training_steps = args.training_steps // args.batch_size
    args.validation_steps = args.validation_steps // args.batch_size

    profiles = f'''
    {time.strftime('%Y-%m-%d %H:%M:%S')}
    {args.name}
    unet_threshold: {args.unet_threshold}
    classifier_threshold: {args.classifier_threshold}
    max_epochs: {args.max_epochs}
    batch_size: {args.batch_size}
    num_workers: {args.num_workers}
    group_num: {args.group_num}
    unet_gamma: {args.unet_gamma}
    classifier_alpha: {args.classifier_alpha}
    classifier_gamma: {args.classifier_gamma}
    rank: {args.rank}
    learning_rate: {args.learning_rate}
    momentum: {args.momentum}
    classifier_blocks: {args.classifier_blocks}
    encoder_heights: {args.encoder_heights}
    unet_heights: {args.unet_heights}
    loss_weight: {args.loss_weight}
    axial_atrous_rates: {args.axial_atrous_rates}
    depth_atrous_rates: {args.depth_atrous_rates}
    drop_rate: {args.drop_rate}
    weight_decay: {args.weight_decay}
    patch_size: {args.patch_size}
    eta_min: {args.eta_min}
    pretrain: {args.pretrain}
    training_dir: {args.training_dir}
    '''

    print('*' * 50)
    print(profiles)
    print('*' * 50)

    writer.add_text("profile", profiles)

    args.rank = 0

    model, optimizer, scheduler, unet_criterion, classifier_criterion, iou, best_iou, epochs = get_unet_classifier(args)

    transform = transforms.Compose([normalize])
    training_set = UNetClassifierDataset(args.training_dir, True, transform)
    validation_set = UNetClassifierDataset(args.validation_dir, False, transform)
    training_loader = DataLoader(training_set, args.batch_size, True, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch_size, False, num_workers=args.num_workers,
                                   pin_memory=True)

    train(args, model, optimizer, writer, scheduler, best_iou, training_loader, validation_loader, epochs, iou,
          unet_criterion, classifier_criterion)

    writer.close()


if __name__ == '__main__':
    run()

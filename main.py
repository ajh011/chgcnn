# from nn.cgcnn import CGCNN
# from nn.defect_model import DefectEncoder
# from data import DefectGraphDataset
# from torch_geometric.loader import DataLoader
#
# dataset = DefectGraphDataset('dataset/C2DB/cifs', 'dataset/C2DB/defects')
# data0 = dataset[0]
# model = DefectEncoder(data0[0].x.size(-1), data0[0].edge_attr.size(-1))
#
# loader = DataLoader(dataset, batch_size=2, shuffle=True)
# defect, host = next(iter(loader))
#
# out = model(defect, host)

import argparse
import wandb
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from sklearn import metrics
from chgcnn import CHGCNN
from data import DefectCalcDataset

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(model, device, train_loader, loss_criterion, optimizer, epoch, logwandb, task):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    if task == 'regression':
        mae_errors = AverageMeter('MAE', ':.4f')
        progress = ProgressMeter(len(train_loader),
            [batch_time, data_time, losses, mae_errors],
            prefix='Epoch: [{}]'.format(epoch))
    else:
        accuracies = AverageMeter('Accuracy', ':.4f')
        precisions = AverageMeter('Precision', ':.4f')
        recalls = AverageMeter('Recall', ':.4f')
        fscores = AverageMeter('F1', ':.4f')
        auc_scores = AverageMeter('AUC', ':.4f')
        progress = ProgressMeter(len(train_loader),
            [batch_time, data_time, losses, accuracies, precisions, recalls, fscores, auc_scores],
            prefix='Epoch: [{}]'.format(epoch))

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)

        output = model(data)
        target = data.y

        loss = loss_criterion(output, target)
        losses.update(loss.item(), target.size(0))

        if task == 'regression':
            mae_error = F.l1_loss(output, target)
            mae_errors.update(mae_error.item(), target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.detach().cpu().numpy(), target.detach().cpu().numpy())
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if logwandb:
            if task == 'regression':
                wandb.log({'train_loss': losses.avg, 'train_mae': mae_errors.avg})
            else:
                wandb.log({'train_loss': losses.avg, 'train_accuracy': accuracies.avg,
                           'train_precision': precisions.avg, 'train_recall': recalls.avg,
                           'train_f1': fscores.avg, 'train_auc': auc_scores.avg})
        else:
            if i % 10 == 0:
                progress.display(i)


def validate(model, device, test_loader, loss_criterion, logwandb, task):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    if task == 'regression':
        mae_errors = AverageMeter('MAE', ':.4f')
        progress = ProgressMeter(len(test_loader),
            [batch_time, losses, mae_errors],
            prefix='Val: ')
    else:
        accuracies = AverageMeter('Accuracy', ':.4f')
        precisions = AverageMeter('Precision', ':.4f')
        recalls = AverageMeter('Recall', ':.4f')
        fscores = AverageMeter('F1', ':.4f')
        auc_scores = AverageMeter('AUC', ':.4f')
        progress = ProgressMeter(len(test_loader),
            [batch_time, losses, accuracies, precisions, recalls, fscores, auc_scores],
            prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):

            data = data.to(device, non_blocking=True)
            output = model(data)
            target = data.y

            loss = loss_criterion(output, target)
            losses.update(loss.item(), target.size(0))

            if task == 'regression':
                mae_error = F.l1_loss(output, target)
                mae_errors.update(mae_error.item(), target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if logwandb:
                if task == 'regression':
                    wandb.log({'val_loss': losses.avg, 'val_mae': mae_errors.avg})
                else:
                    wandb.log({'val_loss': losses.avg, 'val_accuracy': accuracies.avg,
                               'val_precision': precisions.avg, 'val_recall': recalls.avg,
                               'val_f1': fscores.avg, 'val_auc': auc_scores.avg})

            else:
                if i % 10 == 0:
                    progress.display(i)


    if task == 'regression':
        return mae_errors.avg
    else:
        return auc_scores.avg


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='trainning ratio')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',
                        help='choose an optimizer, SGD or Adam, (default:Adam)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--drop-last', default=False, type=bool)
    parser.add_argument('--pin-memory', default=True, type=bool)
    parser.add_argument('--logwandb', default=False, action='store_true')
    parser.add_argument('--project', type=str, default='defect-project')
    parser.add_argument('--entity', type=str, default='wayne833')
    parser.add_argument('--task', type=str, default='classification')

    args = parser.parse_args()

    best_error = 1e6 if args.task == 'regression' else 0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dataset = DefectCalcDataset(task=args.task)
    data0 = dataset[0]

    n_data = len(dataset)
    train_split = int(n_data * args.train_ratio)
    dataset_train, dataset_val = random_split(
        dataset,
        [train_split, len(dataset) - train_split],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory,
    )

    model = CGCNN(data0.x.size(-1), data0.edge_attr.size(-1), task=args.task)
    model.to(device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer must be SGD or Adam.')

    loss_criterion = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.CrossEntropyLoss()

    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        best_error = checkpoint['best_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    if args.logwandb:
        wandb.init(project=args.project, entity=args.entity)
        wandb.watch(model, log='all', log_freq=100)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(model, device, train_loader, loss_criterion, optimizer, epoch, args.logwandb, args.task)
        val_error = validate(model, device, val_loader, loss_criterion, args.logwandb, args.task)
        # scheduler.step()

        if args.task == 'regression':
            is_best = val_error < best_error
            best_error = min(val_error, best_error)
        else:
            is_best = val_error > best_error
            best_error = max(val_error, best_error)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def class_eval(prediction, target_label):
    prediction = np.exp(prediction)
    pred_label = np.argmax(prediction, axis=1)
    assert prediction.shape[1] == 2
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average='binary')
    auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
    accuracy = metrics.accuracy_score(target_label, pred_label)

    return accuracy, precision, recall, fscore, auc_score


if __name__ == '__main__':
    main()


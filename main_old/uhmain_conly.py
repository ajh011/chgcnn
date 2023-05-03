import time
import torch, shutil, argparse
import torch.optim as optim
import numpy as np
import wandb
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from uh_model_conly import HeteroRelConv 
import torch_geometric.transforms as T

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle




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


def train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer, epoch, task):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accus],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        output = model(data.x_dict, data.edge_index_dict)
        if task == 'regression':
            target = data.y.view(1)
        else:
            target = data.y.long()
        loss = loss_criterion(output, target)
        accu = accuracy_criterion(output, target)

        losses.update(loss.item(), target.size(0))
        accus.update(accu.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
        wandb.log({'train-mse': losses.val, 'train-mse-avg': losses.avg, 'train-mae': accus.val, 'train-mae-avg': accus.avg, 'epoch': epoch, 'batch-time': batch_time.val}) 
    return losses.avg, accus.avg


def validate(model, device, test_loader, loss_criterion, accuracy_criterion, task):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, accus],
        prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):

            data = data.to(device, non_blocking=True)
            output = model(data.x_dict, data.edge_index_dict)
            if task == 'regression':
                target = data.y.view(1)
            else:
                target = data.y.long()

            loss = loss_criterion(output,target)
            accu = accuracy_criterion(output,target)

            losses.update(loss.item(), target.size(0))
            accus.update(accu.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
            wandb.log({'val-mse': losses.val, 'val-mse-avg': losses.avg, 'val-mae': accus.val, 'val-mae-avg': accus.avg, 'i': i, 'batch-time': batch_time.val}) 
    return accus.avg


def main():
    parser = argparse.ArgumentParser(description='CGCNN')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='trainning ratio')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='Adam',
                        help='choose an optimizer, SGD or Adam, (default:Adam)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--task', default='regression', type=str)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--drop-last', default=False, type=bool)
    parser.add_argument('--pin-memory', default=True, type=bool)
    parser.add_argument('--dir', default='cifs', type=str)


    args = parser.parse_args()

    best_accu = 1e6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)


    torch.backends.cudnn.benchmark = True
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="chgcnn",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "TransformerConv" ,
        "dataset": "Formation Energy per Atom",
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }
    )



#########################################################################################################################################

    pkl_loc = 'und_hetero_relgraph_list.pkl'
    with open(pkl_loc, 'rb') as storage:
        relgraphs = pickle.load(storage)

    print(f'Importing data from {pkl_loc}')
    #### CLEANING UP MESSY HETERODATA
    dataset = []
    for graph in relgraphs:
        graph['atom'].x  = graph.x_dict['atom'].float()
        graph['bond'].x  = graph.x_dict['bond'].float()
        graph['motif'].x  = graph.x_dict['motif'].float()
        dataset.append(graph)
    print('Initializing model...') 
    ##Convert to undirected edges
    #dataset = [T.ToUndirected()(data) for data in dataset]
    data0 = dataset[0]
    data0.to(device)
    model = HeteroRelConv().to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data0.x_dict, data0.edge_index_dict)
#####################################################################################################################################

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
        pin_memory=False, #args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    #######################################################################################################################################

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer must be SGD or Adam.')

    if args.task == 'regression':
        loss_criterion = torch.nn.MSELoss()
        accuracy_criterion = torch.nn.L1Loss()
        print('Using MSE accuracy and L1 for training loss')
    elif args.num_class == 2 and args.task != 'regression':
        print('CLASSIFICATION NOT CURRENTLY SUPPORTED\n\n\n\nIGNORE FOLLOWING')
        loss_criterion = torch.nn.NLLLoss()
        accuracy_criterion = torch.nn.NLLLoss()
        print('Using NLL for training loss and accuracy')
    else:
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_criterion = torch.nn.CrossEntropyLoss()
        print('Using cross entropy for training loss and accuracy')

    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_accu = train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer,
                                       epoch, args.task)
        val_accu = validate(model, device, val_loader, loss_criterion, accuracy_criterion, args.task)
        # scheduler.step()

        is_best = train_accu < best_accu
        best_accu = min(train_accu, best_accu)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }, is_best)



    ckpt = torch.load('model_best.pth.tar')
    print(ckpt['best_accu'])
    wandb.finish()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()

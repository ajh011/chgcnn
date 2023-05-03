import time
import torch, shutil, argparse
import torch.optim as optim
import numpy as np
import wandb
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from hetero_model import HeteroRelConv 
import torch_geometric.transforms as T
import torch.nn.functional as F


try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeteroRelConv().to(device)


with open('hetero_relgraph_list_test.pkl', 'rb') as storage:
    relgraphs = pickle.load(storage)

dataset = relgraphs
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out,data.y)
    loss.backward()
    optimizer.step()
    return float(loss)

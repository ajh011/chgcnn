import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero


class HeteroRelConv(torch.nn.Module):
    def __init__(self, h_dim = 64, n_layers = 3, orders = ['atom','bond','motif']):
        super().__init__()

        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = HeteroConv({
                ('cell','contains','atom'): SAGEConv((-1,-1), h_dim),
                ('cell','contains','bond'): SAGEConv((-1,-1), h_dim),
                ('cell','contains','motif'): SAGEConv((-1,-1), h_dim),
                }, aggr='sum')
            self.convs.append(conv)
        self.lins = torch.nn.ModuleList()
        for i in range(len(orders)):
            self.lins.append(nn.Linear(-1,h_dim))
        self.proj = nn.Linear(h_dim,h_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(h_dim,1)
 
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict,edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        x = x_dict['cell']
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       
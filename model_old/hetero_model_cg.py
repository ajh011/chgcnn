import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero


class HeteroRelConv(torch.nn.Module):
    def __init__(self, adim = 92, bdim = 7, mdim=35, cdim = 64 ,h_dim = 64, n_layers = 1, orders = ['atom','bond','motif']):
        super().__init__()

        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = HeteroConv({
                ('atom','bonds','atom'): CGConv(h_dim),
                ('atom','in','bond'): CGConv(h_dim),
                ('atom','in','motif'): CGConv(h_dim),
                ('bond','touches','bond'): CGConv(h_dim),
                ('bond','in','motif'): CGConv(h_dim),
                ('motif','touches','motif'): CGConv(h_dim),
                ('atom','in','cell'): CGConv(h_dim),
                ('bond','in','cell'): CGConv(h_dim),
                ('motif','in','cell'): CGConv(h_dim),
                }, aggr='sum')
            self.convs.append(conv)
        self.lina = nn.Linear(adim,h_dim)
        self.linb = nn.Linear(bdim,h_dim)
        self.linm = nn.Linear(mdim,h_dim)
        self.proj = nn.Linear(h_dim,h_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(h_dim,1)
 
    def forward(self, x_dict, edge_index_dict):
        x_dict['atom'] = self.lina(x_dict['atom'])
        x_dict['bond'] = self.linb(x_dict['bond'])
        x_dict['motif']= self.linm(x_dict['motif'])
        for conv in self.convs:
            print(x_dict['atom'][0])
            x_dict = conv(x_dict,edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        x = x_dict['cell']
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

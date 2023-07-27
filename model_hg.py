import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv, HypergraphConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero


class HeteroRelConv(torch.nn.Module):
    def __init__(self, h_dim = 35, hout_dim = 64, n_layers = 3):
        super().__init__()

        self.embed = nn.Linear(92, 35)
        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = HypergraphConv(-1, h_dim, use_attention = True)
            self.convs.append(conv)
        self.proj = nn.Linear(h_dim,hout_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(hout_dim,1)
 
    def forward(self, x, hyperedge_index, hyperedge_attr, batch):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
        x = scatter(x, batch, dim=0, reduce='mean')
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero
from torch_scatter import scatter

class HeteroRelConv(torch.nn.Module):
    def __init__(self, h_dim = 128, hout_dim = 256, n_layers = 3, orders = ['atom','bond','motif','cell'], a_dim= 92, b_dim=25, m_dim=35, c_dim=64):
        super().__init__()
        
        self.orders = orders

        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = HeteroConv({
                ('bond','contains','atom'): TransformerConv(h_dim, h_dim, batch_norm=True),
                ('atom','bonds','atom'): TransformerConv(h_dim, h_dim, batch_norm=True),
#                ('atom','in','cell'): CGConv(h_dim, batch_norm=True),
#                ('bond','in','cell'): CGConv(h_dim, batch_norm=True),
#                ('motif','in','cell'): CGConv(h_dim, batch_norm=True),
#                ('cell','contains','atom'): CGConv(h_dim, batch_norm=True),
#                ('cell','contains','bond'): CGConv(h_dim , batch_norm=True),
#                ('cell','contains','motif'): CGConv(h_dim , batch_norm=True), 
                }, aggr='mean')
            self.convs.append(conv)
        for i in [a_dim,b_dim]:
            self.lins.append(nn.Linear(i,h_dim))
        self.proj = nn.Linear(h_dim,hout_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(hout_dim,1)
 
    def forward(self, x_dict, edge_index_dict, batch):
        x_dict = {key: lin(x) for (key, x), lin in zip(x_dict.items(),self.lins)}
        for conv in self.convs:
            bond_x = x_dict['bond']
            x_dict = conv(x_dict,edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict['bond'] = bond_x 
#        x = x_dict['cell']
        x = x_dict['atom']
        x = scatter(x, batch, dim=0, reduce='mean') 
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

import copy
import torch
import numpy as np
from torch_scatter import scatter
from chgconv import CHGConv
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv, HypergraphConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero



import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr


class CHGConv(MessagePassing):
    def __init__(self, node_fea_dim=92, hedge_fea_dim=35, out_dim=92, hedge_agg_method = 'mean', node_agg_method = 'mean', batch_norm = False):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_f = Linear(2*node_fea_dim+hedge_fea_dim, out_dim)
        self.lin_c = Linear(2*node_fea_dim+hedge_fea_dim, out_dim)

        if batch_norm == True:
            self.bn_f = BatchNorm1d(out_dim)
            self.bn_c = BatchNorm1d(out_dim)
            self.bn_o = BatchNorm1d(out_dim)

        if hedge_agg_method == 'mean':
            self.hedge_agg = aggr.MeanAggregation()

        if node_agg_method == 'mean':
            self.node_agg = aggr.MeanAggregation()


    def forward(self, x, hedge_index, hedge_attr):
        xs = []
        print(f'x0: {x[0]}')
        for i in hedge_index[0]:
            xs.append(x[i])
        hedge_index_xs = torch.stack(xs, dim = 0)
        hedge_index_xs = self.hedge_agg(hedge_index_xs, hedge_index[1], dim = 0)

        print(f'Num hedges: {torch.max(hedge_index[1])}')
        print(f'Num nodes: {torch.max(hedge_index[0])}')
        print(f'hedge_index: {hedge_index.shape}')
        print(f'index_xs_aggr: {hedge_index_xs.shape}')
        print(f'x: {x.shape}')
        print(f'hedge_attr: {hedge_attr.shape}')
        message_holder = []
        for x_index, h_index in zip(hedge_index[0],hedge_index[1]):
            z = torch.cat([x[x_index], hedge_attr[h_index], hedge_index_xs[h_index]], dim =0)
            message_holder.append(z)
#            print(f'lin f shape: {self.lin_f}')
#            print(f'z shape: {z.shape}')
#            print(f'z: {z}')

        message_holder = torch.stack(message_holder, dim = 0)
        z = self.node_agg(message_holder, hedge_index[0], dim = 0)

        print(f'z after agg: {z.shape}')
        z_f = self.lin_f(z)
        z_c = self.lin_c(z)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        out = z_f.sigmoid() * F.softplus(z_c)
        if self.batch_norm == True:
            out = self.bn_o(out)
        out = F.softplus(out + x)
        return out



class CrystalHypergraphConv(torch.nn.Module):
    def __init__(self, h_dim = 64, hout_dim = 128, n_layers = 3):
        super().__init__()

        self.embed = nn.Linear(92, h_dim)
        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = CHGConv(node_fea_dim = h_dim, out_dim = h_dim)
            self.convs.append(conv)
        self.proj = nn.Linear(h_dim,hout_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(hout_dim,1)
 
    def forward(self, x, hyperedge_index, hyperedge_attr, batch):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, hyperedge_index, hyperedge_attr)
        print(hyperedge_attr[0])
        
        x = scatter(x, batch, dim=0, reduce='mean')
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

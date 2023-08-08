import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv, HypergraphConv
import torch_geometric.nn as nn
from torch_geometric.nn import to_hetero

import time


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
        time1 = time.perf_counter()
        '''
        x:            torch tensor (of type float) of node attributes

                      [[node1_feat],[node2_feat],...]
                      dim([num_nodes,node_fea_dim])

        hedge_index:  torch tensor (of type long) of
                      hyperedge indices (as in HypergraphConv)

                      [[node_indxs,...],[hyperedge_indxs,...]]
                      dim([2,num nodes in all hedges])

        hedge attr:   torch tensor (of type float) of
                      hyperedge attributes (with first index algining with 
                      hedges overall hyperedge_indx in hedge_index)

                      [[hedge1_feat], [hedge2_feat],...]
                      dim([num_hedges,hyperedge_feat_dim])
        '''
        '''
        The goal is to generalize the CGConv gated convolution structure to hyperedges. The 
        primary problem with such a generalization is the variable number of nodes contained 
        in each hyperedge (hedge). I propose we simply aggregate the nodes contained within 
        each hedge to complete the message, and then concatenate that with the hyperedge feature 
        to form the message.

        Below, I build a list of the node attributes of all contained 
        within each hedge. These node attributes are then aggregated according to the 
        hyperedge indices.
        '''
        xs = []
        for i in hedge_index[0]:
            xs.append(x[i])
        time2 = time.perf_counter()
        print(f'hedge_xs comp time: {time2-time1}')
        hedge_index_xs = torch.stack(xs, dim = 0)
        hedge_index_xs = self.hedge_agg(hedge_index_xs, hedge_index[1], dim = 0)

        '''
        To finish forming the message, I loop through all x_indxs listed in the hedge_index and 
        concatenate the origin node feature with the hedge feature and the corresponding aggregate neighborhood
        feature. This is currently very SLOW!!! accounts for probably 80%-90% of compute time.
        '''

        time3 = time.perf_counter()
        print(f'hedge_x agg time: {time3-time2}')
        message_holder = []
        for x_index, h_index in zip(hedge_index[0],hedge_index[1]):
            z = torch.cat([x[x_index], hedge_attr[h_index], hedge_index_xs[h_index]], dim =-1)
            message_holder.append(z)

        '''
        We then can aggregate the messages and add to node features after some activation 
        functions and linear layers.
        '''

        time4 = time.perf_counter()
        print(f'message cat time: {time4-time3}')
        message_holder = torch.stack(message_holder, dim = 0)
        z = self.node_agg(message_holder, hedge_index[0], dim = 0)
        
        time5 = time.perf_counter()
        print(f'message agg time: {time5-time4}')
        z_f = self.lin_f(z)
        z_c = self.lin_c(z)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        out = z_f.sigmoid() * F.softplus(z_c)
        if self.batch_norm == True:
            out = self.bn_o(out)
        out = F.softplus(out + x)

        time6 = time.perf_counter()
        print(f'gate time: {time6-time5}')
        print(f'(total) out time: {time6-time1}')
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
        
        x = scatter(x, batch, dim=0, reduce='mean')
        x = self.proj(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

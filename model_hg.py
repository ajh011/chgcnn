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
    def __init__(self, node_fea_dim=92, hedge_fea_dim=35, out_dim=92, batch_norm = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_f1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim)
        self.lin_c1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim)
        self.lin_f2 = Linear(node_fea_dim+hedge_fea_dim, out_dim)
        self.lin_c2 = Linear(node_fea_dim+hedge_fea_dim, out_dim)

        self.aggr = aggr.MeanAggregation()

        if batch_norm == True:
            self.bn_f = BatchNorm1d(hedge_fea_dim)
            self.bn_c = BatchNorm1d(hedge_fea_dim)

            self.bn_o = BatchNorm1d(out_dim)

    def forward(self, data):
        x = data.x
        hyperedge_index = data.hyperedge_index
        hedge_attr = data.hyperedge_attr
        '''
        x:              torch tensor (of type float) of node attributes

                        [[node1_feat],[node2_feat],...]
                        dim([num_nodes,node_fea_dim])

        hedge_index:    torch tensor (of type long) of
                        hyperedge indices (as in HypergraphConv)

                        [[node_indxs,...],[hyperedge_indxs,...]]
                        dim([2,num nodes in all hedges])

        hedge attr:     torch tensor (of type float) of
                        hyperedge attributes (with first index algining with 
                        hedges overall hyperedge_indx in hedge_index)
  
                        [[hedge1_feat], [hedge2_feat],...]
                        dim([num_hedges,hyperedge_feat_dim])

        node_hedge_adj: torch tensor (of type sparse) of node_hedge adjacency 
                        (a redundant hedge_index) that greatly speeds up message 
                        forming. Each row represents the nodes contained in 
                        a hedge (nonzero if in hedge, zero if not), and is normalized 
                        to one so that multiplication by this matrix represents 
                        averaging over all node attributes contained in each hedge

                        [[node1_in_hedge1, node2_in_hedge1,...],[node1_in_hedge2,...],...]
                        dim([num_hedges, num_nodes])
        )
        '''

        '''
        The goal is to generalize the CGConv gated convolution structure to hyperedges. The 
        primary problem with such a generalization is the variable number of nodes contained 
        in each hyperedge (hedge). I propose we simply aggregate the nodes contained within 
        each hedge to complete the message, and then concatenate that with the hyperedge feature 
        to form the message.

        Below, I multiply the node_hedge_adj matrix by the x matrix, effectively aggregating all 
        attributes of contained nodes in each hyperedge
        '''

        hedge_index_xs = x[hyperedge_index[0]]
        hedge_index_xs = self.aggr(hedge_index_xs, hyperedge_index[1])

        '''
        To finish forming the message, I concatenate these aggregated neighborhoods with their 
        corresponding hedge features.
        '''

        message_holder = torch.cat([hedge_index_xs, hedge_attr], dim = 1)
        '''
        We then can aggregate the messages and add to node features after some activation 
        functions and linear layers.
        '''
        z_f = self.lin_f1(message_holder)
        z_c = self.lin_c1(message_holder)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        hyperedge_attrs = z_f.sigmoid() * F.softplus(z_c)
        x_i = x[hyperedge_index[0]]  # Target node features
        x_j = hyperedge_attrs[hyperedge_index[1]]  # Source node features
        z = torch.cat([x_i,x_j], dim=-1)  # Form reverse messages (for origin node messages)
        out = self.lin_f2(z).sigmoid()*F.softplus(self.lin_c2(z)) # Apply CGConv like structure
        out = self.aggr(out, hyperedge_index[0]) #aggregate according to node
 
        #out = self.propagate(hyperedge_index, x=x, size=(num_hedges, num_nodes), flow='source_to_target') # Propagate hyperedge attributes to node features
        #out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_hedges, num_nodes))
        if self.batch_norm == True:
            out = self.bn_o(out)
        out = F.softplus(out + x)
        data.x = out
        data.hyperedge_attr = hyperedge_attrs

        return data



class CrystalHypergraphConv(torch.nn.Module):
    def __init__(self, h_dim = 64, hout_dim = 128, n_layers = 3):
        super().__init__()

        self.embed = nn.Linear(92, h_dim)
        self.convs = torch.nn.ModuleList() 
        for i in range(n_layers):
            conv = CHGConv(node_fea_dim = h_dim, out_dim = h_dim)
            self.convs.append(conv)
        self.l1 = nn.Linear(h_dim, h_dim)
        self.l2 = nn.Linear(h_dim,hout_dim)
        self.activation = torch.nn.Softplus()
        self.out = nn.Linear(hout_dim,1)
 
    def forward(self, data):

        data.x = self.embed(data.x)
        for conv in self.convs:
            data = conv(data)
#        x = self.l1(x) 
#        x = self.activation(x)
        x = scatter(data.x, data.batch, dim=0, reduce='mean')
        x = self.l2(x)
        x = self.activation(x)
        output = self.out(x)
        return output


       

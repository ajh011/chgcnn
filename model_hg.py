import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, SAGEConv, TransformerConv, CGConv, HypergraphConv
import torch_geometric.nn as nn
import torch
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

        self.lin_f1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim+node_fea_dim)
        self.lin_c1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim)
        self.lin_f2 = Linear(2*node_fea_dim+hedge_fea_dim, 2*out_dim)

        self.softplus_hedge = torch.nn.Softplus()
        self.sigmoid_filter = torch.nn.Sigmoid()
        self.softplus_core = torch.nn.Softplus()
        self.softplus_out = torch.nn.Softplus()


        self.hedge_aggr = aggr.SoftmaxAggregation(learn = True)
        self.node_aggr = aggr.SoftmaxAggregation(learn = True)

        if batch_norm == True:
            self.bn_f = BatchNorm1d(node_fea_dim)
            self.bn_c = BatchNorm1d(node_fea_dim)

            self.bn_o = BatchNorm1d(out_dim)

    def forward(self, x, hyperedge_index, hedge_attr, num_nodes):
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

        '''

        '''
        The goal is to generalize the CGConv gated convolution structure to hyperedges. The 
        primary problem with such a generalization is the variable number of nodes contained 
        in each hyperedge (hedge). I propose we simply aggregate the nodes contained within 
        each hedge to complete the message, and then concatenate that with the hyperedge feature 
        to form the message.

        Below, the node attributes are first placed in order with their hyperedge_indices
        and then aggregated according to their hyperedges to form a component of the message corresponding to 
        each hyperedge
        '''

        hedge_index_xs = x[hyperedge_index[0]]
        hedge_index_xs = self.hedge_aggr(hedge_index_xs, hyperedge_index[1])

        '''
        To finish forming the message, I concatenate these aggregated neighborhoods with their 
        corresponding hedge features.
        '''

        message_holder = torch.cat([hedge_index_xs, hedge_attr], dim = 1)
        '''
        We then can aggregate the messages and add to node features after some activation 
        functions and linear layers.
        '''
        hyperedge_attrs = self.lin_c1(message_holder)
        hyperedge_attrs = self.softplus_hedge(hedge_attr + hyperedge_attrs)
        message_holder = self.lin_f1(message_holder)
        x_i = x[hyperedge_index[0]]  # Target node features
        x_j = message_holder[hyperedge_index[1]]  # Source node features
        z = torch.cat([x_i,x_j], dim=-1)  # Form reverse messages (for origin node messages)
        z = self.lin_f2(z)
        z_f, z_c = z.chunk(2, dim = -1)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        out = self.sigmoid_filter(z_f)*self.softplus_core(z_c) # Apply CGConv like structure
        out = self.node_aggr(out, hyperedge_index[0], dim_size = num_nodes) #aggregate according to node
 
        #out = self.propagate(hyperedge_index, x=x, size=(num_hedges, num_nodes), flow='source_to_target') # Propagate hyperedge attributes to node features
        #out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_hedges, num_nodes))
        if self.batch_norm == True:
            out = self.bn_o(out)

        out = self.softplus_out(out + x)

        return out, hyperedge_attrs



class CrystalHypergraphConv(torch.nn.Module):
    def __init__(self, classification, h_dim = 64, hedge_dim=40, hout_dim = 128, n_layers = 3):
        super().__init__()

        self.classification = classification

        self.embed = nn.Linear(92, h_dim)
        self.bembed = nn.Linear(35, hedge_dim)
        self.tembed = nn.Linear(35, hedge_dim)
        self.membed = nn.Linear(35, hedge_dim)
        self.bconvs = torch.nn.ModuleList() 
        self.tconvs = torch.nn.ModuleList() 
        self.mconvs = torch.nn.ModuleList() 
        for i in range(n_layers):
            bconv = CHGConv(node_fea_dim = h_dim, out_dim = h_dim, hedge_fea_dim = hedge_dim)
            mconv = CHGConv(node_fea_dim = h_dim, out_dim = h_dim, hedge_fea_dim = hedge_dim)
            tconv = CHGConv(node_fea_dim = h_dim, out_dim = h_dim, hedge_fea_dim = hedge_dim)
            self.bconvs.append(bconv)
            self.mconvs.append(mconv)
            self.tconvs.append(tconv)
        self.l1 = nn.Linear(h_dim, h_dim)
        self.l2 = nn.Linear(h_dim,hout_dim)
        self.activation = torch.nn.Softplus()
        if self.classification:
            self.out = nn.Linear(hout_dim, 2)
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout()
        else:
            self.out = nn.Linear(hout_dim,1)
 
    def forward(self, data):
        num_nodes = data.num_nodes
        batch = data.batch
        x = data.x
        motif_hyperedge_index = data.motif_hyperedge_index
        #triplet_hyperedge_index = data.triplet_hyperedge_index
        bond_hyperedge_index = data.bond_hyperedge_index
        motif_hyperedge_attr = self.membed(data.motif_hyperedge_attr)
        #triplet_hyperedge_attr = self.tembed(data.triplet_hyperedge_attr)
        bond_hyperedge_attr = self.bembed(data.bond_hyperedge_attr)
        x = self.embed(x)
        for bconv,mconv,tconv in zip(self.bconvs,self.mconvs,self.tconvs):
            x, bond_hyperedge_attr = bconv(x, bond_hyperedge_index, bond_hyperedge_attr, num_nodes)
            #x, triplet_hyperedge_attr = tconv(x, triplet_hyperedge_index, triplet_hyperedge_attr, num_nodes)
            x, motif_hyperedge_attr = mconv(x, motif_hyperedge_index, motif_hyperedge_attr, num_nodes)
            x = x.relu()
        x = scatter(x, batch, dim=0, reduce='mean')
        x = self.l2(x)
        if self.classification:
            x = self.dropout(x)
        x = self.activation(x)
        output = self.out(x)
        if self.classification:
            output = self.sigmoid(output)
        return output


       

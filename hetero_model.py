import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import SAGEConv, TransformerConv, CGConv
import torch.nn as nn


class CHGCNN(nn.Module):

    def __init__(self, conv_arch='cg',  node_dim=64, num_layers=3, h_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(-1, node_dim)

        conv_layer = CGConv((node_dim, node_dim))
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        self.fc = nn.Linear(node_dim, h_dim)
        self.activation = nn.Softplus()
        self.fc_out = nn.Linear(h_dim, 1)

    def forward(self, x, edge_index):
        x = embedding(x)

        # graph convolution layers
        for layer in self.layers:
            x = layer(x, edge_index)



        # graph level mean pooling
        #x = torch.mean(x, dim = 0)
        x = scatter(x, data.batch, dim=0, reduce='mean')
       
 
        # output layers
        x = self.fc(x)
        x = self.activation(x)
        x = self.fc_out(x)

        return x

import copy
import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import SAGEConv
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self, atom_fea_dims,  node_dim=64, num_layers=3, h_dim=128):
        super().__init__()

        self.atom_fea_dims = atom_fea_dims
        self.node_dim = node_dim
        self.num_layers = num_layers

        self.embeddings = []

        for atom_fea_dim in atom_fea_dims:
            self.embeddings.append(nn.Linear(atom_fea_dim, node_dim))

        conv_layer = SAGEConv(node_dim, node_dim)
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        self.fc = nn.Linear(node_dim, h_dim)
        self.activation = nn.Softplus()
        self.fc_out = nn.Linear(h_dim, 1)

    def forward(self, data):
        feats, edge_index = data.feats, data.edge_index
        xs = []
        # an embedding reducing the input dimension from atom_fea_dim to node_dim
        for feat, embedding in zip(feats[0],self.embeddings):
            subset = []
            for i in feat[1]:
                x = torch.tensor(np.nan_to_num(np.array(i, dtype=float))).float()
                x = embedding(x)
                subset.append(x)
            prop_type = torch.stack(subset, axis = -1)
            xs.append(prop_type)


        x = torch.cat(xs, axis = -1)
        x = torch.swapaxes(x, 0 , 1)

        # graph convolution layers
        for layer in self.layers:
            x = layer(x, edge_index)



        # graph level mean pooling
        x = torch.mean(x, dim = 0)
        #x = scatter(x, data.batch, dim=0, reduce='mean')
       
 
        # output layers
        x = self.fc(x)
        x = self.activation(x)
        x = self.fc_out(x)

        return x

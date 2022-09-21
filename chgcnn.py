import copy

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv 
from torch_scatter import scatter


class CHGCNN(nn.Module):
    def __init__(self, atom_fea_dim, edge_dim, node_dim=64, num_layers=3, h_dim=128,
                 task='regression', num_class=2, pool='mean'):
        super().__init__()

        self.atom_fea_dim = atom_fea_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(atom_fea_dim, node_dim)

        conv_layer = HypergraphConv(node_dim, node_dim, batch_norm=True)
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        #Regression test case
        self.fc = nn.Linear(node_dim, h_dim)
        self.softplus = nn.Softplus()
        self.fc_out = nn.Linear(h_dim, 1)

    def forward(self, data):
        x = self.embedding(data.x)
        for layer in self.layers:
            x = layer(x, data.hyperedge_index)
            #Add hypergraph attributes later
        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = self.softplus(x)
        x = self.fc(x)
        x = self.softplus(x)
        x = self.fc_out(x)

        return x


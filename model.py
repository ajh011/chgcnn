import copy
from torch_scatter import scatter
from torch_geometric.nn.conv import CGConv
import torch.nn as nn
import torch.nn.functional as F

import torch



class max_graph_node_features(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ex_lis, splitter):
        crystal = []
        loop = []
        crystal_set = []
        a=0
        for i, j in zip(ex_lis, splitter):
            if j==a:
                crystal.append(i)
            else:
                crystal = torch.stack(crystal, dim = -1)
                crystal = torch.max(crystal, dim = -1)[0]
                crystal_set.append(crystal)
                crystal = []
                a+=1
                crystal.append(i)
        crystal = torch.stack(crystal, dim = -1)
        crystal = torch.max(crystal, dim = -1)[0]
        crystal_set.append(crystal)
        return torch.stack(crystal_set, dim=0)
        
class ProjectionLayer(nn.Module):
    def __init__(self, feats, hidden_dims):
        input_dims = []
        for feat_type in feats:
            super(ProjectionLayer, self).__init__()
            input_dims.append(len(feat_type[1][0]))
        self.linears = nn.ModuleList([nn.Linear(in_dim, hidden_dims, dtype=float) for in_dim in input_dims])
    def forward(self, feats):
        feat_homog = []
        for feat_type, feat_proj in zip(feats, self.linears):
            feat_homog.append(feat_proj(feat_type))
        feat_list = torch.cat(feat_homog, dim = 0)
        return feat_list.float()



class MyModel(nn.Module):

    def __init__(self, atom_fea_dim,  edge_dim, node_dim=64, num_layers=3, h_dim=128, task = 'regression', num_class=2):
        super().__init__()

        self.atom_fea_dim = atom_fea_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.task = task
        self.num_class = num_class

        self.embedding = nn.Linear(atom_fea_dim, node_dim)

        conv_layer = CGConv(node_dim, edge_dim, batch_norm=True)
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        self.pool = max_graph_node_features()

        self.fc = nn.Linear(node_dim, h_dim)
        self.activation = nn.Softplus()
        if task == 'regression':
            self.fc_out = nn.Linear(h_dim, 1)
        else:
            self.fc_out = nn.Linear(h_dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # an embedding reducing the input dimension from atom_fea_dim to node_dim
        x = self.embedding(x)

        # graph convolution layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # graph level max pooling
        #x = scatter(x, data.batch, dim=0, reduce='max')
        x = self.pool(x, data.batch)

        # output layers
        x = self.fc(x)
        x = self.activation(x)
        x = self.fc_out(x)
        if self.task != 'regression' and self.num_class==2:
            x = F.log_softmax(x, dim=1)
        else:
            pass
        return x

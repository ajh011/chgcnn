
import copy

from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class CGAGGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (str, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F_{t})` if bipartite
    """
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 35,
                 aggr: str = 'add', batch_norm: bool = True,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(int(sum(channels)/2 + dim), channels[1], bias=bias)
        self.lin_s = Linear(int(sum(channels)/2 + dim), channels[1], bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1]
        return out


    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = 0.5*(x_i + x_j)
        else:
            z = torch.cat([0.5*(x_i+x_j), edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'


class CrystalGraphConv(nn.Module):

    def __init__(self, atom_fea_dim=92, edge_dim=35, node_dim=64, num_layers=3, h_dim=128, classification=False, num_class=2):
        super().__init__()

        self.atom_fea_dim = atom_fea_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.classification = classification

        self.embedding = nn.Linear(atom_fea_dim, node_dim)
        self.l1 = nn.Linear(node_dim, node_dim)

        conv_layer = CGAGGConv(node_dim, edge_dim, batch_norm=True)
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        if not self.classification:
            self.fc = nn.Linear(node_dim, h_dim)
            self.softplus = nn.Softplus()
            self.fc_out = nn.Linear(h_dim, 1)
        else:
            self.fc = nn.Linear(node_dim, h_dim)
            self.softplus = nn.Softplus()
            self.fc_out = nn.Linear(h_dim, num_class)
            self.dropout = nn.Dropout()

    def forward(self, data):
        x = self.embedding(data.x)

        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)

        x = self.l1(x)
        x = scatter(x, data.batch, dim=0, reduce='mean')

        if not self.classification:
            x = self.softplus(x)
            x = self.fc(x)
            x = self.softplus(x)
            x = self.fc_out(x)
        else:
            x = self.softplus(x)
            x = self.fc(x)
            x = self.softplus(x)
            x = self.dropout(x)
            x = self.fc_out(x)
            x = F.log_softmax(x, dim=1)

        return x    

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr


class CHGConv(MessagePassing):
    def __init__(self, node_fea_dim, hedge_fea_dim, hedge_agg_method = 'mean', node_agg_method = 'mean'):
        super().__init__(aggr=agg_method)
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_f = Linear(2*node_fea_dim+hedge_fea_dim, node_fea_dim)
        self.lin_c = Linear(2*node_fea_dim+hedge_fea_dim, node_fea_dim)
        self.bn_f = BatchNorm1d(node_fea_dim)
        self.bn_c = BatchNorm1d(node_fea_dim)
        self.bn_o = BatchNorm1d(node_fea_dim)

        if hedge_agg_method == 'mean':
            self.hedge_agg = aggr.MeanAggregation()

        if node_agg_method == 'mean':
            self.node_agg = aggr.MeanAggregation()

    def aggregate_hedge_xs(x, hedge_index):
        xs = []
        for i in hedge_index[0]:
            xs.append(x[i])

        hedge_index_xs = torch.cat(xs, 0)
        hedge_index_xs = self.hedge_agg(hedge_index_xs, index=hedge_index[1])
        return hedge_index_xs


    def forward(self, x, hedge_index, hedge_attr):
        message_holder = []
        hedge_index_xs = self.aggregate_hedge_xs(x, hedge_index)
        for x_index, h_index in hedge_index:
            z = torch.cat([x[x_index], hedge_attr[h_index], hedge_index_xs[h_index]], dim =-1)
            z_f = self.lin_f(z)
            z_f = self.bn_f(z_f)
            z_c = self.lin_c(z)
            z_c = self.bn_c(z_c)
            message_holder.append(z_f.sigmoid() * F.softplus(z_c))

        out = self.node_agg(message_holder, index=hedge_index[0])
            
        #out = self.propagate(hedge_index, x=x, hedge_attr=hedge_attr, hedge_index_xs=hedge_index_xs)
        out = self.bn_o(out)
        out = F.softplus(out + x)
        return out
    
         

#    def message(self, x_i, hedge_attr, hedge_index_xs):
#        z = torch.cat([x_i, x_j, hedge_attr], dim=-1)

#        z_f = self.lin_f(z)
#        z_f = self.bn_f(z_f)

#        z_c = self.lin_c(z)
#        z_c = self.bn_c(z_c)

#        return z_f.sigmoid() * F.softplus(z_c)



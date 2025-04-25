import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.utils import scatter

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels, custom_kernel=None):
        """
        MPNN Layer for 'sum' mode using fused linear operations.
        
        Here, we fuse the separate linear transformations for target and source nodes
        into a single linear layer that outputs 2*out_channels. We then split the result
        into xs (target features) and xt (source features). The edge features are processed
        separately.
        """
        super(MPNNLayer, self).__init__(aggr='add')
        # Fused linear layer for both target and source nodes.
        self.linear_1 = Linear(in_channels, out_channels)
        self.linear_2 = Linear(in_channels, out_channels)
        # Linear layer for edge attributes.
        self.linear_3 = Linear(edge_channels, out_channels)
        self.custom_kernel = custom_kernel
        self.linear_out = Linear(out_channels+in_channels, out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels


    def forward(self, x, edge_index, edge_attr, pos):
        """
        Forward pass for 'sum' mode.
        
        Args:
            x (Tensor): Node features (N, in_channels).
            edge_index (Tensor): Edge indices (2, E).
            edge_attr (Tensor): Edge features.
        
        Returns:
            Tensor: Updated node features.
        """
        x_s = self.linear_1(x)
        x_t = self.linear_2(x)

        if self.edge_channels > 0:
            edge_attr = self.linear_3(edge_attr)
        
        return self.propagate(xs=x_s, xt=x_t,x = x, edge_attr=edge_attr, pos=pos, edge_index=edge_index)

    def message(self, xs_i, xt_j, pos, edge_attr, edge_index):
        """
        Message function for 'sum' mode.
        
        Uses in-place addition to reduce temporary allocations.
        
        Args:
            xs_i (Tensor): Transformed features for target nodes.
            xt_j (Tensor): Transformed features for source nodes.
            edge_attr (Tensor): Transformed edge features.
        
        Returns:
            Tensor: Message tensor.
        """

        #euclidean distance
        if self.custom_kernel is not None:
            dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            # Reshape dist to 2d
            dist = dist.view(-1, 1)
            euclid_dist = self.custom_kernel(dist)
            
            if self.edge_channels > 0:
                euclid_dist  = torch.add(edge_attr, euclid_dist)

        
            return F.relu(torch.add(torch.add(xs_i , xt_j ), euclid_dist ))
        
        else:
            # Default behavior: sum of transformed features.
            return F.relu(torch.add(xs_i , xt_j ))

    def update(self, aggr_out, x):
        """
        Update function: Applies a ReLU non-linearity on the aggregated messages.
        
        Args:
            aggr_out (Tensor): Aggregated messages.
        
        Returns:
            Tensor: Updated node features.
        """
        return F.relu(self.linear_out(torch.cat([x, aggr_out], dim=1)))


class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=0, hidden_channels=64, aggr='add', custom_kernel=None):
        super(EGNNLayer, self).__init__(aggr=aggr)


        self.src_linear = Linear(in_channels, hidden_channels)
        self.dst_linear = Linear(in_channels, hidden_channels)
        self.edge_linear = Linear(edge_channels, hidden_channels) if edge_channels > 0 else None
        self.activation = nn.SiLU()
        if custom_kernel is not None:
            self.geometric_linear = custom_kernel
        else:
            self.geometric_linear = Linear(1, hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        x: Node features (N, F)
        pos: Node coordinates (N, 3)
        edge_index: Edge indices (2, E)
        edge_attr: Edge features (E, D) or None
        """
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # Compute squared distance
        diff = pos_i - pos_j
        radial = torch.sum(diff ** 2, dim=1, keepdim=True)

        # Concatenate features
        if edge_attr is not None:
            src = self.src_linear(x_i)
            dst = self.dst_linear(x_j)
            edge = self.edge_linear(edge_attr)
            g_dist = self.geometric_linear(radial)
            #sum the results and use silu activation
            edge_res = src + dst + edge + g_dist
            edge_res = self.activation(edge_res)
        else:
            src = self.src_linear(x_i)
            dst = self.dst_linear(x_j)
            g_dist = self.geometric_linear(radial)
            edge_res = src + dst + g_dist
            edge_res = self.activation(edge_res)

        e_ij = self.edge_mlp(edge_res)
        coord_update = diff * self.coord_mlp(e_ij)
        return {'e_ij': e_ij, 'coord_update': coord_update}

    def aggregate(self, inputs, index, dim_size):
        e_ij = scatter(inputs['e_ij'], index, dim=0, dim_size=dim_size, reduce=self.aggr)
        coord_update = scatter(inputs['coord_update'], index, dim=0, dim_size=dim_size, reduce='mean')
        return {'e_ij': e_ij, 'coord_update': coord_update}

    def update(self, aggr_out, x, pos):
        h = self.node_mlp(torch.cat([x, aggr_out['e_ij']], dim=1))
        pos = pos + aggr_out['coord_update']
        return h, pos

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

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

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

class MPNNLayerCat(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        """
        MPNN Layer for 'cat' mode.
        
        In this mode, the same node features are used for source and target nodes.
        The message function concatenates the target node features (x_i),
        source node features (x_j), and edge features, then applies a linear transformation.
        """
        super(MPNNLayerCat, self).__init__(aggr='add')  # Aggregation type: sum
        # The linear layer expects concatenated features of x_i, x_j, and edge_attr.
        self.linear = Linear((in_channels * 2) + edge_channels, out_channels)
        self.linear_out = Linear(out_channels+in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for 'cat' mode.
        
        Args:
            x (Tensor): Node features (N, F).
            edge_index (Tensor): Edge indices (2, E).
            edge_attr (Tensor): Edge features.
        
        Returns:
            Tensor: Updated node features.
        """
        # Propagate using the same x for both source and target.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Message function for 'cat' mode.
        
        Concatenates target node features (x_i), source node features (x_j),
        and edge features along the feature dimension and applies a linear transformation.
        
        Args:
            x_i (Tensor): Features of target nodes.
            x_j (Tensor): Features of source nodes.
            edge_attr (Tensor): Edge features.
        
        Returns:
            Tensor: Transformed messages.
        """
        # Concatenate features and transform.
        return F.relu(self.linear(torch.cat([x_i, x_j, edge_attr], dim=1)))

    def update(self, aggr_out, x):
        """
        Update function: Applies a ReLU non-linearity on the aggregated messages.
        
        Args:
            aggr_out (Tensor): Aggregated messages.
        
        Returns:
            Tensor: Updated node features.
        """

        return F.relu(self.linear_out(torch.cat([x, aggr_out], dim=1)))


class MPNNLayerSum(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        """
        MPNN Layer for 'sum' mode using fused linear operations.
        
        Here, we fuse the separate linear transformations for target and source nodes
        into a single linear layer that outputs 2*out_channels. We then split the result
        into xs (target features) and xt (source features). The edge features are processed
        separately.
        """
        super(MPNNLayerSum, self).__init__(aggr='add')
        # Fused linear layer for both target and source nodes.
        self.fused_linear = Linear(in_channels, out_channels * 2)
        # Linear layer for edge attributes.
        self.linear_3 = Linear(edge_channels, out_channels, bias=False)
        self.linear_out = Linear(out_channels+in_channels, out_channels)
        # Fused linear transformation for node features.
        # Create two CUDA streams
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for 'sum' mode.
        
        Args:
            x (Tensor): Node features (N, in_channels).
            edge_index (Tensor): Edge indices (2, E).
            edge_attr (Tensor): Edge features.
        
        Returns:
            Tensor: Updated node features.
        """
        
        # Execute function_1 on stream1
        #with torch.cuda.stream(self.stream1):
        fused_out = self.fused_linear(x)  # Shape: (N, 2*out_channels)
        # Split the fused output into target and source node representations.
        xs, xt = fused_out.chunk(2, dim=1)

        # Execute function_2 on stream2
        #with torch.cuda.stream(self.stream2):
        # Propagate using the fused features.
        es = self.linear_3(edge_attr)  # Edge features

        # Synchronize streams (optional, if you need to wait for completion)
        #torch.cuda.synchronize()
        
        
        return self.propagate(edge_index,x=x, xs=xs, xt=xt, edge_attr=es)

    def message(self, xs_i, xt_j, edge_attr):
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
        # In-place addition: xs_i = xs_i + xt_j + edge_attr
        
        return F.relu(xs_i.add_(xt_j).add_(edge_attr))

    def update(self, aggr_out, x):
        """
        Update function: Applies a ReLU non-linearity on the aggregated messages.
        
        Args:
            aggr_out (Tensor): Aggregated messages.
        
        Returns:
            Tensor: Updated node features.
        """
        return F.relu(self.linear_out(torch.cat([x, aggr_out], dim=1)))

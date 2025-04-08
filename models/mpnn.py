import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from models.layers import MPNNLayer
from torch_geometric.nn import GCNConv, global_mean_pool

class MPNN_GC(torch.nn.Module):
    def __init__(self, custom_kernel, in_channels, edge_channels, hidden_channels, out_channels, num_layers=2):
        """
        Initializes an MPNN model.

        Args:
            in_channels (int): Input feature size.
            hidden_channels (int): Hidden layer size.
            out_channels (int): Number of output classes.
            num_layers (int): Number of message-passing layers.
        """
        super(MPNN_GC, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(MPNNLayer(in_channels, edge_channels, hidden_channels, custom_kernel=custom_kernel))


        for _ in range(num_layers - 1):
                self.layers.append(MPNNLayer(hidden_channels, edge_channels, hidden_channels, custom_kernel=custom_kernel))

        self.output_layer = Linear(hidden_channels, out_channels)


    def forward(self, x,  edge_index, batch, edge_attr, pos):
        """
        Forward pass of MPNN.

        Args:
            x (Tensor): Node feature matrix (N, F).
            edge_index (Tensor): Edge index (2, E).

        Returns:
            Tensor: Output logits for classification.
        """

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.output_layer(x)


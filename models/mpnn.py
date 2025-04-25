import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from models.layers import MPNNLayer, EGNNLayer
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MPNN_GC(torch.nn.Module):
    def __init__(self, custom_kernel, in_channels, edge_channels, hidden_channels, out_channels, num_layers=2, task = None):
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
        self.task = task

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
        out = self.output_layer(x)

        return out


class EGNNNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_channels=0, num_layers=2, custom_kernel=None, task = None):


        super(EGNNNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            #make custom kernel a copy, so that each layer has its own copy
            custom = Linear(1, hidden_channels)
            if custom_kernel is not None:
                custom.weight = torch.nn.Parameter(custom_kernel.weight.clone())
                if custom_kernel.bias is not None:
                    custom.bias = torch.nn.Parameter(custom_kernel.bias.clone())
                else:
                    custom.bias = None
            self.layers.append(EGNNLayer(in_c, hidden_channels, edge_channels, custom_kernel=custom))
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.task = task
        print(out_channels)

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.output_mlp(x)

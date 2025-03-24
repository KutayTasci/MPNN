import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from models.layers import MPNNLayerCat, MPNNLayerSum
from torch_geometric.nn import GCNConv, global_mean_pool

class MPNN_QM9(torch.nn.Module):
    def __init__(self, in_channels, edge_channels,hidden_channels, out_channels, num_layers=2, mode='cat'):
        """
        Initializes an MPNN model.

        Args:
            in_channels (int): Input feature size.
            hidden_channels (int): Hidden layer size.
            out_channels (int): Number of output classes.
            num_layers (int): Number of message-passing layers.
        """
        super(MPNN_QM9, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        if mode == 'cat':
            print('Using cat')
            self.layers.append(MPNNLayerCat(in_channels, edge_channels, hidden_channels))
        else:
            print('Using sum')
            self.layers.append(MPNNLayerSum(in_channels, edge_channels, hidden_channels))

        for _ in range(num_layers - 1):
            if mode == 'cat':
                self.layers.append(MPNNLayerCat(hidden_channels, edge_channels, hidden_channels))
            else:
                self.layers.append(MPNNLayerSum(hidden_channels, edge_channels, hidden_channels))

        self.output_layer = Linear(hidden_channels, out_channels)


    def forward(self, x,  edge_index, batch, edge_attr):
        """
        Forward pass of MPNN.

        Args:
            x (Tensor): Node feature matrix (N, F).
            edge_index (Tensor): Edge index (2, E).

        Returns:
            Tensor: Output logits for classification.
        """

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.output_layer(x)

class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, mode="cat"):
        """
        Initializes an MPNN model.

        Args:
            in_channels (int): Input feature size.
            hidden_channels (int): Hidden layer size.
            out_channels (int): Number of output classes.
            num_layers (int): Number of message-passing layers.
        """
        super(MPNN, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        if mode == 'cat':
            self.layers.append(MPNNLayerCat(in_channels,4, hidden_channels))
        else:
            self.layers.append(MPNNLayerSum(in_channels,4, hidden_channels))


        for _ in range(num_layers - 1):
            if mode == 'cat':
                self.layers.append(MPNNLayerCat(hidden_channels,4, hidden_channels))
            else:
                self.layers.append(MPNNLayerSum(hidden_channels,4, hidden_channels))

        self.output_layer = Linear(hidden_channels, out_channels)


    def forward(self, x,  edge_index, edge_attr):
        """
        Forward pass of MPNN.

        Args:
            x (Tensor): Node feature matrix (N, F).
            edge_index (Tensor): Edge index (2, E).

        Returns:
            Tensor: Output logits for classification.
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)            
            x = F.relu(x)

        return self.output_layer(x)
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

from torch.autograd import Function
from torch.utils.cpp_extension import load





cuda_module = load(name="reverse_scatter",
                        sources=["custom_kernels/reverse_scatter.cpp", "custom_kernels/reverse_scatter.cu"])


class ReverseScatter(Function):
    @staticmethod
    def forward(ctx, input, mapping, output):
        ctx.save_for_backward(mapping)
        ctx.input_size = input.size(0)  # Save the input size for use in backward
        return cuda_module.forward(input, mapping, output)

    @staticmethod
    def backward(ctx, grad_output):
        (mapping,) = ctx.saved_tensors
        input_size = ctx.input_size
        grad_input = cuda_module.backward(grad_output.contiguous(), mapping,input_size)
        return grad_input, None, None  # grad w.r.t. input only


class EGNNLayerSum(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=0, hidden_channels=64, aggr='add'):
        super(EGNNLayerSum, self).__init__(aggr=aggr)


        self.src_linear = Linear(in_channels, hidden_channels, bias=False)
        self.dst_linear = Linear(in_channels, hidden_channels, bias=False)
        self.edge_linear = Linear(edge_channels, hidden_channels, bias=False) if edge_channels > 0 else None
        self.activation = nn.SiLU()
        self.geometric_linear = Linear(1, hidden_channels, bias=False)

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

        xi = self.src_linear(x)
        xj = self.dst_linear(x)

        edge = self.edge_linear(edge_attr)
        edge = ReverseScatter.apply(xi, edge_index[0], edge) if edge is not None else None
        edge = ReverseScatter.apply(xj, edge_index[1], edge) if edge is not None else None


        return self.propagate(edge_index,x=x, pos=pos, edge_attr=edge)

    def message(self,x, pos_i, pos_j, edge_attr):
        # Compute squared distance
        diff = pos_i - pos_j
        radial = torch.sum(diff ** 2, dim=1, keepdim=True)
        #edge = self.edge_linear(edge_attr)

        
        g_dist = self.geometric_linear(radial)
 
        e_ij = self.edge_mlp(torch.add(edge_attr, g_dist))
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
    

class EGNNLayerCat(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=0, hidden_channels=64):
        super(EGNNLayerCat, self).__init__(aggr='add')

        concat_size = (2 * in_channels) + edge_channels + 1
        self.mlp = Linear(concat_size, hidden_channels, bias=False)
        self.activation = nn.SiLU()

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
        src = torch.cat([x_i, x_j, edge_attr, radial], dim=1)

        edge_res = self.mlp(src)
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

    

class EGNN(nn.Module):
    def __init__(self, in_channels, edge_channels, hidden_channels, out_channels, num_layers=2, task = None, mode='cat'):


        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            
            layer_to_add = EGNNLayerCat if mode == 'cat' else EGNNLayerSum
            self.layers.append(layer_to_add(in_c, hidden_channels, edge_channels))

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.task = task


    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        
        if batch is not None:
            # Manually compute dim_size to avoid data-dependent guard errors
            dim_size = int(batch.max().item()) + 1
            x = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
        
        if self.task == 'regression':
            x = F.relu(x)
        elif self.task == 'classification':
            x = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")
        return self.output_mlp(x)
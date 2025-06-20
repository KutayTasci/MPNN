import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from typing import Dict

from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

# Merge bindings + dispatcher in single .cu file
with open("custom_kernels/reverse_scatter_bindings.cu", "r") as f:
    cuda_src = f.read()

load_inline(
    name="reverse_scatter",
    cpp_sources="#include <torch/extension.h>\nTORCH_LIBRARY_FRAGMENT(reverse_scatter, m) {}",  # force linkage
    cuda_sources=cuda_src,
    extra_cuda_cflags=[],
    extra_cflags=[],
    functions=[],
    verbose=True
)

@torch._dynamo.disable
def safe_reverse_scatter(input, mapping, output):
    return torch.ops.reverse_scatter.forward(input, mapping, output)


class DimeNetLayerSum(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_rbf=6, num_cbf=6):
        super(DimeNetLayerSum, self).__init__(aggr='add')
        
        self.second_node_mlp = nn.Linear(in_channels, hidden_channels, bias=False)
        self.rbf_mlp = nn.Linear(num_rbf, hidden_channels, bias=False)
        self.cbf_mlp = nn.Linear(num_cbf, hidden_channels, bias=False)

        self.edge_mlp_activation = nn.ReLU()

        
        self.first_node_mlp = nn.Linear(in_channels, hidden_channels, bias=False)
        self.aggregate_mlp = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.edge_mlp_activation_2 = nn.ReLU()
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, edge_attr, triplet_info, rbf=None):
        """
        x: Node features (N, F)
        edge_index: Edge indices (2, E)
        edge_attr: dict with 'distances': [E]
        triplet_info: dict with:
            - 'k_idx': source node of incoming edge to j
            - 'j_idx': shared middle node
            - 'i_idx': target node
            - 'angles': [T]
            - 'cbf': [T, D]
        rbf: Radial basis features per edge [E, D]
        """

        # Preprocess components used in message construction
        x_k = self.second_node_mlp(x)#[triplet_info['k_idx']]           # [T, D]
        x_j = self.rbf_mlp(rbf) if rbf is not None else 0  # [T, D] [triplet_info['j_idx']]
        cbf_emb = self.cbf_mlp(triplet_info['cbf'])                    # [T, D]

        cbf_emb = safe_reverse_scatter(x_k, triplet_info['k_idx'], cbf_emb)  # [E, D]
        triplet_msg = safe_reverse_scatter(x_j, triplet_info['j_idx'], cbf_emb)                            # [T, D]

        # Aggregate triplet messages into edge-level messages
        edge_msg = scatter(triplet_msg, triplet_info['j_idx'], dim=0, dim_size=edge_index.size(1), reduce='add')  # [E, D]

        # Edge-level processing using destination node features
        x_j_edge = self.first_node_mlp(x)            # [E, D]
        edge_out = self.edge_mlp_activation_2(safe_reverse_scatter(x_j_edge,edge_index[1] ,edge_msg))    # [E, D]

        # Node-level aggregation: sum messages for each destination node
        node_aggr = scatter(edge_out, edge_index[1], dim=0, dim_size=x.size(0), reduce='add')  # [N, D]

        # Final node update
        return self.node_mlp(torch.cat([x, node_aggr], dim=-1)) 


class DimeNetLayerCat(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_rbf=6, num_cbf=6):
        super(DimeNetLayerCat, self).__init__(aggr='add')


        self.edge_mlp = nn.Linear(in_channels + num_rbf + num_cbf, hidden_channels, bias=False)
        self.edge_mlp_act = nn.ReLU()


        self.edge_mlp_2 = nn.Linear(hidden_channels + in_channels, hidden_channels, bias=False)
        self.edge_mlp_2_act = nn.ReLU()

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, triplet_info, rbf=None):
        """
        x: Node features (N, F)
        edge_index: Edge indices (2, E)
        edge_attr: dict with 'distances': [E]
        triplet_info: dict with:
            - 'k_idx': source of incoming to j
            - 'j_idx': shared middle node
            - 'i_idx': target node
            - 'angles': [T]
        """

        self.triplet_info = triplet_info
        
        return self.propagate(edge_index, x=x, rbf=rbf)

    def message(self,x, x_j, rbf, index):
        # Triplet data: k → j → i
        k, j, i = self.triplet_info['k_idx'], self.triplet_info['j_idx'], self.triplet_info['i_idx']
        angles = self.triplet_info['angles']  # [T]

        m_kj = x[k]  # [T, F]
        rbf_feat = rbf[j]  # [T, R]
        cbf_feat = self.triplet_info['cbf']  # [T, C]
        
        m_kji = self.edge_mlp(torch.cat([m_kj, rbf_feat, cbf_feat], dim=-1))  # [T, F+R+C]
        edge_message = self.edge_mlp_act(m_kji)  # [T, H]

        # Aggregate over j to get final message for each edge j→i
        aggregated = torch.zeros_like(x_j)
        aggregated = scatter(edge_message, j, dim=0, dim_size=x_j.size(0), reduce='add')

        # Combine with original message
        final_input = torch.cat([x_j, aggregated[index]], dim=-1)
        
        final_input = self.edge_mlp_2_act(self.edge_mlp_2(final_input))  # [E, H]

        return final_input

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))
    


class DimeNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, num_rbf=6, num_cbf=6, task='regression', mode= 'cat'):
        
        super(DimeNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            layer_to_use = DimeNetLayerCat if mode == 'cat' else DimeNetLayerSum
            self.layers.append(
                layer_to_use(in_c, hidden_channels, hidden_channels, num_rbf, num_cbf)
            )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.task = task

    def forward(self, data):
        """
        data should have attributes:
        - x: Node features [N, F]
        - edge_index: Edge indices [2, E]
        - edge_attr: Dictionary with key 'distances' [E]
        - triplet_info: Dictionary with keys 'k_idx', 'j_idx', 'i_idx', 'angles'
        - batch: Batch vector [N]
        """
        
        x, edge_index = data.x, data.edge_index
        rbf = data.rbf
        edge_attr, triplet_info = data.edge_attr, data.triplet_info
        batch = data.batch
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, triplet_info, rbf=rbf)

        if batch is not None:
            # Manually compute dim_size to avoid data-dependent guard errors
            dim_size = int(batch.max().item()) + 1
            x = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')


        if self.task == 'regression':
            x = F.relu(x)
        elif self.task == 'classification':
            x = F.log_softmax(x, dim=-1)
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

        return self.output_mlp(x)
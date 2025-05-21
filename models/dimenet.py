import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class DimeNetLayerSum(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_rbf=6, num_cbf=6):
        super(DimeNetLayerSum, self).__init__(aggr='add')


        self.second_node_mlp = nn.Linear(in_channels, hidden_channels)
        self.rbf_mlp = nn.Linear(num_rbf, hidden_channels)
        self.cbf_mlp = nn.Linear(num_cbf, hidden_channels)

        self.edge_mlp_activation = nn.ReLU()

        
        self.first_node_mlp = nn.Linear(in_channels, hidden_channels)
        self.aggregate_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.edge_mlp_activation_2 = nn.ReLU()
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
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

        x_second = self.second_node_mlp(x)  # [N, H]
        x_rbf = self.rbf_mlp(rbf)
        x_cbf = self.cbf_mlp(self.triplet_info['cbf'])

        x_first = self.first_node_mlp(x)  # [N, H]
        
        return self.propagate(edge_index, x=x, y = x_second, z = x_first, x_rbf=x_rbf, x_cbf=x_cbf)

    def message(self, y, z_j, x_rbf, x_cbf, index):
        # Triplet data: k → j → i
        k, j, i = self.triplet_info['k_idx'], self.triplet_info['j_idx'], self.triplet_info['i_idx']


        m_kj = y[k]  # [T, F]
        rbf_feat = x_rbf[j]  # [T, R]
        
        #m_kji = torch.cat([m_kj, rbf_feat, cbf_feat], dim=-1)  # [T, F+R+C]
        edge_message = self.edge_mlp_activation(torch.add(torch.add(m_kj, rbf_feat), x_cbf))  # [T, H]

        # Aggregate over j to get final message for each edge j→i
        aggregated = torch.zeros_like(z_j)
        aggregated = scatter(edge_message, j, dim=0, dim_size=z_j.size(0), reduce='add')

        # Combine with original message
        final_input = torch.add(z_j, self.aggregate_mlp(aggregated)[index])  # [E, H]

        
        final_input = self.edge_mlp_activation_2(final_input)  # [E, H]

        return final_input

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class DimeNetLayerCat(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_rbf=6, num_cbf=6):
        super(DimeNetLayerCat, self).__init__(aggr='add')


        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + num_rbf + num_cbf, hidden_channels),
            nn.ReLU()
        )


        self.edge_mlp_2 = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
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
        self.edge_rbf = rbf  # [E, num_rbf]
        self.triplet_info = triplet_info
        
        return self.propagate(edge_index, x=x)

    def message(self,x, x_i, x_j, index):
        # Triplet data: k → j → i
        k, j, i = self.triplet_info['k_idx'], self.triplet_info['j_idx'], self.triplet_info['i_idx']
        angles = self.triplet_info['angles']  # [T]

        m_kj = x[k]  # [T, F]
        rbf_feat = self.edge_rbf[j]  # [T, R]
        cbf_feat = self.triplet_info['cbf']  # [T, C]
        
        m_kji = torch.cat([m_kj, rbf_feat, cbf_feat], dim=-1)  # [T, F+R+C]
        edge_message = self.edge_mlp(m_kji)  # [T, H]

        # Aggregate over j to get final message for each edge j→i
        aggregated = torch.zeros_like(x_j)
        aggregated = scatter(edge_message, j, dim=0, dim_size=x_j.size(0), reduce='add')

        # Combine with original message
        final_input = torch.cat([x_j, aggregated[index]], dim=-1)
        
        final_input = self.edge_mlp_2(final_input)  # [E, H]

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

        x = global_mean_pool(x, batch)

        if self.task == 'regression':
            x = F.relu(x)
        elif self.task == 'classification':
            x = F.log_softmax(x, dim=-1)
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

        return self.output_mlp(x)
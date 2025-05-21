import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter


class GemNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_rbf=6, num_cbf=6, num_sbf=6):
        super(GemNetLayer, self).__init__(aggr='add')

        self.edge_mlp_1hop = nn.Sequential(
            nn.Linear(in_channels + num_rbf + num_cbf + num_sbf, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.edge_mlp_2 = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, quadruplet_info, rbf=None):
        self.edge_rbf = rbf
        self.quadruplet_info = quadruplet_info
        return self.propagate(edge_index, x=x)

    def message(self, x, x_i, x_j, index):
        # Indices and basis info for 4-hop message paths
        l, k, i, j = (self.quadruplet_info['l_idx'],
                      self.quadruplet_info['k_idx'],
                      self.quadruplet_info['i_idx'],
                      self.quadruplet_info['j_idx'])
        
        m_lk = x[l]                                # [E, F]

        rbf_feat = self.edge_rbf[k]               # [E, num_rbf]
        cbf_feat = self.quadruplet_info['cbf']    # [Q, num_cbf]
        #print the number of unique cbf rows, eliminate the duplicate rows
        print(f"Unique cbf rows: {len(torch.unique(cbf_feat, dim=0))}")
        sbf_feat = self.quadruplet_info['sbf']    # [Q, num_sbf]
        #print the number of unique sbf rows, eliminate the duplicate rows
        print(f"Unique sbf rows: {len(torch.unique(sbf_feat, dim=0))}")

        
        print(f"m_lk: {m_lk.shape}, rbf_feat: {rbf_feat.shape}, cbf_feat: {cbf_feat.shape}, sbf_feat: {sbf_feat.shape}")
        exit()
        m_jikl = torch.cat([m_lk, rbf_feat, cbf_feat, sbf_feat], dim=-1)

        
        edge_message = self.edge_mlp_1hop(m_jikl)
        
        # Aggregate from 4-hop message path to edge j→i
        aggregated = torch.zeros_like(x_j)
        aggregated = scatter(edge_message, k, dim=0, dim_size=x_j.size(0), reduce='add')

        final_input = torch.cat([x_j, aggregated[index]], dim=-1)
        return self.edge_mlp_2(final_input)

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class GemNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 num_rbf=6, num_cbf=6, num_sbf=6, task='regression'):
        super(GemNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.layers.append(
                GemNetLayer(in_c, hidden_channels, hidden_channels, num_rbf, num_cbf, num_sbf)
            )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.task = task

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        rbf = data.rbf
        edge_attr = data.edge_attr
        quadruplet_info = data.quadruplet_info
        batch = data.batch

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, quadruplet_info, rbf=rbf)

        x = global_mean_pool(x, batch)

        if self.task == 'regression':
            x = F.relu(x)
        elif self.task == 'classification':
            x = F.log_softmax(x, dim=-1)
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

        return self.output_mlp(x)
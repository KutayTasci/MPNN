import torch
from torch_geometric.datasets import MD17
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch import nn

'''
Define a function that creates new fields in the dataset with given dimensions, that will be used as transformation
Override any existing fields in the dataset
x: node features
pos: node positions
y: target values
edge_attr: edge features
'''
x_dim = 64
pos_dim = 3
edge_attr_dim = 64
y_dim = 1

class RBF(nn.Module):
    def __init__(self, num_rbf=6, cutoff=5.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, num_rbf))
        self.gamma = 10.0

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)

class CBF(nn.Module):
    def __init__(self, num_cbf=6):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, torch.pi, num_cbf))  # non-learnable buffer
        self.gamma = 5.0  # fixed scalar

    def forward(self, angles):  # angles: [T]
        angles = angles.unsqueeze(-1)  # [T, 1]
        return torch.exp(-self.gamma * (angles - self.centers) ** 2)  # [T, num_cbf]


class SBF(nn.Module):
    def __init__(self, num_sbf=6):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, torch.pi, num_sbf))
        self.gamma = 5.0

    def forward(self, angles):
        return torch.exp(-self.gamma * (angles.unsqueeze(-1) - self.centers) ** 2)

def compute_triplets(edge_index, pos):
    # edge_index: [2, E], pos: [N, 3]
    row, col = edge_index  # j → i
    E = edge_index.size(1)
    N = pos.size(0)

    # Build adjacency list as a tensor for fast indexing
    adj = [[] for _ in range(N)]
    for idx in range(E):
        j, i = row[idx].item(), col[idx].item()
        adj[j].append(i)

    k_idx_list = []
    j_idx_list = []
    i_idx_list = []
    angles = []

    # Vectorized computation per node j
    for j in range(N):
        neighbors = adj[j]
        if len(neighbors) < 2:
            continue
        neighbors_tensor = torch.tensor(neighbors, dtype=torch.long, device=pos.device)
        v_j = pos[j]
        v_neighbors = pos[neighbors_tensor]  # [num_neighbors, 3]
        v1 = v_j - v_neighbors  # [num_neighbors, 3]
        v2 = v_neighbors - v_j  # [num_neighbors, 3], just -v1

        # Compute all pairs (k, i) with k != i
        idx_k, idx_i = torch.meshgrid(
            torch.arange(len(neighbors), device=pos.device),
            torch.arange(len(neighbors), device=pos.device),
            indexing='ij'
        )
        mask = idx_k != idx_i
        idx_k = idx_k[mask]
        idx_i = idx_i[mask]

        k_idx = neighbors_tensor[idx_k]
        i_idx = neighbors_tensor[idx_i]

        v1_k = v1[idx_k]  # [num_pairs, 3]
        v2_i = v2[idx_i]  # [num_pairs, 3]

        # Compute cosine similarity and angle
        cos_theta = F.cosine_similarity(v1_k, v2_i).clamp(-1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_theta)

        k_idx_list.append(k_idx)
        j_idx_list.append(torch.full_like(k_idx, j))
        i_idx_list.append(i_idx)
        angles.append(angle)

    if k_idx_list:
        k_idx_all = torch.cat(k_idx_list)
        j_idx_all = torch.cat(j_idx_list)
        i_idx_all = torch.cat(i_idx_list)
        angles_all = torch.cat(angles)
    else:
        k_idx_all = torch.tensor([], dtype=torch.long)
        j_idx_all = torch.tensor([], dtype=torch.long)
        i_idx_all = torch.tensor([], dtype=torch.long)
        angles_all = torch.tensor([], dtype=torch.float)

    return {
        'k_idx': k_idx_all,
        'j_idx': j_idx_all,
        'i_idx': i_idx_all,
        'angles': angles_all
    }

def create_new_fields_dimenet(data, num_rbf=6, num_cbf=6, cutoff=5.0):
    data.x = torch.randn(data.num_nodes, x_dim)
    data.pos = torch.randn(data.num_nodes, pos_dim)
    data.y = torch.randn(1, y_dim)
    data.edge_attr = torch.randn(data.num_edges, edge_attr_dim)

    # Ensure undirected graph for triplets
    data.edge_index = to_undirected(data.edge_index)

    # Radial distances
    row, col = data.edge_index
    edge_vec = data.pos[row] - data.pos[col]
    distances = edge_vec.norm(dim=1)  # [E]

    # Compute RBF
    rbf_layer = RBF(num_rbf=num_rbf, cutoff=cutoff)
    rbf_values = rbf_layer(distances)  # [E, num_rbf]

    # Compute triplets + CBF
    triplet_info = compute_triplets(data.edge_index, data.pos)
    cbf_layer = CBF(num_cbf=num_cbf)
    cbf_values = cbf_layer(triplet_info['angles'])  # [T, num_cbf]

    # Store all
    data.distances = distances
    data.rbf = rbf_values
    triplet_info['cbf'] = cbf_values
    data.triplet_info = triplet_info

    return data

def create_new_fields(data):
    data.x = torch.randn(data.num_nodes, x_dim)
    data.pos = torch.randn(data.num_nodes, pos_dim)
    data.y = torch.randn(1, y_dim)
    data.edge_attr = torch.randn(data.num_edges, edge_attr_dim)
    return data

class MD17Dataset:
    def __init__(
        self,
        root: str = 'data/MD17',
        name: str = 'aspirin',
        x_dimt = 64,
        pos_dimt = 3,
        edge_attr_dimt = 64,
        y_dimt = 1,
        dimenet=False
    ):
        """
        Initializes the MD17 dataset with normalized node features x.
        """
        self.root = root
        self.name = name

        x_dim = x_dimt
        pos_dim = pos_dimt
        edge_attr_dim = edge_attr_dimt
        y_dim = y_dimt

        if dimenet:
            basic_transform = T.Compose([
                RadiusGraph(r=6.0),
                create_new_fields_dimenet
            ])
        else:
            basic_transform = T.Compose([
                RadiusGraph(r=6.0),
                create_new_fields
            ])

        print("Applying transforms and caching the dataset...")
        raw_dataset = MD17(
            root=self.root,
            name=self.name,
            train=None,
            transform=basic_transform
        )

        self.dataset = [raw_dataset[i] for i in range(len(raw_dataset))]

        self.num_features = self.dataset[0].x.shape[1]
        self.num_classes = 1  # Regression task
        self.edge_feature_dim = self.dataset[0].edge_attr.shape[1] if self.dataset[0].edge_attr is not None else 0

        self.task = 'regression'
        print(self.dataset[0])

    def get_loaders(self, batch_size=32, shuffle=True):
        '''
        Just returns the dataset
        With an optimized DataLoader
        '''
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True)
        return (train_loader, None, None)
    
    def get_data(self) -> Data:
        return self.dataset

        
import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch import nn
import numpy as np
from tqdm import tqdm


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

class Fourier(nn.Module):
    """Fourier Expansion for angle features."""

    def __init__(self, *, order: int = 5, learnable: bool = False) -> None:
        """
        order: maximum frequency order N in CHGNet eqn 1.
        learnable: if True, frequencies {1…N} are trainable.
        """
        super().__init__()
        self.order = order

        freqs = torch.arange(1, order + 1, dtype=torch.float32)
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand angles/radians tensor x of shape [T] to Fourier features:
          [1/√2, sin(n*x), cos(n*x)] for n=1..order
        Output shape: [T, 1 + 2*order], divided by √π per CHGNet eq (1) :contentReference[oaicite:1]{index=1}.
        """
        T = x.shape[0]
        out = x.new_zeros(T, 1 + 2 * self.order)
        out[:, 0] = 1 / np.sqrt(2.0)

        tmp = x.unsqueeze(1) * self.frequencies.unsqueeze(0)  # [T, order]
        out[:, 1 : self.order + 1] = torch.sin(tmp)
        out[:, self.order + 1 :] = torch.cos(tmp)
        return out / np.sqrt(np.pi)

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

def create_new_fields_chgnet(data, num_rbf=6, num_cbf=6, cutoff=5.0):
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

    # Compute triplets 
    triplet_info = compute_triplets(data.edge_index, data.pos)
    fourier_layer = Fourier(order=num_cbf)  # Fourier expansion for angles
    cbf_values = fourier_layer(triplet_info['angles'])  # [T, num_cbf]

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

class PPI_Dataset:
    def __init__(self, root: str = 'data/PPI', x_dimt=64, pos_dimt=3, edge_attr_dimt=64, y_dimt=1, chgnet=False):
        """
        Initializes the PPI dataset with normalized node features x.
        """
        self.root = root

        x_dim = x_dimt
        pos_dim = pos_dimt
        edge_attr_dim = edge_attr_dimt
        y_dim = y_dimt
        
        if chgnet:
            basic_transform = T.Compose([
                create_new_fields_chgnet
            ])
        else:
            basic_transform = T.Compose([
                create_new_fields
            ])

        raw_dataset = PPI(
            root=self.root,
            split='train',
            transform=basic_transform
        )
        self.dataset = [raw_dataset[i] for i in tqdm(range(len(raw_dataset)))]

        raw_dataset = PPI(
            root=self.root,
            split='val',
            transform=basic_transform
        )
        self.val_dataset = [raw_dataset[i] for i in tqdm(range(len(raw_dataset)))]

        raw_dataset = PPI(
            root=self.root,
            split='test',
            transform=basic_transform
        )

        self.test_dataset = [raw_dataset[i] for i in tqdm(range(len(raw_dataset)))]

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
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        return train_loader, val_loader, test_loader
    def get_data(self) -> Data:
        return self.dataset
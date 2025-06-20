import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import os
import numpy as np
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
    row, col = edge_index  # j → i
    E = edge_index.size(1)

    adj_dict = {}
    for idx in range(E):
        j, i = row[idx].item(), col[idx].item()
        if j not in adj_dict:
            adj_dict[j] = []
        adj_dict[j].append((idx, i))

    k_idx_list, j_idx_list, i_idx_list = [], [], []
    angles = []

    for j in adj_dict:
        neighbors = adj_dict[j]
        for idx1, k in neighbors:
            for idx2, i in neighbors:
                if k == i:
                    continue
                v1 = pos[j] - pos[k]
                v2 = pos[i] - pos[j]
                cos_theta = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).clamp(-1 + 1e-7, 1 - 1e-7)
                angle = torch.acos(cos_theta).item()

                k_idx_list.append(k)
                j_idx_list.append(j)
                i_idx_list.append(i)
                angles.append(angle)

    return {
        'k_idx': torch.tensor(k_idx_list, dtype=torch.long),
        'j_idx': torch.tensor(j_idx_list, dtype=torch.long),
        'i_idx': torch.tensor(i_idx_list, dtype=torch.long),
        'angles': torch.tensor(angles, dtype=torch.float)
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

    # Compute triplets + CBF
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


class QM9Dataset:
    def __init__(self, root: str = "data/QM9", x_dimt=64, pos_dimt=3, edge_attr_dimt=64, y_dimt=1,
                 num_graphs=1000, num_nodes=100, avg_degree=5, chgnet=False, gemnet=False,
                 cache_file='cached_qm9.pt', cache=False):
        """
        Initializes the QM9 dataset with normalized node features x.
        If cache is True and cache_file exists, loads preprocessed data instead of applying transform repeatedly.
        """
        self.root = root
        self.cache_file = cache_file
        self.cache = cache

        x_dim = x_dimt
        pos_dim = pos_dimt
        edge_attr_dim = edge_attr_dimt
        y_dim = y_dimt

        # Choose transform function
        if chgnet:
            basic_transform = T.Compose([create_new_fields_chgnet])
            model = 'chgnet'
        else:
            basic_transform = T.Compose([create_new_fields])
            model = 'egnn'

        cache_file = cache_file+model

        if cache and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            self.dataset = torch.load(cache_file)
        else:
            print("Applying transforms and caching the dataset...")
            raw_dataset = QM9(root=self.root, transform=basic_transform)
            self.dataset = [raw_dataset[i] for i in range(len(raw_dataset))]  # Apply transforms now
            #if cache:
            #    torch.save(self.dataset, cache_file)

        self.num_features = self.dataset[0].x.shape[1]
        self.num_classes = 1  # Regression task
        self.edge_feature_dim = self.dataset[0].edge_attr.shape[1] if self.dataset[0].edge_attr is not None else 0

        # Use only the first label (target 0) for regression
        if self.dataset[0].y is not None and self.dataset[0].y.ndim > 1:
            for data in self.dataset:
                data.y = data.y[:, [0]]
        else:
            raise ValueError("Dataset labels (y) are either None or have insufficient dimensions.")

        self.task = 'regression'
        print("Sample graph:", self.dataset[0])

    def get_loaders(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Returns DataLoaders for train, val, test.
        """
        n_total = len(self.dataset)
        train_ratio = 0.8
        val_ratio = 0.1

        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val, n_test]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)

        return train_loader, val_loader, test_loader

    def get_data(self):
        return self.dataset

class QM9DatasetOriginal:
    def __init__(self, root: str = "data/QM9"):
        """
        Initializes the QM9 dataset with normalized node features x.
        """
        self.root = root

        #self.normalizer = NormalizeX().fit(raw_dataset)
        self.dataset = QM9(root=self.root)
        
        self.num_features = self.dataset[0].x.shape[1]
        self.num_classes = 1  # Regression task
        self.edge_feature_dim = self.dataset[0].edge_attr.shape[1] if self.dataset[0].edge_attr is not None else 0
        # Use only the first label (target 0) for regression
        if self.dataset.data.y is not None and self.dataset.data.y.ndim > 1:
            self.dataset.data.y = self.dataset.data.y[:, [0]]
        else:
            raise ValueError("Dataset labels (y) are either None or have insufficient dimensions.")
        self.task = 'regression'
        print(self.dataset[0])

    def get_loaders(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Returns DataLoaders for train, val, test.
        """
        n_total = len(self.dataset)
        train_ratio = 0.8
        val_ratio = 0.1

        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val, n_test]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, persistent_workers=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, **kwargs)

        return train_loader, val_loader, test_loader

    def get_data(self) -> Data:
        return self.dataset
    

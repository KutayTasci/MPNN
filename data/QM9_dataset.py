import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch import nn
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

def compute_quadruplets(edge_index, pos):
    row, col = edge_index  # j -> i
    E = edge_index.size(1)

    # Create adjacency list
    adj_dict = {}
    for idx in range(E):
        j, i = row[idx].item(), col[idx].item()
        if i not in adj_dict:
            adj_dict[i] = []
        adj_dict[i].append((idx, j))

    l_idx_list, k_idx_list, i_idx_list, j_idx_list = [], [], [], []
    cbf_angles = []
    sbf_angles = []

    for i in adj_dict:
        neighbors_i = adj_dict[i]
        neighbors_i_indices = torch.tensor([n[1] for n in neighbors_i], dtype=torch.long)
        vec_ik = pos[i] - pos[neighbors_i_indices]  # [Ni, 3]

        for k_idx, k in neighbors_i:
            if k == i or k not in adj_dict:
                continue

            neighbors_k = adj_dict[k]
            neighbors_k_indices = torch.tensor([n[1] for n in neighbors_k], dtype=torch.long)
            vec_lk = pos[neighbors_k_indices] - pos[k]  # [Nk, 3]

            # Compute angles ikl
            vec_ik_norm = vec_ik / vec_ik.norm(dim=1, keepdim=True)  # [Ni, 3]
            vec_lk_norm = vec_lk / vec_lk.norm(dim=1, keepdim=True)  # [Nk, 3]
            cos_ikl = torch.mm(vec_ik_norm, vec_lk_norm.T).clamp(-0.999, 0.999)  # [Ni, Nk]
            angles_ikl = torch.acos(cos_ikl)  # [Ni, Nk]

            # Compute dihedral angles jikl
            j_idx = next((j for j in row.tolist() if col[row.tolist().index(j)] == i), i)
            if 0 <= j_idx < pos.size(0):  # Ensure j_idx is a valid index
                vec_ij = pos[i] - pos[j_idx]
                normal1 = torch.cross(vec_ij.unsqueeze(0), vec_ik, dim=1)  # [Ni, 3]
                normal2 = torch.cross(vec_ik.unsqueeze(1), vec_lk.unsqueeze(0), dim=2)  # [Ni, Nk, 3]
                cos_dihedral = F.cosine_similarity(normal1.unsqueeze(1), normal2, dim=2).clamp(-0.999, 0.999)  # [Ni, Nk]
                angles_dihedral = torch.acos(cos_dihedral)  # [Ni, Nk]
            else:
                continue

            # Collect quadruplet indices and angles
            l_idx_list.extend(neighbors_k_indices.tolist())
            k_idx_list.extend([k] * len(neighbors_k_indices))
            i_idx_list.extend([i] * len(neighbors_k_indices))
            j_idx_list.extend([j_idx] * len(neighbors_k_indices))
            cbf_angles.extend(angles_ikl.flatten().tolist())
            sbf_angles.extend(angles_dihedral.flatten().tolist())

    return {
        'l_idx': torch.tensor(l_idx_list, dtype=torch.long),
        'k_idx': torch.tensor(k_idx_list, dtype=torch.long),
        'i_idx': torch.tensor(i_idx_list, dtype=torch.long),
        'j_idx': torch.tensor(j_idx_list, dtype=torch.long),
        'cbf_angles': torch.tensor(cbf_angles, dtype=torch.float),
        'sbf_angles': torch.tensor(sbf_angles, dtype=torch.float)
    }

def create_new_fields_gemnet(data, x_dim=16, pos_dim=3, y_dim=1, edge_attr_dim=8,
                             num_rbf=6, num_cbf=6, num_sbf=6, cutoff=5.0):
    data.x = torch.randn(data.num_nodes, x_dim)
    data.pos = torch.randn(data.num_nodes, pos_dim)
    data.y = torch.randn(1, y_dim)

    # Ensure undirected graph
    data.edge_index = to_undirected(data.edge_index)

    # Compute distances for RBF
    row, col = data.edge_index
    edge_vec = data.pos[row] - data.pos[col]
    distances = edge_vec.norm(dim=1)  # [E]
    rbf_layer = RBF(num_rbf=num_rbf, cutoff=cutoff)
    rbf_values = rbf_layer(distances)  # [E, num_rbf]

    # Compute quadruplets + geometric bases
    quadruplet_info = compute_quadruplets(data.edge_index, data.pos)
    cbf_layer = CBF(num_cbf)
    sbf_layer = SBF(num_sbf)
    cbf = cbf_layer(quadruplet_info['cbf_angles'])
    sbf = sbf_layer(quadruplet_info['sbf_angles'])

    # Store all
    data.distances = distances
    data.rbf = rbf_values
    quadruplet_info['cbf'] = cbf
    quadruplet_info['sbf'] = sbf
    data.quadruplet_info = quadruplet_info

    return data

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


class QM9Dataset:
    def __init__(self, root: str = "data/QM9", x_dimt=64, pos_dimt=3, edge_attr_dimt=64, y_dimt=1, num_graphs=1000, num_nodes=100, avg_degree=5, dimenet=False, gemnet=False):
        """
        Initializes the QM9 dataset with normalized node features x.
        """
        self.root = root
        
        x_dim = x_dimt
        pos_dim = pos_dimt
        edge_attr_dim = edge_attr_dimt
        y_dim = y_dimt

        if dimenet:
            basic_transform = T.Compose([
                create_new_fields_dimenet
            ])
        elif gemnet:
            basic_transform = T.Compose([
                create_new_fields_gemnet
            ])
        else:
            basic_transform = T.Compose([
                create_new_fields
            ])
        
        #self.normalizer = NormalizeX().fit(raw_dataset)
        self.dataset = QM9(root=self.root, transform=basic_transform)
        
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, **kwargs)

        return train_loader, val_loader, test_loader

    def get_data(self) -> Data:
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
    

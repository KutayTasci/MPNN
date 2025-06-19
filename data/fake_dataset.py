import torch
from torch_geometric.datasets import FakeDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
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

class Fake_Dataset:
    def __init__(self, root: str = 'data/FakeDataset', x_dimt=64, pos_dimt=3, edge_attr_dimt=64, y_dimt=1, num_graphs=1000, num_nodes=100, avg_degree=5, dimenet=False, gemnet=False):
        """
        Initializes the Fake dataset with normalized node features x.
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

        raw_dataset= FakeDataset(
            num_graphs=num_graphs,
            avg_num_nodes=num_nodes,
            avg_degree=avg_degree,
            transform=basic_transform
        )
        self.dataset = [raw_dataset[i] for i in tqdm(range(len(raw_dataset)))]
        self.num_features = self.dataset[0].x.shape[1]
        self.num_classes = 1
        self.edge_feature_dim = self.dataset[0].edge_attr.shape[1] if self.dataset[0].edge_attr is not None else 0
        self.task = 'regression'
        print(self.dataset[0])

    def get_loaders(self, batch_size=32, shuffle=True):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        return train_loader, val_loader, test_loader
    def get_data(self) -> Data:
        return self.dataset
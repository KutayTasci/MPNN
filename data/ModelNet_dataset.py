import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


x_dim = 64
pos_dim = 3
edge_attr_dim = 64
y_dim = 1


def create_new_fields(data):
    data.x = torch.randn(data.num_nodes, x_dim)
    data.pos = torch.randn(data.num_nodes, pos_dim)
    data.y = torch.randn(1, y_dim)
    data.edge_attr = torch.randn(data.num_edges, edge_attr_dim)
    return data

class ModelNetDataset:
    def __init__(self, root: str = 'data/ModelNet', name: str = '10', x_dimt=64, pos_dimt=3, edge_attr_dimt=64, y_dimt=1):
        """
        Initializes the ModelNet dataset with normalized node features x.
        """
        self.root = root
        self.name = name

        x_dim = x_dimt
        pos_dim = pos_dimt
        edge_attr_dim = edge_attr_dimt
        y_dim = y_dimt

        basic_transform = T.Compose([
            T.FaceToEdge(remove_faces=False),
            create_new_fields
        ])

        self.dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=True,
            transform=basic_transform
        )

        self.test_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=False,
            transform=basic_transform
        )

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
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        return (train_loader, None, test_loader)
    
    def get_data(self) -> Data:
        return self.dataset
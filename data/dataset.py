import torch
from torch_geometric.datasets import Flickr, QM9, S3DIS, ModelNet
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data,  DataLoader
from torch_geometric.data import Batch
from typing import Optional, Callable
import torch_geometric.transforms as T

def collate_fn(batch):
    data, targets = zip(*batch)
    data = torch.stack(data).to('cuda', non_blocking=True)
    targets = torch.stack(targets).to('cuda', non_blocking=True)
    return data, targets

class FlickrDataset:
    def __init__(self, root: str = "data/Flickr", transform=None):
        """
        Initializes the Flickr dataset.

        Args:
            root (str): Directory to store the dataset.
            transform (callable, optional): Data transformations to apply (e.g., normalization).
        """
        self.root = root
        self.transform = transform if transform else NormalizeFeatures()
        self.dataset = Flickr(root=self.root, transform=self.transform)

    def get_data(self) -> Data:
        """
        Returns the graph data object.

        Returns:
            Data: A single large graph object containing node features, edges, and labels.
        """
        data = self.dataset[0] 
        data.edge_attr = torch.ones(data.edge_index.shape[1], 4)
        return data# Flickr dataset contains a single large graph

    def get_split(self):
        """
        Returns train, validation, and test masks.

        Returns:
            tuple: (train_mask, val_mask, test_mask) boolean masks.
        """
        data = self.get_data()
        return data.train_mask, data.val_mask, data.test_mask

class QM9Dataset:
    def __init__(self, root: str = "data/QM9", transform=None):
        """
        Initializes the QM9 dataset.

        Args:
            root (str): Directory to store the dataset.
            transform (callable, optional): Data transformations to apply (e.g., normalization).
        """
        self.root = root
        self.transform = transform if transform else NormalizeFeatures()
        self.dataset = QM9(root=self.root, transform=self.transform)

    def get_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Returns a DataLoader object for the QM9 dataset. For training, validation, and testing, use the train_mask, val_mask, and test_mask attributes of the Data object.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        n_total = len(self.dataset)
        train_ratio = 0.8
        val_ratio = 0.1

        # Compute the number of examples for each split
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        # Option 1: Using torch.utils.data.random_split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val, n_test]
        )

        # Option 2: Direct slicing (if the dataset is subscriptable)
        # train_dataset = dataset[:n_train]
        # val_dataset = dataset[n_train:n_train+n_val]
        # test_dataset = dataset[n_train+n_val:]

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
   
        return train_loader, val_loader, test_loader
    
    def get_full_graphs(self):
        """
        Returns the dataset as full graphs for train, validation, and test.

        Returns:
            tuple: (train_data, val_data, test_data)
        """
        n_total = len(self.dataset)
        train_ratio = 0.8
        val_ratio = 0.1

        # Compute the number of examples for each split
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        # Split dataset into full graphs
        train_dataset = self.dataset[:n_train]
        val_dataset = self.dataset[n_train:n_train+n_val]
        test_dataset = self.dataset[n_train+n_val:]

        # Create Batch objects for each split
        train_data = Batch.from_data_list(train_dataset)
        val_data = Batch.from_data_list(val_dataset)
        test_data = Batch.from_data_list(test_dataset)

        return train_data, val_data, test_data

    def get_data(self) -> Data:
        """
        Returns the graph data object.

        Returns:
            Data: A single large graph object containing node features, edges, and labels.
        """
        return self.dataset
    
    def get_split(self):
        """
        Returns train, validation, and test masks.

        Returns:
            tuple: (train_mask, val_mask, test_mask) boolean masks.
        """
        data = self.get_data()
        return data.train_mask, data.val_mask, data.test_mask

class S3DISDataset:
    def __init__(self, root: str = "data/S3DIS", transform=None, test_area=1):
        """
        Initializes the S3DIS dataset.

        Args:
            root (str): Directory to store the dataset.
            transform (callable, optional): Data transformations to apply (e.g., normalization).
        """
        self.root = root
        self.transform = transform if transform else NormalizeFeatures()
        self.test_area = test_area
        
    def get_train_data(self) -> Data:
        """
        Returns the graph data object.

        Returns:
            Data: A single large graph object containing node features, edges, and labels.
        """
        self.dataset = S3DIS(root=self.root, transform=self.transform, test_area=self.test_area, train=True)
        return self.dataset
    
    def get_test_data(self) -> Data:
        """
        Returns the graph data object.

        Returns:
            Data: A single large graph object containing node features, edges, and labels.
        """
        self.dataset = S3DIS(root=self.root, transform=self.transform, test_area=self.test_area, train=False)
        return self.dataset
    
class ModelNetDataset:
    def __init__(self, root: str = 'data/ModelNet', name: str = '10', transform=None):
        """
        Initializes the ModelNet dataset.

        Args:
            root (str): Directory to store the dataset.
            name (str): The name of the dataset ('10' for ModelNet10, '40' for ModelNet40).
            transform (callable, optional): Data transformations to apply.
        """
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = T.FaceToEdge(remove_faces=False)  # Convert mesh faces to edge indices

        # Load the training and test datasets
        self.train_dataset = ModelNet(root=self.root, name=self.name, train=True,
                                      transform=self.transform, pre_transform=self.pre_transform)
        
        
        self.test_dataset = ModelNet(root=self.root, name=self.name, train=False,
                                     transform=self.transform, pre_transform=self.pre_transform)

    def get_loaders(self, batch_size: int = 32, shuffle: bool = True):
        """
        Returns DataLoader objects for the training and test datasets.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the training dataset.

        Returns:
            tuple: (train_loader, test_loader) DataLoader objects.
        """

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    

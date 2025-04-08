import torch
from torch_geometric.datasets import QM9, ModelNet, MD17, ZINC, ShapeNet, MoleculeNet, CoMA
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph, NormalizeScale, BaseTransform
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import one_hot
from typing import Optional, Callable, Union, List
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm



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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=True)
   
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


class ModelNetDataset:
    def __init__(self, root: str = 'data/ModelNet', name: str = '10', transform=None):
        """
        Initializes the ModelNet dataset.

        Args:
            root (str): Directory to store the dataset.
            name (str): The name of the dataset ('10' for ModelNet10, '40' for ModelNet40).
            transform (callable, optional): Data transformations to apply.
        """
        def copy_pos_to_x(data):
            data.x = data.pos.clone()
            return data
        self.root = root
        self.name = name

        # Default transform: convert mesh faces to edge indices
        default_transform = T.FaceToEdge(remove_faces=True)
        combined_transform = T.Compose([default_transform, copy_pos_to_x])
        self.transform = transform if transform else combined_transform

        # Load the training and test datasets
        self.train_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=True,
            transform=self.transform
        )

        self.test_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=False,
            transform=self.transform
        )

        self.num_features = self.train_dataset[0].pos.shape[1]
        self.num_classes = int(name)
        self.edge_feature_dim = 0
        self.task = 'classification'  # or 'regression' based on your task


        print(self.train_dataset[0])



    def get_loaders(self, batch_size: int = 32, shuffle: bool = True):
        """
        Returns DataLoader objects for the training and test datasets.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the training dataset.

        Returns:
            tuple: (train_loader, test_loader) DataLoader objects.
        """

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return (train_loader, None, test_loader)
    
    def get_full_graphs(self):
        """
        Returns the dataset as full graphs for train and test.

        Returns:
            tuple: (train_data, test_data)
        """
        train_data = Batch.from_data_list(self.train_dataset)
        test_data = Batch.from_data_list(self.test_dataset)

        return (train_data, None, test_data)
    
class MD17Dataset:
    def __init__(
        self,
        root: str = 'data/MD17',
        name: str = 'aspirin',
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initializes the MD17 dataset.

        Args:
            root (str): Directory to store the dataset.
            name (str): The name of the trajectory to load (e.g., 'aspirin', 'revised aspirin', 'aspirin CCSD').
            train (bool, optional): Whether to load train or test split (only used for CCSD(T) datasets).
            transform (Callable, optional): A function to transform the data.
            pre_transform (Callable, optional): A function to transform the data before saving.
            pre_filter (Callable, optional): A function to filter the data before saving.
            force_reload (bool): If True, forces a re-download and re-process of the dataset.
        """

        def add_node_features(data):
            # One-hot encode the atomic numbers
            atomic_number_features = one_hot(data.z, num_classes=118)
            # Concatenate the one-hot encoded atomic numbers with the positions
            node_features = torch.cat([atomic_number_features, data.pos], dim=-1)
            # Assign to the x attribute
            data.x = node_features
            data.y = data.energy.clone()
            return data

        self.root = root
        self.name = name

        # Default transform: build radius-based graphs
        default_transform = RadiusGraph(r=6.0)
        combined_transform = T.Compose([default_transform, add_node_features])
        self.transform = transform if transform else combined_transform

        self.train_dataset = MD17(
            root=self.root,
            name=self.name,
            train=None,
            transform=self.transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )

        self.num_features = 118 + 3  # 118 for one-hot encoding + 3 for position coordinates
        self.num_classes = 1  # Assuming regression task for energy prediction
        self.edge_feature_dim = 0
        self.task = 'regression'  # or 'classification' based on your task

        print(self.train_dataset[0])


    def get_loaders(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Returns a DataLoader object for the MD17 dataset.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        train_loader =  DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 0)
        )

        return (train_loader, None, None)
    
class ZINCGraphDataset:
    def __init__(self, 
                 root: str = 'data/ZINC', 
                 split: str = 'train', 
                 subset: bool = False, 
                 transform: Optional[Callable] = None,
                 smiles_path: Optional[str] = None):
        """
        Initializes the ZINC dataset and attaches 3D positions using SMILES.

        Args:
            root (str): Directory to store the dataset.
            split (str): Dataset split to load ('train', 'val', or 'test').
            subset (bool): If True, loads the 12k subset version.
            transform (callable, optional): Data transformations to apply.
            smiles_path (str, optional): Path to CSV file containing SMILES strings.
        """

        if smiles_path:
            df = pd.read_csv(smiles_path)
            if 'SMILES' not in df.columns:
                raise ValueError("CSV must have a 'smiles' column")
            self.smiles_list = df['SMILES'].tolist()
        else:
            raise ValueError("You must provide a smiles_path for this dataset.")


        self.root = root
        self.split = split
        self.subset = subset
        self.transform = None

        try:
            self.train_dataset = ZINC(
                root=self.root, 
                split='train', 
                subset=self.subset, 
                transform=self._index_wrapper(self.transform, split='train')
            )
            self.val_dataset = ZINC(
                root=self.root, 
                split='val', 
                subset=self.subset, 
                transform=self._index_wrapper(self.transform, split='val')
            )
            self.test_dataset = ZINC(
                root=self.root, 
                split='test', 
                subset=self.subset, 
                transform=self._index_wrapper(self.transform, split='test')
            )
        except Exception as e:
            print(f"[ZINCGraphDataset] Failed to load dataset: {e}")
            raise

        self.num_features = self.train_dataset[0].x.shape[1]
        self.num_classes = 1  # Regression
        self.edge_feature_dim = 0  # No edge features by default

    def _index_wrapper(self, transform, split):
        """Wraps transform to add index tracking per split."""
        def indexed_transform(data):
            data.idx = data.idx if hasattr(data, 'idx') else data.__dict__.get('idx', 0)
            return transform(data)
        return indexed_transform

    def get_loaders(self, 
                   batch_size: int = 32, 
                   shuffle: bool = True, 
                   **kwargs) -> DataLoader:
        """
        Returns a DataLoader object for the ZINC dataset.

        Args:
            batch_size (int): Number of graphs per batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 0)
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=kwargs.get('num_workers', 0)
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=kwargs.get('num_workers', 0)
        )
        return (train_loader, val_loader, test_loader)
    
class MoleculeNetDataset:
    def __init__(self, root: str, name: str, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        """
        Initializes the MoleculeNet dataset with 3D coordinates.

        Args:
            root (str): Directory where the dataset should be saved.
            name (str): Name of the dataset (e.g., 'ESOL', 'FreeSolv', 'Lipo').
            transform (callable, optional): Function that takes in a Data object and returns a transformed version.
            pre_transform (callable, optional): Function that takes in a Data object and returns a transformed version before saving to disk.
            pre_filter (callable, optional): Function that takes in a Data object and returns a boolean value, indicating whether the data object should be included in the final dataset.
            force_reload (bool, optional): Whether to re-process the dataset. Default is False.
        """
        def generate_3d_coordinates(data):
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                return data

            # Add hydrogen atoms to the molecule
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                return data  # Embedding failed

            # Optimize the molecule's geometry
            AllChem.MMFFOptimizeMolecule(mol)

            # Remove hydrogens to match the heavy atoms in `x`
            mol = Chem.RemoveHs(mol)

            # Extract 3D positions
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            data.pos = torch.tensor(positions, dtype=torch.float)

            return data
        
        self.root = root
        self.name = name
        self.transform = generate_3d_coordinates if transform is None else transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.force_reload = force_reload

        # Load the dataset with the pre-transform function
        self.dataset = MoleculeNet(root=self.root, name=self.name, transform=self.transform,
                                   pre_transform=self.pre_transform, pre_filter=self.pre_filter,
                                   force_reload=self.force_reload)

        # Determine the number of features and classes
        self.num_features = self.dataset.num_features
        print(f"Number of features: {self.num_features}")
        self.num_classes = self.dataset.num_classes
        print(f"Number of classes: {self.num_classes}")
        print(self.dataset[0])

    def get_loaders(self, batch_size: int = 32, shuffle: bool = True, test_split: float = 0.2):
        """
        Returns DataLoader objects for the dataset.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the dataset.
            test_split (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: (train_loader, test_loader) DataLoader objects.
        """
        # Determine the split index
        test_size = int(len(self.dataset) * test_split)
        train_size = len(self.dataset) - test_size

        # Split the dataset
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

class CoMADataset:
    def __init__(
        self,
        root: str = 'data/CoMA',
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initializes the CoMA dataset.

        Args:
            root (str): Directory where the dataset should be saved.
            train (bool): If True, loads the training dataset, otherwise the test dataset.
            transform (Callable, optional): Transform applied at data access time.
            pre_transform (Callable, optional): Transform applied once before saving to disk.
            pre_filter (Callable, optional): Whether to keep a given data object.
            force_reload (bool): If True, forces re-processing of dataset.
        """

        def copy_pos_to_x(data):
            # Copy the positions to the x attribute
            data.x = data.pos.clone()
            return data
        
        self.root = root
        self.train = train
        self.transform = transform
        default_transform = T.FaceToEdge(remove_faces=True)
        combined_transform = T.Compose([default_transform, copy_pos_to_x])
        self.pre_filter = pre_filter
        self.force_reload = force_reload

        # Load dataset
        self.train_dataset = CoMA(
            root=self.root,
            train=True,
            transform=combined_transform,
            pre_transform=combined_transform,
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )

        self.test_dataset = CoMA(
            root=self.root,
            train=False,
            transform=combined_transform,
            pre_transform=combined_transform,
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )

        print(self.train_dataset[0].y)
        self.num_features = self.train_dataset[0].pos.shape[1]  # typically 3 for 3D
        self.num_classes = 12  # unsupervised mesh data; set manually if needed
        self.edge_feature_dim = 0  # no edge features by default
        self.task = 'classification'  # or 'regression' based on your task


    def get_loaders(self, batch_size: int = 32, shuffle: bool = True):
        """
        Returns a DataLoader for the CoMA dataset.

        Args:
            batch_size (int): Number of meshes per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        return (loader, None, test_loader)

class ShapeNetDataset:
    def __init__(
        self,
        root: str = 'data/ShapeNet',
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initializes the ShapeNet part segmentation dataset.

        Args:
            root (str): Directory to store/load the dataset.
            categories (Union[str, List[str]], optional): Categories to include.
            include_normals (bool): Whether to include normal vectors in data.x.
            split (str): Dataset split to load: 'train', 'val', 'trainval', or 'test'.
            transform (callable, optional): Transform applied at access time.
            pre_transform (callable, optional): Transform applied before saving.
            pre_filter (callable, optional): Pre-filter applied before saving.
            force_reload (bool): Force reprocessing the dataset.
        """
        self.root = root
        self.categories = categories
        self.include_normals = include_normals
        self.split = split

        # Default transform: normalize scale for 3D point clouds
        self.transform = transform if transform else NormalizeScale()

        self.dataset = ShapeNet(
            root=self.root,
            categories=self.categories,
            include_normals=self.include_normals,
            split=self.split,
            transform=self.transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )

    def get_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Returns a DataLoader object for the ShapeNet dataset.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=True,
            persistent_workers=True
        )
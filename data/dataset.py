import torch
from torch_geometric.datasets import QM9, ModelNet, MD17, ZINC, ShapeNet, MoleculeNet, CoMA
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph, NormalizeScale, BaseTransform
from torch_geometric.data import Data, DataLoader, Batch
from torch.utils.data import random_split
from torch_geometric.utils import one_hot
from typing import Optional, Callable, Union, List
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

class NormalizePos:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.pos = (data.pos - self.mean) / self.std
        return data


class NormalizeY:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data.y - self.mean) / self.std
        return data

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

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=True)
   
        return train_loader, val_loader, test_loader
    


    def get_data(self) -> Data:
        """
        Returns the graph data object.

        Returns:
            Data: A single large graph object containing node features, edges, and labels.
        """
        return self.dataset



class ModelNetDataset:
    def __init__(self, root: str = 'data/ModelNet', name: str = '10', transform=None, normalize_pos=True):
        self.root = root
        self.name = name

        # Minimal transform to compute stats
        def copy_pos_to_x(data):
            data.x = data.pos.clone()
            return data

        basic_transform = T.Compose([
            T.FaceToEdge(remove_faces=False),
            copy_pos_to_x
        ])

        # Load with basic transform to compute stats
        raw_train_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=True,
            transform=basic_transform
        )
        raw_test_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=False,
            transform=basic_transform
        )

        if normalize_pos:
            mean, std = self._compute_pos_stats(raw_train_dataset + raw_test_dataset)
            norm_transform = NormalizePos(mean, std)
            final_transform = T.Compose([
                T.FaceToEdge(remove_faces=False),
                copy_pos_to_x,
                norm_transform
            ])
        else:
            final_transform = transform if transform else basic_transform

        # Reload datasets with full transform
        self.train_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=True,
            transform=final_transform
        )
        self.test_dataset = ModelNet(
            root=self.root,
            name=self.name,
            train=False,
            transform=final_transform
        )

        self.num_features = self.train_dataset[0].pos.shape[1]
        self.num_classes = int(name)
        self.edge_feature_dim = 0
        self.task = 'classification'
        print(self.train_dataset[0])

    def _compute_pos_stats(self, dataset):
        all_pos = torch.cat([data.pos for data in dataset], dim=0)
        mean = all_pos.mean(dim=0)
        std = all_pos.std(dim=0)
        return mean, std


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
        normalize_pos: bool = True,
        normalize_y: bool = True,
    ):
        """
        Initializes the MD17 dataset with optional per-dataset position normalization.
        """

        def add_node_features(data):
            atomic_number_features = one_hot(data.z, num_classes=118)
            node_features = torch.cat([atomic_number_features, data.pos], dim=-1)
            data.x = node_features
            data.y = data.energy.clone()
            return data

        self.root = root
        self.name = name
        self.normalize_pos = normalize_pos
        self.normalize_y = normalize_y

        # Step 1: Load raw dataset for stat computation
        basic_transform = T.Compose([
            RadiusGraph(r=6.0),
            add_node_features
        ])

        raw_dataset = MD17(
            root=self.root,
            name=self.name,
            train=train,
            transform=basic_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )

        if normalize_pos:
            mean, std = self._compute_pos_stats(raw_dataset)
            normalize = NormalizePos(mean, std)
            

            #self.y_mean, self.y_std = self._compute_y_stats(raw_dataset)
            #normalize_y = NormalizeY(self.y_mean, self.y_std)
            self.transform = T.Compose([
                RadiusGraph(r=6.0),
                add_node_features,
                normalize
            ])
        else:
            self.transform = transform if transform else basic_transform
           

        # Step 2: Reload with transform including normalization
        self.train_dataset = MD17(
            root=self.root,
            name=self.name,
            train=train,
            transform=self.transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )
        

        self.num_features = 118 + 3  # one-hot(atomic_number) + pos
        self.num_classes = 1  # regression
        self.edge_feature_dim = 0
        self.task = 'regression'

        print(self.train_dataset[0])

    def _compute_pos_stats(self, dataset):
        all_pos = torch.cat([data.pos for data in dataset], dim=0)
        mean = all_pos.mean(dim=0)
        std = all_pos.std(dim=0)
        return mean, std

    def _compute_y_stats(self, dataset):
        all_y = torch.cat([data.energy for data in dataset], dim=0)
        return all_y.mean(), all_y.std()

    def get_loaders(self, batch_size: int = 32, shuffle: bool = True, test_ratio=0.1, **kwargs):
        """
        Returns a DataLoader object for the MD17 dataset.

        Args:
            batch_size (int): Number of graphs in each batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        total_len = len(self.train_dataset)
        test_len = max(1, int(total_len * test_ratio))
        train_len = total_len - test_len

        train_subset, test_subset = random_split(self.train_dataset, [train_len, test_len])

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 0)
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=kwargs.get('num_workers', 0)
        )


        return (train_loader, None, test_loader)
    
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
        normalize_pos: bool = True,
    ):
        """
        Initializes the CoMA dataset with optional position normalization.
        """
        class SafeFaceToEdge:
            def __init__(self, remove_faces=False):
                self.remove_faces = remove_faces

            def __call__(self, data):
                if hasattr(data, 'face') and data.face is not None:
                    return T.FaceToEdge(remove_faces=self.remove_faces)(data)
                return data
        def copy_pos_to_x(data):
            data.x = data.pos.clone()
            return data

        self.root = root
        self.train = train
        self.pre_filter = pre_filter
        self.force_reload = force_reload

        # Load base dataset without transform to compute pos stats
        raw_train = CoMA(
            root=self.root,
            train=True,
            transform=None,
            pre_transform=T.Compose([T.FaceToEdge(remove_faces=False), copy_pos_to_x]),
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )
        raw_test = CoMA(
            root=self.root,
            train=False,
            transform=None,
            pre_transform=T.Compose([T.FaceToEdge(remove_faces=False), copy_pos_to_x]),
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )

        if normalize_pos:
            mean, std = self._compute_pos_stats(raw_train + raw_test)
            normalize = NormalizePos(mean, std)
            self.transform = T.Compose([
                SafeFaceToEdge(remove_faces=False),
                copy_pos_to_x,
                normalize
            ])
        else:
            self.transform = transform if transform else T.Compose([
                SafeFaceToEdge(remove_faces=False),
                copy_pos_to_x
            ])

        # Load datasets with final transform
        self.train_dataset = CoMA(
            root=self.root,
            train=True,
            transform=self.transform,
            pre_transform=None,
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )
        self.test_dataset = CoMA(
            root=self.root,
            train=False,
            transform=self.transform,
            pre_transform=None,
            pre_filter=self.pre_filter,
            force_reload=self.force_reload
        )

        print(self.train_dataset[0].y)
        self.num_features = self.train_dataset[0].pos.shape[1]  # typically 3
        self.num_classes = 12  # manual label setting
        self.edge_feature_dim = 0
        self.task = 'classification'

    def _compute_pos_stats(self, dataset):
        all_pos = torch.cat([data.pos for data in dataset], dim=0)
        return all_pos.mean(dim=0), all_pos.std(dim=0)

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

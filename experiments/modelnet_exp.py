import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.mpnn import MPNN_ModelNet
from data.dataset import ModelNetDataset
import warnings
import training.train as train
import yaml
import torch_geometric.transforms as T

warnings.filterwarnings("ignore")


class AddRandomEdgeAttributes(object):
    def __init__(self, num_edge_features):
        """
        Args:
            num_edge_features (int): Number of features to assign to each edge.
        """
        self.num_edge_features = num_edge_features

    def __call__(self, data):
        num_edges = data.edge_index.size(1)
        data.edge_attr = torch.rand((num_edges, self.num_edge_features), dtype=torch.float)
        return data

def Train_On_MPNN(benchmark, dataset, model_type='cat', batch_size=128, no_classes=10):
    cfg = yaml.safe_load(open("configs/modelnet.yaml"))
    # Model Parameters

    # Get DataLoader objects
    train_loader, test_loader = dataset.get_loaders(batch_size=batch_size)
    in_channels = dataset.train_dataset[0].pos.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = 10


    edge_channels = dataset.train_dataset[0].edge_attr.shape[1]
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Model
    model = MPNN_ModelNet(in_channels, edge_channels, hidden_channels, out_channels, num_layers, mode=model_type).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train.Train_ModelNet(model, (train_loader, test_loader), optimizer, F.cross_entropy, device, epochs=epochs, benchmark=benchmark)


def ModelNetExperiment(benchmark=False, model_type='cat'):
    no_classes = 10
    # Initialize the dataset
    transform = T.Compose([T.Distance(norm=True)])

    modelnet_dataset = ModelNetDataset(root='data/ModelNet', name=str(no_classes), transform=transform)

    Train_On_MPNN(benchmark, modelnet_dataset, model_type=model_type, batch_size=32, no_classes=no_classes)



import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.egnn import EGNN
from models.dimenet import DimeNet
from models.gemnet import GemNet
from data.QM9_dataset import QM9Dataset, QM9DatasetOriginal
from data.MD17_dataset import MD17Dataset
from data.ModelNet_dataset import ModelNetDataset
from data.fake_dataset import Fake_Dataset
from data.ppi_dataset import PPI_Dataset
import training.train as train
import torch._dynamo

def Train_On_EGNN(benchmark, dataset, model_type='cat', batch_size=128):
    cfg = {
        "hidden_channels": 64,
        "num_layers": 7,
        "learning_rate": 0.0001,
        "epochs": 10
    }
    
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=batch_size, shuffle=True)
    
    in_channels = dataset.dataset[0].x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = dataset.dataset[0].y.shape[1]
    edge_channels = dataset.dataset[0].edge_attr.shape[1] if dataset.dataset[0].edge_attr is not None else 0
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Initialize Model
    model = EGNN(in_channels, edge_channels, hidden_channels, out_channels,mode=model_type, num_layers=num_layers, task="regression").to(device)
    # compile the model with torch.compile(backend='inductor')
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, backend='inductor', dynamic=True)  
    torch.set_float32_matmul_precision('high')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train EGNN
    train.Train_GraphClassification(model, (train_loader, val_loader, test_loader), optimizer, F.l1_loss, device, epochs=epochs, benchmark=benchmark)

def Train_On_DimeNet(benchmark, dataset, model_type='cat', batch_size=128):
    cfg = {
        "hidden_channels": 64,
        "num_layers": 7,
        "learning_rate": 0.0001,
        "epochs": 10
    }
    
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=batch_size, shuffle=True)
    
    in_channels = dataset.dataset[0].x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = dataset.dataset[0].y.shape[1]
    edge_channels = dataset.dataset[0].edge_attr.shape[1] if dataset.dataset[0].edge_attr is not None else 0
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Initialize Model
    model = DimeNet(in_channels, hidden_channels, out_channels, num_layers=num_layers, task="regression", mode=model_type).to(device)
    # compile the model with torch.compile(backend='inductor')
    #torch._dynamo.config.capture_scalar_outputs = True
    #torch._dynamo.config.suppress_errors = True
    #model = torch.compile(model, backend='inductor', dynamic=True)  
    torch.set_float32_matmul_precision('high')

    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train EGNN
    train.Train_GraphClassification(model, (train_loader, val_loader, test_loader), optimizer, F.l1_loss, device, epochs=epochs, benchmark=benchmark)

def Train_On_GemNet(benchmark, dataset, model_type='cat', batch_size=128):
    cfg = {
        "hidden_channels": 64,
        "num_layers": 2,
        "learning_rate": 0.0001,
        "epochs": 1000
    }
    
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=batch_size, shuffle=True)
    
    in_channels = dataset.dataset[0].x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = dataset.dataset[0].y.shape[1]
    edge_channels = dataset.dataset[0].edge_attr.shape[1] if dataset.dataset[0].edge_attr is not None else 0
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Initialize Model
    model = GemNet(in_channels, hidden_channels, out_channels, num_layers=num_layers, task="regression").to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train EGNN
    train.Train_GraphClassification(model, (train_loader, val_loader, test_loader), optimizer, F.l1_loss, device, epochs=epochs, benchmark=benchmark)

def test_egnn(benchmark=False, model_type='cat', batch_size=1024):
    
    # Load dataset  
    #dataset = QM9Dataset()
    dataset = QM9DatasetOriginal()
    #dataset = MD17Dataset(name='aspirin')
    #dataset = ModelNetDataset()
    #dataset = PPI_Dataset()  
    #dataset = Fake_Dataset(dimenet=True, num_graphs=32, num_nodes=100, avg_degree=50)



    print('Experiment for model type: ', model_type)
    print('Batch Size: ', batch_size)
    # Train EGNN
    #Train_On_EGNN(benchmark, dataset, model_type='cat', batch_size=batch_size)
    Train_On_EGNN(benchmark, dataset, model_type='sum', batch_size=batch_size)
    #Train_On_DimeNet(benchmark, dataset, model_type='cat', batch_size=batch_size)
    #Train_On_DimeNet(benchmark, dataset, model_type='sum', batch_size=batch_size)
    #Train_On_GemNet(benchmark, dataset, model_type=model_type, batch_size=batch_size)
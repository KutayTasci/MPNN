import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.mpnn import MPNN_QM9
from data.dataset import QM9Dataset
import warnings
import training.train as train
import yaml
import torch.profiler
warnings.filterwarnings("ignore")


#Train with minibatches
def Train_On_MPNN(benchmark, dataset, model_type='cat', batch_size=128):
    cfg = yaml.safe_load(open("configs/qm9.yaml"))
    # Model Parameters

    train_loader, val_loader, test_loader = dataset.get_loader(batch_size=batch_size, shuffle=True)
    train_set, val_set, test_set = dataset.get_full_graphs()
    in_channels = dataset.dataset[0].x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = dataset.dataset[0].y.shape[1]
    edge_channels = dataset.dataset[0].edge_attr.shape[1]
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize Model
    model = MPNN_QM9(in_channels, edge_channels, hidden_channels, out_channels, num_layers, mode=model_type).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    #train.Train_QM9(model, (train_loader, val_loader, test_loader), optimizer, F.mse_loss, device, epochs=epochs, benchmark=benchmark)
    train.Train_QM9_FB(model, (train_set, val_set, test_set), optimizer, F.mse_loss, device, epochs=epochs, benchmark=benchmark)

    

def QM9Experiment(benchmark=False, model_type='cat', batch_size=128):
    # Load dataset  
    dataset = QM9Dataset()
    
    print('Experiment for model type: ', model_type)
    print('Batch Size: ', batch_size)
    # Train MPNN
    Train_On_MPNN(benchmark, dataset, model_type=model_type, batch_size=batch_size)
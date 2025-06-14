import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.mpnn import MPNN
from data.dataset import FlickrDataset
import warnings
import training.train as train
import yaml
warnings.filterwarnings("ignore")

def Train_On_MPNN(data, benchmark, model_type='cat'):
    cfg = yaml.safe_load(open("configs/flickr.yaml"))
    # Model Parameters
    in_channels = data.x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = data.y.max().item() + 1  # Number of classes
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Model
    model = MPNN(in_channels, hidden_channels, out_channels, num_layers, mode=model_type).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Move data to device
    data = data.to(device)
    compiled_model = torch.compile(model)
    train.Train_Flickr(compiled_model, data, optimizer, F.cross_entropy, device, epochs=epochs, benchmark=benchmark)
    

def FlickrExperiment(benchmark=False, model_type='cat'):
    # Load dataset  
    dataset = FlickrDataset()
    data = dataset.get_data()

    print('Experiment for model type: ', model_type)
    # Train MPNN
    Train_On_MPNN(data, benchmark, model_type=model_type)
    

    
    
    
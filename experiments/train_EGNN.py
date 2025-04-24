import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch.nn import Linear
from models.mpnn import MPNN_GC
from data.dataset import QM9Dataset
import warnings
import training.train as train
import yaml
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

hidden_channels = 64

#Train with minibatches
def Train_On_EGNN(datasets, batch_size=128):
    pass
    

def TrainEGNN():
    # Load dataset  
    print('Training started', flush=True)
    qm9 = QM9Dataset()
    print(qm9.get_data()[0])


    Train_On_EGNN([qm9], batch_size=16)
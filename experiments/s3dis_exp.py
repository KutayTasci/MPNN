import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.mpnn import MPNN
from data.dataset import S3DISDataset
import warnings
import training.train as train
import yaml
warnings.filterwarnings("ignore")


def Train_On_MPNN(data, benchmark, model_type='cat'):
    pass
    

def S3DISExperiment(benchmark=False, model_type='cat'):
    # Load dataset  
    dataset = S3DISDataset()
    data = dataset.get_train_data()

    print(data)
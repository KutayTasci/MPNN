import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch.nn import Linear
from models.mpnn import MPNN_GC
from data.dataset import QM9Dataset, ModelNetDataset ,MD17Dataset, ShapeNetDataset, CoMADataset
import warnings
import training.train as train
import yaml
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

hidden_channels = 64

#Train with minibatches
def Train_On_MPNN(datasets, batch_size=128):
    custom_kernel = Linear(1, hidden_channels)
    data_loaders = []
    models = []
    loss_ls = [] 
    optimizers = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    learning_rate = 0.0001
    epochs = 30
    num_layers = 3


    for dataset in datasets:
        data_loaders.append(dataset.get_loaders(batch_size=batch_size, shuffle=True))
        model = MPNN_GC(custom_kernel, dataset.num_features, dataset.edge_feature_dim, hidden_channels, dataset.num_classes, num_layers=num_layers, task = dataset.task)
        if dataset.task == 'regression':
            loss_fn = F.mse_loss
        else:
            loss_fn = F.cross_entropy
        
        loss_ls.append(loss_fn)
        optimizers.append(Adam(model.parameters(), lr=learning_rate))
        model = model.to(device)
        models.append(model)


    training_progress = train.Train_Base(models, data_loaders, optimizers, loss_ls, device, epochs=epochs)


    # Access the layer
    layer = custom_kernel

    # Prepare a dictionary with both weight and bias
    weights_and_bias = {
        'weight': layer.weight.tolist(),
        'bias': layer.bias.tolist() if layer.bias is not None else None
    }

    # Write to YAML
    with open('custom_kernel_weights.yaml', 'w') as f:
        yaml.dump(weights_and_bias, f)
    
    #create a directory to save the training progress
    import os
    if not os.path.exists('exp'):
        os.makedirs('exp')
    #visualize the training progress in separate plots and save them
    for i in range(len(training_progress)):
        plt.plot(training_progress[i])
        plt.title(f"Training Progress Model {i+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"exp/training_progress_model_{i+1}.png")
        plt.clf()

def TrainBase():
    # Load dataset  
    print('Training started', flush=True)
    model_net_10 = ModelNetDataset(root='data/ModelNet', name='10')
    print('Dataset_loaded..', flush=True)
 
    model_net_40 = ModelNetDataset(root='data/ModelNet', name='40')
    print('Dataset_loaded..', flush=True)
    md17_aspirin = MD17Dataset(root='data/MD17', name='aspirin')
    print('Dataset_loaded..', flush=True)
    md17_ethanol = MD17Dataset(root='data/MD17', name='ethanol')
    print('Dataset_loaded..', flush=True)
    md17_benzene = MD17Dataset(root='data/MD17', name='benzene')
    print('Dataset_loaded..', flush=True)

    coma = CoMADataset(root='data/CoMA')
    print('Dataset_loaded..', flush=True)


    Train_On_MPNN([model_net_10, model_net_40, md17_aspirin, md17_ethanol, md17_benzene, coma], batch_size=16)
    #Train_On_MPNN([coma], batch_size=128)

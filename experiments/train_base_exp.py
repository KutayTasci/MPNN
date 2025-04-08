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

    learning_rate = 0.001
    epochs = 3
    num_layers = 3

    #print the weights of the custom kernel
    print(f"Custom kernel weights: {custom_kernel.weight}")

    for dataset in datasets:
        data_loaders.append(dataset.get_loaders(batch_size=batch_size, shuffle=True))
        model = MPNN_GC(custom_kernel, dataset.num_features, dataset.edge_feature_dim, hidden_channels, dataset.num_classes, num_layers=num_layers)
        if dataset.task == 'regression':
            loss_fn = F.mse_loss
        else:
            loss_fn = F.cross_entropy
        
        loss_ls.append(loss_fn)
        optimizers.append(Adam(model.parameters(), lr=learning_rate))
        model = model.to(device)
        models.append(model)

    for model in models:
        print(f"Model weights: {model.layers[0].custom_kernel.weight}")

    training_progress = train.Train_Base(models, data_loaders, optimizers, loss_ls, device, epochs=epochs)

    #print the weights of the custom kernel
    print(f"Custom kernel weights: {custom_kernel.weight}")

    #print the weights of the custom kernel in the model
    for model in models:
        print(f"Model weights: {model.layers[0].custom_kernel.weight}")

    #save the custom kernel weights
    with open('custom_kernel_weights.yaml', 'w') as f:
        yaml.dump(custom_kernel.weight.tolist(), f)
    
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
    
    model_net_10 = ModelNetDataset(root='data/ModelNet', name='10')
 
    #model_net_40 = ModelNetDataset(root='data/ModelNet', name='40')

    #md17_aspirin = MD17Dataset(root='data/MD17', name='aspirin')
    #md17_ethanol = MD17Dataset(root='data/MD17', name='ethanol')
    #md17_benzene = MD17Dataset(root='data/MD17', name='benzene')

    coma = CoMADataset(root='data/CoMA')

    #Train_On_MPNN([model_net_10, model_net_40, md17_aspirin, md17_ethanol, md17_benzene, coma], batch_size=4)
    Train_On_MPNN([model_net_10, coma], batch_size=16)

    
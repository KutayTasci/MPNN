import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch.nn import Linear
from models.mpnn import MPNN_GC, EGNNNetwork
from data.dataset import QM9Dataset
import warnings
import training.train as train
import yaml
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

hidden_channels = 64

#Train with minibatches
def Train_On_EGNN(dataset, batch_size=128, custom_kernel=False, target=0):
    if custom_kernel:
        #read the weights from the yaml file
        with open('custom_kernel_weights.yaml', 'r') as f:
            weights_and_bias = yaml.safe_load(f)
        # Create a custom kernel using the weights
        custom_kernel = Linear(1, hidden_channels)
        custom_kernel.weight = torch.nn.Parameter(torch.tensor(weights_and_bias['weight']))
        '''
        if weights_and_bias['bias'] is not None:
            custom_kernel.bias = torch.nn.Parameter(torch.tensor(weights_and_bias['bias']))
        else:
            custom_kernel.bias = None
        '''
    else:
        custom_kernel = None
    print('Custom kernel:', custom_kernel)

    data_loader = dataset.get_loaders(batch_size=batch_size, shuffle=True)
    model = EGNNNetwork(dataset.num_features, hidden_channels, 1, edge_channels=dataset.edge_feature_dim, num_layers=7, custom_kernel=custom_kernel, task=dataset.task)

    if dataset.task == 'regression':
        #mae loss
        loss_fn = F.l1_loss
    else:
        loss_fn = F.cross_entropy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    learning_rate = 0.001
    epochs = 5
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    training_progress = train.Train_EGNN(model, data_loader, optimizer, loss_fn, device, epochs=epochs, target=target)
    return training_progress


def TrainEGNN():
    # Load dataset  
    print('Training started', flush=True)
    qm9 = QM9Dataset()

    for target in range(19):
        print(f'Training on target {target}', flush=True)
        # Train returns a tuple training_progress = (train_loss, val_loss, test_loss, val_mae, test_mae)
        baseline_results =  Train_On_EGNN(qm9, batch_size=1024, target=target)
        custom_kernel = Train_On_EGNN(qm9, batch_size=1024, custom_kernel=True, target=target)

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(baseline_results[0], label='Baseline', color='blue')
        plt.plot(custom_kernel[0], label='Custom Kernel', color='orange')
        plt.title(f'Training Loss for Target {target}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"exp_EGNN/0_training_loss_target_{target}.png")
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.plot(baseline_results[1], label='Baseline', color='blue')
        plt.plot(custom_kernel[1], label='Custom Kernel', color='orange')
        plt.title(f'Validation Loss for Target {target}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"exp_EGNN/1_validation_loss_target_{target}.png")
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.plot(baseline_results[3], label='Baseline', color='blue')
        plt.plot(custom_kernel[3], label='Custom Kernel', color='orange')
        plt.title(f'Validation MAE for Target {target}')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f"exp_EGNN/2_validation_mae_target_{target}.png")
        plt.clf()
        plt.figure(figsize=(10, 5))
        print('Printing test mae ------------')
        print('Baseline Test MAE:', baseline_results[4], 'Custom Kernel Test MAE:', custom_kernel[4])


        # Save the training progress to a file in yaml format
        with open(f"exp_EGNN/training_progress_target_{target}.yaml", 'w') as f:
            yaml.dump({
                'Baseline Training Progress': baseline_results[0],
                'Custom Kernel Training Progress': custom_kernel[0]
            }, f)
        with open(f"exp_EGNN/validation_progress_target_{target}.yaml", 'w') as f:
            yaml.dump({
                'Baseline Validation Progress': baseline_results[1],
                'Custom Kernel Validation Progress': custom_kernel[1]
            }, f)
        with open(f"exp_EGNN/validation_mae_target_{target}.yaml", 'w') as f:
            yaml.dump({
                'Baseline Validation MAE': baseline_results[3],
                'Custom Kernel Validation MAE': custom_kernel[3]
            }, f)
        with open(f"exp_EGNN/test_mae_target_{target}.yaml", 'w') as f:
            yaml.dump({
                'Baseline Test MAE': baseline_results[4],
                'Custom Kernel Test MAE': custom_kernel[4]
            }, f)
        


import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import time
import torch.profiler
import yaml
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, mean_absolute_error
import os



class Trainer_Base:
    def __init__(self, model, optimizer, loss_fn, device, edge_attr=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.edge_attr = False
        self.task = model.task

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        if self.edge_attr:
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device), data.pos.to(self.device))
        else:
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), None, data.pos.to(self.device))
        loss = self.loss_fn(out, data.y)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            if self.edge_attr:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device), data.pos.to(self.device))
            else:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), None, data.pos.to(self.device))
            loss = self.loss_fn(out[0], data.y[0])
            return loss.item()
    def test_accuracy(self, data):
        self.model.eval()
        with torch.no_grad():
            if self.edge_attr:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device), data.pos.to(self.device))
            else:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), None, data.pos.to(self.device))

            preds = out.argmax(dim=1).cpu()
            targets = data.y.cpu()
            acc = accuracy_score(targets, preds)
            return acc

    def test_mae(self, data):
        self.model.eval()
        with torch.no_grad():
            if self.edge_attr:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device), data.pos.to(self.device))
            else:
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), None, data.pos.to(self.device))

            preds = out.view(-1).cpu()
            targets = data.y.view(-1).cpu()
            mae = mean_absolute_error(targets, preds)
            return mae
        

class Trainer_EGNN:
    def __init__(self, model, optimizer, loss_fn, device, target=0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.task = model.task
        self.target = target

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        #calculate the loss on target column
        loss = self.loss_fn(out.view(-1), data.y[:, self.target])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            loss = self.loss_fn(out, data.y[:, self.target])
            return loss.item()
    def test_mae(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            preds = out.view(-1).cpu()
            targets = data.y[:, self.target].view(-1).cpu()
            mae = mean_absolute_error(targets, preds)
            return mae
        


def Train_Base(models, data_loaders, optimizers, loss_fns, device, epochs=3):
    matplotlib.use('Agg')  # Use a non-interactive backend (no GUI)
    trainers = []
    training_progress = [] 
    for i in range(len(models)):
        model = models[i].to(device)
        optimizer = optimizers[i]
        loss_fn = loss_fns[i]
        edge_attr = False
        if hasattr(model, 'edge_attr') and model.edge_attr:
            edge_attr = True
        trainer = Trainer_Base(model, optimizer, loss_fn, device, edge_attr=False)
        trainers.append(trainer)
        training_progress.append([])

    
    for epoch in range(epochs):
        for i in range(len(trainers)):
            trainer = trainers[i]
            traing_loader, val_loader, test_loader = data_loaders[i]
            loss_fn = loss_fns[i]
            loss = 0
            for data in tqdm(traing_loader, desc=f"Training Model {i+1}/{len(trainers)}", leave=False):
                data = data.to(device)
                loss += trainer.train(data)
            training_progress[i].append(loss)
            print(f"Epoch {epoch+1}/{epochs} | Model {i+1}/{len(trainers)} | Loss: {loss:.4f}")
            # Optional evaluation
            if test_loader is not None:
                metric_total = 0
                count = 0
                for data in test_loader:
                    data = data.to(device)
                    if trainer.task == 'classification':
                        metric_total += trainer.test_accuracy(data)
                    else:
                        metric_total += trainer.test_mae(data)
                    count += 1
                metric_avg = metric_total / count

                metric_name = 'Accuracy' if trainer.task == 'classification' else 'MAE'
                print(f"→ Test {metric_name}: {metric_avg:.4f}")
    
        #save the custom kernel weights
        # Access the layer
        layer = models[0].layers[0].custom_kernel

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
            
    return training_progress
                
            
def Train_EGNN(model, dataloader, optimizer, loss_fn, device, epochs=3, target=0):
    """
    Train the EGNN model.
    
    Args:
        model: The EGNN model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for the model.
        loss_fn: Loss function.
        device: Device to train on (CPU or GPU).
        epochs: Number of training epochs.
    """
    train_loss = []
    val_loss = []
    test_loss = []

    val_mae = []
    test_mae = []

    # Create a directory to save the training progress
    if not os.path.exists('exp_EGNN'):
        os.makedirs('exp_EGNN')

    train_loader, val_loader, test_loader = dataloader
    trainer = Trainer_EGNN(model, optimizer, loss_fn, device, target=target)
    for epoch in range(epochs):
        
        loss = 0
        for data in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(device)
            loss += trainer.train(data)
        train_loss.append(loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        # Optional evaluation
        if val_loader is not None:
            metric_total = 0
            count = 0
            for data in val_loader:
                data = data.to(device)
                metric_total += trainer.test_mae(data)
                count += 1
            metric_avg = metric_total / count

            metric_name = 'Accuracy' if trainer.task == 'classification' else 'MAE'
            val_loss.append(metric_avg)
            print(f"→ Validation {metric_name}: {metric_avg:.4f}")
            val_mae.append(metric_avg)

    if test_loader is not None:
        metric_total = 0
        count = 0
        for data in test_loader:
            data = data.to(device)
            metric_total += trainer.test_mae(data)
            count += 1
        metric_avg = metric_total / count

        metric_name = 'Accuracy' if trainer.task == 'classification' else 'MAE'
        print(f"→ Test {metric_name}: {metric_avg:.4f}")
        test_loss.append(metric_avg)
        test_mae.append(metric_avg)
    
    #create a tuple to save the training progress
    training_progress = (train_loss, val_loss, test_loss, val_mae, test_mae)
    return training_progress




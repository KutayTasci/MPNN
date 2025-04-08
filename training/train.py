import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import time
import torch.profiler



class Trainer_Base:
    def __init__(self, model, optimizer, loss_fn, device, edge_attr=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.edge_attr = False

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
        

        
        


def Train_Base(models, data_loaders, optimizers, loss_fns, device, epochs=3):
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
    
    return training_progress
                
            





'''
train_set.to(device)
test_set.to(device)
trainer = Trainer_ModelNet(model, optimizer, loss_fn, device)
if benchmark:
    start = time.time()

for epoch in range(epochs):
    loss = trainer.train(train_set)
    if not benchmark:
        test_loss = trainer.test(test_set)
        print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")
if benchmark:
    end = time.time()
    print(f"Training Time: {end-start:.4f} seconds")
    #print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Troughput: {epochs/(end-start):.3f} epoch/sec")
'''
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import time
import torch.profiler



class Trainer_QM9:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device))
        loss = self.loss_fn(out[0], data.y[0])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device))
            loss = self.loss_fn(out[0], data.y[0])
            return loss.item()
        
class Trainer_Flickr:
    # Flickr trains transductively
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device))
        loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            acc = correct / data.test_mask.sum().item()
            return acc
        
class Trainer_ModelNet:
    # Flickr trains transductively
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.pos.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device),data.edge_attr.to(self.device))
        loss = self.loss_fn(out[0], data.y[0])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.pos.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.edge_attr.to(self.device))
            loss = self.loss_fn(out[0], data.y[0])
            return loss.item()
        
        
    
def Train_QM9(model, data_loader, optimizer, loss_fn, device, epochs=3, benchmark=False):
    train_loader, val_loader, test_loader = data_loader
    trainer = Trainer_QM9(model, optimizer, loss_fn, device)
    if benchmark:
        start = time.time()
    
    for epoch in range(epochs):
        loss = 0
        for data in train_loader:
            data.to(device)
            loss += trainer.train(data)
        if not benchmark:
            test_loss = 0
            for data in test_loader:
                data.to(device)
                test_loss += trainer.test(data)
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")
    if benchmark:
        end = time.time()
        print(f"Training Time: {end-start:.4f} seconds")
        #print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Troughput: {epochs/(end-start):.3f} epoch/sec")
        val_loss = 0
        for data in val_loader:
            data.to(device)
            val_loss += trainer.test(data)
        print(f"Validation Loss: {val_loss:.4f}")

def Train_QM9_FB(model, data_set, optimizer, loss_fn, device, epochs=3, benchmark=False):
    train_set, val_set, test_set = data_set
    train_set.to(device)
    val_set.to(device)
    test_set.to(device)
    trainer = Trainer_QM9(model, optimizer, loss_fn, device)
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
        val_loss = trainer.test(val_set)
        print(f"Validation Loss: {val_loss:.4f}")

def Train_Flickr(model, data, optimizer, loss_fn, device, epochs=3, benchmark=False):
    trainer = Trainer_Flickr(model, optimizer, loss_fn, device)
    if benchmark:
        start = time.time()
    for epoch in range(epochs):
        loss = trainer.train(data) 

        if not benchmark:
            acc = trainer.test(data)
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")
    if benchmark:
        end = time.time()
        print(f"Training Time: {end-start:.4f} seconds")
        #print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Troughput: {epochs/(end-start):.3f} epoch/sec")


def Train_ModelNet(model, data_loader, optimizer, loss_fn, device, epochs=3, benchmark=False):
    train_loader, test_loader = data_loader
    trainer = Trainer_ModelNet(model, optimizer, loss_fn, device)
    if benchmark:
        start = time.time()
    
    for epoch in range(epochs):
        loss = 0
        for data in train_loader:
            data.to(device)
            loss += trainer.train(data)
        
    if benchmark:
        end = time.time()
        print(f"Training Time: {end-start:.4f} seconds")
        #print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Troughput: {epochs/(end-start):.3f} epoch/sec")
        test_loss = 0
        for data in test_loader:
            data.to(device)
            test_loss += trainer.test(data)
        print(f"Validation Loss: {test_loss:.4f}")

def Train_ModelNet_FB(model, data_loader, optimizer, loss_fn, device, epochs=3, benchmark=False):
    train_set, test_set = data_loader
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
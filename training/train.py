import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
import torch_geometric
from tqdm import tqdm
import time
import torch.profiler



class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.loss_fn(out, data.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            loss = self.loss_fn(out, data.y)
            return loss.item()
        

        
def Train_GraphClassification(model, data_loader, optimizer, loss_fn, device, epochs=3, benchmark=False):
    '''
    Every graph needs to be in a batch
    Every grah should have edge index , edge features, node features and node positions 
    x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
    '''
    train_loader, val_loader, test_loader = data_loader
    trainer = Trainer(model, optimizer, loss_fn, device)

    # Perform warmup
    if benchmark:
        print("Warming up GPU...")
        for _ in range(5):
            for data in train_loader:
                data.to(device)
                trainer.train(data)
        torch.cuda.synchronize()
    print("Starting training...")
    comp_time = 0
    if benchmark:
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        loss = 0
        for data in train_loader:
            data.to(device)
            c_start = time.time()
            loss += trainer.train(data)
            c_end = time.time()
            comp_time += (c_end - c_start)
            
        if not benchmark and val_loader is not None:
            test_loss = 0
            ctr = 0
            for data in val_loader:
                data.to(device)
                test_loss += trainer.test(data)
                ctr += 1
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Test Loss: {test_loss/ctr:.4f}")
    

    if benchmark:
        results = {}
        end = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"Training Time: {end-start:.4f} seconds")
        results['training_time'] = end - start
        print(f"Computation Time: {comp_time:.4f} seconds")
        results['computation_time'] = comp_time
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        results['peak_memory'] = peak_memory
        #print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Troughput: {epochs/(end-start):.3f} epoch/sec")
        results['throughput'] = epochs / (end - start)
        print(f"Computation Troughput: {epochs/comp_time:.4f} seconds")
        results['computation_throughput'] = epochs / comp_time
        if test_loader is not None:
            val_loss = 0
            ctr = 0
            for data in test_loader:
                data.to(device)
                val_loss += trainer.test(data)
                ctr += 1
            print(f"Validation Loss: {val_loss/ctr:.4f}")
        return results

    return None


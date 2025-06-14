import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from models.egnn import EGNN
from models.dimenet import DimeNet
from models.gemnet import GemNet
from data.QM9_dataset import QM9Dataset, QM9DatasetOriginal
from data.MD17_dataset import MD17Dataset
from data.ModelNet_dataset import ModelNetDataset
from data.fake_dataset import Fake_Dataset
from data.ppi_dataset import PPI_Dataset
import training.train as train
import torch._dynamo

def Train_On_EGNN(benchmark, dataset, model_type='cat', batch_size=128, hidden_channels=64, num_layers=7, learning_rate=0.0001, epochs=10):
    cfg = {
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "epochs": epochs
    }
    
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=batch_size, shuffle=True)
    
    in_channels = dataset.dataset[0].x.shape[1]
    hidden_channels = cfg["hidden_channels"]
    out_channels = dataset.dataset[0].y.shape[1]
    edge_channels = dataset.dataset[0].edge_attr.shape[1] if dataset.dataset[0].edge_attr is not None else 0
    num_layers = cfg["num_layers"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Initialize Model
    model = EGNN(in_channels, edge_channels, hidden_channels, out_channels,mode=model_type, num_layers=num_layers, task="regression").to(device)
    # compile the model with torch.compile(backend='inductor')
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, backend='inductor', dynamic=True)  
    torch.set_float32_matmul_precision('high')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train EGNN
    results = train.Train_GraphClassification(model, (train_loader, val_loader, test_loader), optimizer, F.l1_loss, device, epochs=epochs, benchmark=benchmark)
    return results


def RealDataset_Experiment(benchmark, dataset, dataset_name, batch_size=128, hidden_channels=64, num_layers=7, learning_rate=0.0001, epochs=10, name='aspirin'):
    
    
    # First, train the EGNN model on cat model
    try:
        cat_results = Train_On_EGNN(
            benchmark,
            dataset,
            model_type='cat',
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('Experiment Failed: CUDA out of memory.')
        else:
            print(f'Experiment Failed: {e}')
        cat_results = {
            'training_time': None,
            'computation_time': None,
            'peak_memory': None,
            'throughput': None,
            'computation_throughput': None,
            'validation_loss': None
        }
        torch.cuda.empty_cache()
    print(cat_results)
    # Then, train the EGNN model on sum model
    try:
        sum_results = Train_On_EGNN(
            benchmark,
            dataset,
            model_type='sum',
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('Experiment Failed: CUDA out of memory.')
        else:
            print(f'Experiment Failed: {e}')
        sum_results = {
            'training_time': None,
            'computation_time': None,
            'peak_memory': None,
            'throughput': None,
            'computation_throughput': None,
            'validation_loss': None
        }
        torch.cuda.empty_cache()
    print(sum_results)
    # create a summary of results for both experiments add them to a dictionary cat_results and sum_results are already dictionaries.
    # Add time stamps dataset name, model_type, batch_size, hidden_channels, num_layers, learning_rate, epochs and name for MD17 dataset as (MD17_name)
    cat_summary = {
        'dataset_name': dataset_name,
        'model_type': 'std_concat',
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'name': name if dataset_name == 'MD17' else None
    }
    cat_summary.update(cat_results)

    sum_summary = {
        'dataset_name': dataset_name,
        'model_type': 'alternative_sum',
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'name': name if dataset_name == 'MD17' else None
    }
    sum_summary.update(sum_results)
    print(cat_summary)
    print(sum_summary)
    return cat_summary, sum_summary


def FakeDataset_Experiment(benchmark, dataset_name, dataset=None ,batch_size=128, hidden_channels=64, num_layers=7, learning_rate=0.0001, epochs=10, num_nodes=1000, density=0.1, num_graphs=100):
    if dataset is None:
        num_edges = int(num_nodes * (num_nodes - 1) * density / 2)  # Calculate number of edges based on density
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        dataset = Fake_Dataset(num_nodes=num_nodes, avg_degree=avg_degree, num_graphs=num_graphs)
    
    try:
        # First, train the EGNN model on cat model
        cat_results = Train_On_EGNN(
            benchmark,
            dataset,
            model_type='cat',
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('Experiment Failed: CUDA out of memory.')
        else:
            print(f'Experiment Failed: {e}')
        cat_results = {
            'training_time': None,
            'computation_time': None,
            'peak_memory': None,
            'throughput': None,
            'computation_throughput': None
        }
        torch.cuda.empty_cache()
    print(cat_results)
    try:
        # Then, train the EGNN model on sum model
        sum_results = Train_On_EGNN(
            benchmark,
            dataset,
            model_type='sum',
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('Experiment Failed: CUDA out of memory.')
        else:
            print(f'Experiment Failed: {e}')
        sum_results = {
            'training_time': None,
            'computation_time': None,
            'peak_memory': None,
            'throughput': None,
            'computation_throughput': None
        }
        torch.cuda.empty_cache()
    print(sum_results)
    # create a summary of results for both experiments add them to a dictionary cat_results and sum_results are already dictionaries.
    # Add time stamps dataset name, model_type, batch_size, hidden_channels, num_layers, learning_rate, epochs and name for MD17 dataset as (MD17_name)
    cat_summary = {
        'dataset_name': dataset_name,
        'model_type': 'std_concat',
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'num_nodes': num_nodes,
        'density': density,
        'num_graphs': num_graphs
    }
    cat_summary.update(cat_results)
    sum_summary = {
        'dataset_name': dataset_name,
        'model_type': 'alternative_sum',
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'num_nodes': num_nodes,
        'density': density,
        'num_graphs': num_graphs
    }
    sum_summary.update(sum_results)
    print(cat_summary)
    print(sum_summary) 
    
    return cat_summary, sum_summary
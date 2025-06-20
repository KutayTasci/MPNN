
import experiments.egnn_experiment as egnn_exp
import experiments.CHGnet_experiment as chgnet_exp
from data.QM9_dataset import QM9Dataset, QM9DatasetOriginal
from data.MD17_dataset import MD17Dataset
from data.ModelNet_dataset import ModelNetDataset
from data.fake_dataset import Fake_Dataset
from data.ppi_dataset import PPI_Dataset
import logging

import os
import torch
import pandas as pd
def get_arch_list():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    arch_list = set()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        # Convert (major, minor) to "x.y" string
        arch_str = f"{cap[0]}.{cap[1]}"
        arch_list.add(arch_str)
    return ";".join(sorted(arch_list))

# Automatically set TORCH_CUDA_ARCH_LIST before importing cpp_extension
os.environ['TORCH_CUDA_ARCH_LIST'] = get_arch_list()
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

#test_exp.test_egnn()

chgnet = True
#test_exp.test_egnn()

experiment_file = 'egnn_fake_experiment.csv'
if chgnet:
    experiment_file = 'chgnet_fake500_experiment.csv'

if chgnet:
    experiment_func = chgnet_exp.FakeDataset_Experiment
else:
    experiment_func = egnn_exp.FakeDataset_Experiment

experiment_fake_50 = {
    'benchmark': True,
    'dataset_name': 'FakeDataset_50',
    'batch_size': [64, 128, 256, 512], # 128
    'hidden_channels': [32, 64, 128],
    'num_layers': 4,
    'learning_rate': 0.0001,
    'epochs': 10,
    'num_nodes': 50,
    'density': [0.1, 0.1, 0.50], 
    'num_graphs': [1024, 16384] # 512
}


experiment_fake_500 = {
    'benchmark': True,
    'dataset_name': 'FakeDataset_500',
    'batch_size': [16, 32, 64, 128],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10,
    'num_nodes': 500,
    'density': [ 0.05], # 0.001, 0.01,
    'num_graphs': [1024]
}

experiment_fake_5000 = {
    'benchmark': True,
    'dataset_name': 'FakeDataset_5000',
    'batch_size': [2, 4, 8, 16],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10,
    'num_nodes': 5000,
    'density': [0.0001, 0.001, 0.01],
    'num_graphs': [64, 2048]
}


experiments = [
    experiment_fake_50,
]

for experiment in experiments:
    batch_sizes = experiment['batch_size']
    hidden_channels = experiment['hidden_channels']
    num_layers = experiment['num_layers']
    learning_rate = experiment['learning_rate']
    epochs = experiment['epochs']
    benchmark = experiment['benchmark']
    dataset_name = experiment['dataset_name']
    density_list = experiment['density']
    num_graphs_list = experiment['num_graphs']

    for num_graphs in num_graphs_list:
        for density in density_list:
            num_nodes = experiment['num_nodes']
            num_edges = int(num_nodes * (num_nodes - 1) * density / 2)  # Calculate number of edges based on density
            avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
            dataset = Fake_Dataset(num_nodes=num_nodes, avg_degree=avg_degree, num_graphs=num_graphs, chgnet=chgnet)
            for batch_size in batch_sizes:
                for hidden_channel in hidden_channels:
                    cat_summary, sum_summary = experiment_func(
                        benchmark=benchmark,
                        dataset_name=dataset_name,
                        dataset = dataset,
                        batch_size=batch_size,
                        hidden_channels=hidden_channel,
                        num_layers=num_layers,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        num_nodes=experiment['num_nodes'],  # Default to 1000 if not specified
                        density=density,
                        num_graphs=num_graphs
                    )
                    # Save results to CSV file
                    results = pd.DataFrame([cat_summary, sum_summary])
                    print(results)
                    if not os.path.exists(experiment_file):
                        results.to_csv(experiment_file, index=False)
                    else:
                        results.to_csv(experiment_file, mode='a', header=False, index=False)

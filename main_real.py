
import experiments.egnn_experiment as egnn_exp
import experiments.dimenet_experiment as dimenet_exp
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
dimenet = True
#test_exp.test_egnn()

experiment_file = 'egnn_experiment.csv'
if dimenet:
    experiment_file = 'dimenet_experiment.csv'

if dimenet:
    experiment_func = dimenet_exp.RealDataset_Experiment
else:
    experiment_func = egnn_exp.RealDataset_Experiment
experiment_qm9 = {
    'benchmark': True,
    'dataset_name': 'QM9',
    'name': [None],
    'batch_size': [512, 1024, 2048, 4096],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10
}


experiment_modelnet = {
    'benchmark': True,
    'dataset_name': 'ModelNet',
    'name': [None],
    'batch_size': [1, 2, 4, 8, 16],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10
}

experiment_ppi = {
    'benchmark': True,
    'dataset_name': 'PPI',
    'name': [None],
    'batch_size': [1, 2, 4, 8, 16],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10
}

experiment_md17 = {
    'benchmark': True,
    'dataset_name': 'MD17',
    'name': ['benzene'], #aspirine
    'batch_size': [64, 128, 256, 512, 1024, 2048],
    'hidden_channels': [32, 64, 128],
    'num_layers': 7,
    'learning_rate': 0.0001,
    'epochs': 10
}

experiments = [
    #experiment_qm9,
    #experiment_ppi,
    #experiment_md17,
    experiment_modelnet
]

for experiment in experiments:
    batch_sizes = experiment['batch_size']
    hidden_channels = experiment['hidden_channels']
    num_layers = experiment['num_layers']
    learning_rate = experiment['learning_rate']
    epochs = experiment['epochs']
    benchmark = experiment['benchmark']
    dataset_name = experiment['dataset_name']
    names = experiment.get('name', [None])  # Default to [None] if 'name' is not provided


    if dataset_name == 'QM9':
        dataset = QM9Dataset(dimenet=dimenet)
    elif dataset_name == 'QM9Original':
        dataset = QM9DatasetOriginal(dimenet=dimenet)
    elif dataset_name == 'MD17':
        pass
    elif dataset_name == 'ModelNet':
        dataset = ModelNetDataset(dimenet=dimenet)
    elif dataset_name == 'PPI':
        dataset = PPI_Dataset(dimenet=dimenet)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    for name in names:
        if dataset_name == 'MD17':
            dataset = MD17Dataset(name=name, dimenet=dimenet)
        for batch_size in batch_sizes:
            for hidden_channel in hidden_channels:
            
                
                cat_summary, sum_summary = experiment_func(
                    benchmark=benchmark,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    batch_size=batch_size,
                    hidden_channels=hidden_channel,
                    num_layers=num_layers,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    name=name
                )
                # Save results to CSV file
                results = pd.DataFrame([cat_summary, sum_summary])
                print(results)
                if not os.path.exists(experiment_file):
                    results.to_csv(experiment_file, index=False)
                else:
                    results.to_csv(experiment_file, mode='a', header=False, index=False)

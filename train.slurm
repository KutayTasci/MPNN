#!/bin/bash
#SBATCH --job-name=mpnn_training
#SBATCH --account=sxk1942
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log     # %j will be replaced with the job ID
#SBATCH --error=logs/train_%j.err      # Optional: separate stderr log

pwd

# Load modules
module load CUDA/12.6.0
module load singularity/3.10.4

singularity exec --nv updated_enviorment.sif export TORCH_CUDA_ARCH_LIST='8.9'
singularity exec --nv updated_enviorment.sif export CUDA_HOME=/usr/local/cuda-12.6
singularity exec --nv updated_enviorment.sif export PATH=$CUDA_HOME/bin:$PATH
singularity exec --nv updated_enviorment.sif python main.py

# Run your training script
python -u main.py
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
module load CUDA/12.1.1
module load Python/3.9.6-GCCcore-11.2.0

# Activate your environment
source /home/kxt437/env/MPNN/bin/activate

# Check that Python is available (for debugging)
which python

# Run your training script
python -u main.py
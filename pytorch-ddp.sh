#!/bin/bash
#SBATCH --job-name=embed-ddp

#SBATCH -p gpu-h100                           # Partition (queue) name
#SBATCH -N 1                                  # Number of nodes (1 for GPU jobs)
#SBATCH --ntasks-per-node=8                   # Number of CPU cores per node
#SBATCH --mem=32000                           # Memory allocation (in MB)
#SBATCH --time=71:59:59                       # Max execution time (hh:mm:ss)
#SBATCH --output=/users/scratch1/jbuler/eucaim/outputs/%x_%j.out




# export NCCL_DEBUG=INFO
# export OMP_NUM_THREADS=4
# torchrun --nproc_per_node=4 t.py


GPU_NUMBER=4
source /users/project1/pt01190/EUCAIM-PG-GUM/.venv_euc/bin/activate

torchrun --nproc_per_node=$GPU_NUMBER /users/project1/pt01190/EUCAIM-PG-GUM/embed_ddp/main.py

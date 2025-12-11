#!/bin/bash
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=8                  # 8 nodes
#SBATCH --ntasks-per-node=1        # 1 task per node
#SBATCH --cpus-per-task=32         # CPUs per task
#SBATCH --time=71:59:00
#SBATCH --job-name=embed_ddp
#SBATCH --output=/outputs/%x_%j.out

source /venv/bin/activate

# Get the first node as master
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

export WORLD_SIZE=$(($SLURM_NNODES * 1))  # 1 GPU per node

echo "=========================================="
echo "SLURM DDP Configuration"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Nodes:           $SLURM_JOB_NODELIST"
echo "Node count:      $SLURM_NNODES"
echo "Master node:     $MASTER_ADDR"
echo "World size:      $WORLD_SIZE"
echo "My node rank:    $SLURM_NODEID"
echo "My proc rank:    $SLURM_PROCID"
echo "=========================================="

# Test GPU access first
echo "Testing GPU visibility..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 nvidia-smi --query-gpu=name --format=csv,noheader

echo "Starting PyTorch DDP training..."

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main.py

echo "Training finished with exit code: $?"
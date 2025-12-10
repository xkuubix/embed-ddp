import os
import torch
import torch.distributed as dist


def init_distributed():
    """
    Initialize DDP for both single-node multi-GPU and multi-node single-GPU scenarios.

    Detects environment variables set by:
    - torchrun (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
    - SLURM (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, etc.)
    - Manual setup

    Returns:
        is_ddp (bool): Whether DDP is enabled
        local_rank (int): Local rank (GPU index on this node)
        rank (int): Global rank across all nodes
        world_size (int): Total number of processes
    """

    # Check if running under torchrun or SLURM
    # torchrun sets these automatically
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"Detected torchrun environment: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    return True, local_rank, rank, world_size

def gather_from_ranks(obj, is_ddp, world_size):
    if not is_ddp:
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out

def cleanup_distributed():
    """Cleanup DDP resources"""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
import torch
import os
import torch.distributed as dist


def init_distributed():
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank, rank, world_size = 0, 0, 1
    return is_ddp, local_rank, rank, world_size


def gather_from_ranks(obj, is_ddp, world_size):
    if not is_ddp:
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out

def cleanup_distributed():
    try:
        dist.destroy_process_group()
    except Exception:
        pass
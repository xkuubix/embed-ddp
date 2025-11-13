import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from models import GatedAttentionMIL

def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.track_running_stats = False
        model.running_mean = None
        model.running_var = None

def build_model(device, is_ddp, rank, local_rank, use_pretrained=True):
    """
    Builds model, handling pretrained-state broadcasting for DDP to avoid duplicated downloads.
    """
    if is_ddp and use_pretrained:
        if rank == 0:
            m = GatedAttentionMIL()
            m.apply(deactivate_batchnorm)
            sd = m.state_dict()
        else:
            m = GatedAttentionMIL()
            m.apply(deactivate_batchnorm)
            sd = None

        obj_list = [sd]
        # broadcast model state dict
        dist.broadcast_object_list(obj_list, src=0)
        sd = obj_list[0]
        m.load_state_dict(sd)
    else:
        m = GatedAttentionMIL()
        m.apply(deactivate_batchnorm)

    print(f"Model {m.__class__.__name__} built on device {device}")
    
    m = m.to(device)
    if is_ddp:
        m = DDP(m, device_ids=[local_rank], output_device=local_rank)
    return m
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(device, is_ddp, rank, local_rank, use_pretrained=True):
    """
    Builds ResNet50, handling pretrained-state broadcasting for DDP to avoid duplicated downloads.
    """
    if is_ddp and use_pretrained:
        if rank == 0:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            m.fc = nn.Linear(m.fc.in_features, 2)
            sd = m.state_dict()
        else:
            m = resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            sd = None

        obj_list = [sd]
        # broadcast model state dict
        dist.broadcast_object_list(obj_list, src=0)
        sd = obj_list[0]
        m.load_state_dict(sd)
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        m = resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, 2)

    m = m.to(device)
    if is_ddp:
        m = DDP(m, device_ids=[local_rank], output_device=local_rank)
    return m
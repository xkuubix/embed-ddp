import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from models import GatedAttentionMIL
from torchvision import models
import torch
import copy


def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.track_running_stats = False
        model.running_mean = None
        model.running_var = None

def build_model(device, is_ddp, rank, local_rank, use_pretrained=True, backbone='convnext_base'):
    """
    Builds model, handling pretrained-state broadcasting for DDP to avoid duplicated downloads.
    """
    if is_ddp and use_pretrained:
        if rank == 0:
            m = GatedAttentionMIL(backbone=backbone, pretrained=use_pretrained)
            m.apply(deactivate_batchnorm)
            sd = m.state_dict()
        else:
            m = GatedAttentionMIL(backbone=backbone, pretrained=use_pretrained)
            m.apply(deactivate_batchnorm)
            sd = None

        obj_list = [sd]
        # broadcast model state dict
        dist.broadcast_object_list(obj_list, src=0)
        sd = obj_list[0]
        m.load_state_dict(sd)
    else:
        m = GatedAttentionMIL(backbone=backbone, pretrained=use_pretrained)
        m.apply(deactivate_batchnorm)
    try:
        print(f"Model {m.__class__.__name__} (with backbone {m.feature_extractor.__class__.__name__}) built on device {device}")
    except:
        print(f"Model {m.__class__.__name__} built on device {device}")
 
    m = m.to(device)
    if is_ddp:
        m = DDP(m, device_ids=[local_rank], output_device=local_rank)
    return m


class ModelEMA:
    def __init__(self, model, decay=0.999):
        # make a copy of the model for EMA
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # update EMA weights after each training step
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1-self.decay)

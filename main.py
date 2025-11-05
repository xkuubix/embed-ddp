import torch
import os
import torch.distributed as dist
from torch import nn
from utils import setup_logging, train_loop
from model_utils import build_model
from ddp_utils import init_distributed
from data_utils import load_and_process_df, preprocess_tensor, create_dataloaders
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

META_DATA_PATH = '/users/scratch1/mg_25/EMBED/tables/EMBED_OpenData_metadata.csv'
CLINICAL_DATA_PATH = '/users/scratch1/mg_25/EMBED/tables/EMBED_OpenData_clinical.csv'


def main():
    logger = setup_logging()

    df = load_and_process_df(META_DATA_PATH, CLINICAL_DATA_PATH)
    logger.info(f"Loaded dataframe with {len(df)} samples")

    is_ddp, local_rank, rank, world_size = init_distributed()
    logger.info(f"DDP mode: {is_ddp} rank={rank} world_size={world_size} local_rank={local_rank}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    main_df = df[['new_path', 'label', 'cohort_num_x']].copy()
    del df
    train_files = main_df[main_df['cohort_num_x'] == 1]
    val_files = main_df[main_df['cohort_num_x'] == 2]
    logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")

    aug = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )
    ])

    def transforms_train(img):
        img = aug(img)
        img = preprocess_tensor(img)
        return img

    def transforms_val(img):
        img = preprocess_tensor(img)
        return img

    transform = {
        'train': transforms_train,
        'val': transforms_val
    }

    train_dl, val_dl, train_sampler = create_dataloaders(train_files, val_files, transform, is_ddp, rank, world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(device, is_ddp, rank, local_rank, use_pretrained=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss().to(device)

    os.makedirs('./checkpoints', exist_ok=True)

    train_loop(model, opt, crit, train_dl, val_dl, train_sampler, is_ddp, rank, world_size, logger, num_epochs=100)

    if is_ddp:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
import torch
import os
from torch import nn
from utils import setup_logging, format_counts, reset_seed, train_loop
from model_utils import build_model
from ddp_utils import init_distributed, cleanup_distributed
import data_utils as du
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

META_DATA_PATH = '/users/scratch1/mg_25/EMBED/tables/EMBED_OpenData_metadata.csv'
CLINICAL_DATA_PATH = '/users/scratch1/mg_25/EMBED/tables/EMBED_OpenData_clinical.csv'

SEED = 42

def main():
    logger = setup_logging()

    is_ddp, local_rank, rank, world_size = init_distributed()
    if rank == 0:
        logger.info(f"DDP mode: {is_ddp} rank={rank} world_size={world_size}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    reset_seed(SEED)
    df = du.load_and_process_df(META_DATA_PATH, CLINICAL_DATA_PATH)
    if rank == 0: logger.info(f"Loaded dataframe with {len(df)} samples")

    main_df = df[['new_path', 'label', 'cohort_num_x', 'empi_anon']].copy()
    train_df = main_df[main_df['cohort_num_x'] == 1]
    test_df = main_df[main_df['cohort_num_x'] == 2]
    train_files, val_files = du.split_by_patient_stratified(
        train_df,
        patient_col='empi_anon',
        label_col='label',
        val_size=0.2,
        random_state=SEED
    )
    del df, main_df, train_df
    if rank == 0:
        logger.info(format_counts(train_files, "Train"))
        logger.info(format_counts(val_files, "Val"))
        logger.info(format_counts(test_df, "Test"))

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
        img = du.preprocess_tensor(img)
        return img

    def transforms_val(img):
        img = du.preprocess_tensor(img)
        return img

    transform = {
        'train': transforms_train,
        'val': transforms_val
    }

    train_dl, val_dl, train_sampler = du.create_dataloaders(
        train_files, val_files, transform, is_ddp, rank, world_size, SEED)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(
        device,
        is_ddp,
        rank,
        local_rank,
        use_pretrained=True
        )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss().to(device)

    os.makedirs('./checkpoints', exist_ok=True)

    train_loop(
        model,
        opt,
        crit,
        train_dl,
        val_dl,
        train_sampler,
        is_ddp,
        rank,
        world_size,
        logger,
        num_epochs=100
    )

    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()
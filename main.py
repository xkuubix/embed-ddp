import torch
import os
from torch import nn
from utils import setup_logging, format_counts, reset_seed, train_loop
from model_utils import build_model
from ddp_utils import init_distributed, cleanup_distributed
import albumentations as A
import data_utils as du
# %%


SEED = 42
prefix = '/users/scratch1/mg_25/EMBED/datathon_tables/'
META_DATA_PATH = f'{prefix}EMBED_OpenData_NUS_Datathon_metadata_reduced.csv'
CLINICAL_DATA_PATH = f'{prefix}EMBED_OpenData_NUS_Datathon_clinical_reduced.csv'


def main():
    logger = setup_logging()

    is_ddp, local_rank, rank, world_size = init_distributed()
    if rank == 0:
        logger.info(f"DDP mode: {is_ddp} rank={rank} local rank={local_rank} world_size={world_size}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    reset_seed(SEED)
    df = du.load_and_process_df(META_DATA_PATH, CLINICAL_DATA_PATH)
    df['target'] = df['label'].map({0: 'negative', 1: 'suspicious'})
    
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

    aug_prior = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Downscale(scale_range=(0.75, 0.95), p=0.125),
        A.OneOf([
            A.RandomToneCurve(scale=0.3, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                                       contrast_limit=(-0.4, 0.5),
                                       brightness_by_max=True, p=0.5)
        ], p=0.5),
        A.OneOf([
            A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.2, 0.2)},
                     scale=(0.85, 1.15), rotate=(-30, 30), p=0.6),
            A.ElasticTransform(alpha=1, sigma=20, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2)
        ], p=0.5),
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.1, 0.25), p=0.25),
    ], p=0.9)

    def transforms_train(img):
        img = aug_prior(image=img)['image']
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
        train_files, val_files, transform, is_ddp, rank, world_size,
        num_workers=2,
        per_gpu_batch=6,
        seed=SEED
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(
        device,
        is_ddp,
        rank,
        local_rank,
        use_pretrained=True,
        backbone='convnext_small'
        )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss().to(device)

    os.makedirs('./checkpoints', exist_ok=True)

    train_loop(
        model,
        opt,
        scheduler,
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
import pandas as pd
import pydicom
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from BreastDataset import BreastDataset
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def load_and_process_df(metadata_csv_path: str, clinical_csv_path: str) -> pd.DataFrame:
    table = pd.read_csv(metadata_csv_path)
    clinical_table = pd.read_csv(clinical_csv_path)

    clin_L = clinical_table[clinical_table['side']=='L']
    clin_R = clinical_table[clinical_table['side']=='R']
    clin_B = clinical_table[clinical_table['side'].isin(['B', None])]

    # left breast
    merged_L = pd.merge(
    clin_L, table,
    left_on=['empi_anon','acc_anon'], right_on=['empi_anon','acc_anon']
    )
    merged_L = merged_L[merged_L['ImageLateralityFinal']=='L']

    # right breast
    merged_R = pd.merge(
    clin_R, table,
    left_on=['empi_anon','acc_anon'], right_on=['empi_anon','acc_anon']
    )
    merged_R = merged_R[merged_R['ImageLateralityFinal']=='R']

    # both/NaN
    merged_B = pd.merge(
    clin_B, table,
    left_on=['empi_anon','acc_anon'], right_on=['empi_anon','acc_anon']
    )
    merged_all = pd.concat([merged_L, merged_R, merged_B], ignore_index=True)

    # normalize values (uppercase, strip) and show original counts
    merged_all['asses_norm'] = merged_all['asses'].astype(str).str.strip().str.upper()
    print("Original counts:\n", merged_all['asses_norm'].value_counts(dropna=False))

    # keep only letters we want to map (drop A -> 0 and P -> 3)
    keep_letters = {'N','B','S','M','K'}   # N=1, B=2, S=4, M=5, K=6
    df = merged_all[merged_all['asses_norm'].isin(keep_letters)].copy()

    # mapping to new labels
    map_letters_to_label = {
    'N': 'negative',   # BIRADS 1
    'B': 'negative',   # BIRADS 2
    'S': 'suspicious', # BIRADS 4
    'M': 'suspicious', # BIRADS 5
    'K': 'suspicious', # BIRADS 6
    }

    df['label'] = df['asses_norm'].map(map_letters_to_label)

    # quick sanity counts
    print("\nKept counts by letter:\n", df['asses_norm'].value_counts())
    print("\nMapped counts by new label:\n", df['label'].value_counts())

    DICOM_BASE_OLD = "/mnt/NAS2/mammo/anon_dicom/"
    DICOM_BASE_NEW = "/users/scratch1/mg_25/EMBED/images/"

    # make new path
    def get_local_path(original_path: str, old_base: str, new_base: str) -> str:
        return original_path.replace(old_base, new_base, 1)

    df["new_path"] = df["anon_dicom_path"].apply(lambda x: get_local_path(x, DICOM_BASE_OLD, DICOM_BASE_NEW))

    print(df)
    return df


def get_pixels_no_voi(ds, apply_voi=True, lut_index=0):
    """
    Returns pixel data as numpy array.
    - If apply_voi=True and VOI LUT (sequence or function) exists, applies it.
    - Otherwise returns raw pixel_array exactly as stored (no rescale, no windowing).

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset.
    apply_voi : bool
        Whether to apply VOI LUT transformation.
    lut_index : int
        Index of LUT to use if multiple VOI LUT Sequence entries exist.

    Returns
    -------
    np.ndarray : The image as uint16 array (0–2**BitsStored-1 range).
    """
    img = ds.pixel_array
    if not apply_voi:
        return img.copy()
    
    bits_stored = int(getattr(ds, 'BitsStored', 12))
    max_val = 2**bits_stored - 1

    # --- Case 1: explicit VOI LUT Sequence ---
    if 'VOILUTSequence' in ds:
        seq = ds.VOILUTSequence
        lut_index = np.clip(lut_index, 0, len(seq)-1)
        lut_item = seq[lut_index]
        
        lut_data = lut_item.LUTData
        lut_desc = lut_item.LUTDescriptor  # [num_entries, first_mapped_pixel_value, bits_per_entry]
        num_entries, first_map, bits_per_entry = [int(x) for x in lut_desc]

        lut_array = np.asarray(lut_data, dtype=np.uint16)
        if len(lut_array) < num_entries:
            lut_array = np.pad(lut_array, (0, num_entries - len(lut_array)), mode='edge')

        img = img.astype(np.int32) - first_map
        img = np.clip(img, 0, num_entries - 1)
        img_voi = lut_array[img]
        return img_voi.astype(np.uint16)

    # --- Case 2: VOI LUT Function (LINEAR / SIGMOID) ---
    voi_lut_func = getattr(ds, 'VOILUTFunction', '').upper()
    if voi_lut_func not in ('LINEAR', 'SIGMOID'):
        # return raw pixels exactly
        return img.copy()

    img = img.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    img = img * slope + intercept


    voi_lut_func = voi_lut_func.upper()
    window_centers = getattr(ds, 'WindowCenter', None)
    window_widths = getattr(ds, 'WindowWidth', None)

    # pick first WC/WW if multiple (["NORMAL", "HARDER", "SOFTER"])
    wc = float(window_centers[0]) if isinstance(window_centers, (list, pydicom.multival.MultiValue)) else float(window_centers or img.mean())
    ww = float(window_widths[0]) if isinstance(window_widths, (list, pydicom.multival.MultiValue)) else float(window_widths or (img.max() - img.min()))

    if voi_lut_func == 'LINEAR':
        img_voi = np.clip((img - (wc - 0.5 - (ww-1)/2)) / (ww - 1), 0, 1)
    elif voi_lut_func == 'SIGMOID':
        img_voi = 1 / (1 + np.exp(-4 * (img - wc) / ww))
    else:
        raise ValueError(f"Unsupported VOI LUT Function: {voi_lut_func}")

    img_voi_bits = np.round(img_voi * max_val).astype(np.uint16)
    
    return img_voi_bits

def load_dicom_as_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    # arr = ds.pixel_array.astype(np.float32)
    arr = get_pixels_no_voi(ds).astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    return arr


def preprocess_tensor(img: torch.Tensor, size=(1024, 1024)):
    """
    Resize and normalize a float tensor in tensor-space.
    - img: torch.Tensor, shape (H,W) or (C,H,W). dtype=float (expected from load_dicom_as_image).
    - Returns: torch.FloatTensor shape (3, size[0], size[1]) normalized by IM_MEAN/IM_STD.
    """
    IM_MEAN = np.array([0.485,0.456,0.406], np.float32)
    IM_STD  = np.array([0.229,0.224,0.225], np.float32)
    # ensure tensor
    if not torch.is_tensor(img):
        img = torch.tensor(img)
    # if H,W -> 1,H,W
    if img.ndim == 2:
        img = img.unsqueeze(0)
    # if single-channel expand to 3 channels by repeating (keeps radiometry)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    # if channels last (H,W,C) — unlikely here — convert (not expected in this code path)
    if img.ndim == 3 and img.shape[0] not in (1,3):
        # assume H,W,C -> C,H,W
        img = img.permute(2,0,1)

    img = img.float()
    # resize in tensor space: expect N,C,H,W for interpolate
    img = F.interpolate(img.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

    # per-sample per-channel scaling to [0,1] (preserves relative contrast)
    c = img.shape[0]
    mins = img.view(c, -1).min(dim=1)[0].view(c,1,1)
    maxs = img.view(c, -1).max(dim=1)[0].view(c,1,1)
    img = (img - mins) / (maxs - mins + 1e-8)

    # normalize using IM_MEAN/IM_STD (create tensors on same device)
    mean = torch.tensor(IM_MEAN, dtype=img.dtype, device=img.device).view(-1,1,1)
    std  = torch.tensor(IM_STD,  dtype=img.dtype, device=img.device).view(-1,1,1)
    img = (img - mean) / (std + 1e-8)
    return img


def create_dataloaders(train_files, val_files, transform, is_ddp, rank, world_size, local_rank, num_workers=12, per_gpu_batch=8):
    """
    Returns train_dl, val_dl, train_sampler (None for non-DDP).
    For DDP uses an oversampled Subset (broadcasted indices) so sampling is consistent across ranks.
    """
    # single-process: WeightedRandomSampler
    if not is_ddp:
        labels = train_files['label'].str.lower().map({'negative': 0, 'suspicious': 1}).astype(np.int64).values
        class_sample_count = np.bincount(labels)
        class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
        weights = 1.0 / class_sample_count
        sample_weights = torch.tensor([weights[int(t)] for t in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_ds = BreastDataset(train_files, transform=transform)
        val_ds = BreastDataset(val_files, transform=transform)

        train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        return train_dl, val_dl, None

    # DDP branch: create balanced subset (upsample minority) on rank 0 and broadcast indices
    train_ds = BreastDataset(train_files, transform=transform)
    val_ds = BreastDataset(val_files, transform=transform)

    train_labels_arr = train_files['label'].str.lower().map({'negative': 0, 'suspicious': 1}).astype(np.int64).values

    SEED = 42
    np.random.seed(SEED)

    if rank == 0:
        pos = np.where(train_labels_arr == 1)[0]
        neg = np.where(train_labels_arr == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            all_idx = np.arange(len(train_ds)).tolist()
        else:
            if len(pos) < len(neg):
                pos_up = np.random.choice(pos, size=len(neg), replace=True)
                all_idx = np.concatenate([neg, pos_up])
            else:
                neg_up = np.random.choice(neg, size=len(pos), replace=True)
                all_idx = np.concatenate([pos, neg_up])
            np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
    else:
        all_idx = None

    obj_list = [all_idx]
    # broadcast indices
    dist.broadcast_object_list(obj_list, src=0)
    all_idx = obj_list[0]
    if isinstance(all_idx, np.ndarray):
        all_idx = all_idx.tolist()
    train_subset = Subset(train_ds, all_idx)

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_dl = DataLoader(train_subset, batch_size=per_gpu_batch, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=per_gpu_batch, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, train_sampler
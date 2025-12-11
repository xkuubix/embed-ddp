import pandas as pd
import pydicom
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from utils import seed_worker
import cv2


def load_and_process_df(metadata_csv_path: str, clinical_csv_path: str, rank: int) -> pd.DataFrame:
    """
    Load and process metadata and clinical data with improved filtering logic
    
    Args:
        metadata_csv_path: Path to image metadata CSV
        clinical_csv_path: Path to clinical data CSV
    
    Returns:
        DataFrame with merged and processed data
    """
    # ===== Load and filter metadata =====
    table = pd.read_csv(metadata_csv_path, low_memory=False)
    if rank == 0:
        print(f"Original metadata rows: {len(table)}")
    
    # Apply filters
    table = table[table['FinalImageType'] == '2D']
    table = table[table['ViewPosition'].isin(['MLO', 'CC'])]
    
    if rank == 0:
        print(f"After FinalImageType and ViewPosition filtering: {len(table)}")
    
    # ===== Load and filter clinical data =====
    clinical_table = pd.read_csv(clinical_csv_path, low_memory=False)
    
    if rank == 0:
        print(f"\nOriginal clinical rows: {len(clinical_table)}")
    
    # Clean and filter BIRADS assessments
    if 'exam_birads' in clinical_table.columns:
        clinical_table['asses_clean'] = clinical_table['exam_birads'].astype(str).str.strip().str.upper()
    else:
        clinical_table['asses_clean'] = clinical_table['asses'].astype(str).str.strip().str.upper()
    if rank == 0:
        print("\nOriginal BIRADS distribution:")
        print(clinical_table['asses_clean'].value_counts(dropna=False))
    
    # Filter to keep only desired BIRADS categories
    keep_letters = {'N', 'B', 'S', 'M', 'K'}  # N=1, B=2, S=4, M=5, K=6
    clinical_filtered = clinical_table[clinical_table['asses_clean'].isin(keep_letters)].copy()
    
    if rank == 0:
        print(f"\nAfter BIRADS filtering: {len(clinical_filtered)}")
        print("Kept BIRADS distribution:")
        print(clinical_filtered['asses_clean'].value_counts())
    
    # Map BIRADS to binary target and label
    birads_to_target = {'N': 0, 'B': 0, 'S': 1, 'M': 1, 'K': 1}
    birads_to_label = {
        'N': 'negative',   # BIRADS 1
        'B': 'negative',   # BIRADS 2
        'S': 'suspicious', # BIRADS 4
        'M': 'suspicious', # BIRADS 5
        'K': 'suspicious', # BIRADS 6
    }
    
    clinical_filtered['target'] = clinical_filtered['asses_clean'].map(birads_to_target)
    clinical_filtered['label'] = clinical_filtered['asses_clean'].map(birads_to_label)
    if rank == 0:
        print(f"\nTarget distribution in clinical data:")
        print(f"Negative (0): {(clinical_filtered['target'] == 0).sum()}")
        print(f"Positive (1): {(clinical_filtered['target'] == 1).sum()}")
    
    clinical_filtered = clinical_filtered.drop_duplicates(
    subset=['empi_anon','acc_anon','side'], keep='first')

    # ===== Split clinical data by laterality =====
    clin_L = clinical_filtered[clinical_filtered['side'] == 'L'].copy()
    clin_R = clinical_filtered[clinical_filtered['side'] == 'R'].copy()
    clin_B = clinical_filtered[clinical_filtered['side'] == 'B'].copy() # empty for datathon .csv
    clin_NaN = clinical_filtered[clinical_filtered['side'].isna()].copy() # empty for datathon .csv
    
    if rank == 0:
        print(f"\nClinical data by side:")
        print(f"  L: {len(clin_L)}")
        print(f"  R: {len(clin_R)}")
        print(f"  B: {len(clin_B)}")
        print(f"  NaN: {len(clin_NaN)}")
    
    # ===== Merge metadata with clinical data by laterality =====
    # Side L: match only with ImageLateralityFinal == 'L'
    merged_L = pd.merge(
        clin_L, 
        table[table['ImageLateralityFinal'] == 'L'], 
        on=['empi_anon', 'acc_anon'],
        how='inner'
    )
    
    # Side R: match only with ImageLateralityFinal == 'R'
    merged_R = pd.merge(
        clin_R, 
        table[table['ImageLateralityFinal'] == 'R'], 
        on=['empi_anon', 'acc_anon'],
        how='inner'
    )
    
    # Side B (bilateral): match with ALL images (both L and R)
    merged_B = pd.merge(
        clin_B, 
        table,  # No laterality filter
        on=['empi_anon', 'acc_anon'],
        how='inner'
    )
    
    # Side NaN: match with ALL images (both L and R)
    merged_NaN = pd.merge(
        clin_NaN, 
        table,  # No laterality filter
        on=['empi_anon', 'acc_anon'],
        how='inner'
    )
    
    if rank == 0:
        print(f"\nMerge results by side:")
        print(f"  L: {len(merged_L)} images")
        print(f"  R: {len(merged_R)} images")
        print(f"  B: {len(merged_B)} images")
        print(f"  NaN: {len(merged_NaN)} images")
    
    # Combine all merged data
    merged_all = pd.concat([merged_L, merged_R, merged_B, merged_NaN], 
                           ignore_index=True)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FINAL DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total images: {len(merged_all)}")
        print(f"Unique patients: {merged_all['empi_anon'].nunique()}")
        print(f"Unique accessions: {merged_all['acc_anon'].nunique()}")
        
        print(f"\nTarget distribution:")
        print(f"  Negative (0): {(merged_all['target'] == 0).sum()} ({(merged_all['target'] == 0).mean():.1%})")
        print(f"  Positive (1): {(merged_all['target'] == 1).sum()} ({(merged_all['target'] == 1).mean():.1%})")
        
        print(f"\nLabel distribution:")
        print(merged_all['label'].value_counts())
        
        print(f"\nBIRADS distribution:")
        print(merged_all['asses_clean'].value_counts())
        
        print(f"\nView distribution:")
        if 'ViewPosition' in merged_all.columns:
            print(merged_all['ViewPosition'].value_counts())
        
        print(f"\nLaterality distribution:")
        if 'ImageLateralityFinal' in merged_all.columns:
            print(merged_all['ImageLateralityFinal'].value_counts())
    
    # ===== Update file paths (if needed) =====
    if 'anon_dicom_path' in merged_all.columns:
        DICOM_BASE_NEW = "/users/scratch1/mg_25/EMBED/images/"
        DICOM_BASE_OLD = "images/" #<- datathon df
        # DICOM_BASE_OLD = "/mnt/NAS2/mammo/anon_dicom/" # <- primary df
        
        merged_all['new_path'] = merged_all['anon_dicom_path'].str.replace(
            DICOM_BASE_OLD, DICOM_BASE_NEW, n=1, regex=False
        )
        if rank == 0:
            print(f"\nUpdated DICOM paths to new base directory.")
    
    if rank == 0:
        print(f"{'='*60}\n")
    dup_count = merged_all.duplicated(subset=['empi_anon', 'acc_anon', 'ImageLateralityFinal', 'ViewPosition']).sum()
    if dup_count > 0:
        if rank == 0:
            print(f"Duplicate image records: {dup_count} of total ({len(merged_all)})")
        merged_all.drop_duplicates(subset=['empi_anon', 'acc_anon', 'ImageLateralityFinal', 'ViewPosition'], inplace=True)
        if rank == 0:
            print(f"Duplicates removed, new count: {len(merged_all)}")
    
    return merged_all


def split_by_patient_stratified(df: pd.DataFrame, patient_col='empi_anon', label_col='label', val_size=0.2, random_state=42):
    """
    Stratified split of patients to train and validation sets.
    All exams/images from a single patient go to the same split.
    
    Returns:
        train_df, val_df
    """
    # get one label per patient (first exam)
    patient_labels = df.groupby(patient_col)[label_col].first().reset_index()

    # stratified split of patients
    train_patients, val_patients = train_test_split(
        patient_labels[patient_col],
        test_size=val_size,
        stratify=patient_labels[label_col],
        random_state=random_state
    )

    train_df = df[df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].reset_index(drop=True)

    return train_df, val_df


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


def crop_to_breast(img, threshold=0.05):
    mask = (img.mean(0) > threshold)
    coords = mask.nonzero(as_tuple=False)
    if len(coords) == 0:
        return img
    y_min, x_min = coords.min(0).values
    y_max, x_max = coords.max(0).values
    return img[:, y_min:y_max+1, x_min:x_max+1]


def pad_to_square(img):
    _, h, w = img.shape
    if h == w:
        return img
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    return F.pad(img, (pad_w, pad_w, pad_h, pad_h), value=0.0)


def crop_to_breast_v2(img, threshold=0.05):
    # img: [C,H,W] or [H,W]
    if img.dim() == 3:
        gray = img.mean(0)
    else:
        gray = img
    
    # binary mask
    mask = gray > threshold
    
    # remove tiny objects by keeping largest connected component
    mask_np = mask.cpu().numpy().astype('uint8')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
    if num_labels <= 1:  # no objects
        return img
    largest_idx = stats[1:, cv2.CC_STAT_AREA].argmax() + 1  # +1 because 0 is background
    largest_mask = (labels == largest_idx)
    
    largest_mask = torch.tensor(largest_mask, dtype=torch.uint8)  # if it's numpy
    coords = largest_mask.nonzero(as_tuple=False)
    mins = coords.min(0)[0]
    maxs = coords.max(0)[0]
    y_min, x_min = mins[0].item(), mins[1].item()
    y_max, x_max = maxs[0].item(), maxs[1].item()
    
    return img[:, y_min:y_max+1, x_min:x_max+1]


def preprocess_tensor(img: torch.Tensor):
    """
    Resize and normalize a float tensor in tensor-space.
    - img: torch.Tensor, shape (H,W) or (C,H,W). dtype=float (expected from load_dicom_as_image).
    - Returns: torch.FloatTensor shape (3, size[0], size[1]) normalized by IM_MEAN/IM_STD.
    """
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

    # per-sample per-channel scaling to [0,1] (preserves relative contrast)
    c = img.shape[0]
    mins = img.view(c, -1).min(dim=1)[0].view(c,1,1)
    maxs = img.view(c, -1).max(dim=1)[0].view(c,1,1)
    img = (img - mins) / (maxs - mins + 1e-8)

    img = crop_to_breast_v2(img, threshold=0.05)
    # img = pad_to_square(img)
    img = F.interpolate(img.unsqueeze(0), size=(2048, 1024), mode="bilinear", align_corners=False)[0]
    return img

class BreastDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.items = []
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx]['new_path']
        y_text = str(self.dataframe.iloc[idx]['label']).strip().lower()
        label_map = {'negative': 0, 'suspicious': 1}
        if y_text not in label_map:
            raise ValueError(f"Unknown label: {y_text}")
        y = label_map[y_text]
        dicom = load_dicom_as_image(path)
        # im=(im-IM_MEAN)/IM_STD

        if self.transform:
            if dicom.ndim == 2:
                dicom = np.stack([dicom]*3, axis=-1)  # (H, W) -> (H, W, 3)
            dicom = self.transform(dicom)
        else:
            if dicom.ndim == 2:
                dicom = np.stack([dicom]*3, axis=0)
            dicom = torch.from_numpy(dicom).float()

        return dicom, torch.tensor(y,dtype=torch.float32).unsqueeze(0)
    

def create_dataloaders(train_files, val_files, transform, is_ddp, rank, world_size, num_workers=12, per_gpu_batch=1, seed=42):
    """
    Returns train_dl, val_dl, train_sampler (None for non-DDP).
    For DDP uses an oversampled Subset (broadcasted indices) so sampling is consistent across ranks.
    """
    # single-process: WeightedRandomSampler
    g = torch.Generator()
    g.manual_seed(seed)

    if not is_ddp:
        labels = train_files['label'].str.lower().map({'negative': 0, 'suspicious': 1}).astype(np.int64).values
        class_sample_count = np.bincount(labels)
        class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
        weights = 1.0 / class_sample_count
        sample_weights = torch.tensor([weights[int(t)] for t in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_ds = BreastDataset(train_files, transform=transform['train'])
        val_ds = BreastDataset(val_files, transform=transform['val'])

        train_dl = DataLoader(
            train_ds,
            batch_size=per_gpu_batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g
            )
        val_dl = DataLoader(
            val_ds,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g
            )
        return train_dl, val_dl, None

    # DDP branch: create balanced subset (upsample minority) on rank 0 and broadcast indices
    train_ds = BreastDataset(train_files, transform=transform['train'])
    val_ds = BreastDataset(val_files, transform=transform['val'])

    train_labels_arr = train_files['label'].str.lower().map({'negative': 0, 'suspicious': 1}).astype(np.int64).values

    if rank == 0:
        all_idx = balance_indices(train_labels_arr, mode='over')
    else:
        all_idx = None

    obj_list = [all_idx]
    # broadcast indices
    dist.broadcast_object_list(obj_list, src=0)
    all_idx = obj_list[0]
    if isinstance(all_idx, np.ndarray):
        all_idx = all_idx.tolist()
    train_subset = Subset(train_ds, all_idx)

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    g = torch.Generator()
    g.manual_seed(42)
    train_dl = DataLoader(
        train_subset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=per_gpu_batch,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_dl, val_dl, train_sampler


def balance_indices(labels, mode='over'):
    assert mode in (None, 'over', 'under'), "mode must be one of None, 'over', 'under'"
    labels = np.array(labels)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]


    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(labels)).tolist()

    if mode == "over":
        if len(pos) < len(neg):
            pos_up = np.random.choice(pos, size=len(neg), replace=True)
            idx = np.concatenate([neg, pos_up])
        else:
            neg_up = np.random.choice(neg, size=len(pos), replace=True)
            idx = np.concatenate([pos, neg_up])

    elif mode == "under":
        m = min(len(pos), len(neg))
        pos_dn = np.random.choice(pos, size=m, replace=False)
        neg_dn = np.random.choice(neg, size=m, replace=False)
        idx = np.concatenate([pos_dn, neg_dn])
    else:
        idx = np.arange(len(labels)).tolist()

    np.random.shuffle(idx)
    return idx

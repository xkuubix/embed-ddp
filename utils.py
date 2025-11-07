from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
import logging
import time
from ddp_utils import gather_from_ranks
import random
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("training")


def format_counts(df, name):
    counts = df['label'].value_counts()
    percents = df['label'].value_counts(normalize=True) * 100
    lines = [f"\t{label}: {counts[label]} ({percents[label]:.1f}%)" for label in counts.index]
    return f"{name} samples: {len(df)}\n" + "\n".join(lines)


def reset_seed(SEED=42):
    """Reset random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_loop(model, opt, crit, train_dl, val_dl, train_sampler, is_ddp, rank, world_size, logger, num_epochs=10):
    
    num_val_steps = 10
    val_interval = max(1, len(train_dl) // num_val_steps)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} starting (rank={rank})")
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running = 0.0
        batch_count = 0
        start_t = time.time()
        for imgs, labels in train_dl:
            model.train()
            imgs = imgs.to(next(model.parameters()).device, non_blocking=True)
            labels = labels.to(next(model.parameters()).device, non_blocking=True)

            opt.zero_grad()
            outputs = model(imgs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()
            running += loss.item()
            batch_count += 1
            
            if batch_count % val_interval == 0:
                res = distributed_mini_validate(model, val_dl, is_ddp, rank, world_size, max_batches=1000)
                if rank == 0 and res is not None:
                    sens, spec, prec, N = res
                    logger.info(
                        f"epoch {epoch+1} batch {batch_count} sens={sens:.3f} spec={spec:.3f} "
                        f"prec={prec:.3f} (N+)={N['pos']:d} (N-)={N['neg']:d}"
                    )

    
        epoch_time = time.time() - start_t
        avg_loss = running / batch_count if batch_count > 0 else 0.0
        logger.info(f"Epoch {epoch+1} training finished (rank={rank}) - avg_loss={avg_loss:.4f} time={epoch_time:.1f}s")
        res = validate(model, val_dl, is_ddp, rank, world_size)
        if res is not None:
            sens, spec, prec, f1, auc_macro, auc_weighted, N = res
            if rank == 0:
                logger.info(
                    f"Epoch {epoch+1} | val sens={sens:.4f} spec={spec:.4f} "
                    f"prec={prec:.4f} f1={f1:.4f} "
                    f"auc_macro={auc_macro if auc_macro is not None else 'N/A'} "
                    f"auc_weighted={auc_weighted if auc_weighted is not None else 'N/A'} "
                    f"(N+)={N['pos']:d} (N-)={N['neg']:d}"
                )
        # # save checkpoint
        # ckpt = {
        #     'epoch': epoch,
        #     'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        #     'optimizer_state_dict': opt.state_dict(),
        # }
        # torch.save(ckpt, f'./checkpoints/best_epoch_{epoch+1}.pth')


def validate(model, val_dl, is_ddp, rank, world_size):
    model.eval()
    local_preds, local_probs, local_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(next(model.parameters()).device, non_blocking=True)
            labels = labels.to(next(model.parameters()).device, non_blocking=True)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            local_preds.extend(preds.cpu().tolist())
            local_probs.extend(probs.cpu().tolist())
            local_labels.extend(labels.cpu().tolist())

    # gather across ranks
    gathered_probs = gather_from_ranks(local_probs, is_ddp, world_size)
    gathered_labels = gather_from_ranks(local_labels, is_ddp, world_size)
    gathered_preds = gather_from_ranks(local_preds, is_ddp, world_size)

    if rank != 0:
        return None

    # flatten lists
    probs = [p for sub in gathered_probs for p in (sub if isinstance(sub, list) else list(sub))]
    labels = [l for sub in gathered_labels for l in (sub if isinstance(sub, list) else list(sub))]
    preds  = [p for sub in gathered_preds for p in (sub if isinstance(sub, list) else list(sub))]

    auc_macro = None
    try:
        auc_macro = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
        auc_weighted = roc_auc_score(labels, probs, average='weighted') if len(set(labels)) > 1 else float('nan')
    except Exception:
        auc_macro = float('nan')

    if len(labels) > 0 and len(set(labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity  / True Positive Rate  /  Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity / True Negative Rate
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision  / Positive Predictive Value
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0 # F1 Score

    N = {'pos': int(sum(labels)), 'neg': int(len(labels) - sum(labels))}
    return sens, spec, prec, f1, auc_macro, auc_weighted, N


def distributed_mini_validate(model, val_dl, is_ddp, rank, world_size, max_batches=-1):
    model.eval()
    local_preds, local_labels = [], []
    device = next(model.parameters()).device

    for i, (x, y) in enumerate(val_dl):
        if max_batches > 0 and i >= max_batches:
            break
        if is_ddp and (i % world_size != rank):
            continue
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        p = (torch.sigmoid(out) >= 0.5).long()
        local_preds.extend(p.cpu().tolist())
        local_labels.extend(y.cpu().tolist())

    if is_ddp:
        gathered_preds = gather_from_ranks(local_preds, is_ddp, world_size)
        gathered_labels = gather_from_ranks(local_labels, is_ddp, world_size)
    else:
        gathered_preds = [local_preds]
        gathered_labels = [local_labels]

    if rank == 0:
        preds_flat = [p for sub in gathered_preds for p in (sub if isinstance(sub, list) else list(sub))]
        labels_flat = [l for sub in gathered_labels for l in (sub if isinstance(sub, list) else list(sub))]
        if len(labels_flat) < 1 or len(set(labels_flat)) < 2:
            return 0.0, 0.0, 0.0, {'pos': 0, 'neg': 0}
        tn, fp, fn, tp = confusion_matrix(labels_flat, preds_flat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        N = {'pos': int(sum(labels_flat)), 'neg': int(len(labels_flat) - sum(labels_flat))}
        return sens, spec, prec, N
    else:
        return None

from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
import logging
import time
from torch.nn.parallel import DistributedDataParallel as DDP
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


def train_loop(model, opt, crit, train_dl, val_dl, train_sampler, is_ddp, rank, world_size, logger, num_epochs=10):
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} starting (rank={rank})")
        model.train()
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running = 0.0
        batch_count = 0
        start_t = time.time()
        for imgs, labels in train_dl:
            imgs = imgs.to(next(model.parameters()).device, non_blocking=True)
            labels = labels.to(next(model.parameters()).device, non_blocking=True)

            opt.zero_grad()
            outputs = model(imgs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()
            running += loss.item()
            batch_count += 1

            if rank == 0 and (batch_count % 50 == 0):
                logger.info(f"Epoch {epoch+1} batch {batch_count} loss={loss.item():.4f}")

        epoch_time = time.time() - start_t
        avg_loss = running / batch_count if batch_count > 0 else 0.0
        logger.info(f"Epoch {epoch+1} training finished (rank={rank}) - avg_loss={avg_loss:.4f} time={epoch_time:.1f}s")

        # validation
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

        if rank == 0:
            probs = [p for sub in gathered_probs for p in (sub if isinstance(sub, list) else list(sub))]
            labels = [l for sub in gathered_labels for l in (sub if isinstance(sub, list) else list(sub))]
            preds  = [p for sub in gathered_preds for p in (sub if isinstance(sub, list) else list(sub))]

            auc = None
            try:
                auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
            except Exception:
                auc = float('nan')

            if len(labels) > 0 and len(set(labels)) > 1:
                tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
            else:
                tn = fp = fn = tp = 0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            logger.info(f"Epoch {epoch+1} | val sens={sens:.4f} spec={spec:.4f} auc={auc if auc is not None else 'N/A'}")

            # save checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                'optimizer_state_dict': opt.state_dict(),
            }
            torch.save(ckpt, f'./checkpoints/best_epoch_{epoch+1}.pth')
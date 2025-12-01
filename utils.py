from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
import logging
import time
from ddp_utils import gather_from_ranks
import random
import numpy as np
from model_utils import ModelEMA


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


def train_loop(model, opt, crit, train_dl, val_dl, train_sampler, is_ddp, rank, world_size, logger, num_epochs=10, accumulation_steps=64):
    
    num_val_steps = 10
    val_interval = max(1, len(train_dl) // num_val_steps)
    use_ddp_no_sync = is_ddp and hasattr(model, "no_sync")

    ema = ModelEMA(model, decay=0.9998)
    for epoch in range(num_epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} started (accum_steps={accumulation_steps})")
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running = 0.0
        batch_count = 0  # number of microbatches processed
        start_t = time.time()
        accum_counter = 0  # counts microbatches accumulated so far

        for imgs, labels in train_dl:
            model.train()
            imgs = imgs.to(next(model.parameters()).device, non_blocking=True)
            labels = labels.to(next(model.parameters()).device, non_blocking=True)

            # zero grads at start of accumulation
            if accum_counter == 0:
                opt.zero_grad()

            outputs = model(imgs)
            epsilon = 0.2
            y_smooth = labels * (1 - epsilon) + (1 - labels) * epsilon
            loss = crit(outputs, y_smooth)
            loss_value = loss.item()
            # scale loss for gradient accumulation
            scaled_loss = loss / float(accumulation_steps)
            # backwards with DDP no_sync when not stepping this microbatch
            if use_ddp_no_sync and (accum_counter != accumulation_steps - 1):
                with model.no_sync():
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            running += loss_value
            batch_count += 1
            accum_counter += 1

            # optimizer step when enough microbatches accumulated
            if accum_counter == accumulation_steps:
                opt.step()
                opt.zero_grad()
                accum_counter = 0
                ema.update(model.module)  # model.module contains the real weights

        # if there are leftover accumulated grads at epoch end, step once
        if accum_counter != 0:
            opt.step()
            opt.zero_grad()
            accum_counter = 0
            ema.update(model.module)  # model.module contains the real weights


        epoch_time = time.time() - start_t
        avg_loss = running / batch_count if batch_count > 0 else 0.0
        if rank == 0:
            logger.info(f"Epoch {epoch+1} training finished (rank={rank}) - L={avg_loss:.4f} time={epoch_time:.1f}s")
        res = validate(ema.ema, val_dl, crit, is_ddp, rank, world_size)
        if res is not None:
            sens, spec, prec, f1, auc_score, val_avg_loss, N = res
            if rank == 0:
                logger.info(
                    f"Epoch {epoch+1} | L={val_avg_loss:.4f}  val sens={sens:.4f} spec={spec:.4f} "
                    f"prec={prec:.4f} f1={f1:.4f} "
                    f"auc={auc_score if auc_score is not None else 'N/A'} "
                    f"(N+)={N['pos']:d} (N-)={N['neg']:d}"
                )
        # # save checkpoint
        # ckpt = {
        #     'epoch': epoch,
        #     'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        #     'optimizer_state_dict': opt.state_dict(),
        # }
        # torch.save(ckpt, f'./checkpoints/best_epoch_{epoch+1}.pth')


def validate(model, val_dl, crit, is_ddp, rank, world_size):
    model.eval()
    local_preds, local_probs, local_labels, local_losses = [], [], [], []

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(next(model.parameters()).device, non_blocking=True)
            labels = labels.to(next(model.parameters()).device, non_blocking=True).view(-1)
            outputs = model(imgs)
            loss = crit(outputs.view(-1), labels)
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs >= 0.5).long().view(-1)
            local_preds.extend(preds.cpu().tolist())
            local_probs.extend(probs.cpu().tolist())
            local_labels.extend(labels.cpu().tolist())
            local_losses.extend([loss.item()] * len(labels))

    # gather across ranks
    gathered_probs = gather_from_ranks(local_probs, is_ddp, world_size)
    gathered_labels = gather_from_ranks(local_labels, is_ddp, world_size)
    gathered_preds = gather_from_ranks(local_preds, is_ddp, world_size)
    gathered_losses = gather_from_ranks(local_losses, is_ddp, world_size)

    if rank != 0:
        return None

    # flatten lists
    probs = [p for sub in gathered_probs for p in (sub if isinstance(sub, list) else list(sub))]
    labels = [l for sub in gathered_labels for l in (sub if isinstance(sub, list) else list(sub))]
    preds  = [p for sub in gathered_preds for p in (sub if isinstance(sub, list) else list(sub))]
    losses = [l for sub in gathered_losses for l in (sub if isinstance(sub, list) else list(sub))]
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    auc_score = None
    try:
        auc_score = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
    except Exception:
        auc_score = float('nan')
    if len(labels) > 0 and len(set(labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity  / True Positive Rate  /  Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity / True Negative Rate
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision  / Positive Predictive Value
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0 # F1 Score

    N = {'pos': int(sum(labels)), 'neg': int(len(labels) - sum(labels))}
    return sens, spec, prec, f1, auc_score, avg_loss, N


"""Shared infrastructure for downstream evaluation tasks.

Provides:
- Feature extraction from pretrained MAE V2 encoder
- Linear probe and fine-tune training loops
- Segmentation decoder head
- Metric computation
- Few-shot sampling
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from config import CHANNELS_V4, MODALITY_GROUPS, MODALITY_GROUP_NAMES


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =====================================================================
# Feature extraction from pretrained encoder
# =====================================================================

def load_pretrained_encoder(checkpoint_path, device="cuda"):
    """Load pretrained MAE V2 encoder weights.

    Returns the model (eval mode) and its config.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("model_config", {})
    size = config.get("size", "base")
    mask_ratio = config.get("mask_ratio", 0.75)
    patch_size = config.get("patch_size", 16)

    from lunar_mae_v2 import lunar_mae_v2_base, lunar_mae_v2_large
    crossmodal_prob = config.get("crossmodal_prob", 0.0)
    contrast_weight = config.get("contrast_weight", 0.0)
    if size == "large":
        model = lunar_mae_v2_large(CHANNELS_V4, mask_ratio=mask_ratio, patch_size=patch_size,
                                   crossmodal_prob=crossmodal_prob, contrast_weight=contrast_weight)
    else:
        model = lunar_mae_v2_base(CHANNELS_V4, mask_ratio=mask_ratio, patch_size=patch_size,
                                  crossmodal_prob=crossmodal_prob, contrast_weight=contrast_weight)

    # The model is rebuilt with the checkpoint's own model_config (crossmodal_prob,
    # contrast_weight, patch_size), so the architecture matches the checkpoint exactly
    # (verified 256/256 keys). Load all weights; only drop a key if its shape genuinely
    # differs (guards against stale/intermediate checkpoints) and warn loudly if many drop.
    sd = ckpt["model"]
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
    dropped = [k for k in sd if k not in filtered]
    if dropped:
        print(f"  WARNING: {len(dropped)} keys not loaded (shape mismatch/extra). "
              f"If this is large, the eval config differs from training: {dropped[:5]}")
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    epoch = ckpt.get("epoch", -1)
    loss = ckpt.get("loss", -1)
    print(f"Loaded pretrained encoder: epoch={epoch}, loss={loss:.4f}, size={size}")
    return model, config


def extract_features(model, pixels, has_group, mode="cls"):
    """Extract features from pretrained encoder.

    Args:
        model: LunarMAEv2 model
        pixels: (B, 28, H, W)
        has_group: (B, 7) bool
        mode: "cls" for global features (B, D) via per-group pooling,
              "patches" for surface group patch tokens (B, 256, D)

    Returns:
        features: (B, D) or (B, 256, D)
    """
    from config import MODALITY_GROUP_NAMES

    # Run encoder without masking (mask_ratio=0)
    # Temporarily disable training mode to prevent crossmodal masking
    was_training = model.training
    model.eval()
    encoded, _, group_token_ranges, _ = model.forward_encoder(
        pixels, has_group, mask_ratio=0.0)
    if was_training:
        model.train()

    if mode == "cls":
        # Pool within each group separately, then average group features.
        # Use has_group to zero out ghost tokens from unavailable groups
        # so they don't pollute the per-batch-element representation.
        B, _, D = encoded.shape
        device = encoded.device
        feat_sum = torch.zeros(B, D, device=device, dtype=encoded.dtype)
        group_count = torch.zeros(B, 1, device=device, dtype=encoded.dtype)

        for g_idx, g_name in enumerate(MODALITY_GROUP_NAMES):
            if g_name not in group_token_ranges:
                continue
            start, end = group_token_ranges[g_name]
            n_tok = end - start
            if n_tok <= 0:
                continue
            g_tokens = encoded[:, start:end]  # (B, n_tok, D)
            g_pooled = g_tokens.mean(dim=1)  # (B, D)
            # Mask: only include for batch elements where this group is available
            g_avail = has_group[:, g_idx].float().unsqueeze(1)  # (B, 1)
            feat_sum = feat_sum + g_pooled * g_avail
            group_count = group_count + g_avail

        # Average over available groups per batch element
        group_count = group_count.clamp(min=1)
        return feat_sum / group_count  # (B, D)

    elif mode == "patches":
        # Return only surface group tokens — always available, 256 patches,
        # preserves spatial grid for SegmentationHead reshape(B, D, 16, 16)
        start, end = group_token_ranges["surface"]
        return encoded[:, start:end]  # (B, 256, D)


@torch.no_grad()
def precompute_features(encoder, loader, device, mode="cls"):
    """Run the frozen encoder once over a loader and cache (features, targets).

    Returns the cached tensors so the head can train for many epochs without
    re-encoding, which is the dominant cost for linear probing. Iterates with a
    single-process loader: forked DataLoader workers deadlock against the HIP/CUDA
    context in some container setups, and worker parallelism gives little here
    since the encoder forward dominates.
    """
    encoder.eval()
    loader = DataLoader(loader.dataset, batch_size=loader.batch_size,
                        shuffle=False, num_workers=0)
    feats, targets = [], []
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        has_group = batch["has_group"].to(device, non_blocking=True).bool()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = extract_features(encoder, pixels, has_group, mode=mode)
        feats.append(f.float().cpu())
        targets.append(batch["target"])
    return torch.cat(feats), torch.cat(targets)


def _feature_loader(features, targets, batch_size, shuffle):
    """DataLoader over cached features, yielding the same dict shape as the
    pixel datasets so the standard loops (which read batch['pixels'] and, when
    encoder is None, call head(pixels)) run unchanged on cached features."""
    ds = TensorDataset(features, targets)

    def collate(items):
        f = torch.stack([it[0] for it in items])
        t = torch.stack([it[1] for it in items])
        return {"pixels": f, "target": t}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# =====================================================================
# Downstream heads
# =====================================================================

class LinearProbe(nn.Module):
    """Simple linear classifier on frozen features."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, features):
        return self.head(features)


class MLPHead(nn.Module):
    """2-layer MLP head for regression/classification."""
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features)


class SegmentationHead(nn.Module):
    """Lightweight segmentation decoder from patch tokens.

    Upsamples patch tokens (B, N, D) back to pixel-level predictions (B, C, H, W).
    Uses 2 transposed conv layers for 16x upsampling (patch_size=16).
    """
    def __init__(self, embed_dim, num_classes, patch_size=16, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        # Reshape patch tokens to spatial grid, then upsample
        self.decoder = nn.Sequential(
            # (B, D, grid, grid) → (B, 128, grid*4, grid*4)
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=4),
            nn.GELU(),
            # (B, 128, grid*4, grid*4) → (B, num_classes, grid*16, grid*16)
            nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=4),
        )

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N, D) where N = grid_size²
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, N, D = patch_tokens.shape
        # Reshape to spatial grid
        x = patch_tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        return self.decoder(x)


# =====================================================================
# Training utilities
# =====================================================================

def get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    """Cosine LR with warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def few_shot_indices(labels, n_per_class, seed=42):
    """Sample n_per_class examples per class for few-shot training."""
    rng = np.random.RandomState(seed)
    unique = np.unique(labels)
    indices = []
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        n = min(n_per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, n, replace=False)
        indices.extend(chosen.tolist())
    return sorted(indices)


# =====================================================================
# Metric helpers
# =====================================================================

def compute_classification_metrics(preds, targets, num_classes):
    """Compute accuracy, macro-F1, per-class F1."""
    correct = (preds == targets).sum().item()
    total = len(targets)
    accuracy = correct / max(total, 1)

    per_class_f1 = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class_f1.append(f1)

    macro_f1 = np.mean(per_class_f1)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class_f1": per_class_f1}


def compute_segmentation_metrics(pred_mask, target_mask, num_classes=2):
    """Compute IoU and F1 per class."""
    results = {}
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        tgt_c = (target_mask == c)
        intersection = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        tp = intersection
        fp = (pred_c & ~tgt_c).sum().item()
        fn = (~pred_c & tgt_c).sum().item()

        iou = intersection / max(union, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        results[f"class_{c}_iou"] = iou
        results[f"class_{c}_f1"] = f1

    results["mean_iou"] = np.mean([results[f"class_{c}_iou"] for c in range(num_classes)])
    results["mean_f1"] = np.mean([results[f"class_{c}_f1"] for c in range(num_classes)])
    return results


def compute_regression_metrics(preds, targets):
    """Compute RMSE, R², Pearson correlation."""
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        pearson = np.corrcoef(preds, targets)[0, 1]
    else:
        pearson = 0.0

    return {"rmse": rmse, "r2": r2, "pearson": pearson, "mse": mse}


# =====================================================================
# Generic training loop
# =====================================================================

def train_downstream(
    encoder, head, train_loader, val_loader, device,
    epochs=50, lr=1e-3, freeze_encoder=True, model_version="v2",
    loss_fn=None, eval_fn=None, task_name="task",
    output_dir="checkpoints/downstream", patience=5,
):
    """Generic downstream training loop.

    Args:
        encoder: Pretrained MAE model (or None for scratch baseline)
        head: Downstream head (LinearProbe, MLPHead, SegmentationHead)
        train_loader, val_loader: DataLoaders
        device: torch device
        epochs: training epochs
        lr: learning rate
        freeze_encoder: if True, only train head; if False, fine-tune all
        model_version: "v2" or "scratch"
        loss_fn: loss function (default: CrossEntropyLoss with label smoothing)
        eval_fn: evaluation function(model_outputs, targets) -> dict
        task_name: name for logging
        output_dir: where to save results
        patience: early stopping patience (0 = disabled)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Linear probing: the frozen encoder produces the same features every epoch,
    # so extract them once and train the head on cached features. Finetuning
    # (freeze_encoder=False) and segmentation heads keep the per-batch path.
    if (encoder is not None and freeze_encoder
            and not isinstance(head, SegmentationHead)):
        print(f"  [{task_name}] precomputing frozen features...")
        tr_f, tr_y = precompute_features(encoder, train_loader, device)
        va_f, va_y = precompute_features(encoder, val_loader, device)
        train_loader = _feature_loader(tr_f, tr_y, train_loader.batch_size, shuffle=True)
        val_loader = _feature_loader(va_f, va_y, val_loader.batch_size, shuffle=False)
        encoder = None  # downstream loops now see cached features as "pixels"

    if loss_fn is None:
        # Label smoothing to prevent overconfident predictions and reduce overfitting
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Feature dropout for finetune mode (prevents overfitting to pretrained features)
    feat_dropout = None
    if encoder is not None and not freeze_encoder:
        feat_dropout = nn.Dropout(0.5).to(device)

    # Setup optimizer — partial fine-tuning to prevent overfitting on small datasets
    if encoder is not None and not freeze_encoder:
        # Freeze early encoder blocks (first 8 of 12), only fine-tune last 4.
        # Attr names differ V2 vs V1 (encoder_blocks/tokenizers/group_embed vs
        # blocks/patch_embeds/channel_embed).
        blocks = getattr(encoder, "encoder_blocks", None) or encoder.blocks
        tokenizers = getattr(encoder, "tokenizers", None) or encoder.patch_embeds
        grp_embed = getattr(encoder, "group_embed", None)
        if grp_embed is None:
            grp_embed = getattr(encoder, "channel_embed", {})
        n_freeze = max(0, len(blocks) - 4)
        for i, block in enumerate(blocks):
            if i < n_freeze:
                for p in block.parameters():
                    p.requires_grad_(False)
        for p in tokenizers.parameters():
            p.requires_grad_(False)
        encoder.pos_embed.requires_grad_(False)
        for name in grp_embed:
            grp_embed[name].requires_grad_(False)
        encoder.cls_token.requires_grad_(False)

        trainable_enc = [p for p in encoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": trainable_enc, "lr": lr * 0.01, "initial_lr": lr * 0.01},
            {"params": list(head.parameters()), "lr": lr, "initial_lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
        params = trainable_enc + list(head.parameters())
    else:
        params = list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        for pg in optimizer.param_groups:
            pg["initial_lr"] = lr
    total_steps = epochs * len(train_loader)
    warmup_steps = min(len(train_loader) * 2, total_steps // 10)

    if encoder is not None and freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    head.train()
    best_metric = None
    no_improve = 0
    logs = []

    for epoch in range(epochs):
        head.train()
        if encoder is not None and not freeze_encoder:
            encoder.train()

        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            # Cosine schedule as fraction of base LR
            lr_scale = get_cosine_lr(step, warmup_steps, total_steps, 1.0, min_lr=0.0)
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", lr) * lr_scale

            pixels = batch["pixels"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if encoder is not None:
                    has_group = batch["has_group"].to(device, non_blocking=True).bool()
                    with torch.set_grad_enabled(not freeze_encoder):
                        features = extract_features(
                            encoder, pixels, has_group,
                            mode="patches" if isinstance(head, SegmentationHead) else "cls")
                    if feat_dropout is not None:
                        features = feat_dropout(features)
                    output = head(features)
                else:
                    # Scratch baseline: head operates directly on pixels
                    output = head(pixels)

                loss = loss_fn(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate_downstream(
            encoder, head, val_loader, device,
            freeze_encoder=True, model_version=model_version,
            loss_fn=loss_fn, eval_fn=eval_fn)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "time": round(elapsed, 1),
            **{f"val_{k}": round(v, 6) if isinstance(v, float) else v
               for k, v in val_metrics.items() if not isinstance(v, list)},
        }
        logs.append(log_entry)

        # Track best + early stopping
        primary_metric = val_metrics.get("macro_f1", val_metrics.get("mean_iou",
                         -val_metrics.get("rmse", -val_metrics.get("loss", 999))))
        if best_metric is None or primary_metric > best_metric:
            best_metric = primary_metric
            no_improve = 0
            torch.save(head.state_dict(), output_dir / f"{task_name}_best.pt")
        else:
            no_improve += 1

        print(f"  [{task_name}] epoch {epoch}: train_loss={avg_loss:.4f}, "
              f"val_loss={val_metrics.get('loss', 0):.4f}, "
              f"elapsed={elapsed:.1f}s")

        # Early stopping
        if patience > 0 and no_improve >= patience:
            print(f"  [{task_name}] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Save logs
    with open(output_dir / f"{task_name}_log.json", "w") as f:
        json.dump(logs, f, indent=2, cls=NumpyEncoder)

    return logs


@torch.no_grad()
def evaluate_downstream(
    encoder, head, loader, device,
    freeze_encoder=True, model_version="v2",
    loss_fn=None, eval_fn=None,
):
    """Evaluate downstream task on a dataloader."""
    head.eval()
    if encoder is not None:
        encoder.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if encoder is not None:
                has_group = batch["has_group"].to(device, non_blocking=True).bool()
                features = extract_features(
                    encoder, pixels, has_group,
                    mode="patches" if isinstance(head, SegmentationHead) else "cls")
                output = head(features)
            else:
                output = head(pixels)

            loss = loss_fn(output, target)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(output.float().cpu())
        all_targets.append(target.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {"loss": avg_loss}
    if eval_fn is not None:
        extra = eval_fn(all_preds, all_targets)
        metrics.update(extra)

    return metrics

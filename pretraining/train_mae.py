"""Pretraining script for the Modality-Grouped MAE (MG-MAE).

Usage:
    # 2-GPU DDP
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 pretraining/train_mae.py \
        --ddp --batch-size 64 --grad-accum 2 \
        --epochs 100 --mask-ratio 0.75 --lr 1.5e-4 --warmup-epochs 10 \
        --crossmodal-prob 0.5 --contrast-weight 0.1 \
        --val-every 5 --val-samples 2000

    # Resume from checkpoint
    python pretraining/train_mae.py --resume checkpoints/latest.pt

    # Quick test
    python pretraining/train_mae.py --test --epochs 2 \
        --epoch-length 256 --batch-size 8
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import CHANNELS_V4, HDF5_V2_PATH, HDF5_V4_PATH, OUTPUT_DIR
from lunar_dataset import (
    GeoTIFFRandomCropDataset,
    HDF5Dataset,
    MmapRandomCropDataset,
    compute_channel_stats,
)


def get_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, device, epoch,
                    global_step, warmup_steps, total_steps, base_lr,
                    grad_accum=1, model_version="v2", log_interval=50,
                    is_main=True):
    """Train for one epoch. Returns avg_loss, per-group losses, throughput."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    group_losses_sum = {}
    group_losses_count = {}
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    n_loader = len(loader)
    # Compute remainder for final accumulation step scaling
    remainder = n_loader % grad_accum

    for batch_idx, batch in enumerate(loader):
        step = global_step + batch_idx
        lr = get_lr(step, warmup_steps, total_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        pixels = batch["pixels"].to(device, non_blocking=True)

        if model_version == "v2":
            has_data = batch["has_group"].to(device, non_blocking=True).bool()
        else:
            has_data = batch["has_channel"].to(device, non_blocking=True).bool()

        # Always divide by grad_accum so each mini-batch contributes equally.
        # The final partial group will have fewer contributions but same scale.
        accum_divisor = grad_accum

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model(pixels, has_data)
            loss = loss / accum_divisor

        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += metrics["loss"]
        n_batches += 1

        # Accumulate per-group losses and recon/contrast losses
        for k, v in metrics.items():
            if k.startswith("loss_") or k in ("recon_loss", "contrast_loss"):
                group_losses_sum[k] = group_losses_sum.get(k, 0.0) + v
                group_losses_count[k] = group_losses_count.get(k, 0) + 1

        if is_main and batch_idx % log_interval == 0:
            elapsed = time.time() - t0
            samples_sec = (batch_idx + 1) * pixels.shape[0] / max(elapsed, 1e-6)
            loss_key = "n_group_losses" if model_version == "v2" else "n_channel_losses"
            n_active = metrics.get(loss_key, 0)
            print(f"  [{epoch}][{batch_idx}/{len(loader)}] "
                  f"loss={metrics['loss']:.4f} "
                  f"active={n_active} "
                  f"lr={lr:.2e} "
                  f"speed={samples_sec:.1f} samples/s")

    # Final grad step if incomplete accumulation
    if remainder > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(n_batches, 1)
    elapsed = time.time() - t0
    throughput = n_batches * loader.batch_size / max(elapsed, 1e-6)

    # Average per-group losses
    avg_group_losses = {
        k: group_losses_sum[k] / max(group_losses_count[k], 1)
        for k in group_losses_sum
    }

    return avg_loss, avg_group_losses, throughput


@torch.no_grad()
def validate(model, dataset, device, n_samples=2000, batch_size=32,
             model_version="v2", seed=42):
    """Compute validation loss on fixed random samples.

    Uses a fixed seed for reproducibility across epochs.
    """
    model.eval()

    # Create a deterministic subset by using a seeded RNG for indices
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(dataset))

    total_loss = 0.0
    n_batches = 0
    group_losses_sum = {}
    group_losses_count = {}

    # Manually batch samples
    batch_pixels = []
    batch_has = []

    for i in range(n):
        sample = dataset[rng.randint(0, len(dataset))]
        batch_pixels.append(sample["pixels"])
        if model_version == "v2":
            batch_has.append(sample["has_group"])
        else:
            batch_has.append(sample["has_channel"])

        if len(batch_pixels) == batch_size or i == n - 1:
            pixels = torch.stack(batch_pixels).to(device)
            has_data = torch.stack(batch_has).to(device).bool()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(pixels, has_data)

            total_loss += metrics["loss"]
            n_batches += 1

            for k, v in metrics.items():
                if k.startswith("loss_") or k in ("recon_loss", "contrast_loss"):
                    group_losses_sum[k] = group_losses_sum.get(k, 0.0) + v
                    group_losses_count[k] = group_losses_count.get(k, 0) + 1

            batch_pixels.clear()
            batch_has.clear()

    # Don't call model.train() here — caller handles it via the DDP wrapper

    avg_loss = total_loss / max(n_batches, 1)
    avg_group_losses = {
        k: group_losses_sum[k] / max(group_losses_count[k], 1)
        for k in group_losses_sum
    }
    return avg_loss, avg_group_losses


def save_checkpoint(model, optimizer, epoch, step, loss, path,
                    model_config=None):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if model_config:
        state["model_config"] = model_config
    torch.save(state, path)
    # Also save as 'latest'
    latest = path.parent / "latest.pt"
    torch.save(state, latest)


def main():
    parser = argparse.ArgumentParser(description="Lunar MAE Pretraining")

    # Data
    parser.add_argument("--data-source", choices=["mmap", "geotiff", "hdf5"], default="mmap")
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--epoch-length", type=int, default=200_000,
                        help="Samples per epoch for mmap/GeoTIFF mode")
    parser.add_argument("--min-valid-fraction", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=8)

    # Model
    parser.add_argument("--model", choices=["base", "large"], default="base")
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--crossmodal-prob", type=float, default=0.0,
                        help="Prob of complementary cross-modal masking (0=disabled)")
    parser.add_argument("--contrast-weight", type=float, default=0.0,
                        help="Weight for cross-modal contrastive loss (0=disabled)")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")

    # Validation
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N epochs (0=disable)")
    parser.add_argument("--val-samples", type=int, default=2000,
                        help="Number of validation samples")

    # Distributed
    parser.add_argument("--ddp", action="store_true")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    # Debug
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    args = parser.parse_args()

    # Setup device
    if args.ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = local_rank == 0
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        world_size = 1

    if is_main:
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Data source: {args.data_source}")
        print(f"Model: v2/{args.model}")
        print(f"Mask ratio: {args.mask_ratio}")
        print(f"Grad accumulation: {args.grad_accum}")
        eff_batch = args.batch_size * world_size * args.grad_accum
        print(f"Effective batch size: {args.batch_size} × {world_size} GPU × {args.grad_accum} accum = {eff_batch}")

    # Load per-channel normalization stats
    stats_path = Path(__file__).parent / "channel_stats.json"
    channel_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            channel_stats = {k: tuple(v) for k, v in json.load(f).items()}
        if is_main:
            print(f"Loaded channel stats from {stats_path}")

    # Dataset
    channel_names = CHANNELS_V4

    if args.data_source == "mmap":
        dataset = MmapRandomCropDataset(
            channel_names=channel_names,
            crop_size=256,
            epoch_length=args.epoch_length,
            min_valid_fraction=args.min_valid_fraction,
        )
    elif args.data_source == "geotiff":
        dataset = GeoTIFFRandomCropDataset(
            channel_names=channel_names,
            crop_size=256,
            epoch_length=args.epoch_length,
            min_valid_fraction=args.min_valid_fraction,
            normalize=True,
            channel_stats=channel_stats,
        )
    else:
        hdf5_path = args.hdf5_path
        if hdf5_path is None:
            hdf5_path = str(HDF5_V4_PATH) if HDF5_V4_PATH.exists() else str(HDF5_V2_PATH)
        dataset = HDF5Dataset(hdf5_path, split="train",
                              normalize=True, channel_stats=channel_stats)

    if args.ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    if is_main:
        print(f"Dataset: {len(dataset)} samples, "
              f"Loader: {len(loader)} batches/epoch")

    # Model
    model_config = {
        "version": "v2",
        "size": args.model,
        "mask_ratio": args.mask_ratio,
        "patch_size": args.patch_size,
        "n_channels": len(channel_names),
        "crossmodal_prob": args.crossmodal_prob,
        "contrast_weight": args.contrast_weight,
    }

    from lunar_mae_v2 import lunar_mae_v2_base, lunar_mae_v2_large
    v2_kwargs = dict(
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        crossmodal_prob=args.crossmodal_prob,
        contrast_weight=args.contrast_weight,
    )
    if args.model == "base":
        model = lunar_mae_v2_base(channel_names, **v2_kwargs).to(device)
    else:
        model = lunar_mae_v2_large(channel_names, **v2_kwargs).to(device)

    if args.ddp:
        # Complementary masking can skip per-group pred_heads (crossmodal_prob > 0),
        # which requires find_unused_parameters.
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=(args.crossmodal_prob > 0))

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params / 1e6:.1f}M")

    # Optimizer (AdamW with betas from MAE paper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model.module if hasattr(model, "module") else model
        missing, unexpected = raw.load_state_dict(ckpt["model"], strict=False)
        if is_main and missing:
            print(f"  Resume: {len(missing)} missing keys (new modules, randomly initialized):")
            for k in missing:
                print(f"    {k}")
        if is_main and unexpected:
            print(f"  Resume: {len(unexpected)} unexpected keys (ignored):")
            for k in unexpected:
                print(f"    {k}")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        if is_main:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training config
    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = ckpt_dir / "train_log.json"
    logs = []
    # Load existing logs if resuming
    if args.resume and log_file.exists():
        with open(log_file) as f:
            logs = json.load(f)

    if is_main:
        print(f"\n=== Starting training ===")
        print(f"  Epochs: {start_epoch} → {args.epochs}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Validation: every {args.val_every} epochs, {args.val_samples} samples")

    for epoch in range(start_epoch, args.epochs):
        if args.ddp:
            sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss, group_losses, throughput = train_one_epoch(
            model, loader, optimizer, device, epoch,
            global_step, warmup_steps, total_steps, args.lr,
            grad_accum=args.grad_accum,
            model_version="v2",
            log_interval=args.log_interval,
            is_main=is_main)
        elapsed = time.time() - t0

        global_step += len(loader)

        if is_main:
            lr_now = get_lr(global_step, warmup_steps, total_steps, args.lr)
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, "
                  f"time={elapsed:.1f}s, throughput={throughput:.1f} samples/s, "
                  f"lr={lr_now:.2e}, step={global_step}")

            log_entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "time": round(elapsed, 1),
                "step": global_step,
                "lr": lr_now,
                "throughput": round(throughput, 1),
            }
            log_entry.update({"train_" + k: round(v, 6) for k, v in group_losses.items()})

            # Validation
            if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
                raw_model = model.module if hasattr(model, "module") else model
                vt0 = time.time()
                val_loss, val_group_losses = validate(
                    raw_model, dataset, device,
                    n_samples=args.val_samples,
                    batch_size=args.batch_size,
                    model_version="v2")
                val_time = time.time() - vt0
                log_entry["val_loss"] = round(val_loss, 6)
                log_entry.update({"val_" + k: round(v, 6) for k, v in val_group_losses.items()})
                print(f"  val_loss={val_loss:.4f} ({val_time:.1f}s)")
                for k, v in sorted(val_group_losses.items()):
                    print(f"    val_{k}: {v:.4f}")

            logs.append(log_entry)

            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    model, optimizer, epoch, global_step, avg_loss,
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                    model_config=model_config)
                print(f"  Saved checkpoint: epoch_{epoch:04d}.pt")

    if is_main:
        print(f"\nTraining complete. {args.epochs - start_epoch} epochs, {global_step} steps.")

    if args.ddp:
        dist.destroy_process_group()

    if hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    main()

"""Task 3: Mare/Highlands Binary Segmentation.

Usage:
    python task_mare.py --checkpoint checkpoints/latest.pt --mode linear
    python task_mare.py --mode scratch
"""

import argparse
import json
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ALIGNED_FILES
from downstream_base import (
    SegmentationHead, load_pretrained_encoder, extract_features,
    compute_segmentation_metrics, get_cosine_lr, NumpyEncoder,
)
from downstream_dataset import SegmentationMmapDataset

NUM_CLASSES = 2


class ScratchUNet(nn.Module):
    def __init__(self, in_channels=28, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


def train_segmentation(encoder, head, train_loader, val_loader, device,
                       epochs, lr, freeze_encoder, task_name, output_dir,
                       patience=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Feature dropout for finetune mode
    feat_dropout = None
    if encoder is not None and not freeze_encoder:
        feat_dropout = nn.Dropout(0.5).to(device)

    if encoder is not None and not freeze_encoder:
        # Freeze early encoder blocks (first 8 of 12), only fine-tune last 4.
        # Attribute names differ V2 vs V1: blocks/tokenizers/group-embed.
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

    if encoder is not None and freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    total_steps = epochs * len(train_loader)
    warmup_steps = min(len(train_loader) * 2, total_steps // 10)
    best_iou = -1
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
                        features = extract_features(encoder, pixels, has_group, mode="patches")
                    if feat_dropout is not None:
                        features = feat_dropout(features)
                    output = head(features)
                else:
                    output = head(pixels)
                loss = loss_fn(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = eval_segmentation(encoder, head, val_loader, device)

        log_entry = {
            "epoch": epoch, "train_loss": round(avg_loss, 6),
            "time": round(time.time() - t0, 1),
            **{f"val_{k}": round(v, 6) if isinstance(v, float) else v
               for k, v in val_metrics.items()},
        }
        logs.append(log_entry)

        if val_metrics.get("mean_iou", 0) > best_iou:
            best_iou = val_metrics["mean_iou"]
            no_improve = 0
            torch.save(head.state_dict(), output_dir / f"{task_name}_best.pt")
        else:
            no_improve += 1

        print(f"  [{task_name}] epoch {epoch}: loss={avg_loss:.4f}, "
              f"val_iou={val_metrics.get('mean_iou', 0):.4f}")

        if patience > 0 and no_improve >= patience:
            print(f"  [{task_name}] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    with open(output_dir / f"{task_name}_log.json", "w") as f:
        json.dump(logs, f, indent=2, cls=NumpyEncoder)
    return logs


@torch.no_grad()
def eval_segmentation(encoder, head, loader, device):
    head.eval()
    if encoder is not None:
        encoder.eval()

    all_preds, all_targets = [], []
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if encoder is not None:
                has_group = batch["has_group"].to(device, non_blocking=True).bool()
                features = extract_features(encoder, pixels, has_group, mode="patches")
                output = head(features)
            else:
                output = head(pixels)
            loss = loss_fn(output, target)
        total_loss += loss.item()
        n += 1
        all_preds.append(output.float().cpu().argmax(dim=1))
        all_targets.append(target.cpu())

    pred_flat = torch.cat(all_preds).reshape(-1)
    tgt_flat = torch.cat(all_targets).reshape(-1)
    metrics = compute_segmentation_metrics(pred_flat, tgt_flat, NUM_CLASSES)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Mare/highlands segmentation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--h5-path", type=str, default="output/lunar_patches_v4.h5")
    parser.add_argument("--mmap-dir", type=str, default="data/mmap")
    parser.add_argument("--mare-tif", type=str, default="data/aligned/mare_mask.tif")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--geo-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading datasets...")
    train_ds = SegmentationMmapDataset(args.mare_tif, h5_path=args.h5_path,
                                        mmap_dir=args.mmap_dir, split=0, geo_split=args.geo_split)
    val_ds = SegmentationMmapDataset(args.mare_tif, h5_path=args.h5_path,
                                      mmap_dir=args.mmap_dir, split=1, geo_split=args.geo_split)
    test_ds = SegmentationMmapDataset(args.mare_tif, h5_path=args.h5_path,
                                       mmap_dir=args.mmap_dir, split=2, geo_split=args.geo_split)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    encoder = None
    if args.mode in ("linear", "finetune") and args.checkpoint:
        encoder, config = load_pretrained_encoder(args.checkpoint, device)
        embed_dim = encoder.embed_dim
        patch_size = config.get("patch_size", 16)
        grid_size = 256 // patch_size
        head = SegmentationHead(embed_dim, NUM_CLASSES,
                                patch_size=patch_size, grid_size=grid_size).to(device)
        freeze = (args.mode == "linear")
    elif args.mode == "scratch":
        head = ScratchUNet(in_channels=28, num_classes=NUM_CLASSES).to(device)
        freeze = True
    else:
        raise ValueError("Pretrained modes require --checkpoint")

    task_name = f"mare_{args.mode}"
    print(f"\nTask: {task_name}")

    lr = args.lr

    logs = train_segmentation(
        encoder=encoder, head=head,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=lr,
        freeze_encoder=freeze, task_name=task_name, output_dir=args.output_dir)

    print("\n=== Test Set Evaluation ===")
    head.load_state_dict(
        torch.load(Path(args.output_dir) / f"{task_name}_best.pt",
                    map_location=device, weights_only=True))
    test_metrics = eval_segmentation(encoder, head, test_loader, device)
    print(f"  Test mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"  Test mean F1:  {test_metrics['mean_f1']:.4f}")

    results = {
        "task": "mare_segmentation", "mode": args.mode,
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in test_metrics.items()},
    }
    out_path = Path(args.output_dir) / f"{task_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

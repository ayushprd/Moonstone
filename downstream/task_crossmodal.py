"""Task 2: Cross-Modal Prediction (predict thermal from surface+gravity+composition).

Evaluates whether pretrained MAE learned cross-modal correlations by predicting
held-out thermal channels from other modalities.

Usage:
    # Pretrained frozen
    python task_crossmodal.py --checkpoint checkpoints/latest.pt --mode linear

    # Pretrained fine-tune
    python task_crossmodal.py --checkpoint checkpoints/latest.pt --mode finetune

    # Scratch baseline
    python task_crossmodal.py --mode scratch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import CHANNELS_V4, MODALITY_GROUPS, MODALITY_GROUP_NAMES
from downstream_base import (
    MLPHead, load_pretrained_encoder, extract_features,
    train_downstream, evaluate_downstream,
    compute_regression_metrics, NumpyEncoder,
)

# Thermal channels to predict (indices in CHANNELS_V4)
THERMAL_CHANNELS = ["diviner_tbol_midnight", "diviner_temp_night",
                     "rock_abundance", "christiansen_feature"]
THERMAL_INDICES = [CHANNELS_V4.index(ch) for ch in THERMAL_CHANNELS]
N_THERMAL = len(THERMAL_INDICES)

# Input groups (everything except thermal)
INPUT_GROUPS = ["surface", "gravity", "composition", "spectral_m3", "hapke", "radar"]


class CrossModalDataset(Dataset):
    """Dataset that loads patches and splits into input/target channels.

    Input: all non-thermal channels (zero out thermal channels)
    Target: mean thermal channel value per patch (regression)
    """

    def __init__(self, mmap_dataset, thermal_indices=THERMAL_INDICES):
        self.mmap_ds = mmap_dataset
        self.thermal_indices = thermal_indices

    def __len__(self):
        return len(self.mmap_ds)

    def __getitem__(self, idx):
        sample = self.mmap_ds[idx]
        pixels = sample["pixels"]  # (28, 256, 256)
        has_group = sample["has_group"]  # (7,)

        # Check thermal group is available (index 1 = thermal)
        thermal_group_idx = MODALITY_GROUP_NAMES.index("thermal")

        # Target: spatial mean of each thermal channel
        target = torch.zeros(N_THERMAL)
        for i, ch_idx in enumerate(self.thermal_indices):
            ch_data = pixels[ch_idx]
            valid = ch_data != 0
            if valid.any():
                target[i] = ch_data[valid].mean()

        # Zero out thermal channels in input
        input_pixels = pixels.clone()
        for ch_idx in self.thermal_indices:
            input_pixels[ch_idx] = 0.0

        # Update has_group: thermal group marked as unavailable
        input_has_group = has_group.clone()
        input_has_group[thermal_group_idx] = 0.0

        return {
            "pixels": input_pixels,
            "target": target,
            "has_group": input_has_group,
        }


class ScratchRegressor(nn.Module):
    """CNN regressor for scratch baseline."""

    def __init__(self, in_channels=28, n_outputs=N_THERMAL):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(256, n_outputs)

    def forward(self, pixels):
        x = self.features(pixels)
        x = x.flatten(1)
        return self.regressor(x)


def regression_eval_fn(preds, targets):
    """Eval function for regression."""
    return compute_regression_metrics(preds.numpy(), targets.numpy())


def main():
    parser = argparse.ArgumentParser(description="Cross-modal thermal prediction")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Use 0 in ROCm/Singularity containers where forked workers deadlock against the HIP context.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Use MmapRandomCropDataset for train/val
    from lunar_dataset import MmapRandomCropDataset

    print("Loading mmap datasets...")
    train_mmap = MmapRandomCropDataset(
        channel_names=CHANNELS_V4, crop_size=256,
        epoch_length=13_000, min_valid_fraction=0.3)
    val_mmap = MmapRandomCropDataset(
        channel_names=CHANNELS_V4, crop_size=256,
        epoch_length=1_600, min_valid_fraction=0.3)

    train_ds = CrossModalDataset(train_mmap)
    val_ds = CrossModalDataset(val_mmap)

    # Filter: only keep samples where thermal group has data
    # (the mmap dataset already ensures valid patches)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    # Setup model
    encoder = None
    if args.mode in ("linear", "finetune") and args.checkpoint:
        encoder, config = load_pretrained_encoder(args.checkpoint, device)
        embed_dim = encoder.embed_dim
        head = MLPHead(embed_dim, embed_dim // 2, N_THERMAL, dropout=0.1).to(device)
        freeze = (args.mode == "linear")
    elif args.mode == "scratch":
        head = ScratchRegressor(in_channels=28, n_outputs=N_THERMAL).to(device)
        freeze = True
    else:
        raise ValueError("Pretrained modes require --checkpoint")

    task_name = f"crossmodal_{args.mode}"
    print(f"\nTask: {task_name}")
    print(f"  Predicting {N_THERMAL} thermal channels from other modalities")

    logs = train_downstream(
        encoder=encoder,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        freeze_encoder=freeze,
        loss_fn=nn.MSELoss(),
        eval_fn=regression_eval_fn,
        task_name=task_name,
        output_dir=args.output_dir,
    )

    # Final results
    best_epoch = min(range(len(logs)), key=lambda i: logs[i].get("val_loss", 999))
    print(f"\nBest epoch: {best_epoch}")
    print(f"  Val RMSE: {logs[best_epoch].get('val_rmse', 'N/A')}")
    print(f"  Val R²:   {logs[best_epoch].get('val_r2', 'N/A')}")
    print(f"  Val Pearson: {logs[best_epoch].get('val_pearson', 'N/A')}")

    results = {
        "task": "crossmodal_thermal_prediction",
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_metrics": {k: v for k, v in logs[best_epoch].items()
                             if k.startswith("val_")},
    }
    out_path = Path(args.output_dir) / f"{task_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {out_path}")

    train_mmap.close()
    val_mmap.close()


if __name__ == "__main__":
    main()

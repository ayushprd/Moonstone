"""Task 5: FeO/TiO2 Composition Prediction.

Predicts LP GRS FeO and TiO2 weight fractions from the other 26 channels.

Usage:
    python task_composition.py --checkpoint checkpoints/latest.pt --mode linear
    python task_composition.py --mode scratch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from downstream_base import (
    MLPHead, load_pretrained_encoder,
    train_downstream, evaluate_downstream,
    compute_regression_metrics, NumpyEncoder,
)
from downstream_dataset import CompositionMmapDataset

TARGET_CHANNELS = ["lpgrs_tio2", "lpgrs_feo"]
N_TARGETS = len(TARGET_CHANNELS)


class ScratchCompositionRegressor(nn.Module):
    def __init__(self, in_channels=28, n_outputs=N_TARGETS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=4, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(256, n_outputs)

    def forward(self, pixels):
        return self.regressor(self.features(pixels).flatten(1))


def composition_eval_fn(preds, targets):
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    overall = compute_regression_metrics(preds_np, targets_np)
    for i, name in enumerate(TARGET_CHANNELS):
        ch_metrics = compute_regression_metrics(preds_np[:, i], targets_np[:, i])
        for k, v in ch_metrics.items():
            overall[f"{name}_{k}"] = v
    return overall


def main():
    parser = argparse.ArgumentParser(description="FeO/TiO2 composition prediction")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--h5-path", type=str, default="output/lunar_patches_v4.h5")
    parser.add_argument("--mmap-dir", type=str, default="data/mmap")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--geo-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Use 0 in ROCm/Singularity containers where forked workers deadlock against the HIP context.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading datasets...")
    train_ds = CompositionMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=0, geo_split=args.geo_split)
    val_ds = CompositionMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=1, geo_split=args.geo_split)
    test_ds = CompositionMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=2, geo_split=args.geo_split)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    encoder = None
    if args.mode in ("linear", "finetune") and args.checkpoint:
        encoder, config = load_pretrained_encoder(args.checkpoint, device)
        embed_dim = encoder.embed_dim
        head = MLPHead(embed_dim, embed_dim // 2, N_TARGETS, dropout=0.1).to(device)
        freeze = (args.mode == "linear")
    elif args.mode == "scratch":
        head = ScratchCompositionRegressor(in_channels=28, n_outputs=N_TARGETS).to(device)
        freeze = True
    else:
        raise ValueError("Pretrained modes require --checkpoint")

    task_name = f"composition_{args.mode}"
    print(f"\nTask: {task_name}")

    logs = train_downstream(
        encoder=encoder, head=head,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr, freeze_encoder=freeze,
        loss_fn=nn.MSELoss(), eval_fn=composition_eval_fn,
        task_name=task_name, output_dir=args.output_dir,
    )

    print("\n=== Test Set Evaluation ===")
    head.load_state_dict(
        torch.load(Path(args.output_dir) / f"{task_name}_best.pt",
                    map_location=device, weights_only=True))
    test_metrics = evaluate_downstream(
        encoder, head, test_loader, device,
        freeze_encoder=True, loss_fn=nn.MSELoss(), eval_fn=composition_eval_fn,
    )
    for ch in TARGET_CHANNELS:
        print(f"  {ch}: RMSE={test_metrics.get(f'{ch}_rmse', 'N/A'):.6f}, "
              f"R²={test_metrics.get(f'{ch}_r2', 'N/A'):.4f}")

    results = {
        "task": "composition_prediction", "mode": args.mode,
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in test_metrics.items()},
    }
    out_path = Path(args.output_dir) / f"{task_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

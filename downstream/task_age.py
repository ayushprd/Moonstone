"""Task 4: Geologic Age Regression (5-class ordinal).

Classifies patches by geologic era:
    0 = Pre-Nectarian (pN) — oldest
    1 = Nectarian (N)
    2 = Imbrian (I, EI)
    3 = Eratosthenian (E)
    4 = Copernican (C) — youngest

Usage:
    python task_age.py --checkpoint checkpoints/latest.pt --mode linear
    python task_age.py --mode scratch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from downstream_base import (
    LinearProbe, MLPHead, load_pretrained_encoder,
    train_downstream, evaluate_downstream,
    compute_classification_metrics, few_shot_indices, NumpyEncoder,
)
from downstream_dataset import AgeMmapDataset

NUM_CLASSES = 5
AGE_NAMES = ["Pre-Nectarian", "Nectarian", "Imbrian", "Eratosthenian", "Copernican"]


class ScratchAgeClassifier(nn.Module):
    def __init__(self, in_channels=28, num_classes=NUM_CLASSES):
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
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, pixels):
        return self.classifier(self.features(pixels).flatten(1))


def age_eval_fn(preds_logits, targets):
    preds = preds_logits.argmax(dim=1)
    metrics = compute_classification_metrics(preds, targets, NUM_CLASSES)
    metrics["ordinal_mae"] = (preds.float() - targets.float()).abs().mean().item()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Geologic age classification (5-class)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--h5-path", type=str, default="output/lunar_patches_v4.h5")
    parser.add_argument("--mmap-dir", type=str, default="data/mmap")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--geo-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Use 0 in ROCm/Singularity containers where forked workers deadlock against the HIP context.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading datasets...")
    train_ds = AgeMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=0, geo_split=args.geo_split)
    val_ds = AgeMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=1, geo_split=args.geo_split)
    test_ds = AgeMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=2, geo_split=args.geo_split)

    if args.few_shot > 0:
        fs_idx = few_shot_indices(train_ds.ages, args.few_shot)
        train_ds = Subset(train_ds, fs_idx)
        print(f"  Few-shot: {args.few_shot}/class → {len(train_ds)} samples")

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
        head = MLPHead(embed_dim, embed_dim // 2, NUM_CLASSES, dropout=0.1).to(device)
        freeze = (args.mode == "linear")
    elif args.mode == "scratch":
        head = ScratchAgeClassifier(in_channels=28, num_classes=NUM_CLASSES).to(device)
        freeze = True
    else:
        raise ValueError("Pretrained modes require --checkpoint")

    task_name = f"age_{args.mode}"
    if args.few_shot > 0:
        task_name += f"_fs{args.few_shot}"
    print(f"\nTask: {task_name}")

    logs = train_downstream(
        encoder=encoder, head=head,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr, freeze_encoder=freeze,
        loss_fn=nn.CrossEntropyLoss(), eval_fn=age_eval_fn,
        task_name=task_name, output_dir=args.output_dir,
    )

    print("\n=== Test Set Evaluation ===")
    head.load_state_dict(
        torch.load(Path(args.output_dir) / f"{task_name}_best.pt",
                    map_location=device, weights_only=True))
    test_metrics = evaluate_downstream(
        encoder, head, test_loader, device,
        freeze_encoder=True, loss_fn=nn.CrossEntropyLoss(), eval_fn=age_eval_fn,
    )
    print(f"  Test accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Test ordinal MAE: {test_metrics['ordinal_mae']:.4f}")

    results = {
        "task": "geologic_age_classification", "mode": args.mode,
        "few_shot": args.few_shot,
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in test_metrics.items()},
    }
    out_path = Path(args.output_dir) / f"{task_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

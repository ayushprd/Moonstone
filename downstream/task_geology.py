"""Task 1: Geologic Unit Classification (49-class).

Evaluates pretrained MAE encoder on classifying lunar geologic units
from the USGS Unified Geologic Map of the Moon v2 (2020).

Usage:
    python task_geology.py --checkpoint checkpoints/latest.pt --mode linear
    python task_geology.py --checkpoint checkpoints/latest.pt --mode finetune
    python task_geology.py --mode scratch
    python task_geology.py --checkpoint checkpoints/latest.pt --mode linear --few-shot 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from downstream_base import (
    LinearProbe, MLPHead, load_pretrained_encoder,
    train_downstream, evaluate_downstream,
    compute_classification_metrics, few_shot_indices, NumpyEncoder,
)
from downstream_dataset import GeologyMmapDataset

NUM_CLASSES = 49


class ScratchClassifier(torch.nn.Module):
    """Simple CNN classifier for scratch baseline."""

    def __init__(self, in_channels=28, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 7, stride=4, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(256, num_classes)

    def forward(self, pixels):
        x = self.features(pixels)
        x = x.flatten(1)
        return self.classifier(x)


def classification_eval_fn(preds_logits, targets):
    preds = preds_logits.argmax(dim=1)
    return compute_classification_metrics(preds, targets, NUM_CLASSES)


def main():
    parser = argparse.ArgumentParser(description="Geology classification (49-class)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--h5-path", type=str, default="output/lunar_patches_v4.h5")
    parser.add_argument("--mmap-dir", type=str, default="data/mmap")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--few-shot", type=int, default=0,
                        help="N samples per class (0=use all)")
    parser.add_argument("--geo-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading datasets...")
    train_ds = GeologyMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=0, geo_split=args.geo_split)
    val_ds = GeologyMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=1, geo_split=args.geo_split)
    test_ds = GeologyMmapDataset(h5_path=args.h5_path, mmap_dir=args.mmap_dir, split=2, geo_split=args.geo_split)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if args.few_shot > 0:
        fs_idx = few_shot_indices(train_ds.unit_ids - 1, args.few_shot)
        train_ds = Subset(train_ds, fs_idx)
        print(f"  Few-shot: {args.few_shot}/class → {len(train_ds)} samples")

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
        head = MLPHead(embed_dim, embed_dim // 2, NUM_CLASSES, dropout=0.1).to(device)
        freeze = (args.mode == "linear")
    elif args.mode == "scratch":
        head = ScratchClassifier(in_channels=28, num_classes=NUM_CLASSES).to(device)
        freeze = True
    else:
        raise ValueError("Pretrained modes require --checkpoint")

    task_name = f"geology_{args.mode}"
    if args.few_shot > 0:
        task_name += f"_fs{args.few_shot}"

    print(f"\nTask: {task_name}")
    print(f"  Mode: {args.mode}, LR: {args.lr}, Epochs: {args.epochs}")

    logs = train_downstream(
        encoder=encoder, head=head,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        freeze_encoder=freeze,
        loss_fn=torch.nn.CrossEntropyLoss(),
        eval_fn=classification_eval_fn,
        task_name=task_name, output_dir=args.output_dir,
    )

    print("\n=== Test Set Evaluation ===")
    head.load_state_dict(
        torch.load(Path(args.output_dir) / f"{task_name}_best.pt",
                    map_location=device, weights_only=True))

    test_metrics = evaluate_downstream(
        encoder, head, test_loader, device,
        freeze_encoder=True,
        loss_fn=torch.nn.CrossEntropyLoss(),
        eval_fn=classification_eval_fn,
    )

    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1: {test_metrics['macro_f1']:.4f}")

    results = {
        "task": "geology_classification",
        "mode": args.mode,
        "few_shot": args.few_shot,
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in test_metrics.items()},
    }
    out_path = Path(args.output_dir) / f"{task_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""Task 6: Large Crater Segmentation (>10 km).

Prerequisites:
    python download_craters.py  # creates data/aligned/crater_mask.tif

Usage:
    python task_craters.py --checkpoint checkpoints/latest.pt --mode linear
    python task_craters.py --mode scratch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import PATCH_SIZE
from downstream_base import (
    SegmentationHead, load_pretrained_encoder,
    compute_segmentation_metrics, NumpyEncoder,
)
from downstream_dataset import SegmentationMmapDataset
from task_mare import train_segmentation, eval_segmentation, ScratchUNet

NUM_CLASSES = 2


def main():
    parser = argparse.ArgumentParser(description="Crater segmentation (>10 km)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["linear", "finetune", "scratch"], default="linear")
    parser.add_argument("--h5-path", type=str, default="output/lunar_patches_v4.h5")
    parser.add_argument("--mmap-dir", type=str, default="data/mmap")
    parser.add_argument("--crater-tif", type=str, default="data/aligned/crater_mask.tif")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--geo-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/downstream")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if not Path(args.crater_tif).exists():
        print("Crater mask not found! Run: python download_craters.py")
        return

    print("Loading datasets...")
    train_ds = SegmentationMmapDataset(args.crater_tif, h5_path=args.h5_path,
                                        mmap_dir=args.mmap_dir, split=0, geo_split=args.geo_split)
    val_ds = SegmentationMmapDataset(args.crater_tif, h5_path=args.h5_path,
                                      mmap_dir=args.mmap_dir, split=1, geo_split=args.geo_split)
    test_ds = SegmentationMmapDataset(args.crater_tif, h5_path=args.h5_path,
                                       mmap_dir=args.mmap_dir, split=2, geo_split=args.geo_split)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if args.few_shot > 0:
        n = min(args.few_shot, len(train_ds))
        rng = np.random.RandomState(42)
        fs_idx = rng.choice(len(train_ds), n, replace=False).tolist()
        train_ds = Subset(train_ds, fs_idx)
        print(f"  Few-shot: {n} training patches")

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

    task_name = f"craters_{args.mode}"
    if args.few_shot > 0:
        task_name += f"_fs{args.few_shot}"
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
    print(f"  Test mean IoU:    {test_metrics['mean_iou']:.4f}")
    print(f"  Test crater IoU:  {test_metrics.get('class_1_iou', 0):.4f}")

    results = {
        "task": "crater_segmentation", "mode": args.mode,
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

"""Fast few-shot evaluation: precompute features, then train lightweight heads.

Precomputes encoder features once per checkpoint, then runs all few-shot
evaluations on cached features — no encoder forward pass during training.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from config import CHANNELS_V4
from downstream_base import (
    load_pretrained_encoder, extract_features,
    compute_classification_metrics, few_shot_indices,
    MLPHead, NumpyEncoder,
)
from downstream_dataset import GeologyMmapDataset, AgeMmapDataset


def precompute_features(encoder, dataset, device, batch_size=64, num_workers=0):
    """Run encoder once over entire dataset, cache features."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(num_workers > 0))
    all_feats = []
    all_targets = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device)
            has_group = batch["has_group"].to(device).bool()
            target = batch["target"]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats = extract_features(encoder, pixels, has_group, mode="cls")
            all_feats.append(feats.float().cpu())
            all_targets.append(target)
    return torch.cat(all_feats), torch.cat(all_targets)


def train_on_cached(train_feats, train_targets, val_feats, val_targets,
                    embed_dim, num_classes, epochs=30, lr=1e-3, device="cuda"):
    """Train MLP head on precomputed features."""
    head = MLPHead(embed_dim, embed_dim // 2, num_classes, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_feats, train_targets)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_f1 = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        head.train()
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            out = head(feats)
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()

        # Validate
        head.eval()
        with torch.no_grad():
            val_out = head(val_feats.to(device))
            val_preds = val_out.argmax(dim=1).cpu()
            metrics = compute_classification_metrics(val_preds, val_targets, num_classes)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    head.load_state_dict(best_state)
    return head, best_f1


def evaluate_cached(head, feats, targets, num_classes, device="cuda"):
    """Evaluate head on cached features."""
    head.eval()
    with torch.no_grad():
        out = head(feats.to(device))
        preds = out.argmax(dim=1).cpu()
    return compute_classification_metrics(preds, targets, num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers; use 0 in ROCm/Singularity containers.")
    parser.add_argument("--output-dir", default="checkpoints_v2/downstream")
    parser.add_argument("--versions", default="v1,v2",
                        help="comma list of model versions to run (v1,v2). "
                             "V1 downstream eval was never released; default skips none.")
    parser.add_argument("--no-scratch", action="store_true",
                        help="skip the few-shot scratch CNN baseline")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _all_ckpts = {
        "v1": "checkpoints/latest.pt",
        "v2": "checkpoints_v2/latest.pt",
    }
    _want = [v.strip() for v in args.versions.split(",") if v.strip()]
    checkpoints = {k: _all_ckpts[k] for k in _want if k in _all_ckpts}
    print(f"Few-shot versions: {list(checkpoints)}  scratch={'no' if args.no_scratch else 'yes'}")

    tasks = {
        "geology": {
            "dataset_cls": GeologyMmapDataset,
            "num_classes": 49,  # some classes may be missing in few-shot
        },
        "age": {
            "dataset_cls": AgeMmapDataset,
            "num_classes": 5,
        },
    }

    few_shots = [5, 10]

    for task_name, task_info in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        ds_cls = task_info["dataset_cls"]
        num_classes = task_info["num_classes"]

        train_ds = ds_cls(split=0)
        val_ds = ds_cls(split=1)
        test_ds = ds_cls(split=2)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Get labels for few-shot sampling (must match each dataset's convention)
        if task_name == "geology":
            train_labels = train_ds.unit_ids - 1
        elif task_name == "age":
            train_labels = train_ds.ages
        else:
            train_labels = np.array([train_ds[i]["target"].item() for i in range(len(train_ds))])

        # Few-shot only ever uses K*C training samples, not the full 11k train split.
        # Precompute train features ONLY for the union of needed indices (huge speedup
        # on a single GCD), keeping val/test full for selection + eval.
        fs_idx_by_k = {fs: few_shot_indices(train_labels, fs) for fs in few_shots}
        union_idx = sorted(set().union(*[set(v) for v in fs_idx_by_k.values()]))
        idx_pos = {orig: pos for pos, orig in enumerate(union_idx)}
        train_sub = Subset(train_ds, union_idx)
        print(f"  Train few-shot features needed: {len(union_idx)} "
              f"(vs {len(train_ds)} full train); val={len(val_ds)} test={len(test_ds)}")

        # Precompute features for each checkpoint
        cached = {}
        for ckpt_name, ckpt_path in checkpoints.items():
            print(f"\n  Precomputing {ckpt_name} features...")
            t0 = time.time()
            encoder, config = load_pretrained_encoder(ckpt_path, device)
            embed_dim = encoder.embed_dim

            sub_feats, sub_targets = precompute_features(encoder, train_sub, device, num_workers=args.num_workers)
            print(f"    train-subset features done ({time.time()-t0:.1f}s)")
            val_feats, val_targets = precompute_features(encoder, val_ds, device, num_workers=args.num_workers)
            print(f"    val features done ({time.time()-t0:.1f}s)")
            test_feats, test_targets = precompute_features(encoder, test_ds, device, num_workers=args.num_workers)

            cached[ckpt_name] = {
                "sub_feats": sub_feats, "sub_targets": sub_targets,
                "val_feats": val_feats, "val_targets": val_targets,
                "test_feats": test_feats, "test_targets": test_targets,
                "embed_dim": embed_dim,
            }
            del encoder
            torch.cuda.empty_cache()
            print(f"    Done in {time.time()-t0:.1f}s")

        # Run few-shot for each setting
        for fs in few_shots:
            fs_idx = fs_idx_by_k[fs]
            # rows in the precomputed train-subset for these few-shot originals
            fs_rows = torch.tensor([idx_pos[i] for i in fs_idx], dtype=torch.long)
            print(f"\n  --- {fs}-shot ({len(fs_idx)} samples) ---")

            # Scratch baseline: train CNN on few-shot pixels (slow, skip — just note)
            # Instead, train MLP on raw flattened pixels would be more fair
            # but scratch CNN is the real baseline from task_*.py

            for ckpt_name in checkpoints:
                c = cached[ckpt_name]
                fs_feats = c["sub_feats"][fs_rows]
                fs_targets = c["sub_targets"][fs_rows]

                t0 = time.time()
                head, best_val_f1 = train_on_cached(
                    fs_feats, fs_targets,
                    c["val_feats"], c["val_targets"],
                    c["embed_dim"], num_classes,
                    epochs=30, lr=1e-3, device=device,
                )
                test_metrics = evaluate_cached(
                    head, c["test_feats"], c["test_targets"],
                    num_classes, device,
                )
                elapsed = time.time() - t0

                print(f"    {ckpt_name} linear: acc={test_metrics['accuracy']:.4f} "
                      f"F1={test_metrics['macro_f1']:.4f} ({elapsed:.1f}s)")

                # Save
                result = {
                    "task": task_name,
                    "mode": "linear",
                    "checkpoint": ckpt_name,
                    "few_shot": fs,
                    "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in test_metrics.items()
                                     if k != "per_class_f1"},
                }
                fname = f"{task_name}_linear_fs{fs}_{ckpt_name}_results.json"
                with open(out_dir / fname, "w") as f:
                    json.dump(result, f, indent=2, cls=NumpyEncoder)

            # Scratch baseline with few-shot
            # Run the actual scratch CNN for fair comparison
            if args.no_scratch:
                continue
            print(f"    Running scratch baseline...")
            t0 = time.time()
            if task_name == "geology":
                from task_geology import ScratchClassifier
                scratch_head = ScratchClassifier(in_channels=28, num_classes=num_classes).to(device)
            else:
                from task_age import ScratchAgeClassifier
                scratch_head = ScratchAgeClassifier(in_channels=28, num_classes=num_classes).to(device)

            fs_train_ds = Subset(train_ds, fs_idx)
            fs_loader = DataLoader(fs_train_ds, batch_size=min(32, len(fs_idx)),
                                   shuffle=True, num_workers=args.num_workers, pin_memory=(args.num_workers > 0), drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

            # Quick train
            optimizer = torch.optim.AdamW(scratch_head.parameters(), lr=1e-3, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()
            best_f1 = -1
            best_state = None
            no_improve = 0

            for epoch in range(30):
                scratch_head.train()
                for batch in fs_loader:
                    pixels = batch["pixels"].to(device)
                    target = batch["target"].to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = scratch_head(pixels)
                        loss = loss_fn(out, target)
                    loss.backward()
                    optimizer.step()

                # Val
                scratch_head.eval()
                all_preds, all_tgts = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        pixels = batch["pixels"].to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out = scratch_head(pixels)
                        all_preds.append(out.float().cpu().argmax(dim=1))
                        all_tgts.append(batch["target"])
                preds = torch.cat(all_preds)
                tgts = torch.cat(all_tgts)
                metrics = compute_classification_metrics(preds, tgts, num_classes)

                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    best_state = {k: v.clone() for k, v in scratch_head.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 10:
                        break

            # Test
            scratch_head.load_state_dict(best_state)
            scratch_head.eval()
            all_preds, all_tgts = [], []
            with torch.no_grad():
                for batch in test_loader:
                    pixels = batch["pixels"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = scratch_head(pixels)
                    all_preds.append(out.float().cpu().argmax(dim=1))
                    all_tgts.append(batch["target"])
            preds = torch.cat(all_preds)
            tgts = torch.cat(all_tgts)
            test_metrics = compute_classification_metrics(preds, tgts, num_classes)
            elapsed = time.time() - t0

            print(f"    scratch:    acc={test_metrics['accuracy']:.4f} "
                  f"F1={test_metrics['macro_f1']:.4f} ({elapsed:.1f}s)")

            result = {
                "task": task_name,
                "mode": "scratch",
                "few_shot": fs,
                "test_metrics": {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in test_metrics.items()
                                 if k != "per_class_f1"},
            }
            fname = f"{task_name}_scratch_fs{fs}_results.json"
            with open(out_dir / fname, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)

            del scratch_head
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All few-shot evaluations complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

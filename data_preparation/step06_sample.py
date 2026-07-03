"""Create stratified train/val/test splits using spatial blocking.

Groups patches into spatial blocks, then assigns blocks to splits
stratified by dominant terrain type to prevent spatial leakage.

Usage:
    python step06_sample.py
    python step06_sample.py --val-fraction 0.1 --test-fraction 0.1
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from config import (
    HDF5_PATH,
    N_PATCHES_X,
    N_PATCHES_Y,
    OUTPUT_DIR,
)


def create_spatial_splits(
    hdf5_path: Path,
    block_size: int = 5,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Create train/val/test splits with spatial blocking.

    Strategy:
    1. Group patches into block_size x block_size spatial blocks
    2. Assign each block a category based on dominant terrain
    3. Stratified random split at block level
    4. All patches in a block go to the same split → no spatial leakage

    Returns dict: {"train": [patch_ids], "val": [patch_ids], "test": [patch_ids]}
    """
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_path, "r") as hf:
        n = hf["patches"].shape[0]
        row_indices = hf["metadata"]["row_idx"][:]
        col_indices = hf["metadata"]["col_idx"][:]

        # Get terrain category per patch (use mare fraction as proxy if geology available)
        has_geology = "geology" in hf["metadata"]
        if has_geology:
            mare_fractions = hf["metadata"]["geology"]["mare_fraction"][:]
        else:
            mare_fractions = np.full(n, 0.5)

        # Categorize patches: 0=highlands, 1=mixed, 2=mare, 3=polar
        center_lats = hf["metadata"]["center_lat"][:]

    categories = np.zeros(n, dtype=np.int32)
    for i in range(n):
        abs_lat = abs(center_lats[i])
        if abs_lat > 80:
            categories[i] = 3  # polar
        elif mare_fractions[i] > 0.7:
            categories[i] = 2  # mare
        elif mare_fractions[i] > 0.3:
            categories[i] = 1  # mixed
        else:
            categories[i] = 0  # highlands

    category_names = {0: "highlands", 1: "mixed", 2: "mare", 3: "polar"}

    # Group patches into spatial blocks
    n_blocks_x = (N_PATCHES_X + block_size - 1) // block_size
    n_blocks_y = (N_PATCHES_Y + block_size - 1) // block_size

    # Map each patch to its block
    patch_to_block = {}
    block_to_patches = defaultdict(list)
    for i in range(n):
        bx = col_indices[i] // block_size
        by = row_indices[i] // block_size
        block_id = by * n_blocks_x + bx
        patch_to_block[i] = block_id
        block_to_patches[block_id].append(i)

    # Assign each block a category (majority vote)
    block_categories = {}
    for block_id, patch_ids in block_to_patches.items():
        cats = categories[patch_ids]
        counter = Counter(cats)
        block_categories[block_id] = counter.most_common(1)[0][0]

    # Group blocks by category
    cat_to_blocks = defaultdict(list)
    for block_id, cat in block_categories.items():
        cat_to_blocks[cat].append(block_id)

    print(f"\n=== Spatial Split Statistics ===")
    print(f"  Total patches: {n}")
    print(f"  Block size: {block_size}x{block_size}")
    print(f"  Total blocks: {len(block_to_patches)}")
    print(f"  Block grid: {n_blocks_x}x{n_blocks_y}")
    print(f"\n  Blocks per category:")
    for cat, blocks in sorted(cat_to_blocks.items()):
        n_patches_in_cat = sum(len(block_to_patches[b]) for b in blocks)
        print(f"    {category_names[cat]}: {len(blocks)} blocks, {n_patches_in_cat} patches")

    # Stratified split: within each category, randomly assign blocks to splits
    splits = {"train": [], "val": [], "test": []}

    for cat, blocks in cat_to_blocks.items():
        rng.shuffle(blocks)
        n_blocks = len(blocks)
        n_test = max(1, int(n_blocks * test_fraction))
        n_val = max(1, int(n_blocks * val_fraction))
        n_train = n_blocks - n_test - n_val

        test_blocks = blocks[:n_test]
        val_blocks = blocks[n_test:n_test + n_val]
        train_blocks = blocks[n_test + n_val:]

        for b in train_blocks:
            splits["train"].extend(block_to_patches[b])
        for b in val_blocks:
            splits["val"].extend(block_to_patches[b])
        for b in test_blocks:
            splits["test"].extend(block_to_patches[b])

    for split_name, ids in splits.items():
        ids.sort()

    print(f"\n  Split sizes:")
    for name, ids in splits.items():
        print(f"    {name}: {len(ids)} patches ({len(ids)/n*100:.1f}%)")

    return splits


def add_splits_to_hdf5(hdf5_path: Path, splits: dict) -> None:
    """Add split assignments to HDF5."""
    with h5py.File(hdf5_path, "a") as hf:
        meta = hf["metadata"]
        n = hf["patches"].shape[0]

        # Create split array: 0=train, 1=val, 2=test
        split_arr = np.zeros(n, dtype=np.uint8)
        split_name_map = {"train": 0, "val": 1, "test": 2}

        for split_name, ids in splits.items():
            for idx in ids:
                split_arr[idx] = split_name_map[split_name]

        if "split" in meta:
            del meta["split"]
        meta.create_dataset("split", data=split_arr)

        # Store split name mapping
        meta.attrs["split_names"] = json.dumps(split_name_map)

    print(f"  Splits saved to {hdf5_path.name}")


def generate_split_report(hdf5_path: Path, output_path: Path) -> None:
    """Generate markdown report of split statistics."""
    with h5py.File(hdf5_path, "r") as hf:
        n = hf["patches"].shape[0]
        splits = hf["metadata"]["split"][:]
        lats = hf["metadata"]["center_lat"][:]
        lons = hf["metadata"]["center_lon"][:]

        has_geology = "geology" in hf["metadata"]
        if has_geology:
            mare_frac = hf["metadata"]["geology"]["mare_fraction"][:]

        split_names = {0: "train", 1: "val", 2: "test"}

        lines = ["# Lunar Dataset Split Report\n"]
        lines.append(f"Total patches: {n}\n")

        for split_id, name in split_names.items():
            mask = splits == split_id
            count = mask.sum()
            lines.append(f"\n## {name.capitalize()} ({count} patches, {count/n*100:.1f}%)\n")
            lines.append(f"- Latitude range: [{lats[mask].min():.1f}, {lats[mask].max():.1f}]")
            lines.append(f"- Longitude range: [{lons[mask].min():.1f}, {lons[mask].max():.1f}]")

            # Latitude band distribution
            polar = (np.abs(lats[mask]) > 80).sum()
            high = ((np.abs(lats[mask]) > 60) & (np.abs(lats[mask]) <= 80)).sum()
            mid = ((np.abs(lats[mask]) > 30) & (np.abs(lats[mask]) <= 60)).sum()
            equat = (np.abs(lats[mask]) <= 30).sum()
            lines.append(f"- Equatorial (±30°): {equat} ({equat/count*100:.1f}%)")
            lines.append(f"- Mid-latitude (30-60°): {mid} ({mid/count*100:.1f}%)")
            lines.append(f"- High-latitude (60-80°): {high} ({high/count*100:.1f}%)")
            lines.append(f"- Polar (>80°): {polar} ({polar/count*100:.1f}%)")

            if has_geology:
                lines.append(f"- Mean mare fraction: {mare_frac[mask].mean():.3f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create stratified spatial splits")
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=5)
    args = parser.parse_args()

    if not args.hdf5.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5}")
        print("Run step04_tile.py first.")
        sys.exit(1)

    splits = create_spatial_splits(
        args.hdf5,
        block_size=args.block_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    add_splits_to_hdf5(args.hdf5, splits)
    generate_split_report(args.hdf5, OUTPUT_DIR / "split_report.md")

    print("\n=== Splitting complete ===")


if __name__ == "__main__":
    main()

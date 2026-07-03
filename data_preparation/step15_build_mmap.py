"""Convert aligned GeoTIFFs to memory-mapped numpy arrays for fast training.

Creates one .npy memmap file per channel (uncompressed, float32),
enabling zero-copy random access at any pixel coordinate.
This is 10-100x faster than HDF5 or compressed GeoTIFF reads.

The output is a directory of .npy files:
    data/mmap/
        wac_morphology.npy      # (23040, 46080) float32
        elevation.npy           # (23040, 46080) float32
        ...
        lpgrs_feo.npy           # (23040, 46080) float32
        channel_index.json      # metadata

Each file is exactly 23040 * 46080 * 4 = 4.24 GB uncompressed.
Total: 28 * 4.24 GB = ~119 GB.

The OS page cache handles caching automatically — frequently accessed
regions stay in RAM without any explicit preloading.

Usage:
    # Convert all channels
    python step15_build_mmap.py

    # Convert specific channels
    python step15_build_mmap.py --channels m3_750 m3_950 elevation

    # Normalize during conversion (recommended for training)
    python step15_build_mmap.py --normalize

    # Verify
    python step15_build_mmap.py --verify
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from config import (
    CHANNELS_V4,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
)
from lunar_dataset import get_all_channel_paths


MMAP_DIR = Path(__file__).parent / "data" / "mmap"


def build_mmap(channel_names: list[str] = None, normalize: bool = False,
               stats_path: str = None, skip_existing: bool = True):
    """Convert GeoTIFFs to memory-mapped numpy arrays."""
    if channel_names is None:
        channel_names = CHANNELS_V4

    MMAP_DIR.mkdir(parents=True, exist_ok=True)

    paths = get_all_channel_paths(channel_names)
    expected_shape = (GLOBAL_HEIGHT, GLOBAL_WIDTH)
    bytes_per_file = GLOBAL_HEIGHT * GLOBAL_WIDTH * 4  # float32

    # Load normalization stats if requested
    channel_stats = {}
    if normalize:
        sp = Path(stats_path) if stats_path else Path(__file__).parent / "channel_stats.json"
        if sp.exists():
            with open(sp) as f:
                channel_stats = {k: tuple(v) for k, v in json.load(f).items()}
            print(f"Loaded normalization stats from {sp}")
        else:
            print(f"WARNING: No stats file found at {sp}, skipping normalization")
            normalize = False

    index = {
        "shape": list(expected_shape),
        "dtype": "float32",
        "channels": {},
        "normalized": normalize,
    }

    for name in tqdm(channel_names, desc="Building mmap"):
        out_path = MMAP_DIR / f"{name}.npy"

        if skip_existing and out_path.exists() and out_path.stat().st_size == bytes_per_file:
            print(f"  Skipped (exists): {name}")
            index["channels"][name] = str(out_path)
            continue

        if name not in paths or not paths[name].exists():
            print(f"  SKIP {name}: source GeoTIFF not found")
            continue

        print(f"  Converting {name}...")
        src_path = paths[name]

        # Read full GeoTIFF
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)

        assert data.shape == expected_shape, \
            f"{name}: shape {data.shape} != expected {expected_shape}"

        # Replace NaN with 0 (mmap doesn't handle NaN well for training)
        nan_mask = ~np.isfinite(data)
        data[nan_mask] = 0.0

        # Normalize if requested
        if normalize and name in channel_stats:
            mean, std = channel_stats[name]
            if std > 0:
                data = (data - mean) / std
            # Re-zero the NaN positions (they should be 0 after norm too)
            data[nan_mask] = 0.0

        # Write as raw binary (not np.save — no header overhead)
        # Actually use np.save format for compatibility
        mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=expected_shape)
        mm[:] = data
        mm.flush()
        del mm

        # Also save the NaN mask as a packed bool array (much smaller)
        mask_path = MMAP_DIR / f"{name}_mask.npy"
        # Pack bool to uint8 for space efficiency
        np.save(mask_path, np.packbits(nan_mask.ravel()))

        valid_pct = (~nan_mask).sum() / data.size * 100
        print(f"    Shape: {data.shape}, valid: {valid_pct:.1f}%, "
              f"range: [{data[~nan_mask].min():.4f}, {data[~nan_mask].max():.4f}]"
              if (~nan_mask).any() else f"    Shape: {data.shape}, all NaN")

        index["channels"][name] = str(out_path)

    # Save index
    index_path = MMAP_DIR / "channel_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved to {index_path}")

    total_size = sum(f.stat().st_size for f in MMAP_DIR.glob("*.npy")) / 1e9
    print(f"Total mmap size: {total_size:.1f} GB")


def verify_mmap():
    """Verify mmap files and print stats."""
    index_path = MMAP_DIR / "channel_index.json"
    if not index_path.exists():
        print("No mmap index found. Run build first.")
        return

    with open(index_path) as f:
        index = json.load(f)

    shape = tuple(index["shape"])
    print(f"Shape: {shape}")
    print(f"Normalized: {index['normalized']}")
    print(f"Channels: {len(index['channels'])}")

    for name, path in index["channels"].items():
        path = Path(path)
        if not path.exists():
            print(f"  {name}: MISSING")
            continue

        mm = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        # Sample a small region
        sample = mm[5000:5100, 20000:20100]
        valid = sample[sample != 0.0]
        if len(valid) > 0:
            print(f"  {name:25s}: range=[{valid.min():.4f}, {valid.max():.4f}], "
                  f"zeros={((sample == 0).sum() / sample.size * 100):.1f}%")
        else:
            print(f"  {name:25s}: all zeros in sample region")
        del mm


def main():
    parser = argparse.ArgumentParser(description="Build mmap training data")
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--normalize", action="store_true",
                        help="Apply per-channel normalization during conversion")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_mmap()
        return

    build_mmap(
        channel_names=args.channels,
        normalize=args.normalize,
        stats_path=args.stats,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()

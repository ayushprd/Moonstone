"""Build 28-channel HDF5 (V4) from existing 16-channel V2 + 12 new GeoTIFFs.

Copies channels 0-15 from the V2 HDF5, reads channels 16-27
from newly aligned GeoTIFFs (GRAIL, Mini-RF, WAC Hapke, Clementine, LP GRS).
Preserves all metadata.

Usage:
    # Test with 100 patches
    python step14_build_v4.py --test 100

    # Full rebuild
    nohup python step14_build_v4.py > build_v4.log 2>&1 &

    # Verify existing V4 HDF5
    python step14_build_v4.py --verify-only
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

from config import (
    CHANNELS_V2,
    CHANNELS_V4,
    HDF5_V2_PATH,
    HDF5_V4_PATH,
    NEW_ALIGNED_FILES,
    NEW_CHANNEL_NAMES,
    NEW_VALUE_RANGES,
    OUTPUT_DIR,
    PATCH_SIZE,
)


def rebuild_hdf5_v4(output_path: Path, v2_hdf5_path: Path,
                     max_patches: int = None) -> int:
    """Build 28-channel HDF5 by extending 16-channel V2 data.

    Copies ch 0-15 from V2, reads ch 16-27 from new GeoTIFFs.
    """
    n_ch_v2 = len(CHANNELS_V2)
    n_ch_v4 = len(CHANNELS_V4)

    print(f"\n=== Building V4 HDF5: {output_path.name} ===")
    print(f"  V2 channels: {n_ch_v2}, V4 channels: {n_ch_v4}")
    print(f"  New channels: {NEW_CHANNEL_NAMES}")

    # Preload new GeoTIFFs into RAM
    new_data = {}
    for name, path in NEW_ALIGNED_FILES.items():
        if path.exists():
            print(f"  Preloading {name} into RAM...")
            with rasterio.open(path) as src:
                new_data[name] = src.read(1)
        else:
            print(f"  WARNING: Missing {name}: {path}")

    print(f"  Loaded {len(new_data)}/{len(NEW_ALIGNED_FILES)} new channels")

    with h5py.File(v2_hdf5_path, "r") as v2_hf:
        n = v2_hf["patches"].shape[0]
        if max_patches:
            n = min(n, max_patches)

        print(f"  Source: {v2_hdf5_path.name} ({v2_hf['patches'].shape})")
        print(f"  Target: ({n}, {n_ch_v4}, {PATCH_SIZE}, {PATCH_SIZE})")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as new_hf:
            # Create patches dataset
            patches_ds = new_hf.create_dataset(
                "patches",
                shape=(n, n_ch_v4, PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
                chunks=(1, n_ch_v4, PATCH_SIZE, PATCH_SIZE),
                compression="gzip",
                compression_opts=4,
            )

            # Copy metadata group structure
            meta = new_hf.create_group("metadata")
            v2_meta = v2_hf["metadata"]

            for key in ["patch_id", "row_idx", "col_idx", "center_lat",
                        "center_lon", "valid_fraction", "mean_elevation",
                        "elevation_range"]:
                if key in v2_meta:
                    data = v2_meta[key][:n]
                    meta.create_dataset(key, data=data)

            if "split" in v2_meta:
                meta.create_dataset("split", data=v2_meta["split"][:n])
                if "split_names" in v2_meta.attrs:
                    meta.attrs["split_names"] = v2_meta.attrs["split_names"]

            if "geology" in v2_meta:
                geo_grp = meta.create_group("geology")
                for key in v2_meta["geology"]:
                    data = v2_meta["geology"][key][:n]
                    geo_grp.create_dataset(key, data=data)

            # Copy M3 subgroup
            if "m3" in v2_meta:
                m3_grp = meta.create_group("m3")
                for key in v2_meta["m3"]:
                    data = v2_meta["m3"][key][:n]
                    m3_grp.create_dataset(key, data=data)

            # Extended has_channel: (n, 28)
            v2_has_ch = v2_meta["has_channel"][:n]  # (n, 16)
            new_has_ch = np.zeros((n, n_ch_v4), dtype=bool)
            new_has_ch[:, :n_ch_v2] = v2_has_ch

            # Read row/col indices for window positioning
            row_indices = v2_meta["row_idx"][:n]
            col_indices = v2_meta["col_idx"][:n]

            # Update attributes
            new_hf.attrs["n_channels"] = n_ch_v4
            new_hf.attrs["patch_size"] = PATCH_SIZE
            if "ppd" in v2_hf.attrs:
                new_hf.attrs["ppd"] = v2_hf.attrs["ppd"]
            new_hf.attrs["channel_names"] = CHANNELS_V4

            # Process patches in batches
            BATCH = 128
            print(f"  Writing {n} patches (batch={BATCH})...")
            for b_start in tqdm(range(0, n, BATCH), desc="Building V4",
                                ncols=80, total=(n + BATCH - 1) // BATCH):
                b_end = min(b_start + BATCH, n)
                bs = b_end - b_start

                # Batch-read channels 0-15 from V2 HDF5
                v2_batch = v2_hf["patches"][b_start:b_end]  # (bs, 16, 256, 256)
                new_batch = np.full((bs, n_ch_v4, PATCH_SIZE, PATCH_SIZE),
                                    np.nan, dtype=np.float32)
                new_batch[:, :n_ch_v2] = v2_batch

                for i in range(bs):
                    idx = b_start + i
                    ry = int(row_indices[idx])
                    rx = int(col_indices[idx])
                    y0 = ry * PATCH_SIZE
                    x0 = rx * PATCH_SIZE
                    y1 = y0 + PATCH_SIZE
                    x1 = x0 + PATCH_SIZE

                    # Read channels 16-27 from preloaded new arrays
                    for ch_offset, ch_name in enumerate(NEW_CHANNEL_NAMES):
                        ch_idx = n_ch_v2 + ch_offset
                        if ch_name in new_data:
                            data = new_data[ch_name][y0:y1, x0:x1].astype(
                                np.float32)
                            new_batch[i, ch_idx] = data
                            new_has_ch[idx, ch_idx] = np.isfinite(data).any()

                patches_ds[b_start:b_end] = new_batch

            meta.create_dataset("has_channel", data=new_has_ch)

    del new_data

    size_gb = output_path.stat().st_size / 1e9
    print(f"\n  V4 HDF5: {output_path.name} ({size_gb:.2f} GB)")
    print(f"  Patches: {n}, Channels: {n_ch_v4}")

    # Per-channel coverage summary
    with h5py.File(output_path, "r") as hf:
        has_ch = hf["metadata/has_channel"][:]
        print("\n  Channel coverage:")
        for i, ch_name in enumerate(CHANNELS_V4):
            pct = has_ch[:, i].sum() / n * 100
            print(f"    ch {i:2d}: {ch_name:25s} {pct:6.1f}%")

    return n


def verify_v4(hdf5_path: Path) -> None:
    """Verify V4 HDF5 integrity and print summary stats."""
    print(f"\n=== Verifying V4 HDF5: {hdf5_path.name} ===")

    with h5py.File(hdf5_path, "r") as hf:
        patches = hf["patches"]
        n, n_ch, h, w = patches.shape
        print(f"  Shape: {patches.shape}")
        print(f"  Channel names: {list(hf.attrs['channel_names'])}")
        print(f"  N channels: {hf.attrs['n_channels']}")

        has_ch = hf["metadata/has_channel"][:]

        print(f"\n  Per-channel statistics (sample of 100 patches):")
        sample_idx = np.linspace(0, n - 1, 100, dtype=int)
        sample = patches[sample_idx]

        for i, ch_name in enumerate(CHANNELS_V4):
            ch_data = sample[:, i]
            valid = np.isfinite(ch_data)
            if valid.any():
                vmin = np.nanmin(ch_data)
                vmax = np.nanmax(ch_data)
                vmean = np.nanmean(ch_data)
            else:
                vmin = vmax = vmean = float("nan")
            coverage = has_ch[:, i].sum() / n * 100
            print(f"    ch {i:2d}: {ch_name:25s}  "
                  f"range=[{vmin:12.4f}, {vmax:12.4f}]  "
                  f"mean={vmean:10.4f}  cov={coverage:5.1f}%")

        # Check V2 channels preserved
        print("\n  Metadata groups:", list(hf["metadata"].keys()))
        if "geology" in hf["metadata"]:
            print("  Geology keys:", list(hf["metadata/geology"].keys()))
        if "split" in hf["metadata"]:
            splits = hf["metadata/split"][:]
            for s in np.unique(splits):
                print(f"    Split {s}: {(splits == s).sum()} patches")


def main():
    parser = argparse.ArgumentParser(description="Build 28-channel V4 HDF5")
    parser.add_argument("--test", type=int, default=None,
                        help="Test with N patches")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing V4 HDF5")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to V2 HDF5 (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to V4 HDF5 (default: from config)")
    args = parser.parse_args()

    v2_path = Path(args.input) if args.input else HDF5_V2_PATH
    v4_path = Path(args.output) if args.output else HDF5_V4_PATH

    if args.verify_only:
        if not v4_path.exists():
            print(f"ERROR: {v4_path} not found")
            sys.exit(1)
        verify_v4(v4_path)
        return

    if not v2_path.exists():
        print(f"ERROR: V2 HDF5 not found at {v2_path}")
        print("Run step12_m3_integrate.py first to build the V2 HDF5.")
        sys.exit(1)

    n = rebuild_hdf5_v4(v4_path, v2_path, max_patches=args.test)
    print(f"\nDone. {n} patches written to {v4_path}")

    if not args.test:
        verify_v4(v4_path)


if __name__ == "__main__":
    main()

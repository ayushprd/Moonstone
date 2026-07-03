"""Tile all 8 aligned/derived channels into 256x256 HDF5 patches.

Reads from aligned GeoTIFFs, creates non-overlapping patches,
and stores in HDF5 with per-patch metadata.

Usage:
    # Test with 10 patches
    python step04_tile.py --samples 10

    # Full tiling
    nohup python step04_tile.py > tile.log 2>&1 &
"""

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

from config import (
    ALIGNED_DIR,
    ALIGNED_FILES,
    CHANNELS,
    DERIVED_DIR,
    DERIVED_FILES,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    HDF5_PATH,
    N_PATCHES_X,
    N_PATCHES_Y,
    OUTPUT_DIR,
    PATCH_SIZE,
    PPD,
)


def get_channel_paths() -> dict:
    """Return ordered dict of channel_name -> GeoTIFF path."""
    return {
        "wac_morphology": ALIGNED_FILES["wac_morphology"],
        "elevation": ALIGNED_FILES["elevation_blended"],
        "slope": DERIVED_FILES["slope"],
        "roughness": DERIVED_FILES["roughness"],
        "diviner_tbol_midnight": ALIGNED_FILES["diviner_tbol_midnight"],
        "diviner_temp_night": ALIGNED_FILES["diviner_temp_night"],
        "rock_abundance": ALIGNED_FILES["rock_abundance"],
        "christiansen_feature": ALIGNED_FILES["christiansen_feature"],
    }


def patch_center_coords(row_idx: int, col_idx: int) -> tuple:
    """Compute center lat/lon for a patch given its grid indices.

    Returns (center_lat, center_lon) in degrees.
    Longitude: -180 to +180°, Latitude: -90 to +90°.
    Grid uses lon_0=0 convention (centered at 0°).
    """
    # Patch covers pixels [col_idx*256 : (col_idx+1)*256] in x
    # and [row_idx*256 : (row_idx+1)*256] in y
    center_col = (col_idx + 0.5) * PATCH_SIZE  # pixel center
    center_row = (row_idx + 0.5) * PATCH_SIZE

    center_lon = -180.0 + center_col / PPD  # -180 to +180
    center_lat = 90.0 - center_row / PPD  # +90 to -90

    return center_lat, center_lon


def tile_to_hdf5(output_path: Path, channel_paths: dict,
                 max_nodata_fraction: float = 0.5,
                 max_patches: int = None,
                 skip_existing: bool = True) -> int:
    """Tile all channels into HDF5 patches.

    Returns number of patches written.
    """
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skipped (exists): {output_path.name}")
        with h5py.File(output_path, "r") as f:
            return f["patches"].shape[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check which channels are available
    available_channels = {}
    missing_channels = []
    for name, path in channel_paths.items():
        if path.exists():
            available_channels[name] = path
        else:
            missing_channels.append(name)

    n_channels = len(CHANNELS)  # Always 8 slots
    print(f"\n=== Tiling to HDF5 ===")
    print(f"  Available channels: {list(available_channels.keys())}")
    if missing_channels:
        print(f"  Missing channels (will be NaN): {missing_channels}")
    print(f"  Patch grid: {N_PATCHES_X} x {N_PATCHES_Y} = {N_PATCHES_X * N_PATCHES_Y} patches")
    print(f"  Max nodata fraction: {max_nodata_fraction}")

    # Open all available channel files
    src_files = {}
    for name, path in available_channels.items():
        src_files[name] = rasterio.open(path)

    # First pass: determine which patches are valid
    total_patches = N_PATCHES_X * N_PATCHES_Y
    if max_patches:
        total_patches = min(total_patches, max_patches)

    # Collect valid patches
    print("  Scanning patches for validity...")
    valid_patches = []

    patch_iter = []
    for ry in range(N_PATCHES_Y):
        for rx in range(N_PATCHES_X):
            patch_iter.append((ry, rx))
            if max_patches and len(patch_iter) >= max_patches:
                break
        if max_patches and len(patch_iter) >= max_patches:
            break

    for ry, rx in tqdm(patch_iter, desc="Scanning", ncols=80):
        # Read a quick check from the first available channel
        check_name = next(iter(available_channels))
        window = rasterio.windows.Window(
            rx * PATCH_SIZE, ry * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
        )

        # Bounds check
        src = src_files[check_name]
        if (rx * PATCH_SIZE + PATCH_SIZE > src.width or
                ry * PATCH_SIZE + PATCH_SIZE > src.height):
            continue

        data = src.read(1, window=window)
        valid_frac = np.isfinite(data).sum() / data.size

        if valid_frac >= (1.0 - max_nodata_fraction):
            valid_patches.append((ry, rx))

    print(f"  Valid patches: {len(valid_patches)} / {len(patch_iter)}")

    if not valid_patches:
        print("  No valid patches found!")
        for name, src in src_files.items():
            src.close()
        return 0

    n_valid = len(valid_patches)

    # Create HDF5 file
    print(f"  Creating HDF5: {output_path}")
    with h5py.File(output_path, "w") as hf:
        # Patch data: (N, 8, 256, 256) float32
        patches_ds = hf.create_dataset(
            "patches",
            shape=(n_valid, n_channels, PATCH_SIZE, PATCH_SIZE),
            dtype=np.float32,
            chunks=(1, n_channels, PATCH_SIZE, PATCH_SIZE),
            compression="gzip",
            compression_opts=4,
        )

        # Metadata datasets
        meta = hf.create_group("metadata")
        patch_id_ds = meta.create_dataset("patch_id", shape=(n_valid,), dtype=np.int32)
        row_idx_ds = meta.create_dataset("row_idx", shape=(n_valid,), dtype=np.int32)
        col_idx_ds = meta.create_dataset("col_idx", shape=(n_valid,), dtype=np.int32)
        center_lat_ds = meta.create_dataset("center_lat", shape=(n_valid,), dtype=np.float32)
        center_lon_ds = meta.create_dataset("center_lon", shape=(n_valid,), dtype=np.float32)
        valid_frac_ds = meta.create_dataset("valid_fraction", shape=(n_valid,), dtype=np.float32)
        mean_elev_ds = meta.create_dataset("mean_elevation", shape=(n_valid,), dtype=np.float32)
        elev_range_ds = meta.create_dataset("elevation_range", shape=(n_valid,), dtype=np.float32)

        # Per-channel availability flags
        has_channel = meta.create_dataset(
            "has_channel",
            shape=(n_valid, n_channels),
            dtype=np.bool_,
        )

        # Store channel names and other attributes
        hf.attrs["n_channels"] = n_channels
        hf.attrs["patch_size"] = PATCH_SIZE
        hf.attrs["ppd"] = PPD
        hf.attrs["channel_names"] = CHANNELS

        # Write patches
        batch_size = 100
        print(f"  Writing {n_valid} patches...")

        for idx, (ry, rx) in enumerate(tqdm(valid_patches, desc="Tiling", ncols=80)):
            window = rasterio.windows.Window(
                rx * PATCH_SIZE, ry * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
            )

            patch = np.full((n_channels, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
            channel_valid = np.zeros(n_channels, dtype=bool)

            for ch_idx, ch_name in enumerate(CHANNELS):
                if ch_name in src_files:
                    data = src_files[ch_name].read(1, window=window).astype(np.float32)
                    patch[ch_idx] = data
                    channel_valid[ch_idx] = np.isfinite(data).sum() > 0

            # Compute metadata
            center_lat, center_lon = patch_center_coords(ry, rx)
            all_valid = np.all(np.isfinite(patch), axis=0)
            valid_fraction = all_valid.sum() / (PATCH_SIZE * PATCH_SIZE)

            elev = patch[CHANNELS.index("elevation")]
            valid_elev = elev[np.isfinite(elev)]
            mean_elev = float(np.mean(valid_elev)) if len(valid_elev) > 0 else np.nan
            elev_range = float(np.ptp(valid_elev)) if len(valid_elev) > 0 else np.nan

            # Write to HDF5
            patches_ds[idx] = patch
            patch_id_ds[idx] = idx
            row_idx_ds[idx] = ry
            col_idx_ds[idx] = rx
            center_lat_ds[idx] = center_lat
            center_lon_ds[idx] = center_lon
            valid_frac_ds[idx] = valid_fraction
            mean_elev_ds[idx] = mean_elev
            elev_range_ds[idx] = elev_range
            has_channel[idx] = channel_valid

    # Close all source files
    for src in src_files.values():
        src.close()

    print(f"  Done: {output_path.name} ({output_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Patches written: {n_valid}")
    return n_valid


def verify_hdf5(path: Path) -> None:
    """Print summary of the HDF5 dataset."""
    print(f"\n=== HDF5 Summary: {path.name} ===")
    with h5py.File(path, "r") as hf:
        print(f"  Patches shape: {hf['patches'].shape}")
        print(f"  Dtype: {hf['patches'].dtype}")
        print(f"  Channels: {list(hf.attrs.get('channel_names', []))}")

        n = hf["patches"].shape[0]
        meta = hf["metadata"]
        print(f"\n  Metadata:")
        print(f"    Lat range: [{meta['center_lat'][:].min():.1f}, {meta['center_lat'][:].max():.1f}]")
        print(f"    Lon range: [{meta['center_lon'][:].min():.1f}, {meta['center_lon'][:].max():.1f}]")
        print(f"    Valid fraction: mean={meta['valid_fraction'][:].mean():.3f}, "
              f"min={meta['valid_fraction'][:].min():.3f}")
        print(f"    Mean elevation: mean={np.nanmean(meta['mean_elevation'][:]):.0f} m")

        # Channel availability
        if "has_channel" in meta:
            has_ch = meta["has_channel"][:]
            channel_names = list(hf.attrs.get("channel_names", CHANNELS))
            print(f"\n  Channel availability:")
            for i, name in enumerate(channel_names):
                pct = has_ch[:, i].sum() / n * 100
                print(f"    {name}: {pct:.1f}%")

        # Sample patch stats
        print(f"\n  Sample patch stats (5 random):")
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=min(5, n), replace=False)
        channel_names = list(hf.attrs.get("channel_names", CHANNELS))
        for idx in sorted(indices):
            patch = hf["patches"][idx]
            lat = meta["center_lat"][idx]
            lon = meta["center_lon"][idx]
            print(f"    Patch {idx} (lat={lat:.1f}, lon={lon:.1f}):")
            for ch_i, ch_name in enumerate(channel_names):
                ch = patch[ch_i]
                valid = ch[np.isfinite(ch)]
                if len(valid) > 0:
                    print(f"      {ch_name}: [{valid.min():.3f}, {valid.max():.3f}] "
                          f"mean={valid.mean():.3f}")
                else:
                    print(f"      {ch_name}: all NaN")


def main():
    parser = argparse.ArgumentParser(description="Tile channels into HDF5 patches")
    parser.add_argument(
        "--output", type=Path, default=HDF5_PATH,
        help="Output HDF5 path"
    )
    parser.add_argument("--max-nodata", type=float, default=0.5,
                        help="Max fraction of nodata pixels per patch")
    parser.add_argument("--samples", type=int, default=None,
                        help="Process only N patches for testing")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing HDF5")
    args = parser.parse_args()

    if args.verify_only:
        if args.output.exists():
            verify_hdf5(args.output)
        else:
            print(f"File not found: {args.output}")
        return

    channel_paths = get_channel_paths()
    n_written = tile_to_hdf5(
        args.output,
        channel_paths,
        max_nodata_fraction=args.max_nodata,
        max_patches=args.samples,
        skip_existing=args.skip_existing,
    )

    if n_written > 0 and args.output.exists():
        verify_hdf5(args.output)

    print("\n=== Tiling complete ===")


if __name__ == "__main__":
    main()

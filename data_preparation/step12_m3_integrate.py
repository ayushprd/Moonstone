"""Build 16-channel HDF5 from existing 8-channel + M3 GeoTIFFs.

Copies channels 0-7 from the original HDF5, reads channels 8-15
from M3 aligned GeoTIFFs, preserves all metadata.

Usage:
    # Test with 100 patches
    python step12_m3_integrate.py --test 100

    # Full rebuild (~2-4 hours)
    nohup python step12_m3_integrate.py > m3_integrate.log 2>&1 &

    # Skip multiangle index
    python step12_m3_integrate.py --skip-multiangle

    # Verify existing V2 HDF5
    python step12_m3_integrate.py --verify-only
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

from config import (
    ALIGNED_DIR,
    CHANNELS,
    CHANNELS_V2,
    HDF5_PATH,
    HDF5_V2_PATH,
    M3_ALIGNED_FILES,
    M3_BAND_NAMES,
    M3_GEOMETRY_FILES,
    M3_MULTIANGLE_PATH,
    M3_VALUE_RANGES,
    OUTPUT_DIR,
    PATCH_SIZE,
)


def get_channel_paths_v2() -> dict:
    """Return ordered dict of all 16 channel_name -> GeoTIFF path."""
    from config import ALIGNED_FILES, DERIVED_FILES

    paths = {
        "wac_morphology": ALIGNED_FILES["wac_morphology"],
        "elevation": ALIGNED_FILES["elevation_blended"],
        "slope": DERIVED_FILES["slope"],
        "roughness": DERIVED_FILES["roughness"],
        "diviner_tbol_midnight": ALIGNED_FILES["diviner_tbol_midnight"],
        "diviner_temp_night": ALIGNED_FILES["diviner_temp_night"],
        "rock_abundance": ALIGNED_FILES["rock_abundance"],
        "christiansen_feature": ALIGNED_FILES["christiansen_feature"],
    }
    # Add M3 bands
    for name, path in M3_ALIGNED_FILES.items():
        paths[name] = path

    return paths


def rebuild_hdf5_v2(output_path: Path, old_hdf5_path: Path,
                    max_patches: int = None,
                    skip_existing: bool = True) -> int:
    """Build 16-channel HDF5 by extending existing 8-channel data.

    Strategy: copy ch 0-7 from old HDF5, read ch 8-15 from M3 GeoTIFFs.
    """
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skipped (exists): {output_path.name}")
        with h5py.File(output_path, "r") as hf:
            return hf["patches"].shape[0]

    print(f"\n=== Building V2 HDF5: {output_path.name} ===")

    n_ch_old = len(CHANNELS)
    n_ch_new = len(CHANNELS_V2)

    # Preload fixed GeoTIFFs and M3 bands into RAM for fast patch extraction
    channel_paths_v2 = get_channel_paths_v2()
    reread_from_tif = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        if ch_name in ("diviner_temp_night", "roughness"):  # fixed GeoTIFFs
            path = channel_paths_v2.get(ch_name)
            if path and path.exists():
                print(f"  Preloading ch {ch_idx} ({ch_name}) into RAM...")
                with rasterio.open(path) as src:
                    reread_from_tif[ch_idx] = (ch_name, src.read(1))

    # Preload M3 GeoTIFFs into RAM
    m3_data = {}
    m3_missing = []
    for name, path in M3_ALIGNED_FILES.items():
        if path.exists():
            print(f"  Preloading {name} into RAM...")
            with rasterio.open(path) as src:
                m3_data[name] = src.read(1)
        else:
            m3_missing.append(name)
            print(f"  WARNING: Missing M3 band: {path.name}")

    # Geometry files are written separately, not needed for patch extraction

    with h5py.File(old_hdf5_path, "r") as old_hf:
        n = old_hf["patches"].shape[0]
        if max_patches:
            n = min(n, max_patches)

        print(f"  Source: {old_hdf5_path.name} ({old_hf['patches'].shape})")
        print(f"  Target: ({n}, {n_ch_new}, {PATCH_SIZE}, {PATCH_SIZE})")
        print(f"  M3 bands available: {list(m3_data.keys())}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as new_hf:
            # Create patches dataset
            patches_ds = new_hf.create_dataset(
                "patches",
                shape=(n, n_ch_new, PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
                chunks=(1, n_ch_new, PATCH_SIZE, PATCH_SIZE),
                compression="gzip",
                compression_opts=4,
            )

            # Copy metadata group structure
            meta = new_hf.create_group("metadata")

            # Copy scalar metadata datasets
            old_meta = old_hf["metadata"]
            for key in ["patch_id", "row_idx", "col_idx", "center_lat",
                         "center_lon", "valid_fraction", "mean_elevation",
                         "elevation_range"]:
                if key in old_meta:
                    data = old_meta[key][:n]
                    meta.create_dataset(key, data=data)

            # Copy split
            if "split" in old_meta:
                meta.create_dataset("split", data=old_meta["split"][:n])
                if "split_names" in old_meta.attrs:
                    meta.attrs["split_names"] = old_meta.attrs["split_names"]

            # Copy geology subgroup
            if "geology" in old_meta:
                geo_grp = meta.create_group("geology")
                for key in old_meta["geology"]:
                    data = old_meta["geology"][key][:n]
                    geo_grp.create_dataset(key, data=data)

            # Extended has_channel
            old_has_ch = old_meta["has_channel"][:n]  # (n, 8)
            new_has_ch = np.zeros((n, n_ch_new), dtype=bool)
            new_has_ch[:, :n_ch_old] = old_has_ch

            # M3 coverage per patch
            m3_valid_fraction = np.zeros(n, dtype=np.float32)

            # Read row/col indices for M3 window positioning
            row_indices = old_meta["row_idx"][:n]
            col_indices = old_meta["col_idx"][:n]

            # Update attributes
            new_hf.attrs["n_channels"] = n_ch_new
            new_hf.attrs["patch_size"] = PATCH_SIZE
            if "ppd" in old_hf.attrs:
                new_hf.attrs["ppd"] = old_hf.attrs["ppd"]
            new_hf.attrs["channel_names"] = CHANNELS_V2

            # Process patches in batches for faster I/O
            BATCH = 128
            print(f"  Writing {n} patches (batch={BATCH})...")
            for b_start in tqdm(range(0, n, BATCH), desc="Building V2",
                                ncols=80, total=(n + BATCH - 1) // BATCH):
                b_end = min(b_start + BATCH, n)
                bs = b_end - b_start

                # Batch-read channels 0-7 from old HDF5
                old_batch = old_hf["patches"][b_start:b_end]  # (bs, 8, 256, 256)
                new_batch = np.full((bs, n_ch_new, PATCH_SIZE, PATCH_SIZE),
                                    np.nan, dtype=np.float32)
                new_batch[:, :n_ch_old] = old_batch

                for i in range(bs):
                    idx = b_start + i
                    ry = int(row_indices[idx])
                    rx = int(col_indices[idx])
                    y0 = ry * PATCH_SIZE
                    x0 = rx * PATCH_SIZE
                    y1 = y0 + PATCH_SIZE
                    x1 = x0 + PATCH_SIZE

                    # Override specific channels from preloaded arrays
                    for ch_i, (ch_name, arr) in reread_from_tif.items():
                        data = arr[y0:y1, x0:x1].astype(np.float32)
                        new_batch[i, ch_i] = data
                        new_has_ch[idx, ch_i] = np.isfinite(data).sum() > 0

                    # Read channels 8-15 from preloaded M3 arrays
                    m3_any_valid = False
                    for ch_offset, band_name in enumerate(M3_BAND_NAMES):
                        ch_idx = n_ch_old + ch_offset
                        if band_name in m3_data:
                            data = m3_data[band_name][y0:y1, x0:x1].astype(
                                np.float32)
                            new_batch[i, ch_idx] = data
                            has_data = np.isfinite(data).sum() > 0
                            new_has_ch[idx, ch_idx] = has_data
                            if has_data:
                                m3_any_valid = True

                    # M3 coverage
                    if m3_any_valid:
                        first_m3 = new_batch[i, n_ch_old]
                        m3_valid_fraction[idx] = (
                            np.isfinite(first_m3).sum() / (PATCH_SIZE * PATCH_SIZE))

                # Batch-write to HDF5
                patches_ds[b_start:b_end] = new_batch

            # Save extended has_channel
            meta.create_dataset("has_channel", data=new_has_ch)

            # Save M3-specific metadata
            m3_meta = meta.create_group("m3")
            m3_meta.create_dataset("m3_valid_fraction", data=m3_valid_fraction)

    # Free preloaded arrays
    del m3_data
    del reread_from_tif

    size_gb = output_path.stat().st_size / 1e9
    m3_coverage = (m3_valid_fraction > 0).sum() / n * 100
    print(f"\n  V2 HDF5: {output_path.name} ({size_gb:.2f} GB)")
    print(f"  Patches: {n}, Channels: {n_ch_new}")
    print(f"  M3 coverage: {m3_coverage:.1f}% of patches have M3 data")

    return n


def build_multiangle_index(output_path: Path, hdf5_v2_path: Path) -> None:
    """Build multiangle geometry index from M3 geometry GeoTIFFs."""
    print(f"\n=== Building Multiangle Index ===")

    n_obs_path = M3_GEOMETRY_FILES.get("m3_n_observations")
    inc_path = M3_GEOMETRY_FILES.get("m3_mean_incidence")
    emi_path = M3_GEOMETRY_FILES.get("m3_mean_emission")
    pha_path = M3_GEOMETRY_FILES.get("m3_mean_phase")

    missing = [p for p in [n_obs_path, inc_path, emi_path, pha_path]
               if not p.exists()]
    if missing:
        print(f"  Missing geometry files: {[p.name for p in missing]}")
        print("  Skipping multiangle index.")
        return

    n_obs_src = rasterio.open(n_obs_path)
    inc_src = rasterio.open(inc_path)
    emi_src = rasterio.open(emi_path)
    pha_src = rasterio.open(pha_path)

    with h5py.File(hdf5_v2_path, "r") as hf:
        n = hf["patches"].shape[0]
        row_indices = hf["metadata"]["row_idx"][:]
        col_indices = hf["metadata"]["col_idx"][:]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as out_hf:
        geo_grp = out_hf.create_group("patch_geometry")

        nobs_ds = geo_grp.create_dataset(
            "n_observations", shape=(n, PATCH_SIZE, PATCH_SIZE),
            dtype=np.uint8, chunks=(1, PATCH_SIZE, PATCH_SIZE),
            compression="gzip", compression_opts=4)
        inc_ds = geo_grp.create_dataset(
            "mean_incidence", shape=(n, PATCH_SIZE, PATCH_SIZE),
            dtype=np.float16, chunks=(1, PATCH_SIZE, PATCH_SIZE),
            compression="gzip", compression_opts=4)
        emi_ds = geo_grp.create_dataset(
            "mean_emission", shape=(n, PATCH_SIZE, PATCH_SIZE),
            dtype=np.float16, chunks=(1, PATCH_SIZE, PATCH_SIZE),
            compression="gzip", compression_opts=4)
        pha_ds = geo_grp.create_dataset(
            "mean_phase", shape=(n, PATCH_SIZE, PATCH_SIZE),
            dtype=np.float16, chunks=(1, PATCH_SIZE, PATCH_SIZE),
            compression="gzip", compression_opts=4)

        for idx in tqdm(range(n), desc="Multiangle", ncols=80):
            ry = int(row_indices[idx])
            rx = int(col_indices[idx])
            window = rasterio.windows.Window(
                rx * PATCH_SIZE, ry * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

            nobs = n_obs_src.read(1, window=window)
            nobs_ds[idx] = np.clip(nobs, 0, 255).astype(np.uint8)
            inc_ds[idx] = inc_src.read(1, window=window).astype(np.float16)
            emi_ds[idx] = emi_src.read(1, window=window).astype(np.float16)
            pha_ds[idx] = pha_src.read(1, window=window).astype(np.float16)

    n_obs_src.close()
    inc_src.close()
    emi_src.close()
    pha_src.close()

    size_gb = output_path.stat().st_size / 1e9
    print(f"  Multiangle index: {output_path.name} ({size_gb:.2f} GB)")


def verify_v2(hdf5_path: Path) -> None:
    """Verify the V2 HDF5 dataset."""
    print(f"\n=== Verifying V2 HDF5: {hdf5_path.name} ===")

    with h5py.File(hdf5_path, "r") as hf:
        shape = hf["patches"].shape
        print(f"  Shape: {shape}")
        channel_names = list(hf.attrs.get("channel_names", CHANNELS_V2))
        print(f"  Channels: {channel_names}")

        n = shape[0]
        n_ch = shape[1]

        # Check channels
        if n_ch != len(CHANNELS_V2):
            print(f"  WARNING: Expected {len(CHANNELS_V2)} channels, got {n_ch}")

        # Sample stats
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=min(50, n), replace=False)

        print(f"\n  Per-channel stats (from {len(sample_idx)} sampled patches):")
        print(f"  {'Channel':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'NaN%':>8}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for ch_i, ch_name in enumerate(channel_names):
            mins, maxs, means, nan_counts = [], [], [], []
            for idx in sample_idx:
                data = hf["patches"][idx, ch_i]
                finite = data[np.isfinite(data)]
                if len(finite) > 0:
                    mins.append(finite.min())
                    maxs.append(finite.max())
                    means.append(finite.mean())
                nan_counts.append((~np.isfinite(data)).sum())

            if mins:
                ch_min = min(mins)
                ch_max = max(maxs)
                ch_mean = np.mean(means)
                nan_pct = np.mean(nan_counts) / (PATCH_SIZE**2) * 100
            else:
                ch_min = ch_max = ch_mean = float("nan")
                nan_pct = 100.0

            print(f"  {ch_name:<25} {ch_min:>10.4f} {ch_max:>10.4f} "
                  f"{ch_mean:>10.4f} {nan_pct:>7.1f}%")

        # Check metadata
        meta = hf.get("metadata", {})
        print(f"\n  Metadata keys: {list(meta.keys())}")
        if "has_channel" in meta:
            has_ch = meta["has_channel"][:]
            print(f"  has_channel shape: {has_ch.shape}")
            for ch_i, ch_name in enumerate(channel_names):
                pct = has_ch[:, ch_i].sum() / n * 100
                print(f"    {ch_name}: {pct:.1f}%")

        if "split" in meta:
            splits = meta["split"][:]
            for s, name in {0: "train", 1: "val", 2: "test"}.items():
                count = (splits == s).sum()
                print(f"  Split {name}: {count} ({count/n*100:.1f}%)")

        if "m3" in meta:
            m3_cov = meta["m3"]["m3_valid_fraction"][:]
            print(f"\n  M3 coverage: {(m3_cov > 0).sum()}/{n} patches "
                  f"({(m3_cov > 0).sum()/n*100:.1f}%)")
            print(f"  Mean M3 valid fraction: {m3_cov[m3_cov > 0].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build 16-channel V2 HDF5 with M3 data"
    )
    parser.add_argument("--old-hdf5", type=Path, default=HDF5_PATH,
                        help="Path to existing 8-channel HDF5")
    parser.add_argument("--output", type=Path, default=HDF5_V2_PATH,
                        help="Output V2 HDF5 path")
    parser.add_argument("--test", type=int, default=None,
                        help="Process only N patches for testing")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    parser.add_argument("--skip-multiangle", action="store_true",
                        help="Skip multiangle geometry index")
    parser.add_argument("--multiangle-only", action="store_true",
                        help="Only build multiangle index")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing V2 HDF5")
    args = parser.parse_args()

    if args.verify_only:
        if args.output.exists():
            verify_v2(args.output)
        else:
            print(f"File not found: {args.output}")
        return

    if not args.multiangle_only:
        if not args.old_hdf5.exists():
            print(f"ERROR: Source HDF5 not found: {args.old_hdf5}")
            print("Run step04_tile.py first.")
            sys.exit(1)

        # Check M3 GeoTIFFs exist
        available = [n for n, p in M3_ALIGNED_FILES.items() if p.exists()]
        if not available:
            print("ERROR: No M3 GeoTIFFs found. Run step10_m3_mosaic.py first.")
            sys.exit(1)
        print(f"  M3 bands available: {available}")

        n = rebuild_hdf5_v2(
            args.output, args.old_hdf5,
            max_patches=args.test,
            skip_existing=args.skip_existing,
        )

    if not args.skip_multiangle:
        v2_path = args.output
        if v2_path.exists():
            build_multiangle_index(M3_MULTIANGLE_PATH, v2_path)
        else:
            print("  V2 HDF5 not found, skipping multiangle index.")

    # Verify
    if args.output.exists():
        verify_v2(args.output)

    print("\n=== Integration complete ===")


if __name__ == "__main__":
    main()

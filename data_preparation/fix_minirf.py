"""Fix Mini-RF S1 (and check CPR) extreme outliers.

S1 backscatter has a heavy-tailed distribution (p50=44, p99=375K, max=1M).
Apply log1p transform to the raw GeoTIFF, then rebuild the mmap files
with updated normalization stats.

Strategy:
  1. Read raw aligned GeoTIFF
  2. Apply log1p(x) transform to valid pixels
  3. Write back to GeoTIFF (overwrite)
  4. Recompute channel stats for minirf_s1 and minirf_cpr
  5. Rebuild mmap for these two channels
"""

import json
import numpy as np
import rasterio
from pathlib import Path

from config import ALIGNED_DIR, GLOBAL_HEIGHT, GLOBAL_WIDTH

MMAP_DIR = Path(__file__).parent / "data" / "mmap"
STATS_PATH = Path(__file__).parent / "channel_stats.json"


def reset_minirf_from_raw():
    """Re-align Mini-RF aligned tifs from the raw .img so log1p is applied exactly once.

    fix_minirf is NOT idempotent on its own: re-running double-applies log1p to the
    already-transformed aligned GeoTIFF. To make the whole step safe to re-run (e.g.
    when the build pipeline restarts), regenerate the aligned tifs from raw first.
    """
    import step13_align_new_datasets as s13
    print("=== Resetting Mini-RF aligned tifs from raw (ensures single log1p) ===")
    for n in ("minirf_s1", "minirf_cpr"):
        p = ALIGNED_DIR / f"{n}.tif"
        if p.exists():
            p.unlink()
    s13.align_minirf(skip_existing=False)


def diagnose_channel(name, data, valid_mask):
    """Print distribution stats for a channel."""
    valid = data[valid_mask]
    if len(valid) == 0:
        print(f"  {name}: no valid pixels")
        return
    pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    vals = np.percentile(valid, pcts)
    print(f"  {name}: n={len(valid):,}, mean={valid.mean():.4f}, std={valid.std():.4f}")
    print(f"    percentiles: {', '.join(f'p{p}={v:.4f}' for p, v in zip(pcts, vals))}")
    print(f"    min={valid.min():.4f}, max={valid.max():.4f}")


def fix_s1():
    """Apply log1p transform to Mini-RF S1 GeoTIFF."""
    s1_path = ALIGNED_DIR / "minirf_s1.tif"
    print(f"=== Fixing Mini-RF S1: {s1_path} ===\n")

    with rasterio.open(s1_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    valid_mask = np.isfinite(data) & (data != 0)
    print("BEFORE log1p transform:")
    diagnose_channel("minirf_s1", data, valid_mask)

    # Apply log1p: log(1 + x) — handles zeros gracefully, compresses heavy tail
    data[valid_mask] = np.log1p(data[valid_mask]).astype(np.float32)

    # Any negative values from raw data → set to 0 (nodata)
    data[data < 0] = 0

    print("\nAFTER log1p transform:")
    diagnose_channel("minirf_s1", data, valid_mask)

    # Write back
    profile.update(dtype="float32")
    with rasterio.open(s1_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"\n  Written: {s1_path} ({s1_path.stat().st_size / 1e9:.2f} GB)")


def check_cpr():
    """Check if CPR also needs fixing."""
    cpr_path = ALIGNED_DIR / "minirf_cpr.tif"
    print(f"\n=== Checking Mini-RF CPR: {cpr_path} ===\n")

    with rasterio.open(cpr_path) as src:
        data = src.read(1).astype(np.float32)

    valid_mask = np.isfinite(data) & (data != 0)
    diagnose_channel("minirf_cpr", data, valid_mask)

    valid = data[valid_mask]
    frac_gt5 = (valid > 5).sum() / len(valid) * 100
    frac_gt10 = (valid > 10).sum() / len(valid) * 100
    print(f"\n  Fraction > 5: {frac_gt5:.3f}%")
    print(f"  Fraction > 10: {frac_gt10:.3f}%")

    # Apply log1p to CPR unconditionally. Supp 2.4 applies log(1+x) to BOTH radar
    # channels, and a log compression of heavy-tailed circular-polarization-ratio
    # backscatter is the scientifically standard normalization regardless of whether
    # this particular (correctly little-endian-read) mosaic shows the extreme tail.
    print("  Applying log1p to CPR (both radar channels per Supp 2.4)")
    data[valid_mask] = np.log1p(data[valid_mask]).astype(np.float32)
    data[data < 0] = 0
    print("\n  AFTER log1p:")
    diagnose_channel("minirf_cpr", data, valid_mask)

    profile = rasterio.open(cpr_path).profile.copy()
    profile.update(dtype="float32")
    with rasterio.open(cpr_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"  Written: {cpr_path}")
    return True


def recompute_stats(channel_names):
    """Recompute normalization stats for specified channels from GeoTIFFs."""
    print(f"\n=== Recomputing stats for {channel_names} ===\n")

    with open(STATS_PATH) as f:
        all_stats = json.load(f)

    from lunar_dataset import get_all_channel_paths

    paths = get_all_channel_paths(channel_names)
    rng = np.random.RandomState(42)

    for name in channel_names:
        path = paths.get(name)
        if path is None or not path.exists():
            print(f"  {name}: not found")
            continue

        with rasterio.open(path) as src:
            h, w = src.height, src.width

        values = []
        n_samples = 200
        crop_size = 256

        for i in range(n_samples):
            r = rng.randint(0, h - crop_size)
            c = rng.randint(0, w - crop_size)
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(c, r, crop_size, crop_size)
                data = src.read(1, window=window).astype(np.float32)
            valid = data[np.isfinite(data) & (data != 0)]
            if len(valid) > 0:
                values.append(valid)

        if values:
            all_valid = np.concatenate(values)
            mean = float(all_valid.mean())
            std = float(all_valid.std())
            print(f"  {name}: mean={mean:.6f}, std={std:.6f} (n={len(all_valid):,})")
            all_stats[name] = [mean, std]
        else:
            print(f"  {name}: no valid samples!")

    with open(STATS_PATH, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Updated: {STATS_PATH}")


def rebuild_mmap(channel_names):
    """Rebuild mmap files for specified channels."""
    print(f"\n=== Rebuilding mmap for {channel_names} ===\n")

    with open(STATS_PATH) as f:
        channel_stats = {k: tuple(v) for k, v in json.load(f).items()}

    from lunar_dataset import get_all_channel_paths
    paths = get_all_channel_paths(channel_names)
    expected_shape = (GLOBAL_HEIGHT, GLOBAL_WIDTH)

    for name in channel_names:
        path = paths.get(name)
        if path is None or not path.exists():
            print(f"  {name}: source not found")
            continue

        out_path = MMAP_DIR / f"{name}.npy"
        print(f"  Converting {name}...")

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

        nan_mask = ~np.isfinite(data) | (data == 0)
        data[nan_mask] = 0.0

        # Normalize
        if name in channel_stats:
            mean, std = channel_stats[name]
            if std > 0:
                valid_mask = ~nan_mask
                data[valid_mask] = (data[valid_mask] - mean) / std
                data[nan_mask] = 0.0

        mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=expected_shape)
        mm[:] = data
        mm.flush()
        del mm

        # Save NaN mask
        mask_path = MMAP_DIR / f"{name}_mask.npy"
        np.save(mask_path, np.packbits(nan_mask.ravel()))

        valid_pct = (~nan_mask).sum() / data.size * 100
        valid_data = data[~nan_mask]
        if len(valid_data) > 0:
            print(f"    Shape: {data.shape}, valid: {valid_pct:.1f}%, "
                  f"norm range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
        print(f"    Written: {out_path}")


if __name__ == "__main__":
    # 0. Reset aligned tifs from raw so log1p is applied exactly once (idempotent re-run)
    reset_minirf_from_raw()

    # 1. Fix S1
    fix_s1()

    # 2. Check and optionally fix CPR
    cpr_fixed = check_cpr()

    # 3. Recompute stats
    channels_to_fix = ["minirf_s1"]
    if cpr_fixed:
        channels_to_fix.append("minirf_cpr")
    recompute_stats(channels_to_fix)

    # 4. Rebuild mmap
    rebuild_mmap(channels_to_fix)

    print("\n=== Done! ===")

"""End-to-end validation and dataset card generation.

Performs quality checks, computes statistics, and generates visualizations.

Usage:
    python step07_validate.py
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from config import CHANNELS, CHANNELS_V2, HDF5_PATH, M3_VALUE_RANGES, OUTPUT_DIR

# Expected physical value ranges per channel
VALUE_RANGES = {
    "wac_morphology": (0.0, 1.0, "reflectance I/F"),
    "elevation": (-9000.0, 11000.0, "meters from 1737.4 km sphere"),
    "slope": (0.0, 90.0, "degrees"),
    "roughness": (0.0, 5000.0, "meters (local std)"),
    "diviner_tbol_midnight": (50.0, 300.0, "Kelvin (bolometric midnight)"),
    "diviner_temp_night": (20.0, 300.0, "Kelvin (nighttime surface)"),
    "rock_abundance": (0.0, 1.0, "fraction"),
    "christiansen_feature": (7.0, 9.5, "micrometers"),
    **M3_VALUE_RANGES,
}


def validate_hdf5(hdf5_path: Path) -> dict:
    """Run comprehensive validation checks on the HDF5 dataset."""
    print(f"\n=== Validating {hdf5_path.name} ===")
    issues = []

    with h5py.File(hdf5_path, "r") as hf:
        # Check structure
        if "patches" not in hf:
            issues.append("CRITICAL: /patches dataset missing")
            return {"issues": issues}

        patches = hf["patches"]
        n, n_ch, ph, pw = patches.shape
        print(f"  Shape: ({n}, {n_ch}, {ph}, {pw})")

        if n_ch not in (len(CHANNELS), len(CHANNELS_V2)):
            issues.append(f"WARNING: Expected {len(CHANNELS)} or {len(CHANNELS_V2)} channels, got {n_ch}")
        if ph != 256 or pw != 256:
            issues.append(f"WARNING: Expected 256x256 patches, got {ph}x{pw}")

        # Check metadata
        meta = hf.get("metadata")
        if meta is None:
            issues.append("CRITICAL: /metadata group missing")
        else:
            for key in ["patch_id", "row_idx", "col_idx", "center_lat",
                         "center_lon", "valid_fraction"]:
                if key not in meta:
                    issues.append(f"WARNING: /metadata/{key} missing")

            if "center_lat" in meta:
                lats = meta["center_lat"][:]
                if lats.min() < -90 or lats.max() > 90:
                    issues.append(f"ERROR: Latitude out of range [{lats.min():.1f}, {lats.max():.1f}]")

            if "center_lon" in meta:
                lons = meta["center_lon"][:]
                if lons.min() < -180 or lons.max() > 180:
                    issues.append(f"ERROR: Longitude out of range [{lons.min():.1f}, {lons.max():.1f}]")

            if "split" in meta:
                splits = meta["split"][:]
                for s in [0, 1, 2]:
                    count = (splits == s).sum()
                    pct = count / n * 100
                    names = {0: "train", 1: "val", 2: "test"}
                    print(f"  Split {names[s]}: {count} ({pct:.1f}%)")

        # Validate value ranges (sample-based to avoid loading all data)
        print(f"\n  Checking value ranges (sampling {min(100, n)} patches)...")
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n, size=min(100, n), replace=False)

        channel_names = list(hf.attrs.get("channel_names", CHANNELS))
        channel_stats = {ch: {"mins": [], "maxs": [], "means": [],
                              "nan_counts": []} for ch in channel_names}

        for idx in sample_indices:
            patch = patches[idx]
            for ch_i, ch_name in enumerate(channel_names):
                ch_data = patch[ch_i]
                finite = ch_data[np.isfinite(ch_data)]
                if len(finite) > 0:
                    channel_stats[ch_name]["mins"].append(finite.min())
                    channel_stats[ch_name]["maxs"].append(finite.max())
                    channel_stats[ch_name]["means"].append(finite.mean())
                channel_stats[ch_name]["nan_counts"].append(
                    (~np.isfinite(ch_data)).sum()
                )

        print(f"\n  Per-channel statistics (from sampled patches):")
        print(f"  {'Channel':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'NaN%':>8} {'Status'}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

        for ch_name in channel_names:
            stats = channel_stats[ch_name]
            if stats["mins"]:
                ch_min = min(stats["mins"])
                ch_max = max(stats["maxs"])
                ch_mean = np.mean(stats["means"])
                nan_pct = np.mean(stats["nan_counts"]) / (256 * 256) * 100
            else:
                ch_min = ch_max = ch_mean = float("nan")
                nan_pct = 100.0

            # Check against expected ranges
            status = "OK"
            if ch_name in VALUE_RANGES:
                exp_min, exp_max, _ = VALUE_RANGES[ch_name]
                if np.isfinite(ch_min) and ch_min < exp_min * 1.5:
                    if ch_min < exp_min * 0.5:
                        status = "WARN (low)"
                if np.isfinite(ch_max) and ch_max > exp_max * 1.5:
                    status = "WARN (high)"
                if nan_pct > 90:
                    status = "WARN (sparse)"
            else:
                status = "?"

            print(f"  {ch_name:<25} {ch_min:>10.3f} {ch_max:>10.3f} "
                  f"{ch_mean:>10.3f} {nan_pct:>7.1f}% {status}")

        # Check for all-NaN patches
        all_nan_count = 0
        for idx in sample_indices:
            patch = patches[idx]
            if not np.any(np.isfinite(patch)):
                all_nan_count += 1

        if all_nan_count > 0:
            issues.append(f"WARNING: {all_nan_count}/{len(sample_indices)} sampled patches are all NaN")

    # Summary
    print(f"\n  Issues found: {len(issues)}")
    for issue in issues:
        print(f"    {issue}")

    return {"n_patches": n, "n_issues": len(issues), "issues": issues}


def compute_global_stats(hdf5_path: Path, output_path: Path) -> None:
    """Compute per-channel global statistics and save as CSV."""
    print(f"\n=== Computing Global Statistics ===")

    with h5py.File(hdf5_path, "r") as hf:
        n = hf["patches"].shape[0]
        channel_names = list(hf.attrs.get("channel_names", CHANNELS))

        # Accumulate stats in streaming fashion
        stats = {}
        for ch_name in channel_names:
            stats[ch_name] = {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "values_for_percentile": [],
            }

        batch_size = 100
        n_batches = (n + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, n, batch_size), desc="Stats", ncols=80):
            batch_end = min(batch_start + batch_size, n)
            batch = hf["patches"][batch_start:batch_end]  # (B, C, H, W)

            for ch_i, ch_name in enumerate(channel_names):
                ch_data = batch[:, ch_i]  # (B, H, W)
                finite = ch_data[np.isfinite(ch_data)]
                if len(finite) > 0:
                    stats[ch_name]["count"] += len(finite)
                    stats[ch_name]["sum"] += finite.sum()
                    stats[ch_name]["sum_sq"] += (finite.astype(np.float64) ** 2).sum()
                    stats[ch_name]["min"] = min(stats[ch_name]["min"], finite.min())
                    stats[ch_name]["max"] = max(stats[ch_name]["max"], finite.max())

                    # Subsample for percentile estimation
                    if len(finite) > 1000:
                        rng = np.random.default_rng(batch_start + ch_i)
                        subsample = rng.choice(finite, size=1000, replace=False)
                    else:
                        subsample = finite
                    stats[ch_name]["values_for_percentile"].append(subsample)

        # Compute final stats
        lines = ["channel,min,max,mean,std,p01,p99,nodata_fraction"]
        total_pixels = n * 256 * 256

        for ch_name in channel_names:
            s = stats[ch_name]
            if s["count"] > 0:
                mean = s["sum"] / s["count"]
                var = s["sum_sq"] / s["count"] - mean ** 2
                std = np.sqrt(max(0, var))
                nodata_frac = 1.0 - s["count"] / total_pixels

                all_vals = np.concatenate(s["values_for_percentile"])
                p01 = np.percentile(all_vals, 1)
                p99 = np.percentile(all_vals, 99)
            else:
                mean = std = p01 = p99 = float("nan")
                nodata_frac = 1.0

            lines.append(
                f"{ch_name},{s['min']:.6f},{s['max']:.6f},{mean:.6f},"
                f"{std:.6f},{p01:.6f},{p99:.6f},{nodata_frac:.6f}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  Stats saved: {output_path}")

        # Print table
        print(f"\n  {'Channel':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'NoData%':>8}")
        for line in lines[1:]:
            parts = line.split(",")
            print(f"  {parts[0]:<25} {float(parts[1]):>10.3f} {float(parts[2]):>10.3f} "
                  f"{float(parts[3]):>10.3f} {float(parts[4]):>10.3f} {float(parts[6])*100:>7.1f}%")


def generate_dataset_card(hdf5_path: Path, output_path: Path) -> None:
    """Generate a comprehensive dataset card."""
    print(f"\n=== Generating Dataset Card ===")

    with h5py.File(hdf5_path, "r") as hf:
        n = hf["patches"].shape[0]
        shape = hf["patches"].shape
        channel_names = list(hf.attrs.get("channel_names", CHANNELS))

        card = f"""# Lunar Foundation Model Dataset

## Overview
- **Patches**: {n}
- **Shape**: {shape}
- **Resolution**: 128 pixels per degree (~237 m/pixel at equator)
- **Patch size**: 256x256 pixels (~60 km x 60 km at equator)
- **Projection**: Equirectangular (simple cylindrical), Moon_ME frame
- **Sphere radius**: 1,737,400 m

## Channels
| Index | Name | Description |
|-------|------|-------------|
"""
        descriptions = {
            "wac_morphology": "LROC WAC panchromatic reflectance (643 nm, I/F)",
            "elevation": "Blended SLDEM2015+LOLA elevation (meters from 1737.4 km sphere)",
            "slope": "Surface slope (degrees, latitude-corrected)",
            "roughness": "Surface roughness (local std of elevation, 5x5 window)",
            "diviner_temp_day": "Diviner daytime bolometric temperature (Kelvin)",
            "diviner_temp_night": "Diviner nighttime bolometric temperature (Kelvin)",
            "rock_abundance": "Rock abundance (fraction, from thermal inertia)",
            "christiansen_feature": "Christiansen Feature wavelength (micrometers)",
        }
        for i, ch in enumerate(channel_names):
            desc = descriptions.get(ch, "")
            card += f"| {i} | {ch} | {desc} |\n"

        card += """
## Data Sources
- **WAC Morphology**: LROC WAC Global Morphology Mosaic, 100 m/pixel (ASU/USGS)
- **LOLA DEM**: LRO LOLA Global DEM, 118 m/pixel (GSFC)
- **SLDEM2015**: Merged LOLA+Kaguya TC DEM, 128 ppd, ±60° lat (Barker et al. 2015)
- **Diviner**: LRO Diviner GDR Level 3, 128 ppd (UCLA/JPL)
- **Geologic Map**: USGS Unified Geologic Map of the Moon v2 (Fortezzo et al. 2020)

## HDF5 Structure
```
/patches              (N, 8, 256, 256) float32 — patch data
/metadata/
  patch_id            (N,) int32
  row_idx, col_idx    (N,) int32 — patch grid position
  center_lat          (N,) float32 — selenographic latitude
  center_lon          (N,) float32 — selenographic longitude (0-360°E)
  valid_fraction      (N,) float32 — fraction of valid (non-NaN) pixels
  mean_elevation      (N,) float32 — mean elevation in patch
  elevation_range     (N,) float32 — max-min elevation in patch
  has_channel         (N, 8) bool — per-channel availability
  split               (N,) uint8 — 0=train, 1=val, 2=test
  geology/
    mare_fraction     (N,) float32
    n_geo_units       (N,) int16
    dominant_unit_id  (N,) int16
    dominant_unit_name (N,) string
```

## Known Limitations
- SLDEM2015 covers only ±60° latitude; beyond that, LOLA-only elevation is used
- Diviner products may have gaps at high latitudes (>80°)
- WAC morphology is single-band (643 nm); 7-band reflectance may be added later
- Equirectangular projection distorts areas at high latitudes
- Patches at extreme poles may have significant nodata

## Citations
- Barker, M.K., et al. (2016). A new lunar digital elevation model from the LOLA and Kaguya TC. Icarus, 273, 346-355.
- Fortezzo, C.M., et al. (2020). Release of the digital unified global geologic map of the Moon at 1:5,000,000-scale.
- Paige, D.A., et al. (2010). Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region. Science, 330(6003), 479-482.
- Robinson, M.S., et al. (2010). Lunar Reconnaissance Orbiter Camera (LROC) instrument overview. Space Science Reviews, 150(1-4), 81-124.
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(card)
        print(f"  Dataset card saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset and generate reports")
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "validation")
    args = parser.parse_args()

    if not args.hdf5.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5}")
        sys.exit(1)

    # Validate
    results = validate_hdf5(args.hdf5)

    # Global statistics
    compute_global_stats(args.hdf5, args.output_dir / "global_stats.csv")

    # Dataset card
    generate_dataset_card(args.hdf5, args.output_dir / "dataset_card.md")

    print(f"\n=== Validation complete ===")
    print(f"  Issues: {results['n_issues']}")
    print(f"  Reports in: {args.output_dir}")


if __name__ == "__main__":
    main()

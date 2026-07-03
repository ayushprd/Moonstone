"""Compute DEM-derived products: slope and roughness.

Computes from the blended elevation GeoTIFF at 128 ppd.
Processes in overlapping strips for memory efficiency.

Usage:
    # Test first 1024 rows
    python step03_derive.py --test

    # Full computation
    nohup python step03_derive.py > derive.log 2>&1 &
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from config import (
    ALIGNED_FILES,
    DERIVED_DIR,
    DERIVED_FILES,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LUNAR_CRS_PROJ4,
    PIXEL_SIZE_M,
    PPD,
)


def compute_slope(dem_path: Path, output_path: Path,
                  skip_existing: bool = True,
                  test_rows: int = None) -> Path:
    """Compute surface slope in degrees from the DEM.

    Uses numpy.gradient with latitude-corrected east-west pixel size.
    East-west distance per pixel = pixel_size * cos(lat).
    """
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skipped (exists): {output_path.name}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Computing Slope ---")
    print(f"  Source: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        out_height = test_rows if test_rows else src.height
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            height=out_height,
            compress="lzw",
            bigtiff="yes",
            nodata=np.nan,
        )

        strip_h = 1024
        overlap = 2  # Need 1 row overlap on each side for gradient
        n_strips = math.ceil(out_height / strip_h)

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, row_start in enumerate(range(0, out_height, strip_h)):
                h = min(strip_h, out_height - row_start)

                # Read with overlap for gradient computation
                read_start = max(0, row_start - overlap)
                read_end = min(src.height, row_start + h + overlap)
                read_h = read_end - read_start
                window = rasterio.windows.Window(0, read_start, src.width, read_h)
                dem = src.read(1, window=window).astype(np.float64)

                # Compute latitude for each row
                row_indices = np.arange(read_start, read_end)
                lats_deg = 90.0 - (row_indices + 0.5) / PPD
                lats_rad = np.radians(lats_deg)
                cos_lat = np.cos(lats_rad)
                # Clamp to avoid division by zero at poles
                cos_lat = np.maximum(cos_lat, 0.01)

                # East-west pixel size varies with latitude
                dx_sizes = PIXEL_SIZE_M * cos_lat  # (rows,)
                dy_size = PIXEL_SIZE_M

                # Compute gradients
                # North-south gradient (constant pixel size)
                dy = np.gradient(dem, dy_size, axis=0)

                # East-west gradient (variable pixel size)
                dx_raw = np.gradient(dem, axis=1)  # in pixel units
                dx = dx_raw / dx_sizes[:, np.newaxis]  # scale by actual pixel width

                # Handle NaN regions
                nan_mask = ~np.isfinite(dem)
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                slope_deg = np.degrees(slope_rad).astype(np.float32)
                slope_deg[nan_mask] = np.nan

                # Extract the non-overlapping portion
                local_start = row_start - read_start
                local_end = local_start + h
                slope_out = slope_deg[local_start:local_end]

                out_window = rasterio.windows.Window(0, row_start, src.width, h)
                dst.write(slope_out, 1, window=out_window)

                if (i + 1) % 5 == 0 or i == n_strips - 1:
                    pct = (row_start + h) / out_height * 100
                    valid = slope_out[np.isfinite(slope_out)]
                    if len(valid) > 0:
                        print(f"    Strip {i+1}/{n_strips} ({pct:.0f}%) — "
                              f"mean slope: {valid.mean():.1f}°, "
                              f"max: {valid.max():.1f}°")

    print(f"  Done: {output_path.name} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path


def compute_roughness(dem_path: Path, output_path: Path,
                      window_size: int = 5,
                      skip_existing: bool = True,
                      test_rows: int = None) -> Path:
    """Compute surface roughness as local std of elevation in a moving window.

    Uses uniform_filter to compute local mean, then std = sqrt(mean(x²) - mean(x)²).
    """
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skipped (exists): {output_path.name}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Computing Roughness (window={window_size}) ---")
    print(f"  Source: {dem_path.name}")

    half_win = window_size // 2

    with rasterio.open(dem_path) as src:
        out_height = test_rows if test_rows else src.height
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            height=out_height,
            compress="lzw",
            bigtiff="yes",
            nodata=np.nan,
        )

        strip_h = 1024
        overlap = half_win + 1
        n_strips = math.ceil(out_height / strip_h)

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, row_start in enumerate(range(0, out_height, strip_h)):
                h = min(strip_h, out_height - row_start)

                read_start = max(0, row_start - overlap)
                read_end = min(src.height, row_start + h + overlap)
                read_h = read_end - read_start
                window = rasterio.windows.Window(0, read_start, src.width, read_h)
                dem = src.read(1, window=window).astype(np.float64)

                # Replace NaN with 0 for filtering, track mask
                nan_mask = ~np.isfinite(dem)
                dem_clean = np.where(nan_mask, 0.0, dem)
                valid_float = (~nan_mask).astype(np.float64)

                # Roughness = local std = sqrt(E[x²] - E[x]²)
                # uniform_filter computes sum/N over the window. To correct for
                # NaN→0 bias at boundaries, we compute sum(valid)/count(valid)
                # instead of sum(all)/N.
                N = window_size ** 2
                # fraction of valid pixels in each window
                valid_frac = uniform_filter(valid_float, size=window_size, mode="nearest")
                valid_frac = np.maximum(valid_frac, 1.0 / N)  # at least 1 valid pixel
                # uniform_filter gives sum/N; divide by valid_frac to get sum/n_valid
                local_mean = uniform_filter(dem_clean, size=window_size, mode="nearest") / valid_frac
                local_sq_mean = uniform_filter(dem_clean**2, size=window_size, mode="nearest") / valid_frac
                variance = local_sq_mean - local_mean**2
                variance = np.maximum(variance, 0.0)  # numerical safety
                roughness = np.sqrt(variance).astype(np.float32)
                roughness[nan_mask] = np.nan

                # Extract non-overlapping portion
                local_start = row_start - read_start
                local_end = local_start + h
                rough_out = roughness[local_start:local_end]

                out_window = rasterio.windows.Window(0, row_start, src.width, h)
                dst.write(rough_out, 1, window=out_window)

                if (i + 1) % 5 == 0 or i == n_strips - 1:
                    pct = (row_start + h) / out_height * 100
                    valid = rough_out[np.isfinite(rough_out)]
                    if len(valid) > 0:
                        print(f"    Strip {i+1}/{n_strips} ({pct:.0f}%) — "
                              f"mean roughness: {valid.mean():.1f} m, "
                              f"max: {valid.max():.1f} m")

    print(f"  Done: {output_path.name} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path


def verify_derived(path: Path) -> None:
    """Print summary stats for a derived product."""
    with rasterio.open(path) as src:
        print(f"\n  === {path.name} ===")
        print(f"    Shape: {src.width} x {src.height}")

        # Sample from multiple rows
        sample_rows = [0, src.height // 4, src.height // 2, 3 * src.height // 4]
        vals = []
        for r in sample_rows:
            win = rasterio.windows.Window(0, r, src.width, 1)
            row = src.read(1, window=win)
            valid = row[np.isfinite(row)]
            if len(valid) > 0:
                vals.append(valid)

        if vals:
            all_vals = np.concatenate(vals)
            print(f"    Min: {np.min(all_vals):.4f}")
            print(f"    Max: {np.max(all_vals):.4f}")
            print(f"    Mean: {np.mean(all_vals):.4f}")
            print(f"    Std: {np.std(all_vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compute DEM-derived products")
    parser.add_argument(
        "--product", choices=["slope", "roughness", "all"], default="all",
        help="Which product to compute"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--test", action="store_true",
        help="Process only first 1024 rows for testing"
    )
    parser.add_argument("--window", type=int, default=5, help="Roughness window size")
    args = parser.parse_args()

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = ALIGNED_FILES["elevation_blended"]

    if not dem_path.exists():
        # Fall back to LOLA-only if blended not available
        dem_path = ALIGNED_FILES["elevation_lola"]
    if not dem_path.exists():
        print(f"ERROR: No aligned DEM found. Run step02_align.py first.")
        sys.exit(1)

    print(f"Using DEM: {dem_path}")
    test_rows = 1024 if args.test else None

    if args.product in ("slope", "all"):
        result = compute_slope(dem_path, DERIVED_FILES["slope"],
                               args.skip_existing, test_rows)
        if result and result.exists():
            verify_derived(result)

    if args.product in ("roughness", "all"):
        result = compute_roughness(dem_path, DERIVED_FILES["roughness"],
                                   args.window, args.skip_existing, test_rows)
        if result and result.exists():
            verify_derived(result)

    print("\n=== Derivation complete ===")


if __name__ == "__main__":
    import sys
    main()

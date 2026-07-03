"""Cross-register M3 750nm mosaic against WAC morphology.

Detects systematic spatial offset using phase cross-correlation
at multiple sample regions, and optionally applies correction.

Usage:
    # Check alignment (read-only)
    python step11_m3_register.py --check-only

    # Apply correction if needed
    python step11_m3_register.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from config import (
    ALIGNED_FILES,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    M3_ALIGNED_FILES,
    M3_GEOMETRY_FILES,
    OUTPUT_DIR,
    PPD,
)


# Sample regions for cross-correlation: (name, center_lat, center_lon)
# Chosen for strong albedo features visible in both M3 750nm and WAC
SAMPLE_REGIONS = [
    ("Aristarchus", 23.7, -47.4),
    ("Copernicus", 9.6, -20.1),
    ("Tycho", -43.3, -11.2),
    ("Kepler", 8.1, -38.0),
    ("Proclus", 16.1, 46.8),
    ("Euler", 23.3, -29.2),
    ("Langrenus", -8.9, 61.0),
    ("Stevinus", -32.5, 54.1),
    ("Byrgius_A", -24.6, -63.7),
    ("Dionysius", 2.8, 17.3),
    ("Mare_Imbrium_edge", 32.0, -20.0),
    ("Mare_Serenitatis_edge", 25.0, 18.0),
    ("Mare_Crisium_edge", 17.0, 59.0),
    ("SPA_edge", -50.0, 170.0),
    ("Orientale_edge", -19.0, -95.0),
    ("Humboldt", -27.0, 80.6),
    ("Petavius", -25.3, 60.4),
    ("Clavius", -58.4, -14.4),
    ("Grimaldi", -5.2, -68.6),
    ("Plato", 51.6, -9.4),
]


def latlon_to_pixel(lat: float, lon: float) -> tuple:
    """Convert lat/lon to pixel coordinates on 128 ppd grid."""
    col = int((lon + 180.0) * PPD)
    row = int((90.0 - lat) * PPD)
    return row, col


def extract_patch(src, center_lat: float, center_lon: float,
                  size: int = 512) -> np.ndarray:
    """Extract a patch centered on given lat/lon."""
    row, col = latlon_to_pixel(center_lat, center_lon)
    half = size // 2

    # Bounds check
    r0 = max(0, row - half)
    c0 = max(0, col - half)
    r1 = min(GLOBAL_HEIGHT, r0 + size)
    c1 = min(GLOBAL_WIDTH, c0 + size)

    if r1 - r0 < size // 2 or c1 - c0 < size // 2:
        return None

    window = Window(c0, r0, c1 - c0, r1 - r0)
    data = src.read(1, window=window).astype(np.float32)
    return data


def phase_cross_correlation(ref: np.ndarray, tgt: np.ndarray) -> tuple:
    """Compute subpixel offset using phase cross-correlation.

    Returns (dy, dx, peak_value).
    """
    # Normalize
    ref = ref.copy()
    tgt = tgt.copy()

    ref_finite = np.isfinite(ref)
    tgt_finite = np.isfinite(tgt)
    both_valid = ref_finite & tgt_finite

    if both_valid.sum() < ref.size * 0.3:
        return None, None, 0.0

    ref[~both_valid] = 0
    tgt[~both_valid] = 0

    ref_mean = ref[both_valid].mean()
    ref_std = ref[both_valid].std()
    tgt_mean = tgt[both_valid].mean()
    tgt_std = tgt[both_valid].std()

    if ref_std < 1e-10 or tgt_std < 1e-10:
        return None, None, 0.0

    ref = (ref - ref_mean) / ref_std
    tgt = (tgt - tgt_mean) / tgt_std
    ref[~both_valid] = 0
    tgt[~both_valid] = 0

    # FFT-based cross-correlation
    f_ref = np.fft.fft2(ref)
    f_tgt = np.fft.fft2(tgt)

    cross_power = f_ref * np.conj(f_tgt)
    denom = np.abs(cross_power)
    denom[denom < 1e-10] = 1e-10
    cross_power /= denom

    cc = np.fft.ifft2(cross_power).real

    # Find peak
    peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
    peak_val = cc[peak_idx]

    dy = peak_idx[0]
    dx = peak_idx[1]

    # Wrap around
    if dy > cc.shape[0] // 2:
        dy -= cc.shape[0]
    if dx > cc.shape[1] // 2:
        dx -= cc.shape[1]

    return float(dy), float(dx), float(peak_val)


def compute_cross_registration(m3_path: Path, wac_path: Path) -> dict:
    """Compute offset between M3 750nm and WAC at multiple regions."""
    print(f"\n=== Cross-Registration: M3 vs WAC ===")
    print(f"  M3:  {m3_path.name}")
    print(f"  WAC: {wac_path.name}")

    results = []
    with rasterio.open(m3_path) as m3_src, rasterio.open(wac_path) as wac_src:
        for name, lat, lon in SAMPLE_REGIONS:
            m3_patch = extract_patch(m3_src, lat, lon, 512)
            wac_patch = extract_patch(wac_src, lat, lon, 512)

            if m3_patch is None or wac_patch is None:
                continue
            if m3_patch.shape != wac_patch.shape:
                continue

            dy, dx, peak = phase_cross_correlation(wac_patch, m3_patch)

            if dy is not None:
                results.append({
                    "region": name,
                    "lat": lat, "lon": lon,
                    "dy_pixels": dy, "dx_pixels": dx,
                    "correlation_peak": peak,
                })
                print(f"  {name:25s}: dy={dy:+.1f}  dx={dx:+.1f}  "
                      f"peak={peak:.3f}")

    if not results:
        print("  No valid regions found!")
        return {"dx_pixels": 0, "dy_pixels": 0, "per_region": []}

    # Robust median (reject outliers)
    dys = [r["dy_pixels"] for r in results]
    dxs = [r["dx_pixels"] for r in results]

    # Simple outlier rejection: within 2*MAD of median
    def robust_mean(vals):
        vals = np.array(vals)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 0.5:
            return float(med)
        mask = np.abs(vals - med) < 3 * mad
        return float(np.mean(vals[mask])) if mask.sum() > 0 else float(med)

    dy_robust = robust_mean(dys)
    dx_robust = robust_mean(dxs)

    pixel_size_m = 237.0  # ~237 m/pixel at equator
    dy_m = dy_robust * pixel_size_m
    dx_m = dx_robust * pixel_size_m

    print(f"\n  Robust offset: dy={dy_robust:+.2f} px, dx={dx_robust:+.2f} px")
    print(f"  In meters: dy={dy_m:+.0f} m, dx={dx_m:+.0f} m")
    print(f"  Total offset: {np.sqrt(dy_m**2 + dx_m**2):.0f} m")

    return {
        "dy_pixels": dy_robust,
        "dx_pixels": dx_robust,
        "dy_meters": dy_m,
        "dx_meters": dx_m,
        "total_offset_m": float(np.sqrt(dy_m**2 + dx_m**2)),
        "n_regions_used": len(results),
        "per_region": results,
    }


def apply_shift(offset: dict, max_offset_px: int = 10) -> bool:
    """Apply integer pixel shift to all M3 GeoTIFFs if needed."""
    dy = offset["dy_pixels"]
    dx = offset["dx_pixels"]

    if abs(dy) < 0.5 and abs(dx) < 0.5:
        print("\n  Offset < 0.5 pixels — no correction needed.")
        return False

    dy_int = int(round(dy))
    dx_int = int(round(dx))

    if abs(dy_int) > max_offset_px or abs(dx_int) > max_offset_px:
        print(f"\n  WARNING: Offset ({dy_int}, {dx_int}) exceeds max "
              f"({max_offset_px} px). Refusing to correct.")
        return False

    print(f"\n  Applying shift: dy={dy_int}, dx={dx_int} pixels")

    all_paths = list(M3_ALIGNED_FILES.values()) + list(M3_GEOMETRY_FILES.values())
    for path in all_paths:
        if not path.exists():
            continue

        print(f"    Shifting {path.name}...")
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            data = src.read(1)

        # Apply shift
        shifted = np.full_like(data, np.nan)
        h, w = data.shape

        # Source and destination slices
        src_r = slice(max(0, dy_int), min(h, h + dy_int))
        dst_r = slice(max(0, -dy_int), min(h, h - dy_int))
        src_c = slice(max(0, dx_int), min(w, w + dx_int))
        dst_c = slice(max(0, -dx_int), min(w, w - dx_int))

        shifted[dst_r, dst_c] = data[src_r, src_c]

        # Write back (atomic via temp file)
        tmp_path = path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(shifted, 1)
        tmp_path.rename(path)

    print("  Shift applied to all M3 GeoTIFFs.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Cross-register M3 mosaic against WAC"
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Only check alignment, don't apply correction")
    parser.add_argument("--max-offset", type=int, default=10,
                        help="Maximum allowed offset in pixels")
    args = parser.parse_args()

    m3_path = M3_ALIGNED_FILES.get("m3_750")
    wac_path = ALIGNED_FILES.get("wac_morphology")

    if not m3_path or not m3_path.exists():
        print(f"ERROR: M3 750nm mosaic not found: {m3_path}")
        print("Run step10_m3_mosaic.py first.")
        sys.exit(1)

    if not wac_path or not wac_path.exists():
        print(f"ERROR: WAC morphology not found: {wac_path}")
        sys.exit(1)

    offset = compute_cross_registration(m3_path, wac_path)

    # Save report
    report_path = OUTPUT_DIR / "validation" / "m3_registration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(offset, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    if args.check_only:
        print("\n=== Check-only mode, no correction applied ===")
        return

    # Apply correction if needed
    corrected = apply_shift(offset, max_offset_px=args.max_offset)

    if corrected:
        # Re-check after correction
        print("\n--- Verification after correction ---")
        offset2 = compute_cross_registration(m3_path, wac_path)
        print(f"  Residual offset: dy={offset2['dy_pixels']:+.2f}, "
              f"dx={offset2['dx_pixels']:+.2f}")

    print("\n=== Cross-registration complete ===")


if __name__ == "__main__":
    main()

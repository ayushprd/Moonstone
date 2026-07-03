"""Convert and align all datasets to the common 128 ppd lunar grid.

Handles:
- PDS .LBL+.IMG → GeoTIFF conversion (SLDEM2015, Diviner)
- Reprojection/resampling to 128 ppd equirectangular
- SLDEM+LOLA DEM blending for full global coverage

Usage:
    # Test on small region first
    python step02_align.py --test

    # Align a single product
    python step02_align.py --product wac

    # Align everything
    nohup python step02_align.py > align.log 2>&1 &
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from config import (
    ALIGNED_DIR,
    ALIGNED_FILES,
    DERIVED_DIR,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LUNAR_CRS_PROJ4,
    LUNAR_RADIUS_M,
    PIXEL_SIZE_M,
    PPD,
    RAW_DIR,
    RAW_FILES,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

LUNAR_CRS = CRS.from_proj4(LUNAR_CRS_PROJ4)

# Target affine: pixel (0,0) top-left = (x_min, y_max) = (lon=0, lat=+90)
TARGET_TRANSFORM = Affine(
    (X_MAX - X_MIN) / GLOBAL_WIDTH,    # pixel width in meters
    0.0,
    X_MIN,                              # x origin
    0.0,
    -(Y_MAX - Y_MIN) / GLOBAL_HEIGHT,  # pixel height (negative = north-up)
    Y_MAX,                              # y origin
)


def convert_pds_to_geotiff(lbl_path: Path, output_tif: Path,
                           skip_existing: bool = True) -> Path:
    """Convert PDS3 .LBL+.IMG to GeoTIFF using gdal_translate.

    The .LBL file is the entry point; GDAL's PDS driver reads it and
    locates the companion .IMG binary data.
    """
    if skip_existing and output_tif.exists() and output_tif.stat().st_size > 0:
        print(f"  Skipped (exists): {output_tif.name}")
        return output_tif

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # The GDAL PDS driver resolves the label's ^IMAGE pointer relative to the
    # label directory. Our downloaded .img is renamed (e.g. diviner_temp_night.img)
    # but the label points at the original PDS name (e.g. DGDR_ST_AVG_CYL_128_IMG.IMG).
    # Create a symlink with the pointed-to name so GDAL can locate the data.
    info = parse_pds_label(lbl_path)
    pointer = (info.get("^IMAGE") or info.get("FILE_NAME") or "").strip().strip('"')
    # ^IMAGE may be "(\"NAME.IMG\", N)" — take the quoted filename.
    if pointer and pointer.startswith("("):
        import re as _re
        m = _re.search(r'"([^"]+)"', pointer)
        pointer = m.group(1) if m else ""
    if pointer:
        target = lbl_path.with_name(pointer)
        actual = lbl_path.with_suffix(".img")
        if not target.exists() and actual.exists():
            try:
                target.symlink_to(actual.name)
                print(f"  Linked {target.name} -> {actual.name}")
            except OSError:
                import shutil as _sh
                _sh.copyfile(actual, target)

    cmd = [
        "gdal_translate",
        "-if", "PDS",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        str(lbl_path),
        str(output_tif),
    ]
    print(f"  Converting: {lbl_path.name} → {output_tif.name}")
    print(f"    Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        raise RuntimeError(f"gdal_translate failed: {result.stderr}")

    print(f"  Converted: {output_tif.name} ({output_tif.stat().st_size / 1e9:.2f} GB)")
    return output_tif


def parse_pds_label(lbl_path: Path) -> dict:
    """Parse key-value pairs from a PDS3 .LBL file."""
    info = {}
    with open(lbl_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"')
                info[key] = val
    return info


def apply_scaling(input_tif: Path, output_tif: Path,
                  scale_factor: float, offset: float,
                  skip_existing: bool = True) -> Path:
    """Apply DN * scale_factor + offset to convert scaled integers to physical values."""
    if skip_existing and output_tif.exists() and output_tif.stat().st_size > 0:
        print(f"  Skipped (exists): {output_tif.name}")
        return output_tif

    print(f"  Applying scaling: DN * {scale_factor} + {offset}")

    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", compress="lzw", bigtiff="yes")

        with rasterio.open(output_tif, "w", **profile) as dst:
            # Process in strips
            strip_h = 1024
            for row_start in range(0, src.height, strip_h):
                h = min(strip_h, src.height - row_start)
                window = rasterio.windows.Window(0, row_start, src.width, h)
                data = src.read(1, window=window).astype(np.float32)

                # Apply scaling
                nodata_mask = (data == src.nodata) if src.nodata is not None else np.zeros_like(data, dtype=bool)
                data = data * scale_factor + offset
                data[nodata_mask] = np.nan

                dst.write(data, 1, window=window)

    print(f"  Scaled: {output_tif.name}")
    return output_tif


def reproject_to_grid(src_path: Path, dst_path: Path,
                      resampling_method: str = "bilinear",
                      skip_existing: bool = True,
                      test_rows: int = None) -> Path:
    """Reproject a raster to the 128 ppd lunar equirectangular grid.

    Processes in strips to manage memory (~200 MB per strip per channel).
    """
    if skip_existing and dst_path.exists() and dst_path.stat().st_size > 0:
        print(f"  Skipped (exists): {dst_path.name}")
        return dst_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    resamp = getattr(Resampling, resampling_method)
    out_height = test_rows if test_rows else GLOBAL_HEIGHT
    out_width = GLOBAL_WIDTH

    print(f"  Reprojecting: {src_path.name} → {dst_path.name}")
    print(f"    Output: {out_width} x {out_height}, resampling={resampling_method}")

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        print(f"    Source: {src.width} x {src.height}, CRS={src.crs}, dtype={src.dtypes[0]}")
        print(f"    Source bounds: {src.bounds}")
        print(f"    Source nodata: {src_nodata}")

        dst_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": out_width,
            "height": out_height,
            "count": 1,
            "crs": LUNAR_CRS,
            "transform": TARGET_TRANSFORM,
            "compress": "lzw",
            "bigtiff": "yes",
            "nodata": np.nan,
        }

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            strip_h = 1024
            n_strips = math.ceil(out_height / strip_h)

            for i, row_start in enumerate(range(0, out_height, strip_h)):
                h = min(strip_h, out_height - row_start)
                dst_window = rasterio.windows.Window(0, row_start, out_width, h)
                dst_transform = rasterio.windows.transform(dst_window, TARGET_TRANSFORM)

                dst_data = np.full((1, h, out_width), np.nan, dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=LUNAR_CRS,
                    resampling=resamp,
                    src_nodata=src_nodata,
                    dst_nodata=np.nan,
                )

                dst.write(dst_data[0], 1, window=dst_window)

                if (i + 1) % 5 == 0 or i == n_strips - 1:
                    pct = (row_start + h) / out_height * 100
                    valid = np.isfinite(dst_data).sum()
                    total = dst_data.size
                    print(f"    Strip {i+1}/{n_strips} ({pct:.0f}%) — "
                          f"valid: {valid}/{total} ({valid/total*100:.1f}%)")

    print(f"  Done: {dst_path.name} ({dst_path.stat().st_size / 1e9:.2f} GB)")
    return dst_path


def verify_aligned(path: Path) -> dict:
    """Print summary stats for an aligned GeoTIFF."""
    with rasterio.open(path) as src:
        print(f"\n  === {path.name} ===")
        print(f"    Shape: {src.width} x {src.height}")
        print(f"    CRS: {src.crs}")
        print(f"    Transform: {src.transform}")

        # Sample some rows for stats
        sample_rows = [0, src.height // 4, src.height // 2, 3 * src.height // 4, src.height - 1]
        vals = []
        for r in sample_rows:
            if r < src.height:
                win = rasterio.windows.Window(0, r, src.width, 1)
                row_data = src.read(1, window=win)
                valid = row_data[np.isfinite(row_data)]
                if len(valid) > 0:
                    vals.append(valid)

        if vals:
            all_vals = np.concatenate(vals)
            print(f"    Sample min: {np.min(all_vals):.4f}")
            print(f"    Sample max: {np.max(all_vals):.4f}")
            print(f"    Sample mean: {np.mean(all_vals):.4f}")

    return {"path": str(path), "width": src.width, "height": src.height}


def align_wac(skip_existing: bool = True, test_rows: int = None) -> Path:
    """Align WAC morphology mosaic to 128 ppd grid."""
    print("\n--- Aligning WAC Morphology ---")
    src = RAW_FILES["wac_morphology"]
    dst = ALIGNED_FILES["wac_morphology"]
    if not src.exists():
        print(f"  Source not found: {src}")
        return None
    return reproject_to_grid(src, dst, "bilinear", skip_existing, test_rows)


def align_lola(skip_existing: bool = True, test_rows: int = None) -> Path:
    """Align LOLA DEM to 128 ppd grid.

    LOLA DEM from USGS stores values in half-meters (int16).
    Must multiply by 0.5 to get meters relative to 1737.4 km sphere.
    """
    print("\n--- Aligning LOLA DEM ---")
    src = RAW_FILES["lola_dem"]
    dst = ALIGNED_FILES["elevation_lola"]
    if not src.exists():
        print(f"  Source not found: {src}")
        return None

    # LOLA DEM values are in half-meters (scale factor 0.5)
    # Always create a scaled version
    scaled_src = RAW_DIR / "lola_dem_scaled.tif"
    if not scaled_src.exists() or scaled_src.stat().st_size == 0:
        print("  Applying 0.5 scale factor to LOLA (half-meters → meters)...")
        apply_scaling(src, scaled_src, 0.5, 0, skip_existing=False)
    else:
        print(f"  Using existing scaled LOLA: {scaled_src.name}")

    return reproject_to_grid(scaled_src, dst, "bilinear", skip_existing, test_rows)


def align_sldem(skip_existing: bool = True, test_rows: int = None) -> Path:
    """Align SLDEM2015 to 128 ppd grid.

    SLDEM is already 128 ppd equirectangular but only covers ±60° latitude.
    Convert from PDS format, scale from km to meters, and assign proper CRS.

    PDS label says UNIT=KILOMETER, so values need *1000 to get meters.
    """
    print("\n--- Aligning SLDEM2015 ---")
    lbl = RAW_FILES["sldem2015_lbl"]
    dst = ALIGNED_FILES["elevation_sldem"]

    if not lbl.exists():
        print(f"  Source not found: {lbl}")
        return None

    # Convert PDS → GeoTIFF first
    converted = RAW_DIR / "sldem2015_128ppd_converted.tif"
    convert_pds_to_geotiff(lbl, converted, skip_existing)

    # Scale from kilometers to meters
    scaled = RAW_DIR / "sldem2015_128ppd_meters.tif"
    if not scaled.exists() or scaled.stat().st_size == 0:
        print("  Scaling SLDEM from km to meters (*1000)...")
        apply_scaling(converted, scaled, 1000.0, 0.0, skip_existing=False)
    else:
        print(f"  Using existing scaled SLDEM: {scaled.name}")

    # SLDEM2015 at 128ppd covers 60S-60N = 120 degrees = 15360 rows
    # It's already at the right resolution, just needs CRS assignment
    # and may need longitude shift
    return reproject_to_grid(scaled, dst, "bilinear", skip_existing, test_rows)


def blend_dem(skip_existing: bool = True) -> Path:
    """Blend SLDEM2015 (±60° lat) with LOLA (global) for full coverage.

    Uses SLDEM within ±58°, LOLA beyond ±62°, linear blend in 58-62° transition.
    """
    print("\n--- Blending DEM (SLDEM+LOLA) ---")
    dst = ALIGNED_FILES["elevation_blended"]

    if skip_existing and dst.exists() and dst.stat().st_size > 0:
        print(f"  Skipped (exists): {dst.name}")
        return dst

    lola_path = ALIGNED_FILES["elevation_lola"]
    sldem_path = ALIGNED_FILES["elevation_sldem"]

    if not lola_path.exists():
        print(f"  LOLA aligned not found: {lola_path}")
        return None
    if not sldem_path.exists():
        print(f"  SLDEM aligned not found: {sldem_path}")
        print("  Using LOLA only as fallback")
        # Just copy LOLA
        import shutil
        shutil.copy2(lola_path, dst)
        return dst

    BLEND_START_DEG = 58.0  # Start blending at ±58°
    BLEND_END_DEG = 62.0    # Fully LOLA beyond ±62°

    with rasterio.open(lola_path) as lola_ds, rasterio.open(sldem_path) as sldem_ds:
        profile = lola_ds.profile.copy()
        profile.update(dtype="float32", compress="lzw", bigtiff="yes", nodata=np.nan)

        with rasterio.open(dst, "w", **profile) as dst_ds:
            strip_h = 1024
            n_strips = math.ceil(GLOBAL_HEIGHT / strip_h)

            for i, row_start in enumerate(range(0, GLOBAL_HEIGHT, strip_h)):
                h = min(strip_h, GLOBAL_HEIGHT - row_start)
                window = rasterio.windows.Window(0, row_start, GLOBAL_WIDTH, h)

                lola_data = lola_ds.read(1, window=window).astype(np.float32)

                # Compute latitude for each row in this strip
                row_indices = np.arange(row_start, row_start + h)
                lats = 90.0 - (row_indices + 0.5) / PPD  # center of pixel
                abs_lats = np.abs(lats)

                # Read SLDEM (may be NaN outside ±60°)
                sldem_data = sldem_ds.read(1, window=window).astype(np.float32)

                # Compute blend weights per row
                for j, abs_lat in enumerate(abs_lats):
                    if abs_lat < BLEND_START_DEG:
                        # Pure SLDEM
                        row = sldem_data[j]
                        valid = np.isfinite(row)
                        if valid.any():
                            lola_data[j][valid] = row[valid]
                        # Where SLDEM is NaN, keep LOLA
                    elif abs_lat < BLEND_END_DEG:
                        # Blend zone
                        w_lola = (abs_lat - BLEND_START_DEG) / (BLEND_END_DEG - BLEND_START_DEG)
                        w_sldem = 1.0 - w_lola
                        row = sldem_data[j]
                        valid = np.isfinite(row) & np.isfinite(lola_data[j])
                        if valid.any():
                            lola_data[j][valid] = (
                                w_sldem * row[valid] + w_lola * lola_data[j][valid]
                            )
                    # else: abs_lat >= BLEND_END_DEG → pure LOLA (already in lola_data)

                dst_ds.write(lola_data, 1, window=window)

                if (i + 1) % 5 == 0 or i == n_strips - 1:
                    pct = (row_start + h) / GLOBAL_HEIGHT * 100
                    print(f"    Blend strip {i+1}/{n_strips} ({pct:.0f}%)")

    print(f"  Done: {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")
    return dst


def align_diviner(product_name: str, skip_existing: bool = True,
                  test_rows: int = None) -> Path:
    """Align a Diviner product to the 128 ppd grid.

    Diviner GDR L3 products are already 128 ppd equirectangular,
    but stored as scaled integers in PDS format.
    The GHRM bolometric temp uses PDS4 XML labels.
    """
    print(f"\n--- Aligning Diviner: {product_name} ---")

    # Map product name to aligned file paths
    product_map = {
        "diviner_tbol_midnight": ALIGNED_FILES["diviner_tbol_midnight"],
        "diviner_temp_night": ALIGNED_FILES["diviner_temp_night"],
        "rock_abundance": ALIGNED_FILES["rock_abundance"],
        "christiansen_feature": ALIGNED_FILES["christiansen_feature"],
    }

    if product_name not in product_map:
        print(f"  Unknown product: {product_name}")
        return None

    dst_path = product_map[product_name]

    # Find the raw files — try .lbl then .xml
    # Build filename: add diviner_ prefix only if not already present
    prefix = "" if product_name.startswith("diviner_") else "diviner_"
    base_name = f"{prefix}{product_name}"
    lbl_path = RAW_DIR / f"{base_name}.lbl"
    xml_path = RAW_DIR / f"{base_name}.xml"
    img_path = RAW_DIR / f"{base_name}.img"

    label_path = None
    if lbl_path.exists():
        label_path = lbl_path
    elif xml_path.exists():
        label_path = xml_path
    else:
        # Search for any matching label
        candidates = (
            list(RAW_DIR.glob(f"*{product_name}*.lbl")) +
            list(RAW_DIR.glob(f"*{product_name}*.LBL")) +
            list(RAW_DIR.glob(f"*{product_name}*.xml"))
        )
        if candidates:
            label_path = candidates[0]

    if label_path is None:
        print(f"  Label file not found for {product_name}")
        return None

    if not img_path.exists():
        print(f"  IMG not found for {product_name}")
        return None

    # For PDS3 (.lbl), parse scaling; for PDS4 (.xml), try gdal_translate directly
    converted = RAW_DIR / f"{base_name}_converted.tif"

    if label_path.suffix.lower() == ".lbl":
        label_info = parse_pds_label(label_path)
        scale_factor = float(label_info.get("SCALING_FACTOR", "1.0"))
        offset = float(label_info.get("OFFSET", "0.0"))
        print(f"  PDS3 label: SCALING_FACTOR={scale_factor}, OFFSET={offset}")
        convert_pds_to_geotiff(label_path, converted, skip_existing)
    else:
        # PDS4 XML — GDAL can read directly, try without -if PDS flag
        print(f"  PDS4 XML label: {label_path.name}")
        scale_factor = 1.0
        offset = 0.0
        # The XML <file_name> points at the original PDS data name; symlink our
        # renamed .img to it so GDAL's PDS4 driver can locate the data.
        try:
            import re as _re
            xtxt = label_path.read_text(errors="ignore")
            m = _re.search(r"<file_name>\s*([^<\s]+)\s*</file_name>", xtxt)
            if m:
                target = label_path.with_name(m.group(1))
                if not target.exists() and img_path.exists():
                    try:
                        target.symlink_to(img_path.name)
                        print(f"  Linked {target.name} -> {img_path.name}")
                    except OSError:
                        import shutil as _sh
                        _sh.copyfile(img_path, target)
        except Exception as _e:  # noqa: BLE001
            print(f"  (xml file_name symlink skipped: {_e})")
        if not (skip_existing and converted.exists() and converted.stat().st_size > 0):
            cmd = [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                "-co", "BIGTIFF=YES",
                str(label_path),
                str(converted),
            ]
            print(f"    Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Fallback: try with the .img file directly
                print(f"    XML conversion failed, trying IMG directly...")
                cmd[4] = str(img_path)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    FAILED: {result.stderr}")
                    return None

    # If scaling needed, apply it
    if scale_factor != 1.0 or offset != 0.0:
        scaled = RAW_DIR / f"{base_name}_scaled.tif"
        apply_scaling(converted, scaled, scale_factor, offset, skip_existing)
        src = scaled
    else:
        src = converted

    # Reproject to target grid
    return reproject_to_grid(src, dst_path, "bilinear", skip_existing, test_rows)


def main():
    parser = argparse.ArgumentParser(description="Align datasets to 128 ppd grid")
    parser.add_argument(
        "--product",
        choices=["wac", "lola", "sldem", "blend", "diviner_tbol_midnight",
                 "diviner_temp_night", "rock_abundance", "christiansen_feature",
                 "all"],
        default="all",
        help="Product to align"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--test", action="store_true",
        help="Process only first 1024 rows for testing"
    )
    args = parser.parse_args()

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = 1024 if args.test else None

    products_to_run = []
    if args.product in ("wac", "all"):
        products_to_run.append(("wac", lambda: align_wac(args.skip_existing, test_rows)))
    if args.product in ("lola", "all"):
        products_to_run.append(("lola", lambda: align_lola(args.skip_existing, test_rows)))
    if args.product in ("sldem", "all"):
        products_to_run.append(("sldem", lambda: align_sldem(args.skip_existing, test_rows)))
    if args.product in ("blend", "all"):
        products_to_run.append(("blend", lambda: blend_dem(args.skip_existing)))

    diviner_products = ["diviner_tbol_midnight", "diviner_temp_night",
                        "rock_abundance", "christiansen_feature"]
    for dp in diviner_products:
        if args.product in (dp, "all"):
            products_to_run.append(
                (dp, lambda p=dp: align_diviner(p, args.skip_existing, test_rows))
            )

    for name, func in products_to_run:
        try:
            result = func()
            if result and result.exists():
                verify_aligned(result)
        except Exception as e:
            print(f"\n  ERROR aligning {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Alignment complete ===")


if __name__ == "__main__":
    main()

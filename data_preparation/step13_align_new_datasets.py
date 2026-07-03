"""Align all new datasets to the common 128 ppd lunar grid.

Handles:
- GRAIL gravity GeoTIFFs (16 ppd → 128 ppd, bilinear upsampling)
- Mini-RF radar PDS IMG (128 ppd, reformat + align)
- WAC Hapke 7-band tiles (400m, mosaic + resample)
- Clementine UVVIS 750nm GeoTIFF (118m → 128ppd)
- LP GRS element abundance (2-deg ASCII → 128ppd, bilinear)
- Diviner bolometric temp (already 128ppd, just convert)

Usage:
    python step13_align_new_datasets.py --product grail
    python step13_align_new_datasets.py --product minirf
    python step13_align_new_datasets.py --product wac_hapke
    python step13_align_new_datasets.py --product clementine
    python step13_align_new_datasets.py --product lpgrs
    python step13_align_new_datasets.py --all
"""

import argparse
import math
import re
import struct
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
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LUNAR_CRS_PROJ4,
    LUNAR_RADIUS_M,
    PPD,
    RAW_DIR,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

LUNAR_CRS = CRS.from_proj4(LUNAR_CRS_PROJ4)

TARGET_TRANSFORM = Affine(
    (X_MAX - X_MIN) / GLOBAL_WIDTH,
    0.0,
    X_MIN,
    0.0,
    -(Y_MAX - Y_MIN) / GLOBAL_HEIGHT,
    Y_MAX,
)

# Directories
GRAIL_DIR = RAW_DIR / "grail"
MINIRF_DIR = RAW_DIR / "minirf"
WAC_HAPKE_DIR = RAW_DIR / "wac_hapke"
CLEMENTINE_DIR = RAW_DIR / "clementine"
LPGRS_DIR = RAW_DIR / "lp_hydrogen"


def write_aligned_tif(output_path, data, nodata=np.nan):
    """Write a float32 GeoTIFF on the 128ppd global grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": GLOBAL_WIDTH,
        "height": GLOBAL_HEIGHT,
        "count": 1,
        "crs": LUNAR_CRS,
        "transform": TARGET_TRANSFORM,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "yes",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"  Written: {output_path.name} ({output_path.stat().st_size / 1e9:.2f} GB)")


def reproject_to_grid(src_path, dst_path, resampling=Resampling.bilinear,
                      skip_existing=True, src_nodata=None):
    """Reproject any georeferenced raster to our 128ppd grid."""
    if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1000:
        print(f"  Skipped (exists): {dst_path.name}")
        return

    dst_data = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        nd = src_nodata if src_nodata is not None else src.nodata
        if nd is not None:
            src_data[src_data == nd] = np.nan

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=LUNAR_CRS,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    write_aligned_tif(dst_path, dst_data)


# =====================================================================
# GRAIL gravity (16 ppd GeoTIFFs → 128 ppd)
# =====================================================================

def align_grail(skip_existing=True):
    """Align GRAIL gravity GeoTIFFs to 128ppd grid."""
    print("\n=== Aligning GRAIL gravity products ===")

    products = {
        "grail_freeair": GRAIL_DIR / "grail_freeair_l660.tif",
        "grail_bouguer": GRAIL_DIR / "grail_bouguer_l660.tif",
        "grail_uncertainty": GRAIL_DIR / "grail_uncertainty_l660.tif",
    }

    for name, src_path in products.items():
        if not src_path.exists():
            print(f"  SKIP {name}: source not found at {src_path}")
            continue
        dst_path = ALIGNED_DIR / f"{name}.tif"
        print(f"  Aligning {name}...")

        # GRAIL GeoTIFFs are in planetocentric lat/lon with radius=1737400
        # They use 0-360 longitude convention. Need to handle this.
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None:
                data[data == nd] = np.nan
            h, w = data.shape
            print(f"    Source: {w}x{h}, range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")

        # GRAIL files are 5760x2880 (16ppd), simple cylindrical 0-360
        # Shift to -180..180 by rolling half the width
        half_w = w // 2
        data = np.roll(data, half_w, axis=1)

        # Create source geo: -180..180, -90..90 in lunar equirectangular
        src_transform = Affine(
            (X_MAX - X_MIN) / w, 0.0, X_MIN,
            0.0, -(Y_MAX - Y_MIN) / h, Y_MAX,
        )

        # Bilinear upsample to 128ppd
        dst_data = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan, dtype=np.float32)
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=LUNAR_CRS,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=LUNAR_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        valid = np.isfinite(dst_data)
        print(f"    Output: range=[{np.nanmin(dst_data):.2f}, {np.nanmax(dst_data):.2f}], "
              f"coverage={valid.sum()/dst_data.size*100:.1f}%")
        write_aligned_tif(dst_path, dst_data)

    print("  GRAIL alignment complete.")


# =====================================================================
# Mini-RF radar (128ppd PDS IMG → GeoTIFF)
# =====================================================================

def read_pds_img_float32(img_path, lbl_path=None):
    """Read a PDS IMG file as float32 array.

    Mini-RF 128ppd global mosaics are 46080x23040 float32, simple cylindrical
    0-360 lon, -90..90 lat. The PDS label declares SAMPLE_TYPE; the LRO Mini-RF
    global mosaics use PC_REAL (IEEE 754, LITTLE-endian). Endianness and
    dimensions are read from the .lbl when present rather than hardcoded.
    """
    # Defaults at 128ppd
    width = 46080
    height = 23040
    dtype = "<f4"  # PC_REAL = little-endian IEEE float32

    # Locate/parse the PDS label for authoritative dims + sample type
    if lbl_path is None:
        cand = img_path.with_suffix(".lbl")
        lbl_path = cand if cand.exists() else None
    if lbl_path is not None and lbl_path.exists():
        txt = lbl_path.read_text(errors="ignore")
        m_lines = re.search(r"\bLINES\s*=\s*(\d+)", txt)
        m_samps = re.search(r"\bLINE_SAMPLES\s*=\s*(\d+)", txt)
        m_type = re.search(r"\bSAMPLE_TYPE\s*=\s*(\S+)", txt)
        if m_lines and m_samps:
            height, width = int(m_lines.group(1)), int(m_samps.group(1))
        if m_type:
            st = m_type.group(1).upper()
            # MSB/IEEE_REAL/SUN_REAL = big-endian; PC_REAL/LSB = little-endian
            if "MSB" in st or st in ("IEEE_REAL", "SUN_REAL"):
                dtype = ">f4"
            elif "LSB" in st or st == "PC_REAL":
                dtype = "<f4"
        print(f"    Label: {width}x{height}, SAMPLE_TYPE={m_type.group(1) if m_type else '?'} "
              f"-> dtype={dtype}")

    expected_size = width * height * 4  # float32
    file_size = img_path.stat().st_size
    if file_size != expected_size:
        print(f"  WARNING: {img_path.name} size={file_size} != expected {expected_size}")
        for w, h in [(46080, 23040), (11520, 5760)]:
            if file_size == w * h * 4:
                width, height = w, h
                print(f"    Detected {w}x{h}")
                break
        else:
            raise ValueError(f"Cannot determine dimensions for {img_path.name}")

    data = np.fromfile(img_path, dtype=dtype)
    data = data.reshape(height, width).astype(np.float32)
    return data, width, height


def align_minirf(skip_existing=True):
    """Align Mini-RF radar mosaics to 128ppd grid."""
    print("\n=== Aligning Mini-RF radar products ===")

    products = {
        "minirf_cpr": MINIRF_DIR / "minirf_cpr_128ppd.img",
        "minirf_s1": MINIRF_DIR / "minirf_s1_128ppd.img",
    }

    for name, src_path in products.items():
        if not src_path.exists():
            print(f"  SKIP {name}: source not found at {src_path}")
            continue
        dst_path = ALIGNED_DIR / f"{name}.tif"
        if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1000:
            print(f"  Skipped (exists): {dst_path.name}")
            continue

        print(f"  Reading {name}...")
        data, w, h = read_pds_img_float32(src_path)

        # Replace PDS nodata values
        # Mini-RF uses -3.4e+38 as nodata and large positive sentinels (>1e10)
        data[data <= 0] = np.nan
        data[data > 1e6] = np.nan  # filter nodata sentinels and outliers
        # CPR physically ranges ~0-5 (circular polarization ratio)
        # S1 backscatter in power units, reasonable range < 1e4
        if "cpr" in name:
            data[data > 50] = np.nan  # CPR > 50 is unphysical

        # Shift from 0-360 to -180..180
        half_w = w // 2
        data = np.roll(data, half_w, axis=1)

        valid = np.isfinite(data)
        print(f"    Range: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}], "
              f"coverage={valid.sum()/data.size*100:.1f}%")

        if w == GLOBAL_WIDTH and h == GLOBAL_HEIGHT:
            # Already at 128ppd, just write
            write_aligned_tif(dst_path, data)
        else:
            # Resample
            src_transform = Affine(
                (X_MAX - X_MIN) / w, 0.0, X_MIN,
                0.0, -(Y_MAX - Y_MIN) / h, Y_MAX,
            )
            dst_data = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan, dtype=np.float32)
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=LUNAR_CRS,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=LUNAR_CRS,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            write_aligned_tif(dst_path, dst_data)

    print("  Mini-RF alignment complete.")


# =====================================================================
# WAC Hapke 7-band (400m tiles → 128ppd mosaics)
# =====================================================================

def align_wac_hapke(skip_existing=True):
    """Mosaic and align WAC Hapke normalized tiles to 128ppd grid.

    Each band has 8 tiles covering 70°S-70°N. The tiles are PDS IMG
    files in simple cylindrical projection at ~400m resolution.
    We mosaic each band into a single global GeoTIFF at 128ppd.
    """
    print("\n=== Aligning WAC Hapke multispectral bands ===")

    # We select 4 bands that complement our existing 643nm WAC
    # Skip 321nm, 360nm (UV, noisy) and 643nm (we already have WAC morph)
    bands = {
        "wac_hapke_415nm": "415NM",
        "wac_hapke_566nm": "566NM",
        "wac_hapke_604nm": "604NM",
        "wac_hapke_689nm": "689NM",
    }

    # Tile naming: WAC_HAPKE_{band}_E350{N|S}{0450|1350|2250|3150}.IMG
    tile_specs = [
        ("E350N0450", 0.0, 70.0, 0.0, 90.0),      # NE quadrants
        ("E350N1350", 0.0, 70.0, 90.0, 180.0),
        ("E350N2250", 0.0, 70.0, 180.0, 270.0),
        ("E350N3150", 0.0, 70.0, 270.0, 360.0),
        ("E350S0450", -70.0, 0.0, 0.0, 90.0),      # SE quadrants
        ("E350S1350", -70.0, 0.0, 90.0, 180.0),
        ("E350S2250", -70.0, 0.0, 180.0, 270.0),
        ("E350S3150", -70.0, 0.0, 270.0, 360.0),
    ]

    for name, band_suffix in bands.items():
        dst_path = ALIGNED_DIR / f"{name}.tif"
        if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1000:
            print(f"  Skipped (exists): {dst_path.name}")
            continue

        print(f"  Mosaicking {name}...")

        # Accumulate into global grid
        dst_data = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan, dtype=np.float32)

        for tile_name, lat_min, lat_max, lon_min_0360, lon_max_0360 in tile_specs:
            tile_path = WAC_HAPKE_DIR / f"WAC_HAPKE_{band_suffix}_{tile_name}.IMG"
            if not tile_path.exists():
                print(f"    SKIP tile {tile_path.name}: not found")
                continue

            # Read PDS IMG - these are typically 16-bit or float32
            # WAC Hapke tiles are 16-bit unsigned, scaled to reflectance
            file_size = tile_path.stat().st_size
            # Typical tile: ~139 MB = 139,000,000 bytes
            # At 16-bit: pixels = 139M/2 = ~69.5M pixels
            # At ~400m: 90° lon ~= 90*128*237/400 ~= 6804 px
            #           70° lat ~= 70*128*237/400 ~= 5293 px
            # From the Hapke README: 25344x17920 per tile at 0.0003472 deg/px
            # That's ~1.25 ppd -> let's try reading with rasterio+GDAL PDS driver
            try:
                with rasterio.open(tile_path) as src:
                    tile_data = src.read(1).astype(np.float32)
                    src_transform = src.transform

                    # Handle nodata (PDS uses very large negative float)
                    nd = src.nodata
                    if nd is not None:
                        tile_data[tile_data <= nd * 0.9] = np.nan
                    # Also filter physically impossible reflectance
                    tile_data[tile_data < 0] = np.nan
                    tile_data[tile_data > 2.0] = np.nan

                    # Wrap 0-360 tiles into -180..180 space
                    # If tile origin x > X_MAX, subtract full circumference
                    full_circ = 2 * math.pi * LUNAR_RADIUS_M
                    if src_transform.c > X_MAX * 0.9:
                        src_transform = Affine(
                            src_transform.a, src_transform.b,
                            src_transform.c - full_circ,
                            src_transform.d, src_transform.e,
                            src_transform.f,
                        )

                    # Reproject into a temp buffer, then merge (avoid overwriting)
                    tile_dst = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan,
                                       dtype=np.float32)
                    reproject(
                        source=tile_data,
                        destination=tile_dst,
                        src_transform=src_transform,
                        src_crs=src.crs,
                        dst_transform=TARGET_TRANSFORM,
                        dst_crs=LUNAR_CRS,
                        resampling=Resampling.bilinear,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                    )
                    # Merge: copy valid pixels from tile into global mosaic
                    valid_tile = np.isfinite(tile_dst)
                    dst_data[valid_tile] = tile_dst[valid_tile]
                    n_valid = valid_tile.sum()
                    print(f"    {tile_path.name}: {n_valid:,} valid pixels")
            except Exception as e:
                print(f"    ERROR reading {tile_path.name}: {e}")
                # Fallback: read as raw binary
                try:
                    _read_wac_hapke_tile_raw(tile_path, dst_data,
                                             lat_min, lat_max,
                                             lon_min_0360, lon_max_0360)
                except Exception as e2:
                    print(f"    FALLBACK ERROR: {e2}")
                    continue

        # Shift from 0-360 to -180..180 if needed
        # Check if data is predominantly in 0-360 space
        left_half = np.isfinite(dst_data[:, :GLOBAL_WIDTH // 2]).sum()
        right_half = np.isfinite(dst_data[:, GLOBAL_WIDTH // 2:]).sum()
        if right_half > left_half * 5:
            # Data is in 0-360, shift
            dst_data = np.roll(dst_data, GLOBAL_WIDTH // 2, axis=1)
            print(f"    Shifted 0-360 → -180..180")

        valid = np.isfinite(dst_data)
        if valid.sum() > 0:
            print(f"    Range: [{np.nanmin(dst_data):.4f}, {np.nanmax(dst_data):.4f}], "
                  f"coverage={valid.sum()/dst_data.size*100:.1f}%")
            write_aligned_tif(dst_path, dst_data)
        else:
            print(f"    WARNING: No valid data for {name}")

    print("  WAC Hapke alignment complete.")


def _read_wac_hapke_tile_raw(tile_path, dst_data, lat_min, lat_max,
                              lon_min_0360, lon_max_0360):
    """Fallback: read WAC Hapke PDS IMG as raw 16-bit and place into grid."""
    # WAC Hapke tiles: 25344 x 17920 pixels, 16-bit unsigned, MSB
    # Resolution: 0.003472222 deg/pixel (~1/288 deg)
    file_size = tile_path.stat().st_size
    # Guess dimensions
    n_pixels = file_size // 2  # 16-bit
    # Common sizes: 25344*17920 = 454,164,480 pixels -> 908,328,960 bytes ~= 908 MB
    # But our files are ~139 MB -> 139M/2 = ~69.5M pixels
    # Try smaller: maybe 8-bit? 139M/1 = 139M pixels
    # Or: 9504*7296 = 69,341,184 pixels -> ~139 MB at 16-bit... close!
    # Actually try: 139,460,608 / 2 = 69,730,304
    # Various: 9504*7340 = 69,759,360 (close)

    # Just skip raw fallback for now
    raise RuntimeError("Cannot read tile as raw, need GDAL PDS driver")


# =====================================================================
# Clementine UVVIS 750nm (118m GeoTIFF → 128ppd)
# =====================================================================

def align_clementine(skip_existing=True):
    """Align Clementine UVVIS 750nm mosaic to 128ppd grid."""
    print("\n=== Aligning Clementine UVVIS 750nm ===")

    src_path = CLEMENTINE_DIR / "clementine_uvvis_750nm.tif"
    dst_path = ALIGNED_DIR / "clementine_uvvis_750nm.tif"

    if not src_path.exists():
        print(f"  SKIP: source not found at {src_path}")
        return
    if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1000:
        print(f"  Skipped (exists): {dst_path.name}")
        return

    print(f"  Reprojecting Clementine...")
    reproject_to_grid(src_path, dst_path, Resampling.bilinear, skip_existing=False)


# =====================================================================
# LP GRS element abundance (2-deg ASCII → 128ppd)
# =====================================================================

def align_lpgrs(skip_existing=True):
    """Align LP GRS element maps to 128ppd grid.

    Reads the 2-degree ASCII table and creates gridded maps for
    key elements: TiO2, FeO (most useful for composition mapping).
    """
    print("\n=== Aligning LP GRS element abundance ===")

    tab_path = LPGRS_DIR / "lpgrs_elements_2deg.tab"
    if not tab_path.exists():
        print(f"  SKIP: source not found at {tab_path}")
        return

    # Parse the table: columns 1-4 are lat/lon bounds, 5+ are elements
    # Col 7=MgO, 8=Al2O3, 9=SiO2, 10=CaO, 11=TiO2, 12=FeO, 13=K(ppm), 14=Th(ppm)
    elements = {
        "lpgrs_tio2":  {"col": 11, "desc": "TiO2 weight fraction"},
        "lpgrs_feo":   {"col": 12, "desc": "FeO weight fraction"},
    }

    print("  Reading LP GRS table...")
    rows = []
    with open(tab_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                row = [float(x) for x in parts[:15]]
                rows.append(row)
            except ValueError:
                continue
    rows = np.array(rows)
    print(f"  Parsed {len(rows)} pixels")

    for name, info in elements.items():
        dst_path = ALIGNED_DIR / f"{name}.tif"
        if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1000:
            print(f"  Skipped (exists): {dst_path.name}")
            continue

        col = info["col"]
        print(f"  Gridding {name} (col {col}: {info['desc']})...")

        # Create a coarse lat/lon grid and fill with values
        # LP GRS uses equal-area pixels at varying lon widths per lat band
        # We'll create a lat/lon scatter and interpolate

        from scipy.interpolate import griddata

        # Center lat/lon for each pixel
        lat_c = (rows[:, 1] + rows[:, 2]) / 2.0
        lon_c = (rows[:, 3] + rows[:, 4]) / 2.0
        vals = rows[:, col]

        # Filter out obvious nodata (negative abundances)
        valid = vals > -0.01
        lat_c = lat_c[valid]
        lon_c = lon_c[valid]
        vals = vals[valid]

        # Create target grid in degrees
        lons = np.linspace(-180 + 0.5 / PPD, 180 - 0.5 / PPD, GLOBAL_WIDTH)
        lats = np.linspace(90 - 0.5 / PPD, -90 + 0.5 / PPD, GLOBAL_HEIGHT)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Interpolate using linear then fill with nearest
        print(f"    Interpolating {len(vals)} points to {GLOBAL_WIDTH}x{GLOBAL_HEIGHT}...")
        dst_data = griddata(
            (lon_c, lat_c), vals,
            (lon_grid, lat_grid),
            method="linear",
        ).astype(np.float32)

        # Fill NaN edges with nearest-neighbor
        mask_nan = np.isnan(dst_data)
        if mask_nan.any():
            dst_nearest = griddata(
                (lon_c, lat_c), vals,
                (lon_grid[mask_nan], lat_grid[mask_nan]),
                method="nearest",
            )
            dst_data[mask_nan] = dst_nearest

        valid_pct = np.isfinite(dst_data).sum() / dst_data.size * 100
        print(f"    Range: [{np.nanmin(dst_data):.4f}, {np.nanmax(dst_data):.4f}], "
              f"coverage={valid_pct:.1f}%")
        write_aligned_tif(dst_path, dst_data)

    print("  LP GRS alignment complete.")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Align new datasets to 128ppd grid")
    parser.add_argument("--product", type=str, default=None,
                        choices=["grail", "minirf", "wac_hapke", "clementine", "lpgrs"],
                        help="Align a specific product")
    parser.add_argument("--all", action="store_true", help="Align all products")
    parser.add_argument("--no-skip", action="store_true", help="Overwrite existing")
    args = parser.parse_args()

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    skip = not args.no_skip

    if args.all or args.product is None:
        align_grail(skip)
        align_minirf(skip)
        align_wac_hapke(skip)
        align_clementine(skip)
        align_lpgrs(skip)
    elif args.product == "grail":
        align_grail(skip)
    elif args.product == "minirf":
        align_minirf(skip)
    elif args.product == "wac_hapke":
        align_wac_hapke(skip)
    elif args.product == "clementine":
        align_clementine(skip)
    elif args.product == "lpgrs":
        align_lpgrs(skip)

    print("\n=== All alignments complete ===")


if __name__ == "__main__":
    main()

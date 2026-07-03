"""Download Robbins Lunar Crater Database and rasterize to 128 ppd mask.

Downloads the Robbins (2019) crater catalog, filters to craters >10 km,
and rasterizes to a 128 ppd equirectangular grid matching our dataset.

Output: data/aligned/crater_mask.tif
  - Binary mask: 1 = inside crater, 0 = background

Usage:
    python download_craters.py
    python download_craters.py --min-diameter 20  # only craters > 20 km
"""

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from config import (
    ALIGNED_DIR, RAW_DIR,
    GLOBAL_WIDTH, GLOBAL_HEIGHT,
    PPD, LUNAR_RADIUS_M,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
)

CATALOG_URL = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
    "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
    "lunar_crater_database_robbins_2018"
)


def download_catalog(output_path):
    """Download the Robbins crater catalog."""
    import subprocess
    print(f"Downloading Robbins crater catalog to {output_path}...")
    subprocess.run(
        ["wget", "-q", "--user-agent=Mozilla/5.0", "-O", str(output_path), CATALOG_URL],
        check=True)
    print(f"  Downloaded: {output_path.stat().st_size / 1e6:.1f} MB")
    return output_path


def load_craters(catalog_path, min_diameter_km=10.0):
    """Load craters from the catalog, filter by size.

    Returns list of (lat, lon, diameter_km) tuples.
    """
    craters = []

    # The file may be a zip containing CSV, or a direct CSV
    path = Path(catalog_path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                raise ValueError(f"No CSV found in {path}")
            # Pick the largest CSV (the actual data, not inventory files)
            csv_name = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
            print(f"  Reading: {csv_name}")
            with zf.open(csv_name) as f:
                text = io.TextIOWrapper(f, encoding='utf-8')
                reader = csv.DictReader(text)
                craters = _parse_csv(reader, min_diameter_km)
    else:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            craters = _parse_csv(reader, min_diameter_km)

    print(f"  Loaded {len(craters)} craters with diameter >= {min_diameter_km} km")
    return craters


def _parse_csv(reader, min_diameter_km):
    """Parse CSV reader, extracting lat/lon/diameter."""
    craters = []
    # Try common column name patterns
    lat_keys = ["LAT_CIRC_IMG", "LAT_ELLI_IMG", "Lat", "lat", "LATITUDE"]
    lon_keys = ["LON_CIRC_IMG", "LON_ELLI_IMG", "Lon", "lon", "LONGITUDE"]
    diam_keys = ["DIAM_CIRC_IMG", "DIAM_ELLI_MAJOR_IMG", "Diam", "diam", "DIAMETER"]

    first_row = None
    for row in reader:
        if first_row is None:
            first_row = row
            # Find the right column names
            lat_key = next((k for k in lat_keys if k in row), None)
            lon_key = next((k for k in lon_keys if k in row), None)
            diam_key = next((k for k in diam_keys if k in row), None)
            if not all([lat_key, lon_key, diam_key]):
                print(f"  Available columns: {list(row.keys())[:20]}")
                raise ValueError(
                    f"Could not find lat/lon/diam columns. "
                    f"Found lat={lat_key}, lon={lon_key}, diam={diam_key}")
            print(f"  Using columns: lat={lat_key}, lon={lon_key}, diam={diam_key}")

        try:
            lat = float(row[lat_key])
            lon = float(row[lon_key])
            diam = float(row[diam_key])
        except (ValueError, TypeError):
            continue

        if diam >= min_diameter_km:
            craters.append((lat, lon, diam))

    return craters


def rasterize_craters(craters, output_path, width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT):
    """Rasterize crater circles to a binary mask at 128 ppd.

    Each crater is drawn as a filled circle based on its center lat/lon
    and diameter.
    """
    import math

    mask = np.zeros((height, width), dtype=np.uint8)

    for lat, lon, diam_km in craters:
        # Convert diameter to pixels
        # At equator: 1 degree = PPD pixels
        # Diameter in degrees = diam_km / (2 * pi * R / 360)
        diam_deg = diam_km / (2 * math.pi * LUNAR_RADIUS_M / 1000.0 / 360.0)
        radius_px = diam_deg * PPD / 2.0

        # Center pixel coordinates
        # lon: -180 to 180 → 0 to width
        cx = (lon + 180.0) * PPD
        # lat: 90 to -90 → 0 to height
        cy = (90.0 - lat) * PPD

        # Bounding box (clipped to image)
        r_int = int(np.ceil(radius_px))
        y0 = max(0, int(cy) - r_int)
        y1 = min(height, int(cy) + r_int + 1)
        x0 = max(0, int(cx) - r_int)
        x1 = min(width, int(cx) + r_int + 1)

        if y0 >= y1 or x0 >= x1:
            continue

        # Draw filled circle
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        inside = dist_sq <= radius_px ** 2
        mask[y0:y1, x0:x1] |= inside.astype(np.uint8)

    # Handle wrap-around for craters near ±180° longitude
    for lat, lon, diam_km in craters:
        diam_deg = diam_km / (2 * math.pi * LUNAR_RADIUS_M / 1000.0 / 360.0)
        radius_px = diam_deg * PPD / 2.0
        cx = (lon + 180.0) * PPD

        # If crater extends beyond edges, draw on the other side
        if cx - radius_px < 0:
            cx_wrap = cx + width
            r_int = int(np.ceil(radius_px))
            cy = (90.0 - lat) * PPD
            x0 = max(0, int(cx_wrap) - r_int)
            x1 = min(width, int(cx_wrap) + r_int + 1)
            y0 = max(0, int(cy) - r_int)
            y1 = min(height, int(cy) + r_int + 1)
            if y0 < y1 and x0 < x1:
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (xx - cx_wrap) ** 2 + (yy - cy) ** 2
                inside = dist_sq <= radius_px ** 2
                mask[y0:y1, x0:x1] |= inside.astype(np.uint8)

        elif cx + radius_px > width:
            cx_wrap = cx - width
            r_int = int(np.ceil(radius_px))
            cy = (90.0 - lat) * PPD
            x0 = max(0, int(cx_wrap) - r_int)
            x1 = min(width, int(cx_wrap) + r_int + 1)
            y0 = max(0, int(cy) - r_int)
            y1 = min(height, int(cy) + r_int + 1)
            if y0 < y1 and x0 < x1:
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (xx - cx_wrap) ** 2 + (yy - cy) ** 2
                inside = dist_sq <= radius_px ** 2
                mask[y0:y1, x0:x1] |= inside.astype(np.uint8)

    n_crater_px = mask.sum()
    print(f"  Rasterized: {n_crater_px:,} crater pixels "
          f"({100 * n_crater_px / mask.size:.2f}% of surface)")

    # Save as GeoTIFF with same CRS as other aligned files
    transform = from_bounds(X_MIN, Y_MIN, X_MAX, Y_MAX, width, height)
    with rasterio.open(
        str(output_path), "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="uint8",
        crs=f"+proj=eqc +lat_ts=0 +lon_0=0 +a={LUNAR_RADIUS_M} +b={LUNAR_RADIUS_M} +units=m",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)

    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return mask


def main():
    parser = argparse.ArgumentParser(description="Download and rasterize Robbins crater catalog")
    parser.add_argument("--min-diameter", type=float, default=10.0,
                        help="Minimum crater diameter in km")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GeoTIFF path")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    catalog_path = RAW_DIR / "robbins_lunar_craters.zip"
    if not catalog_path.exists():
        download_catalog(catalog_path)
    else:
        print(f"Catalog already exists: {catalog_path}")

    craters = load_craters(catalog_path, min_diameter_km=args.min_diameter)

    output_path = Path(args.output) if args.output else ALIGNED_DIR / "crater_mask.tif"
    rasterize_craters(craters, output_path)


if __name__ == "__main__":
    main()

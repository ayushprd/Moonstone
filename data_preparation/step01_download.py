"""Download raw lunar datasets.

Tier 0: WAC morphology GeoTIFF + LOLA DEM GeoTIFF (direct download)
Tier 1: SLDEM2015 PDS + Diviner PDS + USGS Geologic Map

Usage:
    # Test with a small download first
    python step01_download.py --tier 0 --test

    # Full Tier 0 download (use nohup for multi-GB files)
    nohup python step01_download.py --tier 0 > download_t0.log 2>&1 &

    # Full Tier 1 download
    nohup python step01_download.py --tier 1 > download_t1.log 2>&1 &

    # Everything
    nohup python step01_download.py --tier all > download_all.log 2>&1 &
"""

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    DIVINER_PRODUCTS,
    ODE_API_URL,
    RAW_DIR,
    RAW_FILES,
    URLS,
)


def download_file(url: str, dest: Path, skip_existing: bool = True,
                  chunk_size: int = 65536) -> str:
    """Download a file with progress bar and skip-existing support.

    Returns: 'success', 'skipped', or 'failed'
    """
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        print(f"  Skipped (exists): {dest.name}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(tmp_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True,
                      desc=dest.name, ncols=80) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        tmp_path.rename(dest)
        print(f"  Downloaded: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return "success"

    except Exception as e:
        print(f"  FAILED: {dest.name} — {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return "failed"


def verify_geotiff(path: Path) -> bool:
    """Verify a GeoTIFF can be opened and print basic info."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            print(f"  Verified: {path.name}")
            print(f"    Shape: {src.width} x {src.height}")
            print(f"    CRS: {src.crs}")
            print(f"    Bounds: {src.bounds}")
            print(f"    Dtype: {src.dtypes[0]}")
            print(f"    Bands: {src.count}")
            # Read a small window to check data
            import numpy as np
            win = rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height))
            data = src.read(1, window=win)
            print(f"    Sample min/max: {np.nanmin(data):.4f} / {np.nanmax(data):.4f}")
        return True
    except Exception as e:
        print(f"  VERIFY FAILED: {path.name} — {e}")
        return False


def download_tier0(skip_existing: bool = True) -> dict:
    """Download Tier 0: WAC morphology + LOLA DEM (GeoTIFF)."""
    print("\n=== Tier 0: WAC Morphology + LOLA DEM ===")
    results = {"success": 0, "skipped": 0, "failed": 0}

    tier0_files = [
        (URLS["wac_morphology"], RAW_FILES["wac_morphology"]),
        (URLS["lola_dem"], RAW_FILES["lola_dem"]),
    ]

    # Download in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(download_file, url, dest, skip_existing): dest.name
            for url, dest in tier0_files
        }
        for future in as_completed(futures):
            name = futures[future]
            result = future.result()
            results[result] += 1

    # Verify downloaded files
    for _, dest in tier0_files:
        if dest.exists():
            verify_geotiff(dest)

    return results


def download_tier1(skip_existing: bool = True) -> dict:
    """Download Tier 1: SLDEM2015 + Diviner + Geologic Map."""
    print("\n=== Tier 1: SLDEM2015 + Diviner + Geologic Map ===")
    results = {"success": 0, "skipped": 0, "failed": 0}

    # SLDEM2015 (IMG + LBL pair)
    print("\n--- SLDEM2015 ---")
    for key in ["sldem2015_lbl", "sldem2015_img"]:
        r = download_file(URLS[key], RAW_FILES[key], skip_existing)
        results[r] += 1

    # Diviner GDR L3 products
    print("\n--- Diviner GDR L3 ---")
    diviner_results = download_diviner(skip_existing)
    for k, v in diviner_results.items():
        results[k] += v

    # Geologic Map
    print("\n--- USGS Geologic Map ---")
    r = download_file(URLS["geologic_map"], RAW_FILES["geologic_map_zip"], skip_existing)
    results[r] += 1

    # Unzip geologic map
    if RAW_FILES["geologic_map_zip"].exists():
        geo_dir = RAW_DIR / "geologic_map"
        if not geo_dir.exists() or not list(geo_dir.glob("*.shp")):
            print("  Extracting geologic map...")
            with zipfile.ZipFile(RAW_FILES["geologic_map_zip"], "r") as zf:
                zf.extractall(geo_dir)
            print(f"  Extracted to {geo_dir}")
        else:
            print(f"  Already extracted: {geo_dir}")

    return results


def download_diviner(skip_existing: bool = True) -> dict:
    """Download Diviner products using verified PDS URLs.

    Products (from config.DIVINER_PRODUCTS):
    - diviner_temp_night: GDR L3 nighttime surface temp (1.76 GB)
    - rock_abundance: GDR L3 rock abundance fraction (1.76 GB)
    - christiansen_feature: GDR L3 CF wavelength (1.32 GB)
    - diviner_tbol_midnight: GHRM bolometric midnight temp (3.08 GB)
    """
    results = {"success": 0, "skipped": 0, "failed": 0}

    for product_name, info in DIVINER_PRODUCTS.items():
        print(f"\n  Downloading {product_name} ({info['description']})...")

        # Build filename: add diviner_ prefix only if not already present
        prefix = "" if product_name.startswith("diviner_") else "diviner_"
        base_name = f"{prefix}{product_name}"

        # Download label file (LBL or XML, small)
        lbl_url = info["lbl"]
        lbl_ext = lbl_url.rsplit(".", 1)[-1]
        lbl_path = RAW_DIR / f"{base_name}.{lbl_ext}"
        r = download_file(lbl_url, lbl_path, skip_existing)
        results[r] += 1

        # Download IMG (large)
        img_path = RAW_DIR / f"{base_name}.img"
        r = download_file(info["img"], img_path, skip_existing)
        results[r] += 1

    return results


def query_ode_monthly_mosaics(output_path: Path = None) -> None:
    """Query ODE REST API for WAC monthly mosaic catalog (Tier 1.5 prep)."""
    print("\n=== ODE API: WAC Monthly Mosaics (SDPWMG) ===")

    if output_path is None:
        output_path = RAW_DIR / "wac_monthly_catalog.json"

    try:
        resp = requests.get(ODE_API_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Save raw response
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # Parse summary
        if "ODEResults" in data:
            ode = data["ODEResults"]
            count = ode.get("Count", "?")
            print(f"  Total products found: {count}")
            print(f"  Catalog saved to: {output_path}")

            # Try to extract product details
            products = ode.get("Products", {}).get("Product", [])
            if isinstance(products, dict):
                products = [products]
            if products:
                print(f"  Sample product IDs:")
                for p in products[:5]:
                    pid = p.get("pdsid", p.get("PDSid", "?"))
                    print(f"    {pid}")
        else:
            print(f"  Unexpected response format. Saved raw JSON to {output_path}")

    except Exception as e:
        print(f"  ODE query failed: {e}")


def download_test() -> None:
    """Quick test: download first 1MB of WAC to verify connectivity."""
    print("\n=== Test Download (first 1MB of WAC) ===")
    url = URLS["wac_morphology"]
    test_path = RAW_DIR / "test_download.tmp"

    try:
        resp = requests.get(url, stream=True, timeout=30,
                            headers={"Range": "bytes=0-1048575"})
        print(f"  Status: {resp.status_code}")
        print(f"  Content-Length: {resp.headers.get('content-length', '?')}")
        print(f"  Content-Type: {resp.headers.get('content-type', '?')}")

        if resp.status_code in (200, 206):
            with open(test_path, "wb") as f:
                f.write(resp.content)
            print(f"  Downloaded {test_path.stat().st_size} bytes — OK")
            test_path.unlink()
        else:
            print(f"  Unexpected status code: {resp.status_code}")
    except Exception as e:
        print(f"  Test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download raw lunar datasets"
    )
    parser.add_argument(
        "--tier", choices=["0", "1", "all", "ode"],
        default="0",
        help="Which tier to download (0=GeoTIFF, 1=PDS+geo, all=everything, ode=catalog only)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip files that already exist (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-download even if files exist"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick connectivity test only (download 1MB)"
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        download_test()
        return

    total = {"success": 0, "skipped": 0, "failed": 0}

    if args.tier in ("0", "all"):
        r = download_tier0(args.skip_existing)
        for k in total:
            total[k] += r[k]

    if args.tier in ("1", "all"):
        r = download_tier1(args.skip_existing)
        for k in total:
            total[k] += r[k]

    if args.tier in ("ode", "all"):
        query_ode_monthly_mosaics()

    print(f"\n=== Summary ===")
    print(f"  Success: {total['success']}")
    print(f"  Skipped: {total['skipped']}")
    print(f"  Failed:  {total['failed']}")


if __name__ == "__main__":
    main()

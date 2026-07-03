"""Download M3 Moon Mineralogy Mapper data via ODE REST API.

Queries for L2 reflectance (REFIMG) and L1B geometry (CALIV3) products,
constructs download URLs, and downloads with resume support.

We download only what's needed:
  - L2: RFL.HDR + RFL.IMG (reflectance, ~117 MB each)
  - L1B: LOC.HDR + LOC.IMG + OBS.HDR + OBS.IMG (geometry, ~515 MB each)
  - SKIP: L1B RDN (radiance, 2.7 GB each — not needed)

Usage:
    # Query catalog only (fast, ~2 min)
    python step09_m3_download.py --catalog-only

    # Download everything (L2 + L1B geometry, ~560 GB total, use nohup)
    nohup python step09_m3_download.py --tier all > m3_dl.log 2>&1 &

    # Download L2 reflectance only (~104 GB)
    nohup python step09_m3_download.py --tier l2 > m3_dl_l2.log 2>&1 &

    # Download L1B geometry only (~460 GB)
    nohup python step09_m3_download.py --tier l1b > m3_dl_l1b.log 2>&1 &

    # Test with 5 products
    python step09_m3_download.py --tier all --test 5
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    M3_CATALOG_PATH,
    M3_INSTRUMENT_HOST,
    M3_INSTRUMENT_ID,
    M3_L1B_DIR,
    M3_L2_DIR,
    M3_ODE_API_BASE,
    M3_RAW_DIR,
)


def get_product_count(pt: str) -> int:
    """Query ODE REST API for product count."""
    params = {
        "query": "products", "target": "moon",
        "ihid": M3_INSTRUMENT_HOST, "iid": M3_INSTRUMENT_ID,
        "pt": pt, "output": "JSON", "results": "c",
    }
    resp = requests.get(M3_ODE_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    return int(resp.json()["ODEResults"]["Count"])


def get_all_product_metadata(pt: str) -> list:
    """Get all product metadata (pdsid, External_url) via paginated queries."""
    count = get_product_count(pt)
    print(f"  {pt}: {count} products")

    all_products = []
    limit = 100
    for offset in tqdm(range(1, count + 1, limit),
                       desc=f"Querying {pt}", ncols=80):
        params = {
            "query": "product_files", "target": "moon",
            "ihid": M3_INSTRUMENT_HOST, "iid": M3_INSTRUMENT_ID,
            "pt": pt, "output": "JSON", "results": "m",
            "offset": offset, "limit": limit,
        }
        for attempt in range(3):
            try:
                resp = requests.get(M3_ODE_API_BASE, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n  FAILED at offset {offset}: {e}")
                    continue
                time.sleep(3 * (attempt + 1))

        prods = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(prods, dict):
            prods = [prods]

        for p in prods:
            ext_url = p.get("External_url", "")
            all_products.append({
                "external_url": ext_url,
                "center_lat": float(p.get("Center_latitude", 0)),
                "center_lon": float(p.get("Center_longitude", 0)),
            })

    return all_products


def parse_product_id_and_urls(external_url: str, product_type: str) -> dict:
    """Extract product ID and construct file download URLs from External_url.

    L2 External_url pattern:
      .../CH1M3_0004/DATA/{date_range}/{YYYYMM}/L2/{base_ts}_V01_L2.LBL
    L1B External_url pattern:
      .../CH1M3_0003/DATA/{date_range}/{YYYYMM}/L1B/{base_ts}_V03_L1B.LBL
    """
    parts = external_url.split("/")
    lbl_name = parts[-1]  # e.g., M3G20081118T222604_V01_L2.LBL
    base_ts = lbl_name.split("_V")[0]  # e.g., M3G20081118T222604
    dir_url = "/".join(parts[:-1])

    # The JPL PDS Imaging archive migrated: pds-imaging.jpl.nasa.gov/data/...  ->
    # planetarydata.jpl.nasa.gov/img/data/...  (note the new "/img/" prefix).
    # Map the legacy host+path to the current one (handle http and https forms).
    for host in ("http://pds-imaging.jpl.nasa.gov",
                 "https://pds-imaging.jpl.nasa.gov"):
        dir_url = dir_url.replace(host + "/data/",
                                  "https://planetarydata.jpl.nasa.gov/img/data/")
    # Fallback for any already-https planetarydata URL missing the /img/ prefix.
    dir_url = dir_url.replace("https://planetarydata.jpl.nasa.gov/data/",
                              "https://planetarydata.jpl.nasa.gov/img/data/")

    result = {"product_id": base_ts, "base_dir": dir_url}

    if product_type == "REFIMG":
        result["rfl_hdr"] = f"{dir_url}/{base_ts}_V01_RFL.HDR"
        result["rfl_img"] = f"{dir_url}/{base_ts}_V01_RFL.IMG"
    elif product_type == "CALIV3":
        result["loc_hdr"] = f"{dir_url}/{base_ts}_V03_LOC.HDR"
        result["loc_img"] = f"{dir_url}/{base_ts}_V03_LOC.IMG"
        result["obs_hdr"] = f"{dir_url}/{base_ts}_V03_OBS.HDR"
        result["obs_img"] = f"{dir_url}/{base_ts}_V03_OBS.IMG"

    return result


def extract_timestamp(product_id: str) -> str:
    """Extract timestamp from M3 product ID for L2/L1B pairing."""
    match = re.search(r"(\d{8}T\d{6})", product_id)
    return match.group(1) if match else ""


def build_paired_catalog(l2_metadata: list, l1b_metadata: list) -> list:
    """Build paired catalog with all download URLs."""
    # Parse all product URLs
    l2_parsed = []
    for m in l2_metadata:
        if m["external_url"]:
            parsed = parse_product_id_and_urls(m["external_url"], "REFIMG")
            parsed["center_lat"] = m["center_lat"]
            parsed["center_lon"] = m["center_lon"]
            l2_parsed.append(parsed)

    l1b_parsed = {}
    for m in l1b_metadata:
        if m["external_url"]:
            parsed = parse_product_id_and_urls(m["external_url"], "CALIV3")
            ts = extract_timestamp(parsed["product_id"])
            if ts:
                l1b_parsed[ts] = parsed

    # Pair by timestamp
    pairs = []
    unmatched = 0
    for l2 in l2_parsed:
        ts = extract_timestamp(l2["product_id"])
        if ts and ts in l1b_parsed:
            l1b = l1b_parsed[ts]
            pairs.append({
                "timestamp": ts,
                "product_id": l2["product_id"],
                "center_lat": l2["center_lat"],
                "center_lon": l2["center_lon"],
                # L2 files
                "rfl_hdr": l2["rfl_hdr"],
                "rfl_img": l2["rfl_img"],
                # L1B geometry files
                "loc_hdr": l1b["loc_hdr"],
                "loc_img": l1b["loc_img"],
                "obs_hdr": l1b["obs_hdr"],
                "obs_img": l1b["obs_img"],
            })
        else:
            unmatched += 1

    print(f"  Paired: {len(pairs)}, Unmatched L2: {unmatched}")
    return pairs


def download_file(url: str, dest: Path, skip_existing: bool = True,
                  chunk_size: int = 65536, max_attempts: int = 12) -> str:
    """Download a file with HTTP Range resume + rate-limit-aware retry.

    M3 RFL.IMG files are ~1.2 GB and the JPL archive connection is unstable
    (frequent mid-stream drops) and throttles under load (503/500/502). A
    naive retry-from-zero loop never finishes a large file on a flaky link.
    So we RESUME a partial .tmp via `Range: bytes=N-` instead of restarting,
    and back off (honoring Retry-After) on 5xx/429. The .tmp is preserved
    across attempts; only finalized (renamed) once the full Content-Length
    (total) is present on disk.
    """
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    expected_total = None  # absolute byte size from first successful response

    for attempt in range(max_attempts):
        # Resume from whatever bytes we already have.
        have = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            resp = requests.get(url, stream=True, timeout=180, headers=headers)

            if resp.status_code in (500, 502, 503, 504, 429):
                retry_after = resp.headers.get("retry-after")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) \
                    else min(60, 2 ** attempt + 3)
                resp.close()
                time.sleep(wait)
                continue

            # If we asked for a range but the server ignored it (200, not 206),
            # start the file over from scratch to avoid corruption.
            mode = "ab"
            if have and resp.status_code == 200:
                have = 0
                mode = "wb"
            resp.raise_for_status()

            # Establish the absolute expected size once.
            clen = int(resp.headers.get("content-length", 0))
            if resp.status_code == 206:
                cr = resp.headers.get("content-range", "")
                if "/" in cr:
                    expected_total = int(cr.rsplit("/", 1)[1])
            elif clen and expected_total is None:
                expected_total = clen

            with open(tmp_path, mode) as f:
                with tqdm(total=expected_total, initial=have, unit="B",
                          unit_scale=True, desc=dest.name[:40], ncols=80,
                          leave=False) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            got = tmp_path.stat().st_size
            if expected_total and got < expected_total:
                raise IOError(f"incomplete {got}/{expected_total}")

            tmp_path.rename(dest)
            return "success"

        except Exception:
            # Keep the .tmp so the next attempt resumes; just back off.
            if attempt == max_attempts - 1:
                return "failed"
            time.sleep(min(60, 2 ** attempt + 3))

    return "failed"


def download_pair(pair: dict, tier: str,
                  skip_existing: bool = True) -> dict:
    """Download files for one L2/L1B pair."""
    pid = pair["product_id"]
    results = {"success": 0, "skipped": 0, "failed": 0}

    if tier in ("l2", "all"):
        for key in ("rfl_hdr", "rfl_img"):
            url = pair[key]
            fname = url.split("/")[-1]
            dest = M3_L2_DIR / fname
            r = download_file(url, dest, skip_existing)
            results[r] += 1

    if tier in ("l1b", "all"):
        for key in ("loc_hdr", "loc_img", "obs_hdr", "obs_img"):
            url = pair[key]
            fname = url.split("/")[-1]
            dest = M3_L1B_DIR / fname
            r = download_file(url, dest, skip_existing)
            results[r] += 1

    return results


def save_catalog(pairs: list, output_path: Path) -> None:
    """Save paired product catalog."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"paired_count": len(pairs), "pairs": pairs}, f, indent=2)
    print(f"  Catalog saved: {output_path} ({len(pairs)} pairs)")


def load_catalog(catalog_path: Path) -> list:
    """Load previously saved catalog."""
    with open(catalog_path) as f:
        return json.load(f)["pairs"]


def main():
    parser = argparse.ArgumentParser(
        description="Download M3 Moon Mineralogy Mapper data via ODE REST API"
    )
    parser.add_argument("--tier", choices=["l2", "l1b", "all"],
                        default="all", help="Which data to download")
    parser.add_argument("--catalog-only", action="store_true",
                        help="Only query API and save catalog, no downloads")
    parser.add_argument("--use-catalog", action="store_true",
                        help="Use existing catalog instead of querying API")
    parser.add_argument("--test", type=int, default=None,
                        help="Download only N products for testing")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download workers")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    M3_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get or load catalog
    if args.use_catalog and M3_CATALOG_PATH.exists():
        print("=== Loading existing catalog ===")
        pairs = load_catalog(M3_CATALOG_PATH)
        print(f"  Loaded {len(pairs)} pairs")
    else:
        print("=== Querying ODE REST API ===")
        l2_meta = get_all_product_metadata("REFIMG")
        l1b_meta = get_all_product_metadata("CALIV3")

        print("\n--- Pairing L2/L1B by timestamp ---")
        pairs = build_paired_catalog(l2_meta, l1b_meta)
        save_catalog(pairs, M3_CATALOG_PATH)

    if args.catalog_only:
        print("\n=== Catalog-only mode, no downloads ===")

        # Print download size estimate
        n = len(pairs)
        l2_size_gb = n * 0.117  # ~117 MB per RFL
        l1b_size_gb = n * 0.515  # ~515 MB per LOC+OBS
        print(f"\n  Estimated download sizes:")
        print(f"    L2 reflectance: {n} x 117 MB = {l2_size_gb:.0f} GB")
        print(f"    L1B geometry:   {n} x 515 MB = {l1b_size_gb:.0f} GB")
        print(f"    Total:          {l2_size_gb + l1b_size_gb:.0f} GB")
        return

    # Limit for testing
    if args.test:
        pairs = pairs[:args.test]
        print(f"\n--- Test mode: {args.test} products ---")

    # Step 2: Download
    M3_L2_DIR.mkdir(parents=True, exist_ok=True)
    M3_L1B_DIR.mkdir(parents=True, exist_ok=True)

    total = {"success": 0, "skipped": 0, "failed": 0, "failed_ids": []}
    print(f"\n=== Downloading {len(pairs)} products (tier={args.tier}) ===")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_pair, pair, args.tier, args.skip_existing): pair
            for pair in pairs
        }
        for i, future in enumerate(as_completed(futures)):
            pair = futures[future]
            try:
                r = future.result()
                for k in ("success", "skipped", "failed"):
                    total[k] += r[k]
                if r["failed"] > 0:
                    total["failed_ids"].append(pair["product_id"])
            except Exception as e:
                total["failed"] += 1
                total["failed_ids"].append(pair["product_id"])

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(pairs)} "
                      f"(ok={total['success']}, skip={total['skipped']}, "
                      f"fail={total['failed']})")

    # Save failure log
    if total["failed_ids"]:
        fail_path = M3_RAW_DIR / "m3_download_failures.json"
        with open(fail_path, "w") as f:
            json.dump(total["failed_ids"], f, indent=2)
        print(f"\n  Failures logged: {fail_path}")

    print(f"\n=== Download Summary ===")
    print(f"  Success: {total['success']}")
    print(f"  Skipped: {total['skipped']}")
    print(f"  Failed:  {total['failed']}")


if __name__ == "__main__":
    main()

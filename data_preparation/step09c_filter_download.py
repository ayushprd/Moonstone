"""Fast M3 download with HTTP range-request filtering.

Phase 1: Download OBS.HDR (tiny) + sample incidence from OBS.IMG via range requests
         → Build daytime filter list in ~5-10 minutes
Phase 2: Download RFL+LOC+OBS only for daytime strips (skip ~30% nighttime)

This saves ~30-40% bandwidth compared to downloading everything.

Usage:
    nohup python -u step09c_filter_download.py > m3_smart_dl.log 2>&1 &

    # Filter only (check what's daytime vs nighttime)
    python step09c_filter_download.py --phase filter

    # Download only (using existing filter)
    nohup python -u step09c_filter_download.py --phase download --workers 10 > m3_dl.log 2>&1 &
"""

import argparse
import json
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from config import (
    M3_CATALOG_PATH,
    M3_L1B_DIR,
    M3_L2_DIR,
    M3_MAX_INCIDENCE_DEG,
    M3_RAW_DIR,
)


def download_file(url: str, dest: Path, skip_existing: bool = True,
                  chunk_size: int = 65536) -> str:
    """Download a file with resume support."""
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(tmp_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=dest.name[:40], ncols=80, leave=False) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            tmp_path.rename(dest)
            return "success"

        except Exception as e:
            if attempt == 2:
                if tmp_path.exists():
                    tmp_path.unlink()
                return f"failed: {e}"
            time.sleep(5 * (attempt + 1))

    return "failed"


def parse_envi_header_text(text: str) -> dict:
    """Parse ENVI header text into dict."""
    hdr = {}
    text = re.sub(r"\n\s+", " ", text)
    for line in text.split("\n"):
        line = line.strip()
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()
        hdr[key] = val
    for k in ("samples", "lines", "bands", "data_type", "header_offset"):
        if k in hdr:
            hdr[k] = int(hdr[k])
    return hdr


def check_incidence_remote(obs_url: str, obs_hdr_url: str,
                           timeout: int = 30) -> dict:
    """Check incidence angle of a strip using HTTP range requests.

    Downloads the HDR file (< 1 KB) and samples 3 lines from the OBS.IMG
    to determine if the strip is daytime.
    """
    # Download and parse HDR
    resp = requests.get(obs_hdr_url, timeout=timeout)
    resp.raise_for_status()
    hdr = parse_envi_header_text(resp.text)

    lines = hdr["lines"]
    bands = hdr["bands"]
    samples = hdr["samples"]
    dt_code = hdr.get("data_type", 5)
    item_size = 8 if dt_code == 5 else 4  # float64 or float32
    offset = hdr.get("header_offset", 0)

    # Sample 3 lines: first, middle, last
    sample_lines = [0, lines // 2, lines - 1]
    incidence_values = []

    for line_idx in sample_lines:
        # Offset for line L, band B=1 (To-Sun Zenith = incidence), all samples:
        # BIL: (lines, bands, samples)
        byte_offset = offset + (
            line_idx * bands * samples + 1 * samples
        ) * item_size
        byte_length = samples * item_size

        headers = {
            "Range": f"bytes={byte_offset}-{byte_offset + byte_length - 1}"
        }
        try:
            resp = requests.get(obs_url, headers=headers, timeout=timeout)
            if resp.status_code == 206:
                dt = np.float64 if dt_code == 5 else np.float32
                data = np.frombuffer(resp.content, dtype=dt)
                incidence_values.extend(data.tolist())
        except Exception:
            continue

    if not incidence_values:
        return {"is_daytime": False, "error": "no data", "n_samples": 0}

    inc = np.array(incidence_values)
    valid = (inc > 0) & (inc < M3_MAX_INCIDENCE_DEG)
    n_valid = int(valid.sum())
    n_total = len(inc)

    return {
        "is_daytime": n_valid > 0,
        "n_valid": n_valid,
        "n_samples": n_total,
        "pct_valid": n_valid / n_total * 100 if n_total > 0 else 0,
        "inc_min": float(inc.min()),
        "inc_max": float(inc.max()),
        "inc_median": float(np.median(inc)),
        "lines": lines,
        "samples": samples,
    }


def phase_filter(catalog_path: Path, workers: int = 20) -> dict:
    """Phase 1: Filter strips by incidence angle using range requests."""
    with open(catalog_path) as f:
        catalog = json.load(f)

    pairs = catalog["pairs"]
    total = len(pairs)
    print(f"=== Phase 1: Filter {total} products by incidence angle ===")
    print(f"  Using HTTP range requests (no full downloads needed)")

    daytime_pairs = []
    nighttime_ids = []
    error_ids = []

    def check_one(pair):
        pid = pair["product_id"]
        obs_url = pair["obs_img"]
        obs_hdr_url = pair["obs_hdr"]
        try:
            info = check_incidence_remote(obs_url, obs_hdr_url)
            return pid, pair, info
        except Exception as e:
            return pid, pair, {"is_daytime": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_one, pair): pair for pair in pairs}

        for i, future in enumerate(as_completed(futures)):
            pid, pair, info = future.result()

            if "error" in info:
                error_ids.append(pid)
                # Include errors as daytime (download anyway to be safe)
                daytime_pairs.append(pair)
            elif info["is_daytime"]:
                daytime_pairs.append(pair)
            else:
                nighttime_ids.append(pid)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{total} "
                      f"(day={len(daytime_pairs)}, night={len(nighttime_ids)}, "
                      f"err={len(error_ids)})")

    print(f"\n  Results:")
    print(f"    Daytime: {len(daytime_pairs)} ({len(daytime_pairs)/total*100:.0f}%)")
    print(f"    Nighttime: {len(nighttime_ids)} ({len(nighttime_ids)/total*100:.0f}%)")
    print(f"    Errors: {len(error_ids)}")

    # Save filtered catalog
    filtered_path = M3_RAW_DIR / "m3_daytime_catalog.json"
    with open(filtered_path, "w") as f:
        json.dump({
            "total": total,
            "daytime_count": len(daytime_pairs),
            "nighttime_count": len(nighttime_ids),
            "daytime_pairs": daytime_pairs,
            "nighttime_product_ids": nighttime_ids,
        }, f, indent=2)
    print(f"  Saved: {filtered_path}")

    # Estimate savings
    avg_rfl_mb = 440
    saved_gb = len(nighttime_ids) * avg_rfl_mb / 1024
    total_gb = len(daytime_pairs) * avg_rfl_mb / 1024
    print(f"\n  Estimated download savings: ~{saved_gb:.0f} GB skipped")
    print(f"  Estimated total download: ~{total_gb:.0f} GB")

    return {"daytime": daytime_pairs, "nighttime": nighttime_ids}


def phase_download(workers: int = 10, test: int = None) -> None:
    """Phase 2: Download RFL+LOC+OBS for daytime strips only."""
    filtered_path = M3_RAW_DIR / "m3_daytime_catalog.json"
    if not filtered_path.exists():
        print("ERROR: Run --phase filter first")
        sys.exit(1)

    with open(filtered_path) as f:
        data = json.load(f)

    daytime_pairs = data["daytime_pairs"]
    print(f"=== Phase 2: Download {len(daytime_pairs)} daytime products ===")

    if test:
        daytime_pairs = daytime_pairs[:test]
        print(f"  Test mode: {test} products")

    M3_L2_DIR.mkdir(parents=True, exist_ok=True)
    M3_L1B_DIR.mkdir(parents=True, exist_ok=True)

    # Build download tasks
    tasks = []
    for pair in daytime_pairs:
        for key in ("rfl_hdr", "rfl_img", "loc_hdr", "loc_img",
                     "obs_hdr", "obs_img"):
            url = pair[key]
            fname = url.split("/")[-1]
            if key.startswith("rfl"):
                dest = M3_L2_DIR / fname
            else:
                dest = M3_L1B_DIR / fname
            tasks.append((url, dest, pair["product_id"]))

    # Count how many still need downloading
    todo = [(u, d, p) for u, d, p in tasks if not d.exists()]
    already = len(tasks) - len(todo)
    print(f"  Total files: {len(tasks)}")
    print(f"  Already downloaded: {already}")
    print(f"  Need to download: {len(todo)}")

    if not todo:
        print("  All files already downloaded!")
        return

    total = {"success": 0, "skipped": 0, "failed": 0, "failed_ids": []}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, url, dest): (url, dest, pid)
            for url, dest, pid in todo
        }
        for i, future in enumerate(as_completed(futures)):
            url, dest, pid = futures[future]
            r = future.result()
            if r.startswith("failed"):
                total["failed"] += 1
                if pid not in total["failed_ids"]:
                    total["failed_ids"].append(pid)
            elif r == "skipped":
                total["skipped"] += 1
            else:
                total["success"] += 1

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(todo)} "
                      f"(ok={total['success']}, skip={total['skipped']}, "
                      f"fail={total['failed']})")

    if total["failed_ids"]:
        fail_path = M3_RAW_DIR / "m3_download_failures.json"
        with open(fail_path, "w") as f:
            json.dump(total["failed_ids"], f, indent=2)
        print(f"  Failures saved: {fail_path}")

    print(f"\n=== Download Summary ===")
    print(f"  Success: {total['success']}")
    print(f"  Skipped: {total['skipped']}")
    print(f"  Failed:  {total['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart M3 download with incidence filtering"
    )
    parser.add_argument("--phase", choices=["filter", "download", "both"],
                        default="both")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--test", type=int, default=None)
    args = parser.parse_args()

    if args.phase in ("filter", "both"):
        phase_filter(M3_CATALOG_PATH, workers=args.workers)
        if args.phase == "filter":
            return

    if args.phase in ("download", "both"):
        phase_download(workers=args.workers, test=args.test)


if __name__ == "__main__":
    main()

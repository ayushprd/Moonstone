"""Mosaic M3 strips into 128 ppd global GeoTIFFs.

Single-pass approach: for each strip, scatter valid pixels directly into
memory-mapped accumulation arrays. Much faster than multi-pass latitude-band.

Produces 8 band GeoTIFFs + 4 geometry layers in data/aligned/.

Usage:
    # Test with 10 strips
    python step10_m3_mosaic.py --test 10

    # Full mosaicking (use nohup, ~4-8 hours)
    nohup python step10_m3_mosaic.py > m3_mosaic.log 2>&1 &

    # Resume after interruption
    nohup python step10_m3_mosaic.py --resume > m3_mosaic.log 2>&1 &

    # Preprocess only (extract compact .npz from ENVI)
    python step10_m3_mosaic.py --preprocess-only --test 10
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

from config import (
    ALIGNED_DIR,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LUNAR_CRS_PROJ4,
    M3_ALIGNED_FILES,
    M3_BAND_INDICES,
    M3_BAND_NAMES,
    M3_BAND_SELECTION,
    M3_CATALOG_PATH,
    M3_COMPACT_DIR,
    M3_GEOMETRY_FILES,
    M3_L1B_DIR,
    M3_L2_DIR,
    M3_MAX_EMISSION_DEG,
    M3_MAX_INCIDENCE_DEG,
    M3_NODATA,
    M3_RAW_DIR,
    PPD,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)


# ---- ENVI format readers ----

def parse_envi_header(hdr_path: Path) -> dict:
    """Parse ENVI .hdr file into dict."""
    hdr = {}
    with open(hdr_path) as f:
        text = f.read()

    # Handle multi-line values in { }
    text = re.sub(r"\n\s+", " ", text)

    for line in text.split("\n"):
        line = line.strip()
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()

        # Parse braced lists
        if val.startswith("{") and val.endswith("}"):
            val = val[1:-1].strip()
            val = [v.strip() for v in val.split(",") if v.strip()]
        hdr[key] = val

    # Convert numeric fields
    for k in ("samples", "lines", "bands", "data_type", "byte_order",
              "header_offset"):
        if k in hdr:
            hdr[k] = int(hdr[k])

    return hdr


def envi_dtype(data_type: int) -> np.dtype:
    """Convert ENVI data_type code to numpy dtype."""
    mapping = {
        1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
        5: np.float64, 12: np.uint16, 13: np.uint32, 14: np.int64,
        15: np.uint64,
    }
    return np.dtype(mapping.get(data_type, np.float32))


def read_m3_rfl(img_path: Path, hdr: dict,
                band_indices: list) -> np.ndarray:
    """Read selected bands from M3 L2 reflectance ENVI BIL file.

    Returns array of shape (n_selected_bands, lines, samples), float32.
    Uses memory mapping to avoid loading the full 85-band cube.
    """
    lines = hdr["lines"]
    bands = hdr["bands"]
    samples = hdr["samples"]
    dt = envi_dtype(hdr.get("data_type", 4))
    offset = hdr.get("header_offset", 0)

    # BIL: (lines, bands, samples)
    mmap = np.memmap(img_path, dtype=dt, mode="r", offset=offset,
                     shape=(lines, bands, samples))

    selected = np.empty((len(band_indices), lines, samples), dtype=np.float32)
    for i, bi in enumerate(band_indices):
        selected[i] = mmap[:, bi, :].astype(np.float32)

    del mmap
    return selected


def read_m3_loc(img_path: Path, hdr: dict) -> tuple:
    """Read M3 LOC file: per-pixel longitude, latitude, radius.

    LOC has 3 bands: lon (0-360), lat, radius.
    Returns (lat, lon) as float32 arrays. Lon converted to -180..180.
    """
    lines = hdr["lines"]
    bands = hdr["bands"]
    samples = hdr["samples"]
    dt = envi_dtype(hdr.get("data_type", 5))  # LOC is float64
    offset = hdr.get("header_offset", 0)

    mmap = np.memmap(img_path, dtype=dt, mode="r", offset=offset,
                     shape=(lines, bands, samples))

    lon = mmap[:, 0, :].astype(np.float32)
    lat = mmap[:, 1, :].astype(np.float32)

    del mmap

    # Convert 0-360 to -180..180
    lon = np.where(lon > 180, lon - 360, lon)

    return lat, lon


def read_m3_obs(img_path: Path, hdr: dict) -> dict:
    """Read M3 OBS file: observation geometry.

    OBS has 10 bands (from PDS HDR band names):
        band 0: To-Sun Azimuth (deg)
        band 1: To-Sun Zenith (deg) = solar incidence angle
        band 2: To-M3 Azimuth (deg)
        band 3: To-M3 Zenith (deg) = emission angle
        band 4: Phase (deg)
        band 5: To-Sun Path Length (au)
        band 6: To-M3 Path Length (m)
        band 7: Facet Slope (deg)
        band 8: Facet Aspect (deg)
        band 9: Facet Cos(i) (unitless)
    """
    lines = hdr["lines"]
    bands = hdr["bands"]
    samples = hdr["samples"]
    dt = envi_dtype(hdr.get("data_type", 5))  # OBS is float64
    offset = hdr.get("header_offset", 0)

    mmap = np.memmap(img_path, dtype=dt, mode="r", offset=offset,
                     shape=(lines, bands, samples))

    result = {
        "incidence": mmap[:, 1, :].astype(np.float32),  # To-Sun Zenith
        "emission": mmap[:, 3, :].astype(np.float32),   # To-M3 Zenith
        "phase": mmap[:, 4, :].astype(np.float32),      # Phase angle
    }

    del mmap
    return result


# ---- Preprocessing ----

def find_strip_files(l2_dir: Path, l1b_dir: Path) -> list:
    """Find paired L2/L1B files on disk."""
    # Find L2 reflectance files
    l2_imgs = sorted(l2_dir.glob("*_rfl.img"))
    if not l2_imgs:
        l2_imgs = sorted(l2_dir.glob("*.img"))

    pairs = []
    for rfl_img in l2_imgs:
        pid = rfl_img.stem.replace("_rfl", "")
        rfl_hdr = rfl_img.with_suffix(".hdr")
        if not rfl_hdr.exists():
            rfl_hdr = rfl_img.parent / f"{pid}_rfl.hdr"

        # Find matching L1B LOC and OBS
        loc_img = None
        obs_img = None
        for pattern in [f"{pid}_loc.img", f"*{pid[-15:]}*_loc.img"]:
            matches = list(l1b_dir.glob(pattern))
            if matches:
                loc_img = matches[0]
                break

        for pattern in [f"{pid}_obs.img", f"*{pid[-15:]}*_obs.img"]:
            matches = list(l1b_dir.glob(pattern))
            if matches:
                obs_img = matches[0]
                break

        if rfl_hdr.exists() and loc_img and obs_img:
            pairs.append({
                "product_id": pid,
                "rfl_img": rfl_img,
                "rfl_hdr": rfl_hdr,
                "loc_img": loc_img,
                "loc_hdr": loc_img.with_suffix(".hdr"),
                "obs_img": obs_img,
                "obs_hdr": obs_img.with_suffix(".hdr"),
            })

    return pairs


def find_strip_files_from_catalog(catalog_path: Path,
                                  l2_dir: Path, l1b_dir: Path) -> list:
    """Build strip file pairs from saved catalog.

    The catalog stores full URLs; we derive local filenames from them.
    """
    with open(catalog_path) as f:
        catalog = json.load(f)

    pairs = []
    for entry in catalog.get("pairs", []):
        pid = entry["product_id"]

        # Derive local filenames from the URL filenames
        rfl_hdr_name = entry["rfl_hdr"].split("/")[-1]
        rfl_img_name = entry["rfl_img"].split("/")[-1]
        loc_hdr_name = entry["loc_hdr"].split("/")[-1]
        loc_img_name = entry["loc_img"].split("/")[-1]
        obs_hdr_name = entry["obs_hdr"].split("/")[-1]
        obs_img_name = entry["obs_img"].split("/")[-1]

        rfl_img = l2_dir / rfl_img_name
        rfl_hdr = l2_dir / rfl_hdr_name
        loc_img = l1b_dir / loc_img_name
        loc_hdr = l1b_dir / loc_hdr_name
        obs_img = l1b_dir / obs_img_name
        obs_hdr = l1b_dir / obs_hdr_name

        if all(p.exists() for p in [rfl_img, rfl_hdr, loc_img, loc_hdr,
                                     obs_img, obs_hdr]):
            pairs.append({
                "product_id": pid,
                "rfl_img": rfl_img,
                "rfl_hdr": rfl_hdr,
                "loc_img": loc_img,
                "loc_hdr": loc_hdr,
                "obs_img": obs_img,
                "obs_hdr": obs_hdr,
            })

    return pairs


def preprocess_strip(pair: dict, output_dir: Path,
                     band_indices: list,
                     skip_existing: bool = True) -> Path:
    """Extract selected bands + geometry into compact .npz.

    Saves ~10 MB per strip (vs ~100 MB full ENVI).
    Returns path to .npz file, or None on failure.
    """
    pid = pair["product_id"]
    out_path = output_dir / f"{pid}.npz"

    if skip_existing and out_path.exists():
        return out_path

    try:
        # Read reflectance header and data
        rfl_hdr = parse_envi_header(pair["rfl_hdr"])

        # Validate file size
        expected = (rfl_hdr["lines"] * rfl_hdr["bands"] * rfl_hdr["samples"]
                    * envi_dtype(rfl_hdr.get("data_type", 4)).itemsize)
        actual = pair["rfl_img"].stat().st_size
        if actual < expected * 0.9:
            print(f"  SKIP truncated: {pid} ({actual}/{expected} bytes)")
            return None

        rfl = read_m3_rfl(pair["rfl_img"], rfl_hdr, band_indices)

        # Read LOC
        loc_hdr = parse_envi_header(pair["loc_hdr"])
        lat, lon = read_m3_loc(pair["loc_img"], loc_hdr)

        # Read OBS
        obs_hdr = parse_envi_header(pair["obs_hdr"])
        obs = read_m3_obs(pair["obs_img"], obs_hdr)

        # Quality mask
        valid = np.ones(lat.shape, dtype=bool)
        valid &= obs["incidence"] < M3_MAX_INCIDENCE_DEG
        valid &= obs["incidence"] > 0
        valid &= obs["emission"] < M3_MAX_EMISSION_DEG
        valid &= obs["emission"] > 0
        valid &= np.all(rfl > 0, axis=0)
        valid &= np.all(rfl < 1.0, axis=0)  # reflectance > 1.0 is unphysical
        valid &= np.all(rfl != M3_NODATA, axis=0)
        valid &= np.isfinite(lat) & np.isfinite(lon)

        # Compute grid coordinates for valid pixels
        cols = ((lon + 180.0) * PPD).astype(np.int32)
        rows = ((90.0 - lat) * PPD).astype(np.int32)
        valid &= (cols >= 0) & (cols < GLOBAL_WIDTH)
        valid &= (rows >= 0) & (rows < GLOBAL_HEIGHT)

        n_valid = valid.sum()
        if n_valid == 0:
            return None

        # Extract valid pixels only (flat arrays for compact storage)
        flat_idx = np.where(valid.ravel())[0]
        out_rfl = np.empty((len(band_indices), n_valid), dtype=np.float32)
        for i in range(len(band_indices)):
            out_rfl[i] = rfl[i].ravel()[flat_idx]

        out_rows = rows.ravel()[flat_idx].astype(np.int32)
        out_cols = cols.ravel()[flat_idx].astype(np.int32)
        out_inc = obs["incidence"].ravel()[flat_idx].astype(np.float32)
        out_emi = obs["emission"].ravel()[flat_idx].astype(np.float32)
        out_pha = obs["phase"].ravel()[flat_idx].astype(np.float32)

        # Bounding box for spatial index
        bbox = {
            "min_lat": float(lat[valid].min()),
            "max_lat": float(lat[valid].max()),
            "min_lon": float(lon[valid].min()),
            "max_lon": float(lon[valid].max()),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            rfl=out_rfl,           # (8, N) float32
            rows=out_rows,         # (N,) int32
            cols=out_cols,         # (N,) int32
            incidence=out_inc,     # (N,) float32
            emission=out_emi,      # (N,) float32
            phase=out_pha,         # (N,) float32
            bbox_min_lat=bbox["min_lat"],
            bbox_max_lat=bbox["max_lat"],
            bbox_min_lon=bbox["min_lon"],
            bbox_max_lon=bbox["max_lon"],
            n_valid=n_valid,
        )

        return out_path

    except Exception as e:
        print(f"  ERROR preprocessing {pid}: {e}")
        return None


def _preprocess_one(args):
    """Worker function for parallel preprocessing."""
    pair, out_dir, band_indices, skip_existing = args
    try:
        return preprocess_strip(pair, out_dir, band_indices, skip_existing)
    except Exception as e:
        print(f"  ERROR {pair.get('pid', '?')}: {e}")
        return None


def preprocess_all_strips(pairs: list, band_indices: list,
                          skip_existing: bool = True) -> list:
    """Preprocess all strips into compact .npz files (parallel)."""
    import multiprocessing as mp
    n_workers = min(32, mp.cpu_count(), len(pairs))
    print(f"\n=== Preprocessing {len(pairs)} strips ({n_workers} workers) ===")

    args_list = [(p, M3_COMPACT_DIR, band_indices, skip_existing) for p in pairs]

    npz_paths = []
    failed = 0
    with mp.Pool(n_workers) as pool:
        for path in tqdm(pool.imap_unordered(_preprocess_one, args_list),
                         total=len(pairs), desc="Preprocessing", ncols=80):
            if path:
                npz_paths.append(path)
            else:
                failed += 1

    print(f"  Preprocessed: {len(npz_paths)}, Failed: {failed}")
    return npz_paths


# ---- Mosaicking ----

def init_accumulation_arrays(tmpdir: Path) -> dict:
    """Create memory-mapped accumulation arrays.

    Uses tempfile-backed mmap to avoid 52 GB RAM requirement.
    Total disk: 8 bands * 2 arrays * 4 bytes + 4 geometry * 4 bytes
              = 72 bytes/pixel * 46080 * 23040 = ~76 GB temp disk.

    Optimization: use float32 for all, single weight array shared across bands.
    Total = (8+1+4) * 4 * 46080 * 23040 = ~55 GB.

    Further optimization: since weight is shared across bands, we only
    need 8 weighted_sum + 1 weight + 3 geometry_sum + 1 n_obs = 13 arrays.
    = 13 * 4 * 46080 * 23040 = ~55 GB.

    That's still a lot of temp disk. Let's be smarter:
    We can accumulate directly using numpy add.at with the scatter pattern.
    """
    shape = (GLOBAL_HEIGHT, GLOBAL_WIDTH)
    n_bands = len(M3_BAND_INDICES)

    # Allocate in RAM — but use the fact that most of the grid is sparse.
    # At 46080*23040 = ~1 billion pixels, each float32 array = 4 GB.
    # 8 weighted_sum + 1 weight + 3 geometry + 1 count = 13 * 4 GB = 52 GB.
    #
    # Better: use memory-mapped temp files on disk (3.8 TB free).
    tmpdir.mkdir(parents=True, exist_ok=True)

    arrays = {}

    # Weighted reflectance sum: (8, H, W)
    arrays["weighted_sum"] = np.memmap(
        tmpdir / "weighted_sum.dat", dtype=np.float32, mode="w+",
        shape=(n_bands, GLOBAL_HEIGHT, GLOBAL_WIDTH),
    )

    # Total weight: (H, W)
    arrays["weight_total"] = np.memmap(
        tmpdir / "weight_total.dat", dtype=np.float32, mode="w+",
        shape=shape,
    )

    # Geometry sums
    arrays["incidence_sum"] = np.memmap(
        tmpdir / "incidence_sum.dat", dtype=np.float32, mode="w+",
        shape=shape,
    )
    arrays["emission_sum"] = np.memmap(
        tmpdir / "emission_sum.dat", dtype=np.float32, mode="w+",
        shape=shape,
    )
    arrays["phase_sum"] = np.memmap(
        tmpdir / "phase_sum.dat", dtype=np.float32, mode="w+",
        shape=shape,
    )

    # Observation count
    arrays["n_obs"] = np.memmap(
        tmpdir / "n_obs.dat", dtype=np.int32, mode="w+",
        shape=shape,
    )

    return arrays


def scatter_strip(npz_path: Path, accum: dict) -> int:
    """Scatter one preprocessed strip's pixels into accumulation arrays.

    Returns number of pixels scattered.
    """
    data = np.load(npz_path)
    rfl = data["rfl"]           # (8, N)
    rows = data["rows"]         # (N,)
    cols = data["cols"]         # (N,)
    incidence = data["incidence"]
    emission = data["emission"]
    phase = data["phase"]

    n = len(rows)
    if n == 0:
        return 0

    # Weight = cos(incidence angle)
    weight = np.cos(np.radians(incidence))

    # Use flat indexing for np.add.at
    flat_idx = rows.astype(np.intp) * GLOBAL_WIDTH + cols.astype(np.intp)

    # Accumulate weighted reflectance
    for band_i in range(rfl.shape[0]):
        weighted_rfl = rfl[band_i] * weight
        np.add.at(accum["weighted_sum"][band_i].ravel(), flat_idx, weighted_rfl)

    # Accumulate weight
    np.add.at(accum["weight_total"].ravel(), flat_idx, weight)

    # Accumulate geometry (weighted by same cos(incidence) as reflectance)
    np.add.at(accum["incidence_sum"].ravel(), flat_idx, incidence * weight)
    np.add.at(accum["emission_sum"].ravel(), flat_idx, emission * weight)
    np.add.at(accum["phase_sum"].ravel(), flat_idx, phase * weight)

    # Count
    np.add.at(accum["n_obs"].ravel(), flat_idx, np.ones(n, dtype=np.int32))

    return n


def finalize_and_write(accum: dict) -> None:
    """Compute weighted averages and write output GeoTIFFs."""
    print("\n=== Finalizing mosaics ===")

    # Build GeoTIFF profile matching existing aligned files
    dx = (X_MAX - X_MIN) / GLOBAL_WIDTH
    dy = (Y_MAX - Y_MIN) / GLOBAL_HEIGHT
    transform = Affine(dx, 0, X_MIN, 0, -dy, Y_MAX)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": GLOBAL_WIDTH,
        "height": GLOBAL_HEIGHT,
        "count": 1,
        "crs": LUNAR_CRS_PROJ4,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "yes",
    }

    weight = accum["weight_total"]
    n_obs = accum["n_obs"]
    valid_mask = weight > 0

    # Write band GeoTIFFs
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    for band_i, band_name in enumerate(M3_BAND_NAMES):
        out_path = M3_ALIGNED_FILES[band_name]
        print(f"  Writing {out_path.name}...")

        result = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan, dtype=np.float32)
        result[valid_mask] = (accum["weighted_sum"][band_i][valid_mask]
                              / weight[valid_mask])

        # Write in strips to manage memory
        with rasterio.open(out_path, "w", **profile) as dst:
            strip_h = 1024
            for y0 in range(0, GLOBAL_HEIGHT, strip_h):
                y1 = min(y0 + strip_h, GLOBAL_HEIGHT)
                window = rasterio.windows.Window(0, y0, GLOBAL_WIDTH, y1 - y0)
                dst.write(result[y0:y1], 1, window=window)

        valid_pct = valid_mask.sum() / valid_mask.size * 100
        finite_vals = result[valid_mask]
        if len(finite_vals) > 0:
            print(f"    Range: [{finite_vals.min():.4f}, {finite_vals.max():.4f}], "
                  f"mean={finite_vals.mean():.4f}, coverage={valid_pct:.1f}%")

    # Write geometry layers (incidence/emission/phase are cos(i)-weighted, divide by weight)
    for name, arr_key, divisor in [
        ("m3_n_observations", "n_obs", None),
        ("m3_mean_incidence", "incidence_sum", weight),
        ("m3_mean_emission", "emission_sum", weight),
        ("m3_mean_phase", "phase_sum", weight),
    ]:
        out_path = M3_GEOMETRY_FILES[name]
        print(f"  Writing {out_path.name}...")

        if divisor is None:
            result = accum[arr_key].astype(np.float32)
            result[~valid_mask] = np.nan
        else:
            result = np.full((GLOBAL_HEIGHT, GLOBAL_WIDTH), np.nan,
                             dtype=np.float32)
            div_mask = divisor > 0
            result[div_mask] = accum[arr_key][div_mask] / divisor[div_mask]

        with rasterio.open(out_path, "w", **profile) as dst:
            strip_h = 1024
            for y0 in range(0, GLOBAL_HEIGHT, strip_h):
                y1 = min(y0 + strip_h, GLOBAL_HEIGHT)
                window = rasterio.windows.Window(0, y0, GLOBAL_WIDTH, y1 - y0)
                dst.write(result[y0:y1], 1, window=window)

    print("  Done writing GeoTIFFs.")


def mosaic_all(npz_paths: list, resume: bool = False) -> None:
    """Main mosaicking driver: single-pass scatter into mmap accumulators."""
    checkpoint_path = M3_RAW_DIR / "m3_mosaic_checkpoint.json"
    tmpdir = M3_RAW_DIR / "mosaic_tmp"

    # Initialize or reload accumulators
    if resume and tmpdir.exists() and checkpoint_path.exists():
        print("  Resuming from checkpoint...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        completed = set(checkpoint.get("completed", []))
        remaining = [p for p in npz_paths if p.stem not in completed]
        print(f"  Already done: {len(completed)}, Remaining: {len(remaining)}")

        # Reopen existing mmap files
        n_bands = len(M3_BAND_INDICES)
        shape = (GLOBAL_HEIGHT, GLOBAL_WIDTH)
        accum = {
            "weighted_sum": np.memmap(
                tmpdir / "weighted_sum.dat", dtype=np.float32, mode="r+",
                shape=(n_bands, GLOBAL_HEIGHT, GLOBAL_WIDTH)),
            "weight_total": np.memmap(
                tmpdir / "weight_total.dat", dtype=np.float32, mode="r+",
                shape=shape),
            "incidence_sum": np.memmap(
                tmpdir / "incidence_sum.dat", dtype=np.float32, mode="r+",
                shape=shape),
            "emission_sum": np.memmap(
                tmpdir / "emission_sum.dat", dtype=np.float32, mode="r+",
                shape=shape),
            "phase_sum": np.memmap(
                tmpdir / "phase_sum.dat", dtype=np.float32, mode="r+",
                shape=shape),
            "n_obs": np.memmap(
                tmpdir / "n_obs.dat", dtype=np.int32, mode="r+",
                shape=shape),
        }
    else:
        remaining = npz_paths
        completed = set()
        print(f"\n=== Mosaicking {len(remaining)} strips (single-pass) ===")
        accum = init_accumulation_arrays(tmpdir)

    total_pixels = 0
    for i, npz_path in enumerate(tqdm(remaining, desc="Scattering", ncols=80)):
        try:
            n_px = scatter_strip(npz_path, accum)
            total_pixels += n_px
            completed.add(npz_path.stem)
        except Exception as e:
            print(f"\n  ERROR scattering {npz_path.name}: {e}")
            continue

        # Checkpoint every 50 strips
        if (i + 1) % 50 == 0:
            # Flush mmap
            for arr in accum.values():
                if hasattr(arr, "flush"):
                    arr.flush()
            with open(checkpoint_path, "w") as f:
                json.dump({"completed": list(completed),
                           "total_pixels": total_pixels}, f)

    # Final flush
    for arr in accum.values():
        if hasattr(arr, "flush"):
            arr.flush()

    print(f"\n  Total pixels scattered: {total_pixels:,}")
    coverage = (accum["n_obs"] > 0).sum() / accum["n_obs"].size * 100
    print(f"  Grid coverage: {coverage:.1f}%")

    # Finalize
    finalize_and_write(accum)

    # Cleanup temp files
    print("  Cleaning up temp files...")
    for f in tmpdir.glob("*.dat"):
        f.unlink()
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def verify_mosaics() -> None:
    """Quick verification of output GeoTIFFs."""
    print("\n=== Verifying M3 Mosaics ===")

    for name, path in {**M3_ALIGNED_FILES, **M3_GEOMETRY_FILES}.items():
        if not path.exists():
            print(f"  MISSING: {path.name}")
            continue
        with rasterio.open(path) as src:
            print(f"  {path.name}: {src.width}x{src.height}, "
                  f"CRS={src.crs is not None}")
            # Read a small sample
            win = rasterio.windows.Window(
                GLOBAL_WIDTH // 4, GLOBAL_HEIGHT // 4, 256, 256)
            data = src.read(1, window=win)
            finite = data[np.isfinite(data)]
            if len(finite) > 0:
                print(f"    Sample: [{finite.min():.4f}, {finite.max():.4f}], "
                      f"mean={finite.mean():.4f}")
            else:
                print(f"    Sample: all NaN")


def main():
    parser = argparse.ArgumentParser(
        description="Mosaic M3 strips into 128 ppd GeoTIFFs"
    )
    parser.add_argument("--test", type=int, default=None,
                        help="Process only N strips for testing")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only create compact .npz, don't mosaic")
    parser.add_argument("--mosaic-only", action="store_true",
                        help="Skip preprocessing, mosaic from existing .npz")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted mosaicking")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    if args.verify_only:
        verify_mosaics()
        return

    # Skip the entire (multi-hour) mosaic if all 12 output GeoTIFFs already exist.
    # mosaic_all() deletes its checkpoint on completion, so --resume alone would
    # otherwise re-scatter from scratch on a re-run. This makes step10 idempotent.
    if args.skip_existing and not args.test:
        all_out = {**M3_ALIGNED_FILES, **M3_GEOMETRY_FILES}
        if all(p.exists() and p.stat().st_size > 0 for p in all_out.values()):
            print(f"=== All {len(all_out)} M3 mosaic outputs exist — skipping "
                  f"(use --no-skip-existing to force) ===")
            verify_mosaics()
            return

    # Find strip files
    if not args.mosaic_only:
        print("=== Finding M3 strip files ===")
        if M3_CATALOG_PATH.exists():
            pairs = find_strip_files_from_catalog(
                M3_CATALOG_PATH, M3_L2_DIR, M3_L1B_DIR)
        else:
            pairs = find_strip_files(M3_L2_DIR, M3_L1B_DIR)
        print(f"  Found {len(pairs)} paired strips")

        if not pairs:
            print("  No strips found. Run step09_m3_download.py first.")
            sys.exit(1)

        if args.test:
            pairs = pairs[:args.test]
            print(f"  Test mode: {args.test} strips")

        # Preprocess
        npz_paths = preprocess_all_strips(
            pairs, M3_BAND_INDICES, skip_existing=args.skip_existing)
    else:
        # Load existing .npz files
        npz_paths = sorted(M3_COMPACT_DIR.glob("*.npz"))
        if args.test:
            npz_paths = npz_paths[:args.test]
        print(f"  Found {len(npz_paths)} preprocessed strips")

    if args.preprocess_only:
        print("\n=== Preprocessing complete (--preprocess-only) ===")
        return

    if not npz_paths:
        print("  No preprocessed strips found.")
        sys.exit(1)

    # Mosaic
    mosaic_all(npz_paths, resume=args.resume)

    # Verify
    verify_mosaics()

    print("\n=== Mosaicking complete ===")


if __name__ == "__main__":
    main()

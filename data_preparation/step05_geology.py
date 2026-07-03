"""Rasterize USGS Geologic Map and add per-patch geologic metadata.

Downloads the USGS Unified Geologic Map of the Moon (if not already present),
rasterizes it to the 128 ppd grid, and adds geologic metadata to the HDF5 dataset.

Usage:
    python step05_geology.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import mapping
from tqdm import tqdm

from config import (
    ALIGNED_DIR,
    ALIGNED_FILES,
    CHANNELS,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    HDF5_PATH,
    LUNAR_CRS_PROJ4,
    LUNAR_RADIUS_M,
    N_PATCHES_X,
    N_PATCHES_Y,
    OUTPUT_DIR,
    PATCH_SIZE,
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

# Mare unit prefixes from USGS geologic map
MARE_UNIT_PREFIXES = [
    "Em",   # Eratosthenian mare
    "Im",   # Imbrian mare
    "lIm",  # Late Imbrian mare
    "Cm",   # Copernican mare (rare)
]

# Geologic age categories
AGE_MAP = {
    "C": 0,    # Copernican
    "E": 1,    # Eratosthenian
    "I": 2,    # Imbrian (upper)
    "lI": 3,   # Late Imbrian
    "eI": 4,   # Early Imbrian
    "N": 5,    # Nectarian
    "pN": 6,   # Pre-Nectarian
}


def find_shapefile() -> Path:
    """Find the USGS geologic map shapefile (GeoUnits, not contacts)."""
    geo_dir = RAW_DIR / "geologic_map"
    if geo_dir.exists():
        shps = list(geo_dir.rglob("*.shp"))
        # Prioritize GeoUnits shapefile (polygon units, not line contacts)
        for shp in shps:
            name_lower = shp.stem.lower()
            if "geounits" in name_lower or "geo_units" in name_lower:
                return shp
        # Fallback: look for "unit" but exclude "contact"
        for shp in shps:
            name_lower = shp.stem.lower()
            if "unit" in name_lower and "contact" not in name_lower:
                return shp
        if shps:
            return shps[0]
    raise FileNotFoundError(
        f"Geologic map shapefile not found in {geo_dir}. "
        "Run step01_download.py --tier 1 first."
    )


def rasterize_geologic_map(shapefile_path: Path, output_path: Path,
                           skip_existing: bool = True) -> tuple:
    """Rasterize geologic unit polygons to 128 ppd grid.

    Returns (output_path, unit_mapping) where unit_mapping maps
    unit_code (str) -> integer ID.
    """
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        mapping_path = output_path.with_suffix(".json")
        if mapping_path.exists():
            with open(mapping_path) as f:
                unit_mapping = json.load(f)
            print(f"  Skipped (exists): {output_path.name}")
            return output_path, unit_mapping

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Rasterizing Geologic Map ---")
    print(f"  Source: {shapefile_path}")

    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")

    # Find the geologic unit column. The USGS Unified Geologic Map of the Moon v2
    # shapefile uses 'FIRST_Unit' for the unit symbol.
    unit_col = None
    for col in ["GEO_UNIT", "UNIT", "Unit", "unit", "UnitSymbol", "UNITSYMBOL",
                "MapUnit", "FIRST_Unit"]:
        if col in gdf.columns:
            unit_col = col
            break
    if unit_col is None:
        # Try first non-geometry string-like column (object OR pandas string dtype —
        # newer geopandas reads strings as 'string[python]', not 'object').
        import pandas.api.types as ptypes
        for col in gdf.columns:
            if col != "geometry" and (gdf[col].dtype == object
                                      or ptypes.is_string_dtype(gdf[col])):
                unit_col = col
                print(f"  Using column '{col}' as unit identifier")
                break
    if unit_col is None:
        raise ValueError(f"Cannot find geologic unit column. Columns: {list(gdf.columns)}")

    print(f"  Unit column: {unit_col}")
    units = sorted(gdf[unit_col].dropna().unique())
    print(f"  Unique units: {len(units)}")
    for u in units[:10]:
        print(f"    {u}")
    if len(units) > 10:
        print(f"    ... and {len(units) - 10} more")

    # Create unit -> integer mapping (0 = nodata)
    unit_mapping = {str(u): i + 1 for i, u in enumerate(units)}

    # Handle longitude convention
    # Check bounds to determine if -180..180 or 0..360
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"  Bounds: {bounds}")

    # Check if shapefile is already in projected meters or in geographic degrees
    import math
    is_projected = abs(bounds[2]) > 1000  # if max x > 1000, it's in meters not degrees

    if is_projected:
        print("  Data already in projected meters (equirectangular)")
        # Check if centered at 0 or 180
        if bounds[0] >= 0 and bounds[2] > math.pi * LUNAR_RADIUS_M:
            # 0..360 convention in meters, shift to -180..180
            print("  Shifting from 0..360° meters to -180..180° meters")
            x_full = 2 * math.pi * LUNAR_RADIUS_M
            def shift_geom(geom):
                from shapely.ops import transform as shapely_transform
                def shift_x(x, y, z=None):
                    x_arr = np.asarray(x, dtype=np.float64)
                    x_shifted = np.where(x_arr > math.pi * LUNAR_RADIUS_M,
                                         x_arr - x_full, x_arr)
                    return (x_shifted, y) if z is None else (x_shifted, y, z)
                return shapely_transform(shift_x, geom)
            gdf.geometry = gdf.geometry.apply(shift_geom)
        else:
            print("  Already centered at 0° (x range matches our grid)")
    else:
        # Geographic degrees — need to convert to projected meters
        print("  Converting from geographic degrees to projected meters")
        deg_to_m = math.pi / 180.0 * LUNAR_RADIUS_M

        # First check if 0..360 and shift to -180..180
        if bounds[2] > 180:
            print("  Shifting from 0..360° to -180..180°")
            def shift_geom(geom):
                from shapely.ops import transform as shapely_transform
                def shift_lon(x, y, z=None):
                    x_arr = np.asarray(x, dtype=np.float64)
                    x_shifted = np.where(x_arr > 180, x_arr - 360, x_arr)
                    return (x_shifted, y) if z is None else (x_shifted, y, z)
                return shapely_transform(shift_lon, geom)
            gdf.geometry = gdf.geometry.apply(shift_geom)

        def geo_to_proj(geom):
            from shapely.ops import transform as shapely_transform
            def to_meters(x, y, z=None):
                return (np.asarray(x) * deg_to_m, np.asarray(y) * deg_to_m) if z is None else (np.asarray(x) * deg_to_m, np.asarray(y) * deg_to_m, z)
            return shapely_transform(to_meters, geom)

        gdf.geometry = gdf.geometry.apply(geo_to_proj)

    # Create shapes for rasterization
    shapes = []
    for _, row in gdf.iterrows():
        unit = str(row[unit_col])
        if unit in unit_mapping:
            shapes.append((row.geometry, unit_mapping[unit]))

    print(f"  Rasterizing {len(shapes)} polygons to {GLOBAL_WIDTH}x{GLOBAL_HEIGHT}...")

    # Rasterize in strips to manage memory
    strip_h = 2048
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": GLOBAL_WIDTH,
        "height": GLOBAL_HEIGHT,
        "count": 1,
        "crs": LUNAR_CRS,
        "transform": TARGET_TRANSFORM,
        "compress": "lzw",
        "bigtiff": "yes",
        "nodata": 0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for row_start in tqdm(range(0, GLOBAL_HEIGHT, strip_h), desc="Rasterizing"):
            h = min(strip_h, GLOBAL_HEIGHT - row_start)
            window = rasterio.windows.Window(0, row_start, GLOBAL_WIDTH, h)
            win_transform = rasterio.windows.transform(window, TARGET_TRANSFORM)

            strip = rasterize(
                shapes,
                out_shape=(h, GLOBAL_WIDTH),
                transform=win_transform,
                fill=0,
                dtype="int16",
            )
            dst.write(strip.astype(np.int16), 1, window=window)

    # Save unit mapping
    mapping_path = output_path.with_suffix(".json")
    with open(mapping_path, "w") as f:
        json.dump(unit_mapping, f, indent=2)

    print(f"  Done: {output_path.name}")
    print(f"  Mapping saved: {mapping_path.name}")
    return output_path, unit_mapping


def compute_mare_mask(geo_raster_path: Path, unit_mapping: dict,
                      output_path: Path, skip_existing: bool = True) -> Path:
    """Create binary mare/highlands mask from geologic raster."""
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skipped (exists): {output_path.name}")
        return output_path

    print(f"\n--- Computing Mare Mask ---")

    mare_ids = set()
    for unit_name, unit_id in unit_mapping.items():
        for prefix in MARE_UNIT_PREFIXES:
            if unit_name.startswith(prefix):
                mare_ids.add(unit_id)
                break

    print(f"  Mare unit IDs: {mare_ids}")

    with rasterio.open(geo_raster_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=255)

        with rasterio.open(output_path, "w", **profile) as dst:
            strip_h = 2048
            for row_start in range(0, src.height, strip_h):
                h = min(strip_h, src.height - row_start)
                window = rasterio.windows.Window(0, row_start, src.width, h)
                data = src.read(1, window=window)
                mare = np.isin(data, list(mare_ids)).astype(np.uint8)
                mare[data == 0] = 255  # nodata
                dst.write(mare, 1, window=window)

    # Compute global mare fraction
    with rasterio.open(output_path) as src:
        total = 0
        mare_count = 0
        for row_start in range(0, src.height, 2048):
            h = min(2048, src.height - row_start)
            window = rasterio.windows.Window(0, row_start, src.width, h)
            data = src.read(1, window=window)
            valid = data != 255
            total += valid.sum()
            mare_count += (data[valid] == 1).sum()

        if total > 0:
            print(f"  Global mare fraction: {mare_count / total * 100:.1f}%")

    print(f"  Done: {output_path.name}")
    return output_path


def add_geology_to_hdf5(hdf5_path: Path, geo_raster_path: Path,
                        mare_mask_path: Path, unit_mapping: dict) -> None:
    """Add per-patch geologic metadata to HDF5 dataset."""
    print(f"\n--- Adding Geology to HDF5 ---")

    # Invert mapping: id -> unit_name
    id_to_unit = {v: k for k, v in unit_mapping.items()}

    with h5py.File(hdf5_path, "a") as hf:
        meta = hf["metadata"]
        n = hf["patches"].shape[0]

        # Create geology group if not exists
        if "geology" in meta:
            del meta["geology"]
        geo = meta.create_group("geology")

        mare_frac_ds = geo.create_dataset("mare_fraction", shape=(n,), dtype=np.float32)
        n_units_ds = geo.create_dataset("n_geo_units", shape=(n,), dtype=np.int16)
        dominant_ds = geo.create_dataset("dominant_unit_id", shape=(n,), dtype=np.int16)

        # Variable-length string for dominant unit name
        dt_str = h5py.special_dtype(vlen=str)
        dominant_name_ds = geo.create_dataset("dominant_unit_name", shape=(n,), dtype=dt_str)

        row_indices = meta["row_idx"][:]
        col_indices = meta["col_idx"][:]

        with rasterio.open(geo_raster_path) as geo_src, \
             rasterio.open(mare_mask_path) as mare_src:

            for idx in tqdm(range(n), desc="Geology metadata", ncols=80):
                ry = row_indices[idx]
                rx = col_indices[idx]
                window = rasterio.windows.Window(
                    rx * PATCH_SIZE, ry * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
                )

                # Geologic units
                geo_data = geo_src.read(1, window=window)
                valid = geo_data[geo_data > 0]
                if len(valid) > 0:
                    counter = Counter(valid)
                    dominant_id = counter.most_common(1)[0][0]
                    n_units = len(counter)
                    dominant_name = id_to_unit.get(dominant_id, "unknown")
                else:
                    dominant_id = 0
                    n_units = 0
                    dominant_name = "none"

                # Mare fraction
                mare_data = mare_src.read(1, window=window)
                valid_mare = mare_data[mare_data != 255]
                if len(valid_mare) > 0:
                    mare_frac = (valid_mare == 1).sum() / len(valid_mare)
                else:
                    mare_frac = np.nan

                mare_frac_ds[idx] = mare_frac
                n_units_ds[idx] = n_units
                dominant_ds[idx] = dominant_id
                dominant_name_ds[idx] = dominant_name

        # Store unit mapping as attribute
        geo.attrs["unit_mapping"] = json.dumps(unit_mapping)

    print(f"  Done: geology metadata added to {hdf5_path.name}")


def verify_geology(hdf5_path: Path) -> None:
    """Print geology summary from HDF5."""
    print(f"\n=== Geology Summary ===")
    with h5py.File(hdf5_path, "r") as hf:
        geo = hf["metadata"]["geology"]
        n = hf["patches"].shape[0]

        mare_frac = geo["mare_fraction"][:]
        print(f"  Mean mare fraction: {np.nanmean(mare_frac):.3f}")
        print(f"  Patches with >50% mare: {(mare_frac > 0.5).sum()}")
        print(f"  Patches with >50% highlands: {(mare_frac < 0.5).sum()}")

        n_units = geo["n_geo_units"][:]
        print(f"  Mean geo units per patch: {n_units.mean():.1f}")

        # Top dominant units
        names = [geo["dominant_unit_name"][i].decode() if isinstance(geo["dominant_unit_name"][i], bytes) else geo["dominant_unit_name"][i]
                 for i in range(n)]
        counter = Counter(names)
        print(f"  Top 10 dominant units:")
        for unit, count in counter.most_common(10):
            print(f"    {unit}: {count} patches ({count/n*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Geologic map rasterization and metadata")
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_geology(args.hdf5)
        return

    # Find shapefile
    try:
        shp_path = find_shapefile()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Rasterize geologic map
    geo_raster = ALIGNED_FILES["geologic_units"]
    geo_raster, unit_mapping = rasterize_geologic_map(
        shp_path, geo_raster, args.skip_existing
    )

    # Compute mare mask
    mare_mask = ALIGNED_FILES["mare_mask"]
    compute_mare_mask(geo_raster, unit_mapping, mare_mask, args.skip_existing)

    # Add to HDF5 if it exists
    if args.hdf5.exists():
        add_geology_to_hdf5(args.hdf5, geo_raster, mare_mask, unit_mapping)
        verify_geology(args.hdf5)
    else:
        print(f"\n  HDF5 not found ({args.hdf5}). Run step04_tile.py first.")
        print("  Rasterized layers saved — geology metadata will be added later.")

    print("\n=== Geology processing complete ===")


if __name__ == "__main__":
    main()

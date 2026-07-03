"""PyTorch dataset for Lunar Foundation Model pretraining.

Two modes:
1. HDF5Dataset: Reads from pre-tiled HDF5 (fixed 16,200 patches)
2. GeoTIFFRandomCropDataset: Reads random crops from aligned GeoTIFFs
   (unlimited training samples, configurable per-epoch count)

Usage:
    from lunar_dataset import GeoTIFFRandomCropDataset, HDF5Dataset

    # Random crops from GeoTIFFs (for pretraining)
    ds = GeoTIFFRandomCropDataset(
        channel_names=CHANNELS_V4,
        crop_size=256,
        epoch_length=200_000,
        min_valid_fraction=0.5,
    )

    # Fixed HDF5 patches (for evaluation / fine-tuning)
    ds = HDF5Dataset(hdf5_path, split="train")
"""

import math
import random
from pathlib import Path

import numpy as np
import rasterio

# torch is only needed for the Dataset classes (training/eval). The dataset-BUILD
# pipeline (fix_minirf, step15, compute_stats) only imports get_all_channel_paths,
# which uses no torch. Make torch optional so the CPU-only build env works without it.
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None

    class Dataset:  # minimal stand-in so class definitions below still parse
        pass

from config import (
    ALIGNED_DIR,
    ALIGNED_FILES,
    CHANNELS_V4,
    DERIVED_FILES,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    M3_ALIGNED_FILES,
    MODALITY_GROUPS,
    MODALITY_GROUP_NAMES,
    NEW_ALIGNED_FILES,
    PATCH_SIZE,
    PPD,
)


def build_channel_to_group_indices(channel_names: list[str] = None) -> list[list[int]]:
    """Build list of channel index lists, one per modality group.

    Returns a list of length n_groups, where each element is a list of
    channel indices (into channel_names) belonging to that group.
    Groups follow MODALITY_GROUP_NAMES ordering.
    """
    if channel_names is None:
        channel_names = CHANNELS_V4
    ch_to_idx = {name: i for i, name in enumerate(channel_names)}
    result = []
    for group_name in MODALITY_GROUP_NAMES:
        group_info = MODALITY_GROUPS[group_name]
        indices = [ch_to_idx[ch] for ch in group_info["channels"] if ch in ch_to_idx]
        result.append(indices)
    return result


def get_all_channel_paths(channel_names: list[str] = None) -> dict[str, Path]:
    """Return channel_name -> GeoTIFF path for all available channels."""
    if channel_names is None:
        channel_names = CHANNELS_V4

    # Map channel names to file paths
    path_map = {
        "wac_morphology": ALIGNED_FILES["wac_morphology"],
        "elevation": ALIGNED_FILES["elevation_blended"],
        "slope": DERIVED_FILES["slope"],
        "roughness": DERIVED_FILES["roughness"],
        "diviner_tbol_midnight": ALIGNED_FILES["diviner_tbol_midnight"],
        "diviner_temp_night": ALIGNED_FILES["diviner_temp_night"],
        "rock_abundance": ALIGNED_FILES["rock_abundance"],
        "christiansen_feature": ALIGNED_FILES["christiansen_feature"],
    }
    # M3 bands
    path_map.update(M3_ALIGNED_FILES)
    # New channels
    path_map.update(NEW_ALIGNED_FILES)

    return {name: path_map[name] for name in channel_names if name in path_map}


class GeoTIFFRandomCropDataset(Dataset):
    """Random-crop dataset reading directly from aligned GeoTIFFs.

    Each sample is a random 256x256 crop from the global 46080x23040 grid.
    Crops are filtered to have >= min_valid_fraction of valid pixels in
    at least the first channel (WAC morphology).

    The dataset uses rasterio windowed reads for memory efficiency —
    only the crop window is read from disk, not the full image.
    """

    def __init__(
        self,
        channel_names: list[str] = None,
        crop_size: int = 256,
        epoch_length: int = 200_000,
        min_valid_fraction: float = 0.3,
        lat_range: tuple[float, float] = (-70.0, 70.0),
        normalize: bool = False,
        channel_stats: dict = None,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.epoch_length = epoch_length
        self.min_valid_fraction = min_valid_fraction
        self.normalize = normalize
        self.channel_stats = channel_stats or {}

        if channel_names is None:
            channel_names = CHANNELS_V4
        self.channel_names = channel_names
        self.n_channels = len(channel_names)

        # Convert lat range to row range
        # Row 0 = +90°, row GLOBAL_HEIGHT = -90°
        self.row_min = int((90.0 - lat_range[1]) * PPD)  # top row (N boundary)
        self.row_max = int((90.0 - lat_range[0]) * PPD) - crop_size  # bottom row
        self.col_max = GLOBAL_WIDTH - crop_size

        # Resolve channel paths
        self.channel_paths = get_all_channel_paths(channel_names)
        self.available_channels = list(self.channel_paths.keys())

        # Open rasterio file handles (lazy, kept open for windowed reads)
        self._datasets = {}
        for name, path in self.channel_paths.items():
            if path.exists():
                self._datasets[name] = rasterio.open(path)

        # Build group index mapping
        self._group_ch_indices = build_channel_to_group_indices(channel_names)
        self.n_groups = len(self._group_ch_indices)

        print(f"GeoTIFFRandomCropDataset: {self.n_channels} channels, "
              f"{len(self._datasets)} available, "
              f"crop={crop_size}, epoch={epoch_length}, "
              f"rows=[{self.row_min}, {self.row_max}], groups={self.n_groups}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        """Return a random crop as (C, H, W) float32 tensor."""
        # Retry loop to find valid crops
        for _ in range(20):
            row = random.randint(self.row_min, self.row_max)
            col = random.randint(0, self.col_max)

            window = rasterio.windows.Window(col, row, self.crop_size, self.crop_size)

            # Quick check: read first channel to test validity
            first_ch = self.channel_names[0]
            if first_ch in self._datasets:
                data = self._datasets[first_ch].read(1, window=window)
                valid_frac = np.isfinite(data).sum() / data.size
                if valid_frac < self.min_valid_fraction:
                    continue

            # Read all channels
            crop = np.full((self.n_channels, self.crop_size, self.crop_size),
                           np.nan, dtype=np.float32)

            for i, ch_name in enumerate(self.channel_names):
                if ch_name in self._datasets:
                    crop[i] = self._datasets[ch_name].read(1, window=window)

            # Replace NaN with 0 for the tensor
            np.nan_to_num(crop, copy=False, nan=0.0)

            # Create valid mask (per-channel)
            has_data = np.zeros(self.n_channels, dtype=np.float32)
            for i, ch_name in enumerate(self.channel_names):
                if ch_name in self._datasets:
                    raw = self._datasets[ch_name].read(1, window=window)
                    has_data[i] = float(np.isfinite(raw).any())

            if self.normalize and self.channel_stats:
                for i, ch_name in enumerate(self.channel_names):
                    if ch_name in self.channel_stats:
                        mean, std = self.channel_stats[ch_name]
                        if std > 0:
                            crop[i] = (crop[i] - mean) / std

            # Compute has_group
            has_group = np.zeros(self.n_groups, dtype=np.float32)
            for gi, ch_indices in enumerate(self._group_ch_indices):
                has_group[gi] = float(any(has_data[ci] > 0 for ci in ch_indices))

            return {
                "pixels": torch.from_numpy(crop),
                "has_channel": torch.from_numpy(has_data),
                "has_group": torch.from_numpy(has_group),
            }

        # Fallback: return zeros if no valid crop found after retries
        crop = torch.zeros(self.n_channels, self.crop_size, self.crop_size)
        has_data = torch.zeros(self.n_channels)
        has_group = torch.zeros(self.n_groups)
        return {"pixels": crop, "has_channel": has_data, "has_group": has_group}

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()

    def __del__(self):
        self.close()


class MmapRandomCropDataset(Dataset):
    """Fastest dataset: random crops from memory-mapped numpy arrays.

    Each channel is stored as a raw float32 memmap file (4.24 GB).
    Random crops are pure numpy slicing — zero decompression, zero I/O
    overhead beyond page faults (handled by OS page cache).

    Data is pre-normalized during mmap creation (step15_build_mmap.py
    with --normalize), so no per-sample normalization needed.

    ~10-100x faster than HDF5, ~5-10x faster than compressed GeoTIFF reads.
    """

    def __init__(
        self,
        mmap_dir: str | Path = None,
        channel_names: list[str] = None,
        crop_size: int = 256,
        epoch_length: int = 200_000,
        min_valid_fraction: float = 0.3,
        lat_range: tuple[float, float] = (-70.0, 70.0),
    ):
        super().__init__()
        from config import PPD

        if mmap_dir is None:
            mmap_dir = Path(__file__).parent / "data" / "mmap"
        self.mmap_dir = Path(mmap_dir)
        self.crop_size = crop_size
        self.epoch_length = epoch_length
        self.min_valid_fraction = min_valid_fraction

        if channel_names is None:
            channel_names = CHANNELS_V4
        self.channel_names = channel_names
        self.n_channels = len(channel_names)

        # Load index
        index_path = self.mmap_dir / "channel_index.json"
        import json
        with open(index_path) as f:
            index = json.load(f)
        self.shape = tuple(index["shape"])

        # Row/col bounds from lat range
        self.row_min = int((90.0 - lat_range[1]) * PPD)
        self.row_max = int((90.0 - lat_range[0]) * PPD) - crop_size
        self.col_max = self.shape[1] - crop_size

        # Open memmap file handles
        self._mmaps = {}
        self._masks = {}
        for name in channel_names:
            fpath = self.mmap_dir / f"{name}.npy"
            if fpath.exists():
                self._mmaps[name] = np.memmap(
                    fpath, dtype=np.float32, mode="r", shape=self.shape)
                # Load packed mask if available
                mask_path = self.mmap_dir / f"{name}_mask.npy"
                if mask_path.exists():
                    self._masks[name] = np.load(mask_path)

        # Build group index mapping for has_group computation
        self._group_ch_indices = build_channel_to_group_indices(channel_names)
        self.n_groups = len(self._group_ch_indices)

        print(f"MmapRandomCropDataset: {self.n_channels} channels, "
              f"{len(self._mmaps)} available, crop={crop_size}, "
              f"epoch={epoch_length}, groups={self.n_groups}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        """Return a random crop as (C, H, W) float32 tensor."""
        for _ in range(20):
            row = random.randint(self.row_min, self.row_max)
            col = random.randint(0, self.col_max)

            # Quick validity check on first channel
            first_name = self.channel_names[0]
            if first_name in self._mmaps:
                patch = self._mmaps[first_name][row:row+self.crop_size,
                                                 col:col+self.crop_size]
                # Zeros indicate nodata in mmap (NaN → 0 during conversion)
                nonzero_frac = (patch != 0).sum() / patch.size
                if nonzero_frac < self.min_valid_fraction:
                    continue

            # Read all channels (pure numpy slicing, near-instant)
            crop = np.zeros((self.n_channels, self.crop_size, self.crop_size),
                            dtype=np.float32)
            has_data = np.zeros(self.n_channels, dtype=np.float32)

            for i, name in enumerate(self.channel_names):
                if name in self._mmaps:
                    crop[i] = self._mmaps[name][row:row+self.crop_size,
                                                 col:col+self.crop_size]
                    has_data[i] = float((crop[i] != 0).any())

            # Compute has_group: group has data if ANY channel in it has data
            has_group = np.zeros(self.n_groups, dtype=np.float32)
            for gi, ch_indices in enumerate(self._group_ch_indices):
                has_group[gi] = float(any(has_data[ci] > 0 for ci in ch_indices))

            return {
                "pixels": torch.from_numpy(crop.copy()),
                "has_channel": torch.from_numpy(has_data),
                "has_group": torch.from_numpy(has_group),
            }

        # Fallback
        return {
            "pixels": torch.zeros(self.n_channels, self.crop_size, self.crop_size),
            "has_channel": torch.zeros(self.n_channels),
            "has_group": torch.zeros(self.n_groups),
        }

    def close(self):
        self._mmaps.clear()
        self._masks.clear()

    def __del__(self):
        self.close()


class HDF5Dataset(Dataset):
    """Dataset reading from pre-tiled HDF5 patches.

    Supports split filtering and optional normalization.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = None,
        normalize: bool = False,
        channel_stats: dict = None,
    ):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.normalize = normalize
        self.channel_stats = channel_stats or {}

        self.hf = h5py.File(self.hdf5_path, "r")
        self.patches = self.hf["patches"]
        self.n_total = self.patches.shape[0]
        self.n_channels = self.patches.shape[1]

        # Get channel names
        if "channel_names" in self.hf.attrs:
            self.channel_names = list(self.hf.attrs["channel_names"])
        else:
            self.channel_names = [f"ch_{i}" for i in range(self.n_channels)]

        # Filter by split
        if split is not None and "split" in self.hf["metadata"]:
            split_data = self.hf["metadata/split"][:]
            split_names = list(self.hf["metadata"].attrs.get("split_names", []))
            if split in split_names:
                split_idx = split_names.index(split)
                self.indices = np.where(split_data == split_idx)[0]
            else:
                raise ValueError(f"Unknown split '{split}'. Available: {split_names}")
        else:
            self.indices = np.arange(self.n_total)

        # has_channel mask
        self.has_channel = self.hf["metadata/has_channel"][:]

        print(f"HDF5Dataset: {len(self.indices)} patches "
              f"({self.n_channels} channels, split={split})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        crop = self.patches[real_idx].astype(np.float32)
        has_data = self.has_channel[real_idx].astype(np.float32)

        np.nan_to_num(crop, copy=False, nan=0.0)

        if self.normalize and self.channel_stats:
            for i, ch_name in enumerate(self.channel_names):
                if ch_name in self.channel_stats:
                    mean, std = self.channel_stats[ch_name]
                    if std > 0:
                        crop[i] = (crop[i] - mean) / std

        return {
            "pixels": torch.from_numpy(crop),
            "has_channel": torch.from_numpy(has_data),
        }

    def close(self):
        self.hf.close()

    def __del__(self):
        self.close()


# Need h5py import at module level for HDF5Dataset
import h5py


def compute_channel_stats(hdf5_path: str | Path, n_samples: int = 1000) -> dict:
    """Compute per-channel mean and std from HDF5 for normalization."""
    with h5py.File(hdf5_path, "r") as hf:
        patches = hf["patches"]
        n = patches.shape[0]
        n_ch = patches.shape[1]
        channel_names = list(hf.attrs.get("channel_names", []))

        indices = np.random.choice(n, min(n_samples, n), replace=False)
        indices.sort()

        stats = {}
        for ch_idx in range(n_ch):
            vals = []
            for i in indices:
                data = patches[i, ch_idx]
                valid = data[np.isfinite(data)]
                if len(valid) > 0:
                    vals.append(valid)

            if vals:
                all_vals = np.concatenate(vals)
                mean = float(np.mean(all_vals))
                std = float(np.std(all_vals))
            else:
                mean, std = 0.0, 1.0

            ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch_{ch_idx}"
            stats[ch_name] = (mean, std)
            print(f"  {ch_name:25s}: mean={mean:10.4f}, std={std:10.4f}")

        return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["geotiff", "hdf5", "stats"], default="geotiff")
    parser.add_argument("--hdf5", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "geotiff":
        ds = GeoTIFFRandomCropDataset(epoch_length=args.n_samples)
        for i in range(min(args.n_samples, len(ds))):
            sample = ds[i]
            pixels = sample["pixels"]
            has_ch = sample["has_channel"]
            print(f"Sample {i}: shape={pixels.shape}, "
                  f"channels_available={int(has_ch.sum())}/{len(has_ch)}, "
                  f"range=[{pixels.min():.4f}, {pixels.max():.4f}]")
        ds.close()

    elif args.mode == "hdf5":
        from config import HDF5_V4_PATH, HDF5_V2_PATH
        hdf5_path = Path(args.hdf5) if args.hdf5 else HDF5_V4_PATH
        if not hdf5_path.exists():
            hdf5_path = HDF5_V2_PATH
        ds = HDF5Dataset(hdf5_path)
        for i in range(min(args.n_samples, len(ds))):
            sample = ds[i]
            print(f"Sample {i}: shape={sample['pixels'].shape}, "
                  f"range=[{sample['pixels'].min():.4f}, {sample['pixels'].max():.4f}]")
        ds.close()

    elif args.mode == "stats":
        from config import HDF5_V4_PATH, HDF5_V2_PATH
        hdf5_path = Path(args.hdf5) if args.hdf5 else HDF5_V4_PATH
        if not hdf5_path.exists():
            hdf5_path = HDF5_V2_PATH
        print(f"Computing stats from {hdf5_path}")
        stats = compute_channel_stats(hdf5_path)

"""Mmap-backed dataset for downstream tasks.

Reads fixed patches from the global mmap arrays using row/col indices
from the V4 HDF5 metadata. Much faster than HDF5 random reads.

Labels (geology, mare, craters, etc.) are loaded from metadata once
at init time and served from memory.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from config import (
    CHANNELS_V4, MODALITY_GROUPS, MODALITY_GROUP_NAMES,
    PATCH_SIZE, PPD, GLOBAL_HEIGHT, GLOBAL_WIDTH,
    ALIGNED_DIR,
)
from lunar_dataset import build_channel_to_group_indices


class MmapPatchDataset(Dataset):
    """Fixed-patch dataset backed by global mmap arrays.

    Patches are defined by (row_idx, col_idx) from V4 HDF5 metadata.
    Pixel data is read from per-channel .npy mmap files (no HDF5 at runtime).
    Labels are loaded at init and stored in memory.
    """

    def __init__(
        self,
        h5_path="output/lunar_patches_v4.h5",
        mmap_dir="data/mmap",
        split=0,
        channel_names=None,
        geo_split=False,
    ):
        self.mmap_dir = Path(mmap_dir)
        if channel_names is None:
            channel_names = CHANNELS_V4
        self.channel_names = channel_names
        self.n_channels = len(channel_names)

        # Load metadata from HDF5 (one-time, lightweight)
        with h5py.File(h5_path, "r") as f:
            if geo_split:
                # Paper Table 6 geographic split by latitude band (both hemispheres):
                #   train |lat|<=60, val 60-70, test 70-80. Replaces the stored random split.
                lat = np.abs(np.array(f["metadata"]["center_lat"]))
                if split == 0:
                    sel = lat <= 60
                elif split == 1:
                    sel = (lat > 60) & (lat <= 70)
                else:
                    sel = (lat > 70) & (lat <= 80)
                self._indices = np.where(sel)[0]
            else:
                splits = np.array(f["metadata"]["split"])
                self._indices = np.where(splits == split)[0]
            self.row_idx = np.array(f["metadata"]["row_idx"])[self._indices]
            self.col_idx = np.array(f["metadata"]["col_idx"])[self._indices]
            self.has_channel = np.array(f["metadata"]["has_channel"])[self._indices]

            # Load all geology metadata
            geo = f["metadata"]["geology"]
            self.unit_ids = np.array(geo["dominant_unit_id"])[self._indices]
            self.mare_fraction = np.array(geo["mare_fraction"])[self._indices]

            # Load unit names for age classification
            all_names = geo["dominant_unit_name"]
            self.unit_names = [
                all_names[i].decode() if isinstance(all_names[i], bytes)
                else str(all_names[i])
                for i in self._indices
            ]

        # Open mmap arrays
        index_path = self.mmap_dir / "channel_index.json"
        with open(index_path) as jf:
            index = json.load(jf)
        self.shape = tuple(index["shape"])

        self._mmaps = {}
        for name in channel_names:
            fpath = self.mmap_dir / f"{name}.npy"
            if fpath.exists():
                self._mmaps[name] = np.memmap(
                    fpath, dtype=np.float32, mode="r", shape=self.shape)

        # Group indices
        self._group_ch_indices = build_channel_to_group_indices(channel_names)
        self.n_groups = len(self._group_ch_indices)

        print(f"MmapPatchDataset: split={split}, {len(self._indices)} patches, "
              f"{len(self._mmaps)}/{self.n_channels} channels from mmap")

    def __len__(self):
        return len(self._indices)

    def _read_patch(self, idx):
        """Read a patch from mmap arrays. Returns (pixels, has_data)."""
        r, c = int(self.row_idx[idx]), int(self.col_idx[idx])
        y0 = r * PATCH_SIZE
        x0 = c * PATCH_SIZE

        crop = np.zeros((self.n_channels, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        has_data = np.zeros(self.n_channels, dtype=np.float32)

        for i, ch_name in enumerate(self.channel_names):
            if ch_name in self._mmaps:
                patch = np.array(self._mmaps[ch_name][y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
                valid = np.isfinite(patch) & (patch != 0)
                if valid.any():
                    has_data[i] = 1.0
                patch[~np.isfinite(patch)] = 0.0
                crop[i] = patch

        # has_group
        has_group = np.zeros(self.n_groups, dtype=np.float32)
        for gi, ch_indices in enumerate(self._group_ch_indices):
            has_group[gi] = float(any(has_data[ci] > 0 for ci in ch_indices))

        return crop, has_data, has_group

    def __getitem__(self, idx):
        pixels, has_data, has_group = self._read_patch(idx)
        return {
            "pixels": torch.from_numpy(pixels),
            "has_channel": torch.from_numpy(has_data),
            "has_group": torch.from_numpy(has_group),
        }

    def close(self):
        self._mmaps.clear()


class GeologyMmapDataset(MmapPatchDataset):
    """Geology classification (49-class) from mmap."""

    def __getitem__(self, idx):
        pixels, _, has_group = self._read_patch(idx)
        target = int(self.unit_ids[idx]) - 1  # 0-indexed
        return {
            "pixels": torch.from_numpy(pixels),
            "target": torch.tensor(target, dtype=torch.long),
            "has_group": torch.from_numpy(has_group),
        }


class AgeMmapDataset(MmapPatchDataset):
    """Age classification (5-class) from mmap."""

    AGE_PREFIX_MAP = {"pN": 0, "N": 1, "EI": 2, "I": 2, "E": 3, "C": 4}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert unit names to age classes, filter unknowns
        ages = np.array([self._name_to_age(n) for n in self.unit_names])
        valid = ages >= 0
        # Re-index to valid only
        self._valid_map = np.where(valid)[0]
        self.ages = ages[valid]
        print(f"  AgeMmapDataset: {len(self._valid_map)} valid "
              f"(filtered {(~valid).sum()} unknown)")

    @staticmethod
    def _name_to_age(name):
        for prefix in ["pN", "EI", "N", "I", "E", "C"]:
            if name.startswith(prefix):
                return AgeMmapDataset.AGE_PREFIX_MAP[prefix]
        return -1

    def __len__(self):
        return len(self._valid_map)

    def __getitem__(self, idx):
        real_idx = int(self._valid_map[idx])
        pixels, _, has_group = self._read_patch(real_idx)
        target = int(self.ages[idx])
        return {
            "pixels": torch.from_numpy(pixels),
            "target": torch.tensor(target, dtype=torch.long),
            "has_group": torch.from_numpy(has_group),
        }


class CompositionMmapDataset(MmapPatchDataset):
    """FeO/TiO2 composition prediction from mmap.

    Holds out LP GRS channels, uses spatial mean as regression target.
    """

    TARGET_CHANNELS = ["lpgrs_tio2", "lpgrs_feo"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_indices = [self.channel_names.index(ch) for ch in self.TARGET_CHANNELS]
        # Filter to patches with LP GRS data
        valid = np.ones(len(self._indices), dtype=bool)
        for ch_idx in self.target_indices:
            valid &= self.has_channel[:, ch_idx].astype(bool)
        self._valid_map = np.where(valid)[0]
        print(f"  CompositionMmapDataset: {len(self._valid_map)} with LP GRS data")

    def __len__(self):
        return len(self._valid_map)

    def __getitem__(self, idx):
        real_idx = int(self._valid_map[idx])
        pixels, has_data, has_group = self._read_patch(real_idx)

        # Target: spatial mean of LP GRS channels
        target = np.zeros(len(self.TARGET_CHANNELS), dtype=np.float32)
        for i, ch_idx in enumerate(self.target_indices):
            ch_data = pixels[ch_idx]
            valid = ch_data != 0
            if valid.any():
                target[i] = ch_data[valid].mean()

        # Zero out LP GRS channels from input
        for ch_idx in self.target_indices:
            pixels[ch_idx] = 0.0

        # Update has_group for composition group
        comp_group_idx = MODALITY_GROUP_NAMES.index("composition")
        # Check if Clementine still available
        clem_idx = self.channel_names.index("clementine_uvvis_750nm")
        has_group[comp_group_idx] = float(has_data[clem_idx] > 0)

        return {
            "pixels": torch.from_numpy(pixels),
            "target": torch.from_numpy(target),
            "has_group": torch.from_numpy(has_group),
        }


class SegmentationMmapDataset(MmapPatchDataset):
    """Segmentation dataset that reads masks from a global GeoTIFF."""

    def __init__(self, mask_tif_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with rasterio.open(str(mask_tif_path)) as src:
            self.mask_global = src.read(1)

    def __getitem__(self, idx):
        pixels, _, has_group = self._read_patch(idx)
        r, c = int(self.row_idx[idx]), int(self.col_idx[idx])
        y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
        mask_patch = self.mask_global[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        return {
            "pixels": torch.from_numpy(pixels),
            "target": torch.from_numpy(mask_patch.astype(np.int64)),
            "has_group": torch.from_numpy(has_group),
        }

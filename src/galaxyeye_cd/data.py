from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sample_id: str
    pre_path: Path
    post_path: Path
    mask_path: Path


def resolve_split_dir(path: str | Path) -> Path:
    """Accept either a split root or a parent containing one split subfolder."""
    path = Path(path)
    if (path / "pre-event").is_dir() and (path / "post-event").is_dir() and (path / "target").is_dir():
        return path
    split_name = path.name
    nested = path / split_name
    if (nested / "pre-event").is_dir():
        return nested
    candidates = [p for p in path.iterdir() if p.is_dir() and (p / "pre-event").is_dir()] if path.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve split directory from {path}")


def list_samples(split_dir: str | Path) -> list[Sample]:
    split_dir = resolve_split_dir(split_dir)
    pre_dir = split_dir / "pre-event"
    post_dir = split_dir / "post-event"
    mask_dir = split_dir / "target"
    samples: list[Sample] = []
    for pre_path in sorted(pre_dir.glob("*.tif")):
        post_path = post_dir / pre_path.name
        mask_path = mask_dir / pre_path.name
        if not post_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing post-event or target file for {pre_path.name}")
        samples.append(Sample(pre_path.stem, pre_path, post_path, mask_path))
    if not samples:
        raise RuntimeError(f"No .tif samples found in {split_dir}")
    return samples


def read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def remap_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask >= 2).astype(np.float32)


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        image_size: int | None = 512,
        augment: bool = False,
        positive_crop_prob: float = 0.0,
    ) -> None:
        self.samples = list_samples(split_dir)
        self.image_size = image_size
        self.augment = augment
        self.positive_crop_prob = positive_crop_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        pre = read_tif(sample.pre_path).astype(np.float32) / 255.0
        post = read_tif(sample.post_path).astype(np.float32) / 255.0
        mask = remap_mask(read_tif(sample.mask_path))

        image = np.concatenate([pre, post], axis=-1)
        image, mask = self._crop(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
        return {"image": image_t, "mask": mask_t, "id": sample.sample_id}

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.image_size is None:
            return image, mask
        h, w = mask.shape
        size = self.image_size
        if h < size or w < size:
            raise ValueError(f"Image is {h}x{w}, smaller than configured crop size {size}")
        if self.augment and self.positive_crop_prob > 0 and random.random() < self.positive_crop_prob:
            ys, xs = np.where(mask > 0.5)
            if len(ys) > 0:
                idx = random.randrange(len(ys))
                cy, cx = int(ys[idx]), int(xs[idx])
                y = min(max(cy - random.randint(0, size - 1), 0), h - size)
                x = min(max(cx - random.randint(0, size - 1), 0), w - size)
                return image[y : y + size, x : x + size], mask[y : y + size, x : x + size]
        if self.augment:
            y = random.randint(0, h - size)
            x = random.randint(0, w - size)
        else:
            y = (h - size) // 2
            x = (w - size) // 2
        return image[y : y + size, x : x + size], mask[y : y + size, x : x + size]

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        return image, mask


def estimate_binary_distribution(split_dir: str | Path) -> dict[str, float | int]:
    zeros = 0
    ones = 0
    for sample in list_samples(split_dir):
        mask = remap_mask(read_tif(sample.mask_path))
        ones += int(mask.sum())
        zeros += int(mask.size - mask.sum())
    total = zeros + ones
    return {
        "no_change_pixels": zeros,
        "change_pixels": ones,
        "total_pixels": total,
        "change_fraction": ones / total if total else 0.0,
        "pos_weight": zeros / max(ones, 1),
    }

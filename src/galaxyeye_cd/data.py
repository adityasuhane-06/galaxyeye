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


def scene_id_from_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "scene":
        raise ValueError(f"Could not parse scene id from {name}")
    return parts[1]


def normalize_scene_ids(scenes: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if not scenes:
        return None
    normalized = set()
    for scene in scenes:
        text = str(scene).strip()
        if text.startswith("scene_"):
            text = text.split("_", 1)[1]
        normalized.add(text.zfill(2))
    return normalized


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


def list_samples(split_dir: str | Path, scenes: list[str] | tuple[str, ...] | None = None) -> list[Sample]:
    split_dir = resolve_split_dir(split_dir)
    scene_filter = normalize_scene_ids(scenes)
    pre_dir = split_dir / "pre-event"
    post_dir = split_dir / "post-event"
    mask_dir = split_dir / "target"
    samples: list[Sample] = []
    for pre_path in sorted(pre_dir.glob("*.tif")):
        if scene_filter is not None and scene_id_from_name(pre_path.stem) not in scene_filter:
            continue
        post_path = post_dir / pre_path.name
        mask_path = mask_dir / pre_path.name
        if not post_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing post-event or target file for {pre_path.name}")
        samples.append(Sample(pre_path.stem, pre_path, post_path, mask_path))
    if not samples:
        raise RuntimeError(f"No .tif samples found in {split_dir}")
    return samples


def read_tif(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF file: {path}") from exc
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def remap_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Assignment-mandated binary remap:
    # 0 Background -> 0, 1 Intact -> 0, 2 Damaged -> 1, 3 Destroyed -> 1.
    return (mask >= 2).astype(np.float32)


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        image_size: int | None = 512,
        augment: bool = False,
        positive_crop_prob: float = 0.0,
        scenes: list[str] | tuple[str, ...] | None = None,
        grayscale_prob: float = 0.0,
        sar_speckle_prob: float = 0.0,
        channel_shuffle_prob: float = 0.0,
        brightness_contrast_prob: float = 0.4,
    ) -> None:
        self.samples = list_samples(split_dir, scenes=scenes)
        self.image_size = image_size
        self.augment = augment
        self.positive_crop_prob = positive_crop_prob
        self.grayscale_prob = grayscale_prob
        self.sar_speckle_prob = sar_speckle_prob
        self.channel_shuffle_prob = channel_shuffle_prob
        self.brightness_contrast_prob = brightness_contrast_prob

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

        # Apply ImageNet normalization for EO (first 3 channels), scale SAR (4th channel)
        # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image_t[:3] = (image_t[:3] - mean) / std
        if image_t.shape[0] > 3:
            # Shift the SAR image roughly corresponding to mean=0.5, std=0.5
            image_t[3:] = (image_t[3:] - 0.5) / 0.5

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
        # 1. Geometric Augmentations (applies to both EO and SAR equally)
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

        # 2. EO appearance augmentations for cross-scene generalization.
        if random.random() < self.grayscale_prob:
            gray = image[..., :3].mean(axis=-1, keepdims=True)
            image[..., :3] = np.repeat(gray, 3, axis=-1)

        if random.random() < self.channel_shuffle_prob:
            idx = np.random.permutation(3)
            image[..., :3] = image[..., idx]

        # 3. Gentle EO brightness/contrast jitter, never applied to SAR.
        if random.random() < self.brightness_contrast_prob:
            alpha = random.uniform(0.8, 1.2)  # Contrast multiplier
            beta = random.uniform(-0.1, 0.1)  # Brightness shift
            image_rgb = image[..., :3] * alpha + beta
            image_rgb = np.clip(image_rgb, 0.0, 1.0)
            image[..., :3] = image_rgb

        # 4. SAR multiplicative speckle augmentation.
        if image.shape[-1] > 3 and random.random() < self.sar_speckle_prob:
            speckle = np.random.gamma(4.0, 0.25, image[..., 3:].shape).astype(np.float32)
            image[..., 3:] = np.clip(image[..., 3:] * speckle, 0.0, 1.0)

        return image, mask


def estimate_binary_distribution(
    split_dir: str | Path,
    scenes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float | int]:
    zeros = 0
    ones = 0
    samples = list_samples(split_dir, scenes=scenes)
    for sample in samples:
        mask = remap_mask(read_tif(sample.mask_path))
        ones += int(mask.sum())
        zeros += int(mask.size - mask.sum())
    total = zeros + ones
    return {
        "samples": len(samples),
        "scenes": sorted(normalize_scene_ids(scenes) or []),
        "no_change_pixels": zeros,
        "change_pixels": ones,
        "total_pixels": total,
        "change_fraction": ones / total if total else 0.0,
        "pos_weight": zeros / max(ones, 1),
    }

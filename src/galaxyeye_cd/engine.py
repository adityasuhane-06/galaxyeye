from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import BinaryConfusion
from .visualize import save_prediction_grid


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def _window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    tile_size: int,
    stride: int,
) -> torch.Tensor:
    """Run low-memory tiled inference for one CHW image and return CPU logits."""
    _, h, w = image.shape
    y_starts = _window_starts(h, tile_size, stride)
    x_starts = _window_starts(w, tile_size, stride)
    logits_sum = torch.zeros((1, h, w), dtype=torch.float32)
    counts = torch.zeros((1, h, w), dtype=torch.float32)

    for y in y_starts:
        for x in x_starts:
            tile = image[:, y : y + tile_size, x : x + tile_size].unsqueeze(0)
            tile = _prepare_images(tile, device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                tile_logits = model(tile).squeeze(0).detach().float().cpu()
            logits_sum[:, y : y + tile_size, x : x + tile_size] += tile_logits
            counts[:, y : y + tile_size, x : x + tile_size] += 1.0
    return logits_sum / counts.clamp_min(1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 25,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        images = _prepare_images(batch["image"], device)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if step % log_interval == 0:
            pbar.set_postfix(loss=total_loss / max(total_items, 1))
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    threshold: float = 0.5,
    vis_dir: str | Path | None = None,
    vis_count: int = 0,
    tile_size: int | None = None,
    tile_stride: int | None = None,
    extra_thresholds: list[float] | None = None,
    use_tta: bool = True,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = BinaryConfusion()
    extra_confusions = {t: BinaryConfusion() for t in (extra_thresholds or [])}
    saved = 0
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        if tile_size is not None:
            images = batch["image"]
            masks = batch["mask"]
            logits_list = []
            for i in range(images.size(0)):
                logits_i = sliding_window_logits(
                    model,
                    images[i],
                    device,
                    tile_size=tile_size,
                    stride=tile_stride or tile_size,
                )
                logits_list.append(logits_i)
            logits = torch.stack(logits_list, dim=0)
        else:
            images = _prepare_images(batch["image"], device)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits_orig = model(images)
                if criterion is not None:
                    loss = criterion(logits_orig, masks)

                if use_tta:
                    logits_hf = model(torch.flip(images, dims=[-1]))
                    logits_hf = torch.flip(logits_hf, dims=[-1])

                    logits_vf = model(torch.flip(images, dims=[-2]))
                    logits_vf = torch.flip(logits_vf, dims=[-2])

                    logits_rot = model(torch.rot90(images, k=1, dims=[-2, -1]))
                    logits_rot = torch.rot90(logits_rot, k=-1, dims=[-2, -1])

                    logits = (logits_orig + logits_hf + logits_vf + logits_rot) / 4.0
                else:
                    logits = logits_orig
            if criterion is not None:
                total_loss += float(loss.item()) * images.size(0)
                total_items += images.size(0)

        confusion.update(logits, masks, threshold=threshold)
        for t, extra_confusion in extra_confusions.items():
            extra_confusion.update(logits, masks, threshold=t)
        if vis_dir is not None and saved < vis_count:
            for i, sample_id in enumerate(batch["id"]):
                if saved >= vis_count:
                    break
                save_prediction_grid(images[i], masks[i], logits[i], str(sample_id), vis_dir, threshold)
                saved += 1
    metrics = confusion.compute()
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_items, 1)
    if extra_confusions:
        sweep = []
        best = None
        for t, extra_confusion in extra_confusions.items():
            row = extra_confusion.compute()
            row["threshold"] = t
            sweep.append(row)
            if best is None or row["iou"] > best["iou"]:
                best = row
        metrics["threshold_sweep"] = sweep
        metrics["best_threshold_by_iou"] = best
    return metrics

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
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = BinaryConfusion()
    saved = 0
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        images = _prepare_images(batch["image"], device)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        if criterion is not None:
            loss = criterion(logits, masks)
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
        confusion.update(logits, masks, threshold=threshold)
        if vis_dir is not None and saved < vis_count:
            for i, sample_id in enumerate(batch["id"]):
                if saved >= vis_count:
                    break
                save_prediction_grid(images[i], masks[i], logits[i], str(sample_id), vis_dir, threshold)
                saved += 1
    metrics = confusion.compute()
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_items, 1)
    return metrics

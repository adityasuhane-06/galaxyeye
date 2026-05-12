from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_prediction_grid(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    sample_id: str,
    out_dir: str | Path,
    threshold: float = 0.5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().numpy()
    pre = np.clip(image_np[:3].transpose(1, 2, 0), 0, 1)
    post = np.clip(image_np[3], 0, 1)
    gt = mask.detach().cpu().numpy().squeeze()
    prob = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
    pred = (prob >= threshold).astype(np.float32)
    error = np.zeros((*gt.shape, 3), dtype=np.float32)
    error[(pred == 1) & (gt == 1)] = [0.0, 0.8, 0.0]
    error[(pred == 1) & (gt == 0)] = [1.0, 0.2, 0.0]
    error[(pred == 0) & (gt == 1)] = [0.2, 0.2, 1.0]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    panels = [
        ("Pre-event EO", pre, None),
        ("Post-event SAR", post, "gray"),
        ("Ground truth", gt, "gray"),
        ("Prediction", pred, "gray"),
        ("Errors", error, None),
    ]
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_id}.png", dpi=160)
    plt.close(fig)

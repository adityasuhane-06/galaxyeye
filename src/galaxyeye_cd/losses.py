from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Focal Loss (Handles Extreme Class Imbalance)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

        # Dice Loss (Handles Shape and IoU)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()

        return self.focal_weight * focal + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """
    Advanced Tversky Loss. Specifically designed for Highly Imbalanced datasets
    (like our 1.5% target pixel dataset). By shifting alpha & beta, we heavily penalize
    False Negatives (missed buildings) strictly harder than False Positives.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.3, # Weight on False Positives
        beta: float = 0.7,  # Weight on False Negatives (Harder penalty)
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)

        # True Positives, False Positives & False Negatives
        TP = (probs * targets).sum(dim=dims)
        FP = (probs * (1 - targets)).sum(dim=dims)
        FN = ((1 - probs) * targets).sum(dim=dims)

        # Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1.0 - tversky.mean()

        return self.bce_weight * bce + self.tversky_weight * tversky_loss


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "bce_dice").lower()

    if name == "tversky":
        return TverskyLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.3)),
            tversky_weight=float(loss_cfg.get("tversky_weight", 0.7)),
            alpha=float(loss_cfg.get("alpha", 0.3)),
            beta=float(loss_cfg.get("beta", 0.7)),
            pos_weight=loss_cfg.get("pos_weight"),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    elif name == "focal_dice":
        return FocalDiceLoss(
            focal_weight=float(loss_cfg.get("focal_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            alpha=float(loss_cfg.get("alpha", 0.25)), # Tuning parameter for class frequency
            gamma=float(loss_cfg.get("gamma", 2.0)),  # Tuning parameter for hard vs easy examples
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    elif name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            pos_weight=loss_cfg.get("pos_weight"),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported loss: {name}")

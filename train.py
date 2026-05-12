from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.galaxyeye_cd.config import load_config, save_config
from src.galaxyeye_cd.data import ChangeDetectionDataset, estimate_binary_distribution
from src.galaxyeye_cd.engine import evaluate, train_one_epoch
from src.galaxyeye_cd.losses import build_loss
from src.galaxyeye_cd.model import build_model
from src.galaxyeye_cd.utils import (
    configure_torch_runtime,
    cuda_memory_summary,
    describe_device,
    get_device,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EO-SAR binary change detection model")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument("--train_dir", default=None, help="Override training split directory")
    parser.add_argument("--val_dir", default=None, help="Override validation split directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device: auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_dir:
        config["data"]["train_dir"] = args.train_dir
    if args.val_dir:
        config["data"]["val_dir"] = args.val_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device

    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    configure_torch_runtime(device)
    print(describe_device(device))

    data_cfg = config["data"]
    train_ds = ChangeDetectionDataset(
        data_cfg["train_dir"],
        image_size=data_cfg["image_size"],
        augment=True,
        positive_crop_prob=float(data_cfg.get("positive_crop_prob", 0.0)),
    )
    val_ds = ChangeDetectionDataset(data_cfg["val_dir"], image_size=data_cfg["image_size"], augment=False)
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples")
    if max_train_samples:
        train_ds = Subset(train_ds, range(min(int(max_train_samples), len(train_ds))))
    if max_val_samples:
        val_ds = Subset(val_ds, range(min(int(max_val_samples), len(val_ds))))

    train_stats = estimate_binary_distribution(data_cfg["train_dir"])
    print("Train mask distribution:", train_stats)

    loader_args = {
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
    }
    if loader_args["num_workers"] > 0:
        loader_args["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_args["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if bool(config["training"].get("amp", True)) and device.type == "cuda"
        else None
    )

    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, ckpt_dir / "used_config.yaml")
    write_json(train_stats, ckpt_dir / "train_distribution.json")

    best_iou = -1.0
    history: list[dict] = []
    threshold = float(config["evaluation"]["threshold"])
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            grad_clip_norm=config["training"].get("grad_clip_norm"),
            log_interval=int(config["training"].get("log_interval", 25)),
        )
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=threshold)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        write_json({"history": history}, ckpt_dir / "history.json")

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | val_recall={val_metrics['recall']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "metrics": val_metrics,
        }
        torch.save(state, ckpt_dir / "last.pth")
        if val_metrics["iou"] > best_iou:
            best_iou = float(val_metrics["iou"])
            torch.save(state, ckpt_dir / "best.pth")
            print(f"Saved new best checkpoint: IoU={best_iou:.4f}")
        mem = cuda_memory_summary(device)
        if mem:
            print(
                "CUDA memory | "
                f"allocated={mem['allocated_gb']:.2f} GB | "
                f"reserved={mem['reserved_gb']:.2f} GB | "
                f"max_allocated={mem['max_allocated_gb']:.2f} GB"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.galaxyeye_cd.config import load_config
from src.galaxyeye_cd.data import ChangeDetectionDataset, estimate_binary_distribution
from src.galaxyeye_cd.engine import evaluate
from src.galaxyeye_cd.losses import build_loss
from src.galaxyeye_cd.model import build_model
from src.galaxyeye_cd.utils import configure_torch_runtime, describe_device, get_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EO-SAR binary change detection model")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument("--data_path", required=True, help="Split directory containing pre-event/post-event/target")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", default="outputs/metrics/eval_metrics.json", help="Metrics JSON output")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold")
    parser.add_argument("--sweep_thresholds", action="store_true", help="Report best validation threshold from 0.05 to 0.95")
    parser.add_argument("--device", default=None, help="Override device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--visualize", action="store_true", help="Save qualitative prediction grids")
    parser.add_argument("--vis_dir", default="outputs/visualizations", help="Visualization output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device or config.get("device", "auto"))
    configure_torch_runtime(device)
    print(describe_device(device))
    threshold = args.threshold if args.threshold is not None else float(config["evaluation"]["threshold"])

    dataset = ChangeDetectionDataset(args.data_path, image_size=config["data"]["image_size"], augment=False)
    loader_args = {
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": bool(config["data"].get("pin_memory", True)) and device.type == "cuda",
    }
    if loader_args["num_workers"] > 0:
        loader_args["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
        loader_args["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        **loader_args,
    )

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    criterion = build_loss(config).to(device)

    vis_dir = args.vis_dir if args.visualize else None
    vis_count = int(config["evaluation"].get("visualization_count", 8)) if args.visualize else 0
    metrics = evaluate(model, loader, criterion, device, threshold=threshold, vis_dir=vis_dir, vis_count=vis_count)
    if args.sweep_thresholds:
        sweep = []
        best = None
        for i in range(5, 100, 5):
            t = i / 100
            row = evaluate(model, loader, criterion=None, device=device, threshold=t)
            row["threshold"] = t
            sweep.append(row)
            if best is None or row["iou"] > best["iou"]:
                best = row
        metrics["threshold_sweep"] = sweep
        metrics["best_threshold_by_iou"] = best
    metrics["data_path"] = str(Path(args.data_path))
    metrics["weights"] = str(Path(args.weights))
    metrics["threshold"] = threshold
    metrics["data_distribution"] = estimate_binary_distribution(args.data_path)
    write_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()

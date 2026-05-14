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
    parser.add_argument("--full_image", action="store_true", help="Evaluate full 1024x1024 images with tiled inference")
    parser.add_argument("--tile_size", type=int, default=None, help="Tile size for --full_image")
    parser.add_argument("--tile_stride", type=int, default=None, help="Tile stride for --full_image")
    parser.add_argument("--no_tta", action="store_true", help="Disable flip/rotation TTA for crop-based evaluation")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene ids to evaluate, e.g. 07 08")
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

    eval_image_size = None if args.full_image else config["data"]["image_size"]
    dataset = ChangeDetectionDataset(args.data_path, image_size=eval_image_size, augment=False, scenes=args.scenes)
    loader_args = {
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": bool(config["data"].get("pin_memory", True)) and device.type == "cuda",
    }
    if loader_args["num_workers"] > 0:
        loader_args["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
        loader_args["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))

    loader = DataLoader(
        dataset,
        batch_size=1 if args.full_image else int(config["training"]["batch_size"]),
        shuffle=False,
        **loader_args,
    )

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    criterion = build_loss(config).to(device)

    vis_dir = args.vis_dir if args.visualize else None
    vis_count = int(config["evaluation"].get("visualization_count", 8)) if args.visualize else 0
    tile_size = args.tile_size or int(config["data"]["image_size"])
    tile_stride = args.tile_stride or max(tile_size // 2, 1)

    # Sweep micro-thresholds (for highly imbalanced Focal Loss) and macro-thresholds
    micro_thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
    macro_thresholds = [i / 100 for i in range(5, 100, 5)]
    sweep_thresholds = micro_thresholds + macro_thresholds if args.sweep_thresholds else None

    metrics = evaluate(
        model,
        loader,
        criterion=None if args.full_image else criterion,
        device=device,
        threshold=threshold,
        vis_dir=vis_dir,
        vis_count=vis_count,
        tile_size=tile_size if args.full_image else None,
        tile_stride=tile_stride if args.full_image else None,
        extra_thresholds=sweep_thresholds,
        use_tta=not args.no_tta,
    )
    if args.sweep_thresholds:
        print("Best threshold by IoU:", metrics["best_threshold_by_iou"])
    metrics["data_path"] = str(Path(args.data_path))
    metrics["weights"] = str(Path(args.weights))
    metrics["threshold"] = threshold
    metrics["scenes"] = args.scenes
    metrics["full_image"] = args.full_image
    metrics["tta"] = not args.no_tta and not args.full_image
    if args.full_image:
        metrics["tile_size"] = tile_size
        metrics["tile_stride"] = tile_stride
    metrics["data_distribution"] = estimate_binary_distribution(args.data_path, scenes=args.scenes)
    write_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from src.galaxyeye_cd.data import estimate_binary_distribution, list_samples, read_tif
from src.galaxyeye_cd.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GalaxEye split structure and mask distribution")
    parser.add_argument("--data_path", required=True, help="Split path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_samples(args.data_path)
    first = samples[0]
    pre = read_tif(first.pre_path)
    post = read_tif(first.post_path)
    mask = read_tif(first.mask_path)
    stats = estimate_binary_distribution(args.data_path)
    report = {
        "data_path": args.data_path,
        "num_samples": len(samples),
        "first_sample": first.sample_id,
        "pre_shape": list(pre.shape),
        "post_shape": list(post.shape),
        "mask_shape": list(mask.shape),
        **stats,
    }
    print(report)
    if args.output:
        write_json(report, args.output)


if __name__ == "__main__":
    main()

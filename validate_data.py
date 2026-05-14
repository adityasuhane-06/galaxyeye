from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.galaxyeye_cd.data import list_samples, read_tif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that all EO/SAR/mask TIFF files are readable")
    parser.add_argument("--data_path", required=True, help="Split directory containing pre-event/post-event/target")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene ids to validate, e.g. 07 08")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_samples(args.data_path, scenes=args.scenes)
    bad: list[Path] = []

    for sample in tqdm(samples, desc=f"validating {args.data_path}"):
        for path in (sample.pre_path, sample.post_path, sample.mask_path):
            try:
                read_tif(path)
            except Exception as exc:
                bad.append(path)
                print(f"\nBAD: {path}\n  {exc}")

    if bad:
        print(f"\nFound {len(bad)} unreadable TIFF file(s). Re-extract or re-download this split.")
        raise SystemExit(1)

    print(f"All {len(samples)} samples are readable in {args.data_path}.")


if __name__ == "__main__":
    main()

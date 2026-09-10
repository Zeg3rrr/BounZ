"""Create reproducible train/validation/test folders from YOLO-labelled images."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Annotated image folder")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO .txt label folder")
    parser.add_argument("--output", type=Path, required=True, help="Dataset output root")
    parser.add_argument("--train", type=float, default=0.70, help="Train share (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.20, help="Validation share (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument(
        "--test-prefix",
        action="append",
        default=[],
        help="Filename prefix of a whole clip reserved for test; repeat if needed",
    )
    return parser.parse_args()


def copy_pair(image: Path, labels: Path, output: Path, split: str) -> None:
    image_target = output / "images" / split / image.name
    label_target = output / "labels" / split / f"{image.stem}.txt"
    if image_target.exists() or label_target.exists():
        raise SystemExit(f"Refusing to overwrite files in {output}; use a new empty folder.")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)
    source_label = labels / f"{image.stem}.txt"
    # A missing label is valid in YOLO: it means this is a negative image.
    if source_label.exists():
        shutil.copy2(source_label, label_target)


def main() -> None:
    args = parse_args()
    if not args.images.is_dir() or not args.labels.is_dir():
        raise SystemExit("--images and --labels must both be existing folders")
    if not 0 < args.train < 1 or not 0 < args.val < 1 or args.train + args.val >= 1:
        raise SystemExit("Use positive --train/--val values whose sum is smaller than 1")

    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 10:
        raise SystemExit("Use at least 10 images before splitting a dataset")

    if args.test_prefix:
        test_images = [
            image for image in images
            if any(image.name.startswith(prefix) for prefix in args.test_prefix)
        ]
        remaining_images = [image for image in images if image not in test_images]
        if not test_images:
            raise SystemExit("No images match --test-prefix")
        random.Random(args.seed).shuffle(remaining_images)
        train_share = args.train / (args.train + args.val)
        train_end = round(len(remaining_images) * train_share)
        splits = {
            "train": remaining_images[:train_end],
            "val": remaining_images[train_end:],
            "test": test_images,
        }
    else:
        random.Random(args.seed).shuffle(images)
        train_end = round(len(images) * args.train)
        val_end = train_end + round(len(images) * args.val)
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

    for split, split_images in splits.items():
        for image in split_images:
            copy_pair(image, args.labels, args.output, split)
        print(f"{split}: {len(split_images)} images")


if __name__ == "__main__":
    main()

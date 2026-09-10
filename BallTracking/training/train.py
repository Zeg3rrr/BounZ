"""Train the BounZ basketball/hoop detector on the local YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "yolov8n.pt"
DATASET_CONFIG = Path(__file__).with_name("basketball_dataset.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8, help="Lower this when GPU memory is full")
    parser.add_argument("--device", default=None, help="Use 0 for first GPU, or cpu")
    parser.add_argument("--name", default="basketball_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DEFAULT_MODEL.is_file():
        raise SystemExit(f"Base model not found: {DEFAULT_MODEL}")

    train_options = {
        "data": str(DATASET_CONFIG),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_DIR / "runs"),
        "name": args.name,
        "exist_ok": False,
    }
    if args.device is not None:
        train_options["device"] = args.device

    model = YOLO(str(DEFAULT_MODEL))
    model.train(**train_options)


if __name__ == "__main__":
    # Required for safe DataLoader startup on Windows.
    main()

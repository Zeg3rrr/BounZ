"""Extract a representative set of JPEG frames from one basketball video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source video file")
    parser.add_argument("--output", type=Path, required=True, help="Empty/new frame folder")
    parser.add_argument(
        "--every",
        type=int,
        default=15,
        help="Save one frame for every N video frames (default: 15)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of frames to save (default: 500)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Video not found: {args.input}")
    if args.every < 1 or args.max_frames < 1:
        raise SystemExit("--every and --max-frames must be positive integers")

    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")

    video_frame = 0
    saved = 0
    try:
        while saved < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if video_frame % args.every == 0:
                target = args.output / f"frame_{saved:05d}.jpg"
                if target.exists():
                    raise SystemExit(
                        f"Refusing to overwrite {target}. Choose an empty output folder."
                    )
                if not cv2.imwrite(str(target), frame):
                    raise SystemExit(f"Could not write {target}")
                saved += 1
            video_frame += 1
    finally:
        capture.release()

    print(f"Saved {saved} frames from {video_frame} video frames to {args.output}")


if __name__ == "__main__":
    main()

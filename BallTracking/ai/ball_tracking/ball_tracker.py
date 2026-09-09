from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple, TypedDict

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
TRAIL_LENGTH = 20
WINDOW_NAME = "BounZ Ball Tracking"
DEFAULT_MODEL_NAME = "yolov8n.pt"

BALL_CLASS_NAMES = {"basketball", "sports ball", "ball"}


class Detection(TypedDict):
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


def get_model_path() -> Optional[Path]:
    """Return a local model path if one is configured or already exists."""
    env_path = os.getenv("BOUNZ_MODEL_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"The configured model path does not exist: {candidate}. "
            "Check BOUNZ_MODEL_PATH or point it to a valid .pt file."
        )

    fallback_path = ROOT_DIR / "ai" / "ball_tracking" / "models" / "basketball.pt"
    if fallback_path.exists():
        return fallback_path

    return None


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> YOLO:
    """Load a local YOLO model if available, otherwise use the standard YOLOv8 weights."""
    model_path = get_model_path()

    if model_path is not None:
        print(f"Loading local model from: {model_path}")
        try:
            return YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load the local model: {model_path}. Error: {exc}") from exc

    print(f"No local model found. Downloading the default YOLO model: {model_name}")
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the YOLO model. Check your internet connection and Ultralytics installation."
        ) from exc


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a webcam using a Windows-friendly fallback strategy."""
    camera_candidates: list[int] = [camera_index, 0, 1, 2]
    seen: set[int] = set()
    for index in camera_candidates:
        if index in seen:
            continue
        seen.add(index)

        cap = None
        try:
            if hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(index)
        except Exception:
            cap = None

        if cap is not None and cap.isOpened():
            print(f"Webcam opened successfully on camera index {index}.")
            return cap

        if cap is not None:
            cap.release()

    raise RuntimeError(
        f"Could not open webcam. Check camera permissions or CAMERA_INDEX={camera_index}. "
        "Try another index or close other apps using the camera."
    )


def normalize_class_name(class_name: str) -> str:
    return class_name.lower().replace("_", " ")


def is_ball_class(class_name: str) -> bool:
    normalized = normalize_class_name(class_name)
    return any(label in normalized for label in BALL_CLASS_NAMES)


def calculate_fps(last_time: float) -> Tuple[float, float]:
    """Return the current FPS and the updated time reference."""
    current_time = time.time()
    elapsed = current_time - last_time
    fps = 1.0 / max(elapsed, 0.0001)
    return fps, current_time


def detect_ball(frame: np.ndarray, model: YOLO, confidence_threshold: float) -> Optional[Detection]:
    """Run YOLO on a frame and return the best basketball detection."""
    raw_results: Any = model(frame, verbose=False, conf=confidence_threshold)
    if not raw_results or len(raw_results) == 0:
        return None

    result: Any = raw_results[0]
    best_match: Optional[Detection] = None

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, str(class_id))

        if not is_ball_class(class_name):
            continue

        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detection: Detection = {
            "class_name": class_name,
            "confidence": confidence,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (int(center_x), int(center_y)),
        }

        if best_match is None or confidence > best_match["confidence"]:
            best_match = detection

    return best_match


def draw_tracking_overlay(
    frame: np.ndarray,
    detection: Optional[Detection],
    trail: deque[tuple[int, int]],
    fps: float,
) -> None:
    """Draw the video overlay with tracking information."""
    cv2.putText(
        frame,
        "BounZ Ball Tracking",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        "AI model: YOLOv8",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if detection is None:
        cv2.putText(
            "Basketball: NOT DETECTED",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            "Tracking: SEARCHING",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        center_x, center_y = detection["center"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"Basketball: {class_name} {confidence:.2f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Ball: ({center_x}, {center_y})",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Tracking: ACTIVE",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if len(trail) > 1:
        points = list(trail)
        for index in range(1, len(points)):
            pt1 = points[index - 1]
            pt2 = points[index]
            cv2.line(frame, pt1, pt2, (255, 140, 0), 2)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for beginner-friendly configuration."""
    parser = argparse.ArgumentParser(description="BounZ live basketball tracking")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="Camera index to open (0, 1, 2, etc.)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--trail-length", type=int, default=TRAIL_LENGTH, help="Number of positions kept in the movement trail")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="YOLO model to load, e.g. yolov8n.pt")
    return parser.parse_args()


def main() -> int:
    """Open the webcam, detect a basketball, and show the live tracking overlay."""
    args = parse_arguments()
    camera_index = args.camera_index
    confidence_threshold = args.confidence
    trail_length = args.trail_length
    model_name = args.model_name

    try:
        model = load_model(model_name)
    except Exception as exc:
        print(f"Model error: {exc}")
        return 1

    try:
        cap = open_camera(camera_index)
    except Exception as exc:
        print(str(exc))
        return 1

    trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
    last_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Webcam is open, but no frames are being received. Check the camera connection.")
                return 1

            detection = detect_ball(frame, model, confidence_threshold)
            if detection is not None:
                trail.append(detection["center"])
            else:
                trail.clear()

            fps, last_time = calculate_fps(last_time)
            draw_tracking_overlay(frame, detection, trail, fps)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)

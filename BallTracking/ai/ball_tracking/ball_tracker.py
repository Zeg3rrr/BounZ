from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
TRAIL_LENGTH = 20
WINDOW_NAME = "BounZ Ball Tracking"

BALL_CLASS_NAMES = {"basketball", "sports ball", "ball"}


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


def load_model() -> YOLO:
    """Load a local YOLO model if available, otherwise use the standard YOLOv8 weights."""
    model_path = get_model_path()

    if model_path is not None:
        print(f"Loading local model from: {model_path}")
        try:
            return YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load the local model: {model_path}. Error: {exc}") from exc

    print("No local model found. Downloading the default YOLOv8 Nano model for the first prototype.")
    try:
        return YOLO("yolov8n.pt")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the YOLO model. Check your internet connection and Ultralytics installation."
        ) from exc


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


def detect_ball(frame: np.ndarray, model: YOLO, confidence_threshold: float) -> Optional[dict]:
    """Run YOLO on a frame and return the best basketball detection."""
    results = model(frame, verbose=False, conf=confidence_threshold)
    if not results or len(results) == 0:
        return None

    result = results[0]
    best_match: Optional[dict] = None

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

        detection = {
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
    detection: Optional[dict],
    trail: deque,
    fps: float,
) -> None:
    """Draw the video overlay with tracking information."""
    cv2.putText(
        frame,
        f"BounZ Ball Tracking",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if detection is None:
        cv2.putText(
            frame,
            "Basketball: NOT DETECTED",
            (10, 95),
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
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Ball: ({center_x}, {center_y})",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    if len(trail) > 1:
        points = list(trail)
        for index in range(1, len(points)):
            pt1 = points[index - 1]
            pt2 = points[index]
            cv2.line(frame, pt1, pt2, (255, 140, 0), 2)


def main() -> int:
    """Open the webcam, detect a basketball, and show the live tracking overlay."""
    try:
        model = load_model()
    except Exception as exc:
        print(f"Model error: {exc}")
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam. Check camera permissions or CAMERA_INDEX={CAMERA_INDEX}.")
        return 1

    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LENGTH)
    last_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("Webcam is open, but no frames are being received. Check the camera connection.")
                return 1

            detection = detect_ball(frame, model, CONFIDENCE_THRESHOLD)
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

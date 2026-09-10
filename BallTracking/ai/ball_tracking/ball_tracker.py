# Basketball Shot Detection and Tracking - Based on hi-tech-AI GitHub
# Modified for BounZ application
# Avi Shah (Original) - Adapted for BounZ
#
# ITERATION 2 - BALL TRACKING FOCUS
# ----------------------------------------------------------------------------
# Rim/hoop detection is intentionally left mostly untouched in this pass; it
# will be reworked in the next iteration once ball tracking is solid.
#
# What changed in this pass and why:
#
# 1. Real single-target tracking instead of "append every matching box".
#    A new `BallTracker` class keeps a constant-velocity Kalman filter for
#    the ball. Every frame it:
#      - predicts where the ball should be,
#      - if several "ball"-like boxes were detected, picks the ONE closest
#        to that prediction (weighted a bit by confidence) instead of
#        blindly appending every match (the old code could push several
#        unrelated detections into `ball_pos` in the same frame),
#      - rejects a candidate if the jump is physically implausible, with a
#        tolerance that grows the longer the ball has been missing (instead
#        of a fixed "4x diameter" cutoff that didn't account for occlusion),
#      - survives short occlusions (net, rim, hands) by predicting through
#        gaps instead of just losing the point.
#
# 2. Only ball/hoop-like classes are requested from the model (`classes=`),
#    which speeds up inference and NMS noticeably.
#
# 3. Inference now runs at a smaller `imgsz` while detections are still
#    reported in full original-frame coordinates (Ultralytics does this
#    rescaling internally), so we keep a crisp full-res display without
#    paying full-res inference cost. This, together with (2), is what was
#    actually killing FPS with yolov8l on a full-res webcam feed - not the
#    tracking logic.
#
# 4. Automatic GPU + half-precision use when available, CPU fallback.
#
# 5. Score/attempt detection now reads from the tracker's smoothed
#    trajectory, so the make/miss line-fit is noticeably less jittery.
#
# 6. Removed dead code (`self.class_names` was defined but never actually
#    used anywhere).
#
# A note on the model itself: stock COCO-pretrained yolov8*.pt only has a
# generic "sports ball" class - there is no "basketball" or "hoop" class in
# COCO. That means:
#   - hoop detection cannot work at all with a stock model (this is exactly
#     why it's being deferred to the next pass),
#   - ball detection works, but can occasionally false-positive on other
#     round objects since it's not a basketball-specific class.
# The bundled yolov8n.pt is used by default because it is fast for a webcam.
# Once ball tracking is confirmed solid, the recommended next step for BOTH
# ball and hoop accuracy is setting BOUNZ_MODEL_PATH to a basketball/hoop
# fine-tuned model (or fine-tuning yolov8n yourselves). The rest of the code
# already works from `model.names`, rather than hardcoded class IDs.

from ultralytics import YOLO
import cv2
import math
import numpy as np
import time
import sys
import os
import threading
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# A Windows virtual webcam (Camo, DroidCam, ...) is selected either by its
# DirectShow device name or by its numeric OpenCV index.  Environment variables
# make this usable on another laptop without editing this file.
CAMERA_INDEX = int(os.environ.get("BOUNZ_CAMERA_INDEX", "0"))
CAMERA_NAME = os.environ.get("BOUNZ_CAMERA_NAME", "").strip()
CAMERA_FALLBACK = os.environ.get("BOUNZ_CAMERA_FALLBACK", "1") == "1"
WINDOW_NAME = "BounZ - Basketball Shot Detector"

# Resolve the bundled model relative to this file, rather than relative to
# the terminal's current directory. BOUNZ_MODEL_PATH may point at a custom
# basketball/hoop model without editing this source file.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
MODEL_NAME = os.environ.get("BOUNZ_MODEL_PATH", str(DEFAULT_MODEL_PATH))

# Keep capture small enough for low latency.  The display uses this native
# camera image; upscaling a 1080p webcam stream after a slow inference loop is
# the main reason a live view looks delayed/"blurred" while moving.
CAPTURE_WIDTH = int(os.environ.get("BOUNZ_CAMERA_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("BOUNZ_CAMERA_HEIGHT", "480"))
CAPTURE_FPS = int(os.environ.get("BOUNZ_CAMERA_FPS", "60"))

INFER_WIDTH = int(os.environ.get("BOUNZ_INFER_WIDTH", "512"))
                              # Must be a multiple of 32.  YOLO is only a
                              # fallback for the white-pingpong tracker below.
YOLO_EVERY_N_FRAMES = max(1, int(os.environ.get("BOUNZ_YOLO_EVERY", "3")))

CONFIDENCE_BALL = 0.25              # normal ball confidence threshold
CONFIDENCE_HOOP = 0.45              # unchanged, revisited in the rim pass
CONFIDENCE_BALL_IN_REGION = 0.10    # relaxed threshold near the hoop

# CLAHE performs several full-frame colour conversions. It is disabled for
# real-time use; enable it only in a very dark room with BOUNZ_USE_CLAHE=1.
USE_CLAHE = os.environ.get("BOUNZ_USE_CLAHE", "0") == "1"

MAX_MISSED_FRAMES = 15       # how long the tracker tolerates the ball being
                              # undetected (occlusion) before it resets
TRAJECTORY_HISTORY = 60      # how many accepted ball points we keep
SMOOTHING_ALPHA = 0.6        # EMA smoothing factor used only for the drawn/
                              # fitted trajectory, not for raw gating


# ============================================================================
# CLASS-NAME HELPERS
# ============================================================================

def _normalise_class_name(name):
    """Normalise common dataset class-name styles for reliable comparison."""
    return str(name).lower().replace("_", " ").strip()


BALL_STYLES = {
    "ball": ("SPORTS BALL", "sports_ball", (0, 255, 0)),
    "sports ball": ("SPORTS BALL", "sports_ball", (0, 255, 0)),
    "basketball": ("BASKETBALL", "basketball", (0, 140, 255)),
    "ping pong ball": ("WHITE PINGPONG", "ping_pong", (255, 255, 0)),
    "white ping pong ball": ("WHITE PINGPONG", "ping_pong", (255, 255, 0)),
}


def ball_style(name):
    """Return label, type and BGR display colour for supported ball classes."""
    return BALL_STYLES.get(_normalise_class_name(name))


def is_ball_class(name):
    """Return true only for a ball itself, not e.g. a baseball bat/glove."""
    return ball_style(name) is not None


def is_hoop_class(name):
    """Recognise the common names used by basketball-specific datasets."""
    return _normalise_class_name(name) in {"hoop", "basketball hoop", "rim"}


def find_white_pingpong_candidates(frame):
    """Find small, bright, round objects without running the neural network.

    COCO's ``sports ball`` model was not trained specifically on a small white
    pingpong ball, and it can easily miss one while it is partly covered by a
    hand.  This detector is deliberately cheap (threshold + contours), so it
    can run on *every* camera frame.  The Kalman tracker below chooses the
    candidate that continues the existing path, which prevents a white lamp or
    wall detail from taking over an established track.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White has very little saturation and a high value. The upper saturation
    # limit still accepts a slightly warm/grey ball under indoor lighting.
    mask = cv2.inRange(hsv, (0, 0, 165), (179, 95, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame.shape[:2]
    max_diameter = min(frame_w, frame_h) * 0.18
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 18:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4 or w > max_diameter or h > max_diameter:
            continue

        aspect = min(w, h) / max(w, h)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area / (perimeter * perimeter)
                       if perimeter > 0 else 0)
        if aspect < 0.62 or circularity < 0.48:
            continue

        # A filled circular contour has area ~= pi/4 * bounding-box area.
        # This rejects most slim bright reflections while preserving a ball
        # whose edge is a little smeared by motion.
        fill_ratio = area / max(w * h, 1)
        if fill_ratio < 0.42:
            continue

        confidence = min(0.98, 0.58 + 0.24 * aspect + 0.18 * circularity)
        candidates.append({
            "center": (x + w // 2, y + h // 2),
            "w": w,
            "h": h,
            "conf": confidence,
            "box": (x, y, x + w, y + h),
            "label": "WHITE PINGPONG",
            "kind": "ping_pong",
            "color": (255, 255, 0),
            "source": "colour/circle",
        })

    return candidates


class LatestFrameCapture:
    """Continuously drain the webcam and expose only its newest frame.

    Inference can take longer than the camera's frame interval. Reading in a
    separate thread keeps old frames from accumulating in the driver queue,
    so the image follows the hand instead of being several frames behind it.
    """

    def __init__(self, capture):
        self.capture = capture
        self._frame = None
        self._sequence = 0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _reader(self):
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._sequence += 1

    def read_latest(self):
        with self._lock:
            return (None if self._frame is None
                    else (self._sequence, self._frame.copy()))

    def close(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)
        self.capture.release()


# ============================================================================
# UTILITY FUNCTIONS (rim/shot-scoring logic - unchanged, revisited later)
# ============================================================================

def score(ball_pos, hoop_pos):
    """Detect if a shot is made using linear regression."""
    if len(ball_pos) < 2 or len(hoop_pos) < 1:
        return False

    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]

    # Get first point above rim and first point below rim
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            if i + 1 < len(ball_pos):
                x.append(ball_pos[i + 1][0][0])
                y.append(ball_pos[i + 1][0][1])
            break

    # Create line from two points using linear regression
    if len(x) > 1:
        try:
            m, b = np.polyfit(x, y, 1)
            predicted_x = ((hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]) - b) / m
            rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
            rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]

            if rim_x1 < predicted_x < rim_x2:
                return True
        except Exception:
            pass

    return False


def detect_down(ball_pos, hoop_pos):
    """Detect if ball is below the net."""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return False

    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    return ball_pos[-1][0][1] > y


def detect_up(ball_pos, hoop_pos):
    """Detect if ball is around the backboard (shooting motion)."""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return False

    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1]

    return (x1 < ball_pos[-1][0][0] < x2 and
            y1 < ball_pos[-1][0][1] < y2 - 0.5 * hoop_pos[-1][3])


def in_hoop_region(center, hoop_pos):
    """Check if center point is near the hoop."""
    if len(hoop_pos) < 1:
        return False

    x, y = center
    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    return x1 < x < x2 and y1 < y < y2


def clean_hoop_pos(hoop_pos):
    """Remove inaccurate hoop detections. Unchanged - revisited in rim pass."""
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        if ((dist > max_dist and f_dif < 5) or
                (w2 * 1.3 < h2) or (h2 * 1.3 < w2)):
            # Remove only the newest bad detection. The old separate `if`
            # statements could pop twice and accidentally discard history.
            hoop_pos.pop()

    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos


# ============================================================================
# BALL TRACKER (new) - Kalman filter based single-target tracker
# ============================================================================

class BallTracker:
    """
    Encapsulates all ball-specific tracking logic so it can be reasoned
    about (and tuned) independently of the rest of the pipeline:

      - a constant-velocity Kalman filter predicts where the ball should be
      - when several "ball"-like boxes are detected in one frame, the one
        closest to the prediction is kept (best-candidate selection)
      - a physics gate rejects implausible jumps, with tolerance that grows
        the longer the ball has gone undetected (handles brief occlusion by
        the rim, net or a hand without immediately declaring it "lost")
      - a shape gate rejects clearly non-ball-shaped boxes
      - `positions` holds the accepted, cleaned trajectory in the same
        (center, frame, w, h, conf) tuple format the existing rim/score
        utility functions above already expect, so nothing downstream needs
        to change format-wise
    """

    def __init__(self, max_missed=MAX_MISSED_FRAMES, history_len=TRAJECTORY_HISTORY):
        self.positions = []
        self.max_missed = max_missed
        self.missed_frames = 0
        self.history_len = history_len
        self.kalman = self._init_kalman()
        self.kalman_initialized = False
        # Prediction made immediately before the latest candidate selection.
        # It is kept for the live overlay without advancing the filter again.
        self.last_prediction = None
        self.current_ball_kind = None
        self.current_ball_label = None
        self.current_ball_color = None

    @staticmethod
    def _init_kalman():
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        return kf

    def predict(self):
        if not self.kalman_initialized:
            return None
        pred = self.kalman.predict()
        # OpenCV returns a (4, 1) state vector, so select scalar cells.
        return float(pred[0, 0]), float(pred[1, 0])

    def _correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman_initialized = True
        self.kalman.correct(measurement)

    def update(self, candidates, frame_count):
        """
        candidates: list of dicts {"center": (x, y), "w": w, "h": h, "conf": c}
        for every box classified as "ball" this frame.

        Returns the accepted (center, frame, w, h, conf) tuple, or None if
        nothing was accepted this frame.
        """
        self.last_prediction = self.predict() if self.kalman_initialized else None

        if not candidates:
            self.missed_frames += 1
            return None

        # A basketball's bounding box should be roughly square. Apply this
        # before initialisation too, so the first false positive cannot start
        # an implausible track.
        candidates = [
            candidate for candidate in candidates
            if candidate["w"] > 0 and candidate["h"] > 0
            and candidate["w"] * 1.4 >= candidate["h"]
            and candidate["h"] * 1.4 >= candidate["w"]
        ]
        if not candidates:
            self.missed_frames += 1
            return None

        if not self.positions:
            # No track yet - just start with the most confident detection.
            best = max(candidates, key=lambda c: c["conf"])
        else:
            pred_x, pred_y = self.last_prediction

            def candidate_score(c):
                dx = c["center"][0] - pred_x
                dy = c["center"][1] - pred_y
                dist = math.hypot(dx, dy)
                # distance dominates; confidence only nudges close ties
                return dist - (c["conf"] * 15)

            best = min(candidates, key=candidate_score)

            last_x, last_y = self.positions[-1][0]
            last_frame = self.positions[-1][1]
            last_w, last_h = self.positions[-1][2], self.positions[-1][3]
            f_dif = max(frame_count - last_frame, 1)

            dist = math.hypot(best["center"][0] - last_x, best["center"][1] - last_y)
            # Tolerance grows with frames missed, so a real occlusion
            # doesn't get rejected just because it's been a few frames.
            max_dist = 4 * math.hypot(last_w, last_h) * f_dif

            if dist > max_dist:
                self.missed_frames += 1
                return None

        self._correct(*best["center"])
        self.missed_frames = 0
        self.current_ball_kind = best.get("kind", "sports_ball")
        self.current_ball_label = best.get("label", "BALL")
        self.current_ball_color = best.get("color", (0, 255, 0))

        entry = (best["center"], frame_count, best["w"], best["h"], best["conf"])
        self.positions.append(entry)

        if len(self.positions) > self.history_len:
            self.positions.pop(0)

        return entry

    def get_smoothed_trajectory(self, alpha=SMOOTHING_ALPHA):
        """EMA-smoothed copy of the trajectory, used only for drawing and for
        the score() line-fit - the raw `positions` used for gating is left
        untouched."""
        if not self.positions:
            return []

        smoothed = [self.positions[0]]
        for pos in self.positions[1:]:
            prev = smoothed[-1]
            sx = alpha * pos[0][0] + (1 - alpha) * prev[0][0]
            sy = alpha * pos[0][1] + (1 - alpha) * prev[0][1]
            smoothed.append(((sx, sy), pos[1], pos[2], pos[3], pos[4]))
        return smoothed

    def is_lost(self):
        return self.missed_frames > self.max_missed

    def reset(self):
        self.positions = []
        self.missed_frames = 0
        self.kalman = self._init_kalman()
        self.kalman_initialized = False
        self.last_prediction = None
        self.current_ball_kind = None
        self.current_ball_label = None
        self.current_ball_color = None


# ============================================================================
# SHOT DETECTOR CLASS
# ============================================================================

class ShotDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO(MODEL_NAME)

        self.device, self.use_half = self._pick_device()

        # Only ask the model for classes that could plausibly be a ball or a
        # hoop - cheaper inference/NMS and fewer irrelevant boxes to filter
        # in Python. Falls back to "no filter" if the model has neither.
        self.relevant_class_ids = [
            i for i, n in self.model.names.items()
            if is_ball_class(n) or is_hoop_class(n)
        ] or None

        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # MJPG avoids expensive uncompressed USB frames on many Windows
        # webcams. Drivers that do not support one of these properties simply
        # ignore it, so this remains compatible with built-in cameras.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera: "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{self.cap.get(cv2.CAP_PROP_FPS):.0f} FPS (requested)")
        self.frame_source = LatestFrameCapture(self.cap).start()

        self.ball_tracker = BallTracker()
        self.hoop_pos = []  # untouched for now, see header note

        self.frame_count = 0
        self.frame = None
        self._prev_time = time.time()
        self._last_camera_sequence = -1

        self.makes = 0
        self.attempts = 0

        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

    @staticmethod
    def _open_camera():
        """Open a named virtual camera or an index via low-latency DirectShow.

        Set ``BOUNZ_CAMERA_NAME=Camo`` when Camo is installed. If its exposed
        name differs, use the index printed by ``camera_test.py`` instead. A
        named camera deliberately does not fall back to the school webcam when
        BOUNZ_CAMERA_FALLBACK=0, making configuration mistakes obvious.
        """
        if CAMERA_NAME:
            print(f"Requesting camera by name: {CAMERA_NAME!r}")
            cap = cv2.VideoCapture(f"video={CAMERA_NAME}", cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Camera opened by name: {CAMERA_NAME!r}")
                return cap
            cap.release()
            if not CAMERA_FALLBACK:
                print("Requested camera was not available; webcam fallback is disabled.")
                return None
            print("Requested camera was not available; trying camera indices.")

        indices = [CAMERA_INDEX] + [idx for idx in (0, 1, 2)
                                    if idx != CAMERA_INDEX]
        for idx in indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Camera opened on index {idx}")
                return cap
            cap.release()
        return None

    @staticmethod
    def _pick_device():
        try:
            import torch
            if torch.cuda.is_available():
                return 0, True
        except ImportError:
            pass
        return "cpu", False

    @staticmethod
    def _enhance_frame(frame):
        """Mild contrast enhancement - helps in dim gyms. Toggle USE_CLAHE
        off first if you need more FPS."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def run(self):
        print("Starting Basketball Shot Detection (ball-tracking focus)...")
        print("Press 'q' to exit\n")

        while True:
            latest = self.frame_source.read_latest()
            if latest is None:
                # The camera thread has not delivered its first frame yet.
                time.sleep(0.002)
                continue
            sequence, raw_frame = latest
            if sequence == self._last_camera_sequence:
                # Do not run the Kalman filter repeatedly on one old frame:
                # that would make its timing differ from the actual camera.
                time.sleep(0.001)
                continue
            self._last_camera_sequence = sequence

            self.frame = raw_frame.copy()  # draw on the original, full-res frame

            # This inexpensive detector updates the pingpong position on every
            # displayed frame, including the frames between YOLO inferences.
            ball_candidates = find_white_pingpong_candidates(raw_frame)

            # Generic YOLO remains useful for a basketball, or when the white
            # ball is completely hidden by a hand. While a pingpong track is
            # healthy, it is not needed at all: this is what lets the simple
            # hand-held left/right test run at the camera's FPS on a CPU.
            pingpong_track_is_healthy = (
                self.ball_tracker.current_ball_kind == "ping_pong"
                and self.ball_tracker.missed_frames <= 2
            )
            if (not pingpong_track_is_healthy
                    and self.frame_count % YOLO_EVERY_N_FRAMES == 0):
                infer_frame = self._enhance_frame(raw_frame) if USE_CLAHE else raw_frame
                results = self.model.predict(
                    infer_frame,
                    imgsz=INFER_WIDTH,
                    conf=CONFIDENCE_BALL_IN_REGION,
                    classes=self.relevant_class_ids,
                    device=self.device,
                    half=self.use_half,
                    verbose=False,
                )
                self._append_yolo_candidates(results, ball_candidates)

            accepted_ball = self.ball_tracker.update(ball_candidates, self.frame_count)

            if self.ball_tracker.is_lost():
                # Ball has been gone too long - treat the next detection as a
                # fresh possession rather than dragging along a stale track.
                self.ball_tracker.reset()
                self.up = False
                self.down = False

            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            self._draw_detection_feedback(ball_candidates, accepted_ball)
            self._draw_ball_trajectory()
            self.shot_detection()

            now = time.time()
            fps = 1.0 / max(now - self._prev_time, 1e-6)
            self._prev_time = now
            self.display_score(fps)

            self.frame_count += 1

            cv2.imshow(WINDOW_NAME, self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.frame_source.close()
        cv2.destroyAllWindows()
        print("\nProgram ended")

    def _append_yolo_candidates(self, results, ball_candidates):
        """Merge a periodic YOLO pass into the per-frame pingpong candidates."""
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names.get(cls, "unknown")
                center = (int(x1 + w / 2), int(y1 + h / 2))

                if is_ball_class(name):
                    label, kind, color = ball_style(name)
                    required_conf = (
                        CONFIDENCE_BALL_IN_REGION
                        if in_hoop_region(center, self.hoop_pos)
                        else CONFIDENCE_BALL
                    )
                    if conf >= required_conf:
                        ball_candidates.append({
                            "center": center,
                            "w": w,
                            "h": h,
                            "conf": conf,
                            "box": (x1, y1, x2, y2),
                            "label": label,
                            "kind": kind,
                            "color": color,
                            "source": "YOLO",
                        })

                elif is_hoop_class(name) and conf > CONFIDENCE_HOOP:
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(self.frame, f"Hoop: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def _draw_detection_feedback(self, candidates, accepted):
        """Draw the three tracking stages: YOLO candidates, prediction, result."""
        # Orange boxes are raw YOLO candidates that passed the confidence
        # threshold. They make it clear what the detector saw this frame.
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box"]
            color = candidate.get("color", (0, 165, 255))
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                self.frame,
                f"YOLO {candidate.get('label', 'BALL')} {candidate['conf']:.0%}",
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        # Blue crosshair is where the Kalman filter expected the ball before
        # it inspected the current frame's detections.
        prediction = self.ball_tracker.last_prediction
        if prediction is not None:
            px, py = map(int, prediction)
            cv2.drawMarker(
                           self.frame, (px, py), (255, 180, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(self.frame, "prediction", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

        # The bold class colour is the one candidate accepted by the
        # shape/movement checks.
        if accepted is not None:
            (x, y), _, width, height, confidence = accepted
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            color = self.ball_tracker.current_ball_color or (0, 255, 0)
            label = self.ball_tracker.current_ball_label or "BALL"
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 3)
            cv2.drawMarker(self.frame, (int(x), int(y)), color,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=16, thickness=2)
            cv2.putText(self.frame, f"TRACKED {label} {confidence:.0%}",
                        (x1, max(y1 - 28, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)
        elif candidates:
            cv2.putText(self.frame, "Candidates rejected by tracker", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(self.frame, "No ball detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _draw_ball_trajectory(self):
        trajectory = self.ball_tracker.get_smoothed_trajectory()

        for pos in trajectory:
            point = (int(pos[0][0]), int(pos[0][1]))
            cv2.circle(self.frame, point, 3, (0, 0, 255), -1)

        for i in range(1, len(trajectory)):
            p1 = (int(trajectory[i - 1][0][0]), int(trajectory[i - 1][0][1]))
            p2 = (int(trajectory[i][0][0]), int(trajectory[i][0][1]))
            cv2.line(self.frame, p1, p2, (0, 0, 255), 1)

        if self.hoop_pos:
            cv2.circle(self.frame, self.hoop_pos[-1][0], 5, (128, 128, 0), 2)

    def shot_detection(self):
        trajectory = self.ball_tracker.get_smoothed_trajectory()

        if len(self.hoop_pos) < 1 or len(trajectory) < 1:
            return

        # A pingpong ball is a tracking test object, not a basketball shot.
        if self.ball_tracker.current_ball_kind == "ping_pong":
            return

        if not self.up:
            self.up = detect_up(trajectory, self.hoop_pos)
            if self.up:
                self.up_frame = trajectory[-1][1]

        if self.up and not self.down:
            self.down = detect_down(trajectory, self.hoop_pos)
            if self.down:
                self.down_frame = trajectory[-1][1]

        if self.frame_count % 10 == 0:
            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = False
                self.down = False

                if score(trajectory, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.fade_counter = self.fade_frames
                    print(f"MAKE! ({self.makes}/{self.attempts})")
                else:
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames
                    print(f"MISS! ({self.makes}/{self.attempts})")

    def display_score(self, fps):
        frame_height, frame_width = self.frame.shape[:2]
        panel_right = min(frame_width - 20, 760)
        panel = self.frame.copy()
        cv2.rectangle(panel, (20, 18), (panel_right, 158), (20, 20, 20), -1)
        self.frame = cv2.addWeighted(panel, 0.72, self.frame, 0.28, 0)

        cv2.putText(self.frame, "BOUNZ VISION", (42, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        cv2.putText(self.frame, f"{self.makes} / {self.attempts}", (42, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.putText(self.frame, "MAKES / ATTEMPTS", (44, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)

        status = (
            f"FPS {fps:.1f}   TRACK {len(self.ball_tracker.positions)}   "
            f"MISSED {self.ball_tracker.missed_frames}   HOOPS {len(self.hoop_pos)}"
        )
        cv2.putText(self.frame, status, (285, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1)
        legend_y = min(frame_height - 24, 190)
        cv2.putText(self.frame, "ORANGE basketball  |  CYAN pingpong  |  BLUE prediction  |  RED trail",
                    (42, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color), alpha, 0,
            )
            self.fade_counter -= 1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        detector = ShotDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

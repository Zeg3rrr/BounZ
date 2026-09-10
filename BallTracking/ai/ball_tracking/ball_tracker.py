# Basketball Shot Detection and Tracking - Based on hi-tech-AI GitHub
# Modified for BounZ application
# Avi Shah (Original) - Adapted for BounZ

from ultralytics import YOLO
import cv2
import math
import numpy as np
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_INDEX = 0
WINDOW_NAME = "BounZ - Basketball Shot Detector"
MODEL_NAME = "yolov8n.pt"

# ============================================================================
# UTILITY FUNCTIONS (From reference project utils.py)
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
            # Check if projected line fits between rim ends
            predicted_x = ((hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]) - b) / m
            rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
            rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
            
            if rim_x1 < predicted_x < rim_x2:
                return True
        except:
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


def clean_ball_pos(ball_pos, frame_count):
    """Remove inaccurate ball detections based on motion physics."""
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Ball should not move 4x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            ball_pos.pop()

        # Ball should be relatively square
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """Remove inaccurate hoop detections."""
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

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos


# ============================================================================
# SHOT DETECTOR CLASS (From reference project shot_detector.py)
# ============================================================================

class ShotDetector:
    def __init__(self):
        # Load the YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(MODEL_NAME)
        self.class_names = ['basketball', 'ball', 'sports ball', 'hoop', 'basketball hoop']

        # Use webcam
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            # Try alternative cameras
            for idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    print(f"Camera opened on index {idx}")
                    break

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Position tracking arrays: ((x_pos, y_pos), frame_count, width, height, confidence)
        self.ball_pos = []
        self.hoop_pos = []

        self.frame_count = 0
        self.frame = None

        # Score tracking
        self.makes = 0
        self.attempts = 0

        # Shot detection (UP/DOWN regions)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Visual feedback
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

    def run(self):
        """Main detection loop."""
        print("Starting Basketball Shot Detection...")
        print("Press 'q' to exit\n")

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break

            # YOLO Detection on frame
            results = self.model(self.frame, stream=True, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class name
                    cls = int(box.cls[0])
                    current_class = self.model.names.get(cls, "unknown")

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Basketball detection - lower confidence when near hoop
                    if "ball" in current_class.lower():
                        if (conf > 0.3 or 
                            (in_hoop_region(center, self.hoop_pos) and conf > 0.15)):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            # Draw rectangle
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(self.frame, f"Ball: {conf:.2f}", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Hoop detection - high confidence required
                    if "hoop" in current_class.lower():
                        if conf > 0.5:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            # Draw rectangle
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(self.frame, f"Hoop: {conf:.2f}", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Process detections
            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            cv2.imshow(WINDOW_NAME, self.frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nProgram ended")

    def clean_motion(self):
        """Clean and display ball motion."""
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        
        # Display ball trajectory
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Draw line connecting ball positions
        for i in range(1, len(self.ball_pos)):
            cv2.line(
                self.frame,
                self.ball_pos[i - 1][0],
                self.ball_pos[i][0],
                (0, 0, 255),
                1,
            )

        # Clean hoop motion and display
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 5, (128, 128, 0), 2)

    def shot_detection(self):
        """Detect shot attempts and determine makes/misses."""
        if len(self.hoop_pos) < 1 or len(self.ball_pos) < 1:
            return

        # Detecting when ball is in 'up' and 'down' area
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]

        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        # If ball goes from 'up' area to 'down' area in that order, increase attempt
        if self.frame_count % 10 == 0:
            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = False
                self.down = False

                # Check if it's a make or miss
                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)  # Green for make
                    self.fade_counter = self.fade_frames
                    print(f"🎯 MAKE! ({self.makes}/{self.attempts})")
                else:
                    self.overlay_color = (0, 0, 255)  # Red for miss
                    self.fade_counter = self.fade_frames
                    print(f"❌ MISS! ({self.makes}/{self.attempts})")

    def display_score(self):
        """Display the score on frame."""
        # Score text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(
            self.frame,
            text,
            (50, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            6,
        )
        cv2.putText(
            self.frame,
            text,
            (50, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            3,
        )

        # Status info
        status_text = f"Balls: {len(self.ball_pos)} | Hoops: {len(self.hoop_pos)} | Frame: {self.frame_count}"
        cv2.putText(
            self.frame,
            status_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Gradually fade out make/miss overlay
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame,
                1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha,
                0,
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

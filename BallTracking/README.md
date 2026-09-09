# BounZ - Step 1: Live Basketball Ball Tracking

BounZ is a sport-tech project that turns real basketball training into a digital game experience. The first technical step is to open a webcam, detect a basketball in the live image, and track its center position in real time.

This repository currently focuses on Step 1 only:

- live webcam feed
- YOLO object detection
- basketball bounding box
- confidence score
- ball center tracking
- motion trail
- FPS display
- exit with Q

## What this step does

The program opens the default webcam, runs a YOLO model, and tries to detect a basketball. If a basketball is found, the script draws a bounding box, prints the confidence value, and shows the ball center point as the ball moves.

This is a simple first prototype for later steps like dribble detection, shot detection, score logic, and game integration.

## Requirements

- Python 3.10 or newer
- Windows 10 or 11
- Webcam connected to the PC
- Internet access for the first model download (if no local model is already available)

## Create a virtual environment

From the project root, run:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```powershell
pip install -r ai/requirements.txt
```

## Model choice

For this first prototype, the project uses Ultralytics YOLO. The script will prefer a local custom model if you place one in the project, but it will fall back to the standard YOLOv8 Nano model (`yolov8n.pt`) if no local model is available.

Why this is a good first choice:

- fast enough for live webcam testing
- simple to run for beginners
- works well as a starting point before using a Roboflow-specific basketball model
- easy to swap later for a custom-trained basketball detector

If you later want to use your own Roboflow model, save the exported `.pt` file in a folder such as `ai/ball_tracking/models/` and set the environment variable:

```powershell
$env:BOUNZ_MODEL_PATH = "C:\path\to\your\model.pt"
```

Then run the script again. This keeps the model path configurable without storing secrets in source code.

## Webcam setup

The default camera index is:

```python
CAMERA_INDEX = 0
```

If your webcam is not the default one, you can change it in the script or set another camera index in the code.

Common values:

- `0` = first webcam
- `1` = second webcam
- `2` = third webcam

## Run the program

From the project root:

```powershell
python ai/ball_tracking/ball_tracker.py
```

The program opens the webcam and shows the live detection view. Press `Q` to quit.

## What you should see on screen

- window title: `BounZ Ball Tracking`
- FPS in the upper left corner
- a bounding box around the basketball
- class label and confidence value
- center point of the detected ball
- a short motion trail of previous ball positions
- `Basketball: NOT DETECTED` when no ball is seen

## Common webcam problems

### Camera will not open

Possible causes:

- no webcam is attached
- webcam is in use by another application
- camera permissions are blocked
- `CAMERA_INDEX` is wrong

The script will print a clear error message if the camera cannot open.

### Webcam opens but no frames appear

Possible causes:

- the camera driver is failing
- the webcam is disconnected
- the camera is blocked by an app using it in the background

### Video is frozen

Try closing other camera apps and re-run the script.

## Common model problems

### YOLO cannot load the model

Possible causes:

- no internet connection for the first download
- antivirus or firewall blocking the download
- invalid local model file

The script checks for a local model and then loads the standard YOLO weights if needed.

### The model is not detecting a basketball

Possible causes:

- the ball is too far away or blurred
- the ball is not visible enough
- the room lighting is poor
- the model is a generic object detector and may detect `sports ball` instead of `basketball`

For a production version, a custom basketball-trained model will improve detection quality.

## Project structure

```text
BounZ/
├── ai/
│   ├── ball_tracking/
│   │   ├── models/
│   │   ├── __init__.py
│   │   └── ball_tracker.py
│   └── requirements.txt
├── images/
│   └── test/
├── videos/
├── .gitignore
├── README.md
└── .venv/   (created locally)
```

## Next step after Step 1

Once the webcam and ball tracking work reliably, the natural next step is to build dribble, shot, and ball-motion logic on top of the detection pipeline.

This repository is intentionally focused only on Step 1 and does not yet include Unity, a game, a database, or advanced training analysis.

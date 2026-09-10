"""Find and preview a Windows webcam or virtual iPhone camera.

Run ``python ai/ball_tracking/camera_test.py`` after starting Camo Studio.
It lists the working OpenCV camera indices. Then run it with ``--index N`` to
preview one of them before using that same number in BounZ.
"""

import argparse
import time

import cv2


def open_camera(index):
    """Prefer DirectShow on Windows, with OpenCV's default backend as fallback."""
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    return camera


def list_cameras(limit):
    found = []
    for index in range(limit):
        camera = open_camera(index)
        if not camera.isOpened():
            camera.release()
            continue

        ok, frame = camera.read()
        camera.release()
        if not ok:
            continue

        height, width = frame.shape[:2]
        print(f"Camera-index {index}: {width}x{height}")
        found.append(index)
    return found


def preview(index):
    camera = open_camera(index)
    if not camera.isOpened():
        raise RuntimeError(f"Camera-index {index} kon niet worden geopend.")

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Preview van camera-index {index}. Druk Q om te sluiten.")
    previous = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Kon geen frame van deze camera lezen.")

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - previous, 1e-6)
        previous = now
        cv2.putText(frame, f"Camera {index} | {fps:.1f} FPS | Q = sluiten",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        cv2.imshow("BounZ camera test", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Zoek of bekijk een webcam of Camo/DroidCam-virtuele camera."
    )
    parser.add_argument("--index", type=int,
                        help="toon een live-preview van deze camera-index")
    parser.add_argument("--limit", type=int, default=10,
                        help="hoeveel indices bij zoeken worden gecontroleerd (standaard: 10)")
    args = parser.parse_args()

    if args.index is None:
        found = list_cameras(args.limit)
        if not found:
            print("Geen camera's gevonden. Start Camo Studio en controleer de USB-kabel.")
            return
        print("\nTest bijvoorbeeld een index met:")
        print(f"  python ai/ball_tracking/camera_test.py --index {found[-1]}")
        return

    preview(args.index)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Camera test error: {error}")
        raise SystemExit(1)

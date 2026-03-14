"""Real-time smile detector using Haar cascade classifiers and OpenCV."""

import os
import sys

import cv2


# ---------------------------------------------------------------------------
# Cascade loader – uses OpenCV's bundled cascades (cv2.data.haarcascades),
# falling back to local XML copies shipped with this repo.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade, trying cv2.data first, then the local directory."""
    # Prefer the cascades bundled with the opencv-python package
    cv2_data_path = os.path.join(cv2.data.haarcascades, filename)
    local_path = os.path.join(SCRIPT_DIR, filename)

    cascade = cv2.CascadeClassifier()
    if os.path.isfile(cv2_data_path) and cascade.load(cv2_data_path):
        return cascade
    if os.path.isfile(local_path) and cascade.load(local_path):
        return cascade

    print(f"[ERROR] Could not load cascade: {filename}")
    print(f"  Tried: {cv2_data_path}")
    print(f"  Tried: {local_path}")
    sys.exit(1)


face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
eye_cascade = _load_cascade("haarcascade_eye.xml")
smile_cascade = _load_cascade("haarcascade_smile.xml")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect(gray, frame):
    """Detect faces, eyes, and smiles in *frame* (BGR) using *gray* (grayscale).

    Returns the annotated *frame* with coloured rectangles:
      - Blue  (255, 0, 0) → face
      - Green (0, 255, 0) → eyes
      - Red   (0, 0, 255) → smile
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Smile detection – adjust minNeighbors (default 50) for your
        # lighting / skin-tone conditions.
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=50)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    """Open the default webcam and run smile detection until 'q' is pressed."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (device 0).")
        sys.exit(1)

    print("Smile Detector running – press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] No frame captured – skipping.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = detect(gray, frame)
            cv2.imshow("Smile Detector", canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

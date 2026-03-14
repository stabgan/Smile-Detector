"""
Smile Detector using OpenCV Haar Cascades.

Detects faces, eyes, and smiles in real-time from a webcam feed.
Draws colored bounding boxes around detected regions:
  - Blue: Face
  - Green: Eyes
  - Red: Smile
"""

import os
import sys
import cv2


def load_cascades():
    """Load Haar cascade classifiers from the same directory as this script."""
    cascade_dir = os.path.dirname(os.path.abspath(__file__))

    face_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(cascade_dir, "haarcascade_eye.xml")
    smile_path = os.path.join(cascade_dir, "haarcascade_smile.xml")

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    smile_cascade = cv2.CascadeClassifier(smile_path)

    # Validate that cascades loaded correctly
    if face_cascade.empty():
        sys.exit(f"Error: Could not load face cascade from {face_path}")
    if eye_cascade.empty():
        sys.exit(f"Error: Could not load eye cascade from {eye_path}")
    if smile_cascade.empty():
        sys.exit(f"Error: Could not load smile cascade from {smile_path}")

    return face_cascade, eye_cascade, smile_cascade


def detect(gray, frame, face_cascade, eye_cascade, smile_cascade):
    """
    Detect faces, eyes, and smiles in a frame.

    Args:
        gray: Grayscale version of the frame.
        frame: Original BGR frame from the webcam.
        face_cascade: Haar cascade for face detection.
        eye_cascade: Haar cascade for eye detection.
        smile_cascade: Haar cascade for smile detection.

    Returns:
        The annotated frame with bounding boxes drawn.
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Green rectangles around eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Red rectangles around smiles
        # Adjust minNeighbors (default 50) depending on lighting / skin tone
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=50)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return frame


def main():
    """Run the smile detector on the default webcam."""
    face_cascade, eye_cascade, smile_cascade = load_cascades()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        sys.exit("Error: Could not open webcam (device 0).")

    print("Smile Detector running — press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("Warning: Failed to read frame from webcam. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame, face_cascade, eye_cascade, smile_cascade)
        cv2.imshow("Smile Detector", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

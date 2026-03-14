# Smile Detector

Real-time smile detection using Haar cascade classifiers and OpenCV.  
Detects faces, eyes, and smiles from your webcam feed and draws coloured bounding boxes around each.

## How It Works

The detector uses three pre-trained Haar cascade classifiers shipped with OpenCV:

| Cascade | Detects |
|---------|---------|
| `haarcascade_frontalface_default.xml` | Frontal faces |
| `haarcascade_eye.xml` | Eyes (within each face ROI) |
| `haarcascade_smile.xml` | Smiles (within each face ROI) |

Each frame from the webcam is converted to grayscale, then passed through the cascades in order: face → eyes → smile. Detected regions are highlighted with coloured rectangles (blue = face, green = eyes, red = smile).

## Results

| Not Smiling | Smiling |
|:-----------:|:-------:|
| ![not smiling](https://image.ibb.co/i8HjAn/not_smiling.png) | ![smiling](https://image.ibb.co/gaV4An/smiling.png) |

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 👁 OpenCV (`opencv-python`) | Computer vision & cascade classifiers |

## Getting Started

```bash
# Install dependency
pip install opencv-python

# Run the detector
cd "Smile Detector"
python smile_detector.py
```

Press **q** to quit the webcam window.

### Tuning Smile Sensitivity

If smiles aren't being detected (or too many false positives appear), adjust the `minNeighbors` parameter in `smile_detector.py`:

```python
smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=50)
#                                                                   ^^^^^^^^^^^^^^
# Lower → more sensitive (more detections, more false positives)
# Higher → stricter (fewer detections, fewer false positives)
```

## ⚠️ Known Issues

- Detection accuracy depends heavily on lighting conditions and distance from the camera.
- Haar cascades are lightweight but less accurate than modern deep-learning-based detectors (e.g., MediaPipe, dlib CNN).
- The smile cascade can produce false positives around the mouth area when not smiling.

## License

[MIT](LICENSE)

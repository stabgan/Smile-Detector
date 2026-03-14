# 😄 Smile Detector

Real-time smile detection using **OpenCV Haar Cascades**. The program captures video from your webcam and draws bounding boxes around detected faces (blue), eyes (green), and smiles (red).

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 Language | Python 3.7+ |
| 👁️ Computer Vision | OpenCV (`cv2`) |
| 🧠 Detection | Haar Cascade Classifiers |

## 📦 Dependencies

- **Python** 3.7 or higher
- **opencv-python** (tested with 4.x)

Install with pip:

```bash
pip install opencv-python
```

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/stabgan/Smile-Detector.git
   cd "Smile-Detector/Smile Detector"
   ```

2. Run the detector:

   ```bash
   python smile_detector.py
   ```

3. Press **`q`** to quit the webcam feed.

## ⚙️ Tuning

Smile detection sensitivity depends on lighting and skin tone. Adjust the `minNeighbors` parameter in the `detect()` function:

```python
smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=50)
#                                                                   ^^^^^^^^^^^^^^
#                                                   Lower = more sensitive, Higher = stricter
```

## 📸 Results

| Not Smiling | Smiling |
|:-----------:|:-------:|
| ![not smiling](https://image.ibb.co/i8HjAn/not_smiling.png) | ![smiling](https://image.ibb.co/gaV4An/smiling.png) |

## ⚠️ Known Issues

- Requires a working webcam (device index `0`). If you have multiple cameras, change `cv2.VideoCapture(0)` to the correct index.
- Haar cascades can produce false positives in poor lighting or at extreme angles.
- The smile threshold (`minNeighbors=50`) may need manual tuning per environment.
- Detection runs on CPU only — no GPU acceleration.

## 📄 License

See [LICENSE](LICENSE) for details.

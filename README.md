# YOLO11-Image-Video-Webcam-Detection

YOLO11 for real-time image, video, and webcam object detection.

## Overview
This project utilizes YOLO11, a state-of-the-art object detection model, to detect objects in images, videos, and webcam streams with high accuracy and speed.

## Features
- **Real-time Detection**: Detect objects in images, videos, and live webcam feeds.
- **High Accuracy**: Uses the latest YOLO11 model for precise detection.
- **Easy Deployment**: Run detection with minimal setup.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/YOLO11-Image-Video-Webcam-Detection.git
cd YOLO11-Image-Video-Webcam-Detection

# Install dependencies
pip install ultralytics opencv-python numpy
```

## Usage

### 1. Detect Objects in an Image
```bash
python detect.py --source image.jpg --model yolo11n.pt
```

### 2. Detect Objects in a Video
```bash
python detect.py --source video.mp4 --model yolo11n.pt
```

### 3. Detect Objects Using Webcam
```bash
python detect.py --source 0 --model yolo11n.pt
```

## Example Code
Create a `detect.py` file and add the following code:
```python
from ultralytics import YOLO
import cv2
import argparse

def detect(source, model):
    model = YOLO(model)
    results = model(source, show=True)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, or webcam (0)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO11 model")
    args = parser.parse_args()
    detect(args.source, args.model)
```

## Acknowledgments
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- OpenCV for image processing

## License
This project is licensed under the MIT License.

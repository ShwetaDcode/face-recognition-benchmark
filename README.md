# Face Detection Benchmark and Comparison Tool

This project provides a robust Python script to compare the performance (speed and stability) of three leading face detection methods: **Haar Cascade**, **DNN-SSD (Caffe)**, and **YOLOv8-Nano (Ultralytics)**, using a live webcam feed.

## Features

* **Three Modes**: Run any single detector (`haar`, `dnn`, or `yolo`), run **all three simultaneously** (`compare`), or run a **detailed benchmark** (`benchmark`).
* **Performance Benchmarking**: FPS measurement, stability testing, and **Matplotlib charts** for comparison.
* **Real-time Visualization**: Bounding boxes, confidence scores (DNN/YOLO), and FPS.

## Project Contents

| File Name                                       | Description                                  |
| ----------------------------------------------- | -------------------------------------------- |
| `face_detection_com.py`                         | Main script with detection + benchmark logic |
| `haarcascade_frontalface_default.xml`           | Haar Cascade model                           |
| `deploy.prototxt`                               | DNN-SSD model configuration                  |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | DNN-SSD pretrained weights                   |
| `yolov8n-face.pt`                               | YOLOv8 face detection model                  |
| `requirements.txt`                              | Lists Python dependencies                    |
| `LICENSE`                                       | License information                          |

## Setup and Installation

### Prerequisites

Install **Python 3.x**

### Create & Activate Virtual Environment

```bash
python3 -m venv venv
# Windows
./venv/Scripts/activate
# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure model files are placed in the same directory as `face_detection_com.py`

## Usage

Run from terminal:

```bash
python face_detection_com.py <mode>
```

Press `q` to exit video window.

### Available Modes

| Mode        | Command Example                          | Purpose                        |
| ----------- | ---------------------------------------- | ------------------------------ |
| `haar`      | `python face_detection_com.py haar`      | Haar Cascade detection         |
| `dnn`       | `python face_detection_com.py dnn`       | DNN-SSD face detection         |
| `yolo`      | `python face_detection_com.py yolo`      | YOLOv8-Nano detection          |
| `compare`   | `python face_detection_com.py compare`   | Shows 3 detectors side-by-side |
| `benchmark` | `python face_detection_com.py benchmark` | Tests FPS & shows charts       |

Example:

```bash
python face_detection_com.py benchmark
```

## 🎨 Detection Colors

| Detector     | Color    |
| ------------ | -------- |
| Haar Cascade | 🟩 Green |
| DNN-SSD      | 🟦 Blue  |
| YOLOv8       | 🟥 Red   |

## ⚠️ Troubleshooting

* **Webcam not opening** → Close apps using camera (Zoom/Teams)
* **ImportError** → Install dependencies again: `pip install -r requirements.txt`
* **Benchmark issue** → Script includes a **2-second wait** to reset webcam between tests
* **Missing model files** → Check `.caffemodel`, `.xml`, `.pt` are in correct folder

---

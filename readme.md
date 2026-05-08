# The Aerial Guardian
## Lightweight Drone-Based Multi-Person Detection and Tracking System

## Overview
This project implements a lightweight Computer Vision pipeline for detecting and tracking multiple persons from drone footage using the VisDrone MOT dataset.

The system is designed with a focus on:
- Small object detection
- Real-time performance
- Lightweight deployment
- Multi-object tracking

---

## Features
- YOLOv8-based person detection
- Multi-object tracking with persistent IDs
- Trajectory visualization
- FPS monitoring
- Optimized for drone scenarios
- Lightweight architecture (<300MB)

---

## Dataset
VisDrone2019-MOT Validation Dataset

Target Class:
- Person

---

## System Pipeline
Input Frames → YOLO Detection → Tracking → ID Assignment → Output Video

---

## Installation

### Create Environment
```bash
python -m venv venv
```

### Activate Environment

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Run Project
```bash
python main.py
```

---

## Optimization Techniques
### Small Object Detection
- Increased inference resolution
- Lower confidence threshold
- Person-only detection filtering

### Tracking Stability
- Spatial proximity-based tracking
- Track persistence using missed-frame handling
- Trajectory history visualization

### Performance Optimization
- Lightweight YOLOv8 nano model
- Frame skipping
- Resolution balancing

---

## Hardware Used
- CPU/GPU: (Add your system details)
- OS: Windows/Linux
- Python Version: 3.x

---

## Future Improvements
- DeepSORT integration
- Kalman filtering
- Appearance-based Re-ID
- Edge deployment optimization using TensorRT

---

## Output
The output video contains:
- Bounding boxes
- Unique IDs
- Trajectory lines
- Live FPS
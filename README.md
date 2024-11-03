# Webcam Line Detection with Mask and EMA Smoothing

This project implements a real-time line detection system using OpenCV to capture video from a webcam. The program detects lines within a specified region of interest (ROI) and smooths the central line using Exponential Moving Average (EMA) to reduce noise and stabilize detection.

## Features

- **Region of Interest (ROI)**: Detects lines only within a specific boundary to focus on relevant areas.
- **Line Detection**: Uses edge detection (Canny) and Hough Line Transform for identifying lines.
- **EMA Smoothing**: Applies EMA to the detected central line's slope for stability.
- **Real-Time Processing**: Displays a video feed with detected lines from a webcam.

## Installation

### Requirements

- Python 3.x
- OpenCV
- NumPy

Install the required packages:
```bash
pip install opencv-python numpy

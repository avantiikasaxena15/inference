# inference
# Video Object Detection Inference Pipeline

This project provides a pipeline for detecting and annotating objects in a video using the YOLOv5 model. The pipeline identifies two types of objects, "child" and "therapist", and overlays bounding boxes with unique IDs on the video frames.

## Features

- **Object Detection**: Uses YOLOv5 to detect objects in the video.
- **Bounding Box Annotation**: Draws bounding boxes around detected objects with unique IDs.
- **Output Video**: Generates a video with annotations overlaid on the original frames.

## Requirements

- Python 3.6+
- Required Python packages:
  - `opencv-python-headless`
  - `torch`
  - `torchvision`
  - `pandas`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

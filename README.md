# Object Detector

A Python project using OpenCV and YOLO to identify objects from camera feed.

## Requirements

- Python 3.x
- OpenCV
- pyttsx3 (for text-to-speech)

## Installation

1. Clone or download the project.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLO model files:
   - [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights) (34 MB)
   - [yolov3-tiny.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg)
   - [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/blob/master/data/coco.names)

4. Download the MediaPipe pose landmarker model file:
   - [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_lite.task)

   Place these files in the project directory.

## Usage

1. Ensure your camera is connected.
2. Run the script:
   ```
   python object_detector.py
   ```
3. The script will open a camera feed and display detected objects with green bounding boxes and labels. It will also announce detected objects via text-to-speech when they change.
4. Press 'q' to quit.

## How it works

The script uses YOLOv3 model loaded via OpenCV's DNN module to detect objects in real-time from the camera. It detects various objects (humans, animals, vehicles, etc.) from the COCO dataset, applies non-max suppression to avoid overlapping boxes, labels each detection, and announces changes in detected objects using text-to-speech.

## Customization

- Adjust `threshold_value` in the script to change sensitivity to darkness.
- Modify `min_area` to filter smaller or larger objects.
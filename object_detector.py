import cv2
import numpy as np
import pyttsx3
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image


def classify_human_action(landmarks, image_h):
    """Heuristic action classification based on pose landmarks.

    Uses landmark positions (wrist/shoulder/hip/knee) to guess a small set of
    actions. This is not a full action-recognition model, but it improves
    labeling for simple activities.
    """
    if landmarks is None or len(landmarks) == 0:
        return None

    # Use the first pose detected
    pose_landmarks = landmarks[0]

    # Helper to convert normalized landmark to pixel position
    def to_xy(landmark):
        return landmark.x * image_h, landmark.y * image_h

    # Get landmark points
    l_wrist_x, l_wrist_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.LEFT_WRIST])
    r_wrist_x, r_wrist_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.RIGHT_WRIST])
    l_shoulder_x, l_shoulder_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.LEFT_SHOULDER])
    r_shoulder_x, r_shoulder_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.RIGHT_SHOULDER])
    l_hip_x, l_hip_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.LEFT_HIP])
    r_hip_x, r_hip_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.RIGHT_HIP])
    l_knee_x, l_knee_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.LEFT_KNEE])
    r_knee_x, r_knee_y = to_xy(pose_landmarks[pose_landmarker.PoseLandmark.RIGHT_KNEE])

    # Simple averages
    wrist_y = (l_wrist_y + r_wrist_y) / 2
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
    hip_y = (l_hip_y + r_hip_y) / 2
    knee_y = (l_knee_y + r_knee_y) / 2

    shoulder_dist = abs(r_shoulder_x - l_shoulder_x)
    knee_dist = abs(r_knee_x - l_knee_x)

    # Heuristics (higher in list = higher priority)
    if wrist_y < shoulder_y and hip_y > shoulder_y:
        return "hands up"

    if knee_y < hip_y and hip_y > shoulder_y:
        return "sitting"

    # If knees are widely spaced compared to shoulders, assume running/walking
    if knee_dist > shoulder_dist * 1.3:
        return "running/walking"

    if hip_y < shoulder_y:
        return "bending"

    return "standing"


# Detection parameters
CONFIDENCE_THRESHOLD = 0.05  # Very low for testing
NMS_THRESHOLD = 0.5        # Non-max suppression overlap threshold
INPUT_WIDTH = 416
INPUT_HEIGHT = 416


def detect_objects_from_camera():
    """Detect objects from camera feed using YOLO model."""

    # Check if required files exist
    import os
    required_files = ['yolov3-tiny.weights', 'yolov3-tiny.cfg', 'coco.names', 'pose_landmarker_lite.task']
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Missing required files: {missing}")
        return

    # Load YOLO model
    try:
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        print("YOLOv3-tiny model loaded successfully.")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    # Load class names
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera 0. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera 1 either.")
            return

    # Initialize text-to-speech
    engine = pyttsx3.init()

    # Initialize pose landmarker (MediaPipe Tasks)
    pose = pose_landmarker.PoseLandmarker.create_from_model_path(
        "pose_landmarker_lite.task"
    )

    print("Press 'q' to quit.")

    previous_detected = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Prepare input for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (INPUT_WIDTH, INPUT_HEIGHT), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Debug: print detection details to console (first frame only)
        if 'debug_printed' not in locals():
            print(f"Frame shape: {frame.shape}")
            print(f"Blob shape: {blob.shape}")
            print(f"Output layers: {len(outs)}")
            for i, out in enumerate(outs):
                print(f"Output {i} shape: {out.shape}")
                # Print max confidence in this output
                if out.size > 0:
                    max_conf = np.max(out[:, 5:])
                    print(f"  Max confidence in output {i}: {max_conf:.3f}")
            debug_printed = True

        # Run pose detection once per frame for action classification
        if pose is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_frame = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_frame)
            pose_result = pose.detect(mp_image_frame)
        else:
            pose_result = None

        # Process detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Normalize indexes to a flat list of ints
        if isinstance(indexes, (np.ndarray, list, tuple)) and len(indexes) > 0:
            try:
                indexes = np.array(indexes).flatten().tolist()
            except Exception:
                indexes = [int(i) for i in indexes]
        else:
            indexes = []

        # Debug overlay: show detection counts
        cv2.putText(frame, f"YOLO boxes: {len(boxes)}  NMS: {len(indexes)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {width}x{height}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw bounding boxes
        detected = set()

        for i in indexes:
            if i < len(boxes):
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]

                # If person, attempt simple action classification via pose
                if label == 'person' and pose_result and pose_result.pose_landmarks and pose_result and pose_result.pose_landmarks:
                    action = classify_human_action(pose_result.pose_landmarks, height)
                    if action:
                        label = f"person ({action})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw label background for readability
                label_text = label if label else str(class_ids[i])
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size
                text_x = max(x, 0)
                text_y = max(y - 10, text_h + 2)
                cv2.rectangle(frame, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + 2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                detected.add(label_text)

        # Speak detected objects if changed
        if detected and detected != previous_detected:
            engine.say(f"Detected: {', '.join(detected)}")
            engine.runAndWait()
        previous_detected = detected

        # Display the frame
        cv2.imshow('Object Detection from Camera', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects_from_camera()
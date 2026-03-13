
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from kivy.core.window import Window
from plyer import tts

class ObjectDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Camera widget
        self.camera = Camera(play=True, resolution=(640, 480))
        self.layout.add_widget(self.camera)
        
        # Label for detections
        self.label = Label(text="Detections will appear here", size_hint_y=0.1)
        self.layout.add_widget(self.label)
        
        # Button to start detection
        self.button = Button(text="Start Detection", size_hint_y=0.1)
        self.button.bind(on_press=self.start_detection)
        self.layout.add_widget(self.button)
        
        # Detection flag
        self.detecting = False
        
        # Load YOLO model
        try:
            self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
            self.classes = []
            with open('coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.label.text = "Model loaded successfully"
        except Exception as e:
            self.label.text = f"Model load failed: {e}"
            self.net = None
        
        return self.layout
    
    def start_detection(self, instance):
        if not self.detecting:
            self.detecting = True
            self.button.text = "Stop Detection"
            Clock.schedule_interval(self.process_frame, 1.0 / 30.0)  # 30 FPS
        else:
            self.detecting = False
            self.button.text = "Start Detection"
            Clock.unschedule(self.process_frame)
    
    def process_frame(self, dt):
        if self.camera.texture and self.detecting and self.net is not None:
            # Get frame from camera texture
            texture = self.camera.texture
            size = texture.size
            pixels = texture.pixels
            # Convert to numpy array (RGBA)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 4)
            # Convert to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            height, width, channels = frame.shape
            self.label.text = f"Processing frame: {height}x{width}"
            
            # Prepare input for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if isinstance(indexes, (np.ndarray, list, tuple)) and len(indexes) > 0:
                try:
                    indexes = np.array(indexes).flatten().tolist()
                except Exception:
                    indexes = [int(i) for i in indexes]
            else:
                indexes = []
            
            # Get detected classes
            detected = set(self.classes[class_ids[i]] for i in indexes if i < len(class_ids))
            
            # Update label
            if detected:
                detections_text = ', '.join(detected)
                self.label.text = f"Detected: {detections_text}"
                # Speak detections
                try:
                    tts.speak(detections_text)
                except Exception as e:
                    print(f"TTS error: {e}")
            else:
                self.label.text = f"No detections ({len(boxes)} raw boxes)"
        elif not self.net:
            self.label.text = "Model not loaded"

if __name__ == '__main__':
    ObjectDetectorApp().run()
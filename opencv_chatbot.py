import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

# Load YOLO model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, detected_objects

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_detected = set()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame to speed up
            _, detected = detect_objects(frame)
            all_detected.update(detected)
    cap.release()
    return list(all_detected)

def simple_nlp_response(query, detected):
    query_lower = query.lower()
    if "what" in query_lower and "see" in query_lower or "detect" in query_lower:
        if detected:
            return f"I detected: {', '.join(detected)}. What would you like to know more about?"
        else:
            return "I didn't detect any objects. Try a clearer image or video."
    elif "how many" in query_lower:
        counts = {}
        for obj in detected:
            counts[obj] = counts.get(obj, 0) + 1
        response = "Counts: " + ", ".join([f"{k}: {v}" for k, v in counts.items()])
        return response
    else:
        return f"Based on the analysis, I found: {', '.join(detected) if detected else 'nothing'}. Ask me about what you see!"

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Chatbot with AI Responses")
        self.root.geometry("900x700")

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.response_label = tk.Label(root, text="Welcome! Select an image or video to analyze.", wraplength=800, justify="left")
        self.response_label.pack(pady=10)

        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack(pady=5)
        self.query_entry.insert(0, "Ask me something about the image/video...")

        self.select_image_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=5)

        self.select_video_button = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_video_button.pack(pady=5)

        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.pack(pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=10)

        self.detected_objects = []

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Invalid image file.")
            return
        processed_img, detected = detect_objects(img)
        self.detected_objects = detected
        self.response_label.config(text="Image processed. Ask me a question!")

        # Display image
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(processed_img)
        pil_img.thumbnail((800, 500))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        self.response_label.config(text="Processing video... Please wait.")
        self.root.update()
        detected = process_video(file_path)
        self.detected_objects = detected
        self.response_label.config(text="Video processed. Ask me a question!")
        self.image_label.config(image='')  # Clear image

    def ask_question(self):
        query = self.query_entry.get()
        if not query or query == "Ask me something about the image/video...":
            messagebox.showwarning("Warning", "Please enter a question.")
            return
        response = simple_nlp_response(query, self.detected_objects)
        self.response_label.config(text=response)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
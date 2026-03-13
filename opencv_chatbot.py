import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import pytesseract
from pytesseract import Output

# Set Tesseract path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
if not os.path.exists("yolov3-tiny.weights") or not os.path.exists("yolov3-tiny.cfg"):
    messagebox.showerror("Error", "YOLO model files not found.")
    exit(1)
try:
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
    exit(1)

# Load classes
if not os.path.exists("coco.names"):
    messagebox.showerror("Error", "coco.names file not found.")
    exit(1)
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
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    detected_text = ""
    try:
        detected_text = pytesseract.image_to_string(img)
        # Draw text boxes
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                text = data['text'][i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    except Exception as e:
        detected_text = ""
        messagebox.showwarning("Warning", f"Text detection failed: {e}. Ensure Tesseract is installed and in PATH.")
    return img, detected_objects, detected_text

def is_dark(img, threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < threshold

def is_blur(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_detected = set()
    detected_text = ""
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            _, detected, text = detect_objects(frame)
            all_detected.update(detected)
            if text.strip() and not detected_text:  # Take first non-empty text
                detected_text = text.strip()
    cap.release()
    return list(all_detected), detected_text

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

        ''' Response text with scrollbar
        self.response_frame = tk.Frame(root)
        self.response_frame.pack(pady=10)
        self.response_text = tk.Text(self.response_frame, height=10, wrap=tk.WORD)
        #self.scrollbar = tk.Scrollbar(self.response_frame, command=self.response_text.yview)
        #self.response_text.config(yscrollcommand=self.scrollbar.set)
        self.response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)'''

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
        self.is_dark = False
        self.is_blur = False

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Invalid image file.")
            return
        self.is_dark = is_dark(img)
        self.is_blur = is_blur(img)
        if self.is_dark:
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)  # Brighten dark images
        processed_img, detected, text = detect_objects(img)
        self.detected_objects = detected
        dark_msg = "Image appears dark, detection may be unreliable." if self.is_dark else ""
        blur_msg = "Image appears blurred, detection may be unreliable." if self.is_blur else ""
        warnings = [dark_msg, blur_msg]
        warnings = [w for w in warnings if w]
        warning_msg = " ".join(warnings) if warnings else ""
        detected_msg = f"Detected objects: {', '.join(detected)}." if detected else "No objects detected."
        if "person" in detected:
            detected_msg += " (Includes persons/humans detected.)"
        if text.strip():
            detected_msg += f" Detected text: {text.strip()}."
        auto_response = simple_nlp_response("What do you see?", detected)
        self.response_label.config(text=f"Image processed. {detected_msg} {warning_msg} {auto_response}")

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
        detected, text = process_video(file_path)
        self.detected_objects = detected
        detected_msg = f"Detected objects: {', '.join(detected)}." if detected else "No objects detected."
        if "person" in detected:
            detected_msg += " (Includes persons/humans detected.)"
        if text.strip():
            detected_msg += f" Detected text: {text.strip()}."
        auto_response = simple_nlp_response("What do you see?", detected)
        self.response_label.config(text=f"Video processed. {detected_msg} {auto_response}")
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
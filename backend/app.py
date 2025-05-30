from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from collections import Counter
import datetime

app = Flask(__name__)

# Load the YOLOv8 model once
model_path = "yolov8n.pt"
model = YOLO(model_path)
model_name = model_path

@app.route('/')
def home():
    base_url = request.host_url.rstrip('/')
    return jsonify({
        "made_by": "SimpleCyber",
        "project": "YOLOv8 Object Detection API",
        "description": "Upload an image and get a JSON response with detected object names and their counts.",
        "usage": {
            "POST /detect": {
                "description": "Upload an image file as form-data with key 'image'.",
                "example": f"curl -X POST -F image=@image.png {base_url}/detect"
            },
            "GET /health": {
                "description": "Simple health check to confirm server is running.",
                "example": f"{base_url}/health"
            }
        },
        "model": model_name,
        "server_url": base_url,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded. Use form-data key as 'image'."}), 400

        file = request.files['image']
        image_bytes = file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Resize to reduce memory usage
        image = cv2.resize(image, (640, 640))

        # Run detection
        with torch.no_grad():
            results = model.predict(image, stream=False)
            detections = results[0].boxes
            names = model.names

            if detections is not None and len(detections) > 0:
                class_ids = detections.cls.cpu().numpy().astype(int)
                labels = [names[class_id] for class_id in class_ids]
                counts = Counter(labels)
                response = {
                    "detected_objects": dict(counts),
                    "total": sum(counts.values())
                }
            else:
                response = {
                    "detected_objects": {},
                    "message": "No objects detected"
                }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

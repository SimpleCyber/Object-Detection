from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import datetime
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the YOLOv8 model once
try:
    model = YOLO("yolov8n.pt")
    model_name = "yolov8n.pt"
    logger.info(f"Model {model_name} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

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
        "model": model_name if model else "Model not loaded",
        "server_url": base_url,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if model else "not loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect_objects():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500
        
        # Check if image is uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded. Use form-data key as 'image'."}), 400

        file = request.files['image']
        
        # Check if file is actually selected
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"}), 400

        # Read and decode image
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty file uploaded."}), 400
            
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format or corrupted file."}), 400

        logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")

        # Run detection
        results = model(image, verbose=False)  # Set verbose=False to reduce logs
        detections = results[0].boxes
        names = model.names

        if detections is not None and len(detections) > 0:
            class_ids = detections.cls.cpu().numpy().astype(int)
            confidences = detections.conf.cpu().numpy()
            
            # Filter by confidence threshold (e.g., 0.25)
            confidence_threshold = 0.25
            valid_detections = confidences >= confidence_threshold
            
            if np.any(valid_detections):
                filtered_class_ids = class_ids[valid_detections]
                labels = [names[class_id] for class_id in filtered_class_ids]
                counts = Counter(labels)
                
                response = {
                    "detected_objects": dict(counts),
                    "total": sum(counts.values()),
                    "confidence_threshold": confidence_threshold,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                logger.info(f"Detected {sum(counts.values())} objects: {dict(counts)}")
            else:
                response = {
                    "detected_objects": {},
                    "total": 0,
                    "message": f"No objects detected above {confidence_threshold} confidence threshold",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        else:
            response = {
                "detected_objects": {},
                "total": 0,
                "message": "No objects detected",
                "timestamp": datetime.datetime.now().isoformat()
            }

        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            "error": "Internal server error during object detection",
            "details": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large",
        "message": "The uploaded file exceeds the maximum allowed size"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

if __name__ == '__main__':
    # Set maximum file size to 10MB
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
    app.run(debug=True, host='0.0.0.0', port=5000)
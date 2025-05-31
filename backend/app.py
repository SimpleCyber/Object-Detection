from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import datetime
import logging
import os
import base64

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:*", 
            "http://127.0.0.1:*", 
            "http://0.0.0.0:*",
            "https://object-detection-lpgq.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store model
model = None
model_name = "yolov8n.pt"

def load_model():
    """Load the YOLO model once at startup"""
    global model
    try:
        os.environ['YOLO_VERBOSE'] = 'False'
        model = YOLO(model_name)
        logger.info(f"✅ Model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model at startup
model_loaded = load_model()

@app.route('/health', methods=['GET'])
@cross_origin()
def health():
    try:
        # Simple model check
        model_status = "loaded" if model_loaded and model is not None else "not loaded"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def detect_objects():
    start_time = datetime.datetime.now()
    
    try:
        import psutil
        if psutil.virtual_memory().percent > 90:
            return jsonify({"error": "Server overloaded"}), 503
        
        if not model_loaded or model is None:
            return jsonify({
                "error": "Model not loaded",
                "timestamp": datetime.datetime.now().isoformat()
            }), 500
        
        file = request.files.get('image') or request.files.get('file')
        if not file or file.filename == '':
            return jsonify({
                "error": "No valid image uploaded",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        # Read and decode image
        image_bytes = file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "error": "Invalid image format",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        logger.info(f"Processing image: {file.filename}")

        # Run detection
        detection_start = datetime.datetime.now()
        results = model(image, verbose=False)
        detection_time = (datetime.datetime.now() - detection_start).total_seconds()
        
        # Generate annotated image with boxes
        annotated_img = results[0].plot()
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Process detections
        detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            confidence_threshold = 0.25
            valid_detections = confidences >= confidence_threshold
            
            if np.any(valid_detections):
                filtered_class_ids = class_ids[valid_detections]
                filtered_confidences = confidences[valid_detections]
                labels = [model.names[class_id] for class_id in filtered_class_ids]
                counts = Counter(labels)
                
                # Get average confidence per object type
                object_confidences = {}
                for i, label in enumerate(labels):
                    if label not in object_confidences:
                        object_confidences[label] = []
                    object_confidences[label].append(float(filtered_confidences[i]))

                avg_confidences = {
                    obj: round(float(np.mean(confs)), 3)
                    for obj, confs in object_confidences.items()
                }
                
                # Prepare detection data
                detections = [{
                    'class': label,
                    'confidence': float(conf),
                    'bbox': boxes.xyxy[i].tolist()
                } for i, (label, conf) in enumerate(zip(labels, filtered_confidences))]
        
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        
        response = {
            "status": "success",
            "detections": detections,
            "detected_objects": dict(counts) if 'counts' in locals() else {},
            "total": len(detections),
            "annotated_image": img_base64,
            "confidence_threshold": confidence_threshold,
            "average_confidences": avg_confidences if 'avg_confidences' in locals() else {},
            "processing_time": {
                "detection_ms": round(detection_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2)
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "processing_time": {
                "error_after_ms": round(error_time * 1000, 2)
            },
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
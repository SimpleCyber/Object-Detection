from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import datetime
import logging
import os

app = Flask(__name__)

# Configure CORS more explicitly
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

# Set up logging with better formatting
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
        # Suppress YOLO verbose output during model loading
        os.environ['YOLO_VERBOSE'] = 'False'
        model = YOLO(model_name)
        logger.info(f"✅ Model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model at startup
model_loaded = load_model()

@app.route('/')
@cross_origin()
def home():
    base_url = request.host_url.rstrip('/')
    return jsonify({
        "made_by": "SimpleCyber",
        "project": "YOLOv8 Object Detection API",
        "description": "Upload an image and get a JSON response with detected object names and their counts.",
        "status": "🟢 Online" if model_loaded else "🔴 Model Error",
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
        "model": model_name if model_loaded else "Model not loaded",
        "server_url": base_url,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
@cross_origin()
def health():
    model_status = "loaded" if model_loaded else "not loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def detect_objects():
    start_time = datetime.datetime.now()
    
    try:
        # Check if model is loaded
        if not model_loaded or model is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs.",
                "timestamp": datetime.datetime.now().isoformat()
            }), 500
        
        # Check if image is uploaded - accept both 'image' and 'file' keys
        file = None
        if 'image' in request.files:
            file = request.files['image']
        elif 'file' in request.files:
            file = request.files['file']
        else:
            return jsonify({
                "error": "No image uploaded. Use form-data key as 'image' or 'file'.",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
        
        # Check if file is actually selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected.",
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
        if len(image_bytes) == 0:
            return jsonify({
                "error": "Empty file uploaded.",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
            
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "error": "Invalid image format or corrupted file.",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        logger.info(f"📸 Processing image: {file.filename}, size: {len(image_bytes):,} bytes")

        # Run detection with suppressed verbose output
        detection_start = datetime.datetime.now()
        results = model(image, verbose=False)
        detection_time = (datetime.datetime.now() - detection_start).total_seconds()
        
        detections = results[0].boxes
        names = model.names

        if detections is not None and len(detections) > 0:
            class_ids = detections.cls.cpu().numpy().astype(int)
            confidences = detections.conf.cpu().numpy()
            
            # Filter by confidence threshold
            confidence_threshold = 0.25
            valid_detections = confidences >= confidence_threshold
            
            if np.any(valid_detections):
                filtered_class_ids = class_ids[valid_detections]
                filtered_confidences = confidences[valid_detections]
                labels = [names[class_id] for class_id in filtered_class_ids]
                counts = Counter(labels)
                
                # Get average confidence per object type
                # Get average confidence per object type
                object_confidences = {}
                for i, label in enumerate(labels):
                    if label not in object_confidences:
                        object_confidences[label] = []
                    object_confidences[label].append(float(filtered_confidences[i]))  # Convert to Python float

                avg_confidences = {
                    obj: round(float(np.mean(confs)), 3)  # Ensure Python float type
                    for obj, confs in object_confidences.items()
                }
                
                total_time = (datetime.datetime.now() - start_time).total_seconds()
                
                response = {
                    "detected_objects": dict(counts),
                    "total": sum(counts.values()),
                    "confidence_threshold": confidence_threshold,
                    "average_confidences": avg_confidences,
                    "processing_time": {
                        "detection_ms": round(detection_time * 1000, 2),
                        "total_ms": round(total_time * 1000, 2)
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }
                logger.info(f"✅ Detected {sum(counts.values())} objects: {dict(counts)} (took {total_time:.2f}s)")
            else:
                response = {
                    "detected_objects": {},
                    "total": 0,
                    "message": f"No objects detected above {confidence_threshold} confidence threshold",
                    "processing_time": {
                        "detection_ms": round(detection_time * 1000, 2),
                        "total_ms": round((datetime.datetime.now() - start_time).total_seconds() * 1000, 2)
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }
                logger.info(f"⚠️ No objects detected above confidence threshold")
        else:
            response = {
                "detected_objects": {},
                "total": 0,
                "message": "No objects detected",
                "processing_time": {
                    "detection_ms": round(detection_time * 1000, 2),
                    "total_ms": round((datetime.datetime.now() - start_time).total_seconds() * 1000, 2)
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            logger.info(f"⚠️ No objects detected in image")

        return jsonify(response)
        
    except Exception as e:
        error_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Detection error: {str(e)} (after {error_time:.2f}s)")
        return jsonify({
            "error": "Internal server error during object detection",
            "details": str(e),
            "processing_time": {
                "error_after_ms": round(error_time * 1000, 2)
            },
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large",
        "message": "The uploaded file exceeds the maximum allowed size (10MB)",
        "timestamp": datetime.datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later",
        "timestamp": datetime.datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🚀 Starting YOLOv8 Object Detection API Server...")
    print(f"📊 Model Status: {'✅ Loaded' if model_loaded else '❌ Failed'}")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000")
    print("❤️ Health Check: http://localhost:5000/health")
    print("-" * 50)
    
    # Set maximum file size to 10MB
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
    
    # Run with reduced reloader sensitivity
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        use_reloader=True,
        reloader_type='stat'  # Use stat-based reloader instead of watchdog
    )

"""
app.py
Flask Web Application for Plant Disease Detection.
Provides a modern web UI for leaf photo uploads, real-time disease diagnosis,
confidence score visualization, treatment recommendations, and a RESTful API.
"""

import os
import io
import time
import json
import base64
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model import load_trained_model, predict_leaf_image

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "static", "samples")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DISEASE_INFO_FILE = os.path.join(BASE_DIR, "disease_info.json")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Global model state
MODEL = None
CLASS_TO_IDX = None
IDX_TO_CLASS = None
DEVICE = None
DISEASE_INFO = {}

def load_disease_database():
    """Loads botanical knowledge base."""
    global DISEASE_INFO
    if os.path.exists(DISEASE_INFO_FILE):
        with open(DISEASE_INFO_FILE, "r") as f:
            DISEASE_INFO = json.load(f)
    else:
        DISEASE_INFO = {}

def get_or_load_model():
    """Lazy loader for PyTorch model."""
    global MODEL, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE
    if MODEL is None:
        model_path = os.path.join(MODELS_FOLDER, "plant_disease_model.pth")
        class_path = os.path.join(MODELS_FOLDER, "class_indices.json")
        if not os.path.exists(model_path) or not os.path.exists(class_path):
            return None, None, None, None
        MODEL, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE = load_trained_model(
            model_path=model_path,
            class_indices_path=class_path
        )
    return MODEL, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE

# Load knowledge base on startup
load_disease_database()

def allowed_file(filename):
    """Check if uploaded file extension is supported."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Main dashboard with leaf scanner, sample gallery, and disease info."""
    model, _, idx_to_class, _ = get_or_load_model()
    model_ready = model is not None
    supported_classes = list(idx_to_class.values()) if idx_to_class else []

    # Load samples manifest if available
    samples = []
    samples_manifest_path = os.path.join(SAMPLES_FOLDER, "samples.json")
    if os.path.exists(samples_manifest_path):
        try:
            with open(samples_manifest_path, "r") as f:
                samples = json.load(f)
        except Exception:
            samples = []

    return render_template(
        "index.html",
        model_ready=model_ready,
        supported_classes=supported_classes,
        samples=samples,
        disease_info=DISEASE_INFO
    )

@app.route("/about")
def about():
    """About page with technical architecture, dataset specs, and training metrics."""
    history = []
    history_file = os.path.join(MODELS_FOLDER, "training_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    return render_template(
        "about.html",
        history=history,
        total_diseases=len(DISEASE_INFO)
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Predict disease for an uploaded leaf image file (Form or AJAX)."""
    model, _, idx_to_class, device = get_or_load_model()
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model is not trained or loaded yet. Please run train.py first."
        }), 503

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file format. Please upload JPG, PNG, or WEBP."}), 400

    try:
        t0 = time.time()
        filename = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        saved_filename = f"upload_{timestamp}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        file.save(file_path)

        # Open and run inference
        with Image.open(file_path) as img:
            result = predict_leaf_image(
                model=model,
                image=img,
                idx_to_class=idx_to_class,
                device=device,
                disease_info=DISEASE_INFO,
                top_k=3
            )

        inference_time_ms = round((time.time() - t0) * 1000, 1)
        result["image_url"] = f"/static/uploads/{saved_filename}"
        result["inference_time_ms"] = inference_time_ms
        result["success"] = True

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic leaf disease prediction.
    Accepts:
      - Multipart/form-data with file in 'image' field
      - JSON body with 'image_base64' (Base64 data or data URL)
    """
    model, _, idx_to_class, device = get_or_load_model()
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Ensure plant_disease_model.pth exists."
        }), 503

    image = None
    try:
        t0 = time.time()
        # Check multipart file
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
        # Check JSON payload
        elif request.is_json:
            data = request.get_json()
            if "image_base64" in data:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(img_bytes))

        if image is None:
            return jsonify({
                "status": "error",
                "message": "Missing image. Provide 'image' in multipart form-data or 'image_base64' in JSON."
            }), 400

        result = predict_leaf_image(
            model=model,
            image=image,
            idx_to_class=idx_to_class,
            device=device,
            disease_info=DISEASE_INFO,
            top_k=3
        )
        inference_time_ms = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "status": "success",
            "prediction": {
                "class": result["class_raw"],
                "crop": result["crop"],
                "disease": result["display_name"],
                "is_healthy": result["is_healthy"],
                "confidence_percent": result["confidence"],
                "severity": result["details"].get("severity", "N/A"),
                "pathogen": result["details"].get("pathogen", "N/A"),
                "symptoms": result["details"].get("symptoms", []),
                "treatment": result["details"].get("treatment", []),
                "prevention": result["details"].get("prevention", "")
            },
            "top_candidates": result["top_predictions"],
            "inference_time_ms": inference_time_ms
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/sample/<filename>")
def predict_sample(filename):
    """Run prediction on a preloaded sample image from static/samples."""
    model, _, idx_to_class, device = get_or_load_model()
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    safe_name = secure_filename(filename)
    sample_path = os.path.join(SAMPLES_FOLDER, safe_name)
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": f"Sample '{safe_name}' not found."}), 404

    try:
        t0 = time.time()
        with Image.open(sample_path) as img:
            result = predict_leaf_image(
                model=model,
                image=img,
                idx_to_class=idx_to_class,
                device=device,
                disease_info=DISEASE_INFO,
                top_k=3
            )
        result["image_url"] = f"/static/samples/{safe_name}"
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/diseases")
def api_diseases():
    """List all supported disease conditions and treatments."""
    return jsonify({
        "total_diseases": len(DISEASE_INFO),
        "diseases": DISEASE_INFO
    })

@app.route("/api/samples")
def api_samples():
    """List all available UI demo sample images."""
    samples_path = os.path.join(SAMPLES_FOLDER, "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Plant Disease Detector Web Server")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)

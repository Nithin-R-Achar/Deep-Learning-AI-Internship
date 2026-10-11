import os
import json
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from model import MNISTNet
from preprocess import preprocess_canvas_image

app = Flask(__name__)

# Model state
MODEL_PATH = "mnist_cnn.pth"
META_PATH = "model_meta.json"
device = torch.device("cpu")  # CPU inference is instant (<5ms)
model = None

def load_trained_model():
    global model
    model = MNISTNet().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print("Successfully loaded model weights from", MODEL_PATH)
            return True
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return False
    else:
        print(f"Model checkpoint not found at {MODEL_PATH}. Waiting for training to finish.")
        return False

# Attempt initial load
load_trained_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        if not load_trained_model():
            return jsonify({
                "status": "error",
                "message": "Model is still training or checkpoint not found. Please wait a few moments."
            }), 503

    try:
        data = request.get_json(force=True)
        image_data = data.get("image")
        if not image_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Preprocess
        result = preprocess_canvas_image(image_data)
        if result["is_empty"]:
            return jsonify({
                "status": "empty",
                "message": "Canvas is clear. Draw a digit to see predictions."
            })

        # Inference
        tensor = result["tensor"].to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
            pred_digit = int(torch.argmax(outputs, dim=1).item())
            confidence = probabilities[pred_digit] * 100.0

        # Format probabilities formatted to 2 decimal places
        prob_list = [round(float(p) * 100, 2) for p in probabilities]

        return jsonify({
            "status": "success",
            "prediction": pred_digit,
            "confidence": round(confidence, 1),
            "probabilities": prob_list,
            "preview_image": result["preview_base64"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/meta", methods=["GET"])
def get_meta():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                data = json.load(f)
            return jsonify({"status": "success", "data": data})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "pending", "message": "Training metadata not ready yet."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

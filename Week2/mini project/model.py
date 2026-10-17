"""
model.py
Pre-trained Deep Learning model architecture and inference pipeline
for Plant Disease Detection using PyTorch and MobileNetV2.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ImageNet normalization statistics standard for PyTorch pre-trained backbones
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline for input images
inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Training data augmentation transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def build_model(num_classes, pretrained=True, freeze_backbone=False):
    """
    Constructs MobileNetV2 with custom classification head for plant diseases.
    
    Args:
        num_classes (int): Number of plant disease classes.
        pretrained (bool): Load ImageNet pre-trained weights.
        freeze_backbone (bool): Freeze lower convolutional feature layers.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier with custom head matching num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

def load_trained_model(
    model_path="models/plant_disease_model.pth",
    class_indices_path="models/class_indices.json",
    device=None
):
    """
    Loads saved model weights and class mapping.
    
    Returns:
        tuple: (model, class_to_idx, idx_to_class, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices file not found at: {class_indices_path}")

    with open(class_indices_path, "r") as f:
        idx_to_class_raw = json.load(f)
        # Ensure keys are integers
        idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}

    num_classes = len(idx_to_class)
    model = build_model(num_classes=num_classes, pretrained=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_to_idx = {v: k for k, v in idx_to_class.items()}
    return model, class_to_idx, idx_to_class, device

def format_class_name(raw_name):
    """Converts raw folder class name to human-readable string."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{crop} - {condition}"

def predict_leaf_image(model, image, idx_to_class, device=None, disease_info=None, top_k=3):
    """
    Runs model inference on a PIL image and returns disease predictions and recommendations.
    
    Args:
        model: PyTorch model
        image (PIL.Image): Input leaf image
        idx_to_class (dict): Index to class name mapping
        device: torch.device
        disease_info (dict): Optional loaded disease_info.json dictionary
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Prediction results including top class, confidence, and details.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = inference_transforms(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    num_classes = len(idx_to_class)
    k = min(top_k, num_classes)
    top_probs, top_indices = torch.topk(probabilities, k=k)

    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()

    top_class_raw = idx_to_class[top_indices[0]]
    confidence = float(top_probs[0])

    # Top predictions list
    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        c_raw = idx_to_class[idx]
        is_healthy = "healthy" in c_raw.lower()
        info_entry = disease_info.get(c_raw, {}) if disease_info else {}
        top_predictions.append({
            "class_raw": c_raw,
            "display_name": info_entry.get("disease", format_class_name(c_raw)),
            "crop": info_entry.get("crop", c_raw.split("___")[0].replace("_", " ")),
            "confidence": round(prob * 100, 2),
            "is_healthy": is_healthy,
            "severity": info_entry.get("severity", "Unknown")
        })

    # Disease details from knowledge base
    details = {}
    if disease_info and top_class_raw in disease_info:
        details = disease_info[top_class_raw]
    else:
        is_healthy = "healthy" in top_class_raw.lower()
        details = {
            "crop": top_class_raw.split("___")[0].replace("_", " "),
            "disease": format_class_name(top_class_raw),
            "is_healthy": is_healthy,
            "pathogen": "N/A" if is_healthy else "Pathogen information pending",
            "severity": "None" if is_healthy else "Moderate",
            "symptoms": ["Normal appearance"] if is_healthy else ["Visible spotting or discoloration on leaf surface"],
            "treatment": ["No treatment required"] if is_healthy else ["Isolate affected leaves and monitor surrounding plants"],
            "prevention": "Ensure good cultural sanitation and proper spacing."
        }

    return {
        "class_raw": top_class_raw,
        "display_name": details.get("disease", format_class_name(top_class_raw)),
        "crop": details.get("crop", top_class_raw.split("___")[0].replace("_", " ")),
        "is_healthy": details.get("is_healthy", "healthy" in top_class_raw.lower()),
        "confidence": round(confidence * 100, 2),
        "details": details,
        "top_predictions": top_predictions
    }

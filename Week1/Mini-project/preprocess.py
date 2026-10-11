import base64
import io
import math
import numpy as np
from PIL import Image
import torch

def preprocess_canvas_image(image_data_uri, target_size=28, box_size=20):
    """
    Takes a base64 data URI or raw base64 string from an HTML canvas and processes it
    to match the MNIST dataset specification:
    1. Grayscale conversion and stroke polarity detection (white digit on black background).
    2. Blank canvas detection.
    3. Bounding box cropping.
    4. Aspect ratio-preserving resize into a box_size x box_size frame.
    5. Centering in a 28x28 canvas using Center of Mass (intensity-weighted centroid).
    6. Normalization with MNIST mean (0.1307) and std (0.3081).
    7. Base64 encoding of the 28x28 processed image for visual feedback.
    """
    # 1. Decode base64
    if "," in image_data_uri:
        image_data_uri = image_data_uri.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data_uri)
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGBA
    img = img.convert("RGBA")
    arr = np.array(img)

    # 2. Extract grayscale intensity based on canvas mode
    # Check if canvas uses alpha channel (transparent background with drawn strokes)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    if np.any(alpha < 255):
        # Canvas has transparency. Non-transparent areas are the strokes.
        # Check average brightness of the stroke pixels
        mask = alpha > 20
        if np.any(mask):
            stroke_brightness = np.mean(rgb[mask])
            if stroke_brightness < 128:
                # Black stroke on transparent canvas -> use alpha as digit intensity
                gray = alpha.astype(np.float32)
            else:
                # White stroke on transparent canvas
                gray = (np.mean(rgb, axis=2) * (alpha / 255.0)).astype(np.float32)
        else:
            gray = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
    else:
        # Opaque canvas (RGB)
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        # In MNIST, digits are bright (white) on dark (black) background.
        # If the background is bright (white canvas), invert it.
        # Determine background color by sampling the 4 borders
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        if np.mean(border_pixels) > 127:
            # White background -> invert so digit is white (255)
            gray = 255.0 - gray

    # Threshold very small noise
    gray[gray < 25] = 0.0

    # 3. Blank canvas detection
    non_zero_coords = np.argwhere(gray > 25)
    if non_zero_coords.size == 0 or len(non_zero_coords) < 15:
        return {
            "is_empty": True,
            "tensor": None,
            "preview_base64": None
        }

    # 4. Bounding box cropping
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0)
    cropped = gray[y_min : y_max + 1, x_min : x_max + 1]

    h, w = cropped.shape
    if h == 0 or w == 0:
        return {"is_empty": True, "tensor": None, "preview_base64": None}

    # 5. Aspect-ratio preserving resize into box_size x box_size (e.g. 20x20)
    cropped_img = Image.fromarray(cropped.astype(np.uint8))
    if h > w:
        new_h = box_size
        new_w = max(1, int(round((w * box_size) / h)))
    else:
        new_w = box_size
        new_h = max(1, int(round((h * box_size) / w)))

    resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized, dtype=np.float32)

    # Place resized digit into a blank target_size x target_size (28x28) canvas initially centered
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    start_y = (target_size - new_h) // 2
    start_x = (target_size - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_arr

    # 6. Fine-tune centering by Center of Mass (intensity-weighted centroid)
    total_mass = np.sum(canvas)
    if total_mass > 0:
        y_indices, x_indices = np.indices(canvas.shape)
        cy = np.sum(y_indices * canvas) / total_mass
        cx = np.sum(x_indices * canvas) / total_mass

        target_center = (target_size - 1) / 2.0  # 13.5 for 28x28
        shift_y = int(round(target_center - cy))
        shift_x = int(round(target_center - cx))

        # Perform translation with zero padding
        shifted_canvas = np.zeros_like(canvas)
        src_y_start = max(0, -shift_y)
        src_y_end = min(target_size, target_size - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(target_size, target_size + shift_y)

        src_x_start = max(0, -shift_x)
        src_x_end = min(target_size, target_size - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(target_size, target_size + shift_x)

        if (dst_y_end > dst_y_start) and (dst_x_end > dst_x_start):
            shifted_canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = canvas[src_y_start:src_y_end, src_x_start:src_x_end]
            canvas = shifted_canvas

    # 7. Generate 28x28 visual preview (as base64 PNG)
    preview_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    # Resize preview with nearest neighbor so pixels are crisp when rendered in UI
    preview_display = preview_img.resize((140, 140), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    preview_display.save(buffer, format="PNG")
    preview_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 8. Normalize for MNIST PyTorch model: (x/255.0 - 0.1307) / 0.3081
    normalized = (canvas / 255.0 - 0.1307) / 0.3081
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    return {
        "is_empty": False,
        "tensor": tensor,
        "preview_base64": preview_base64
    }

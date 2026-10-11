# ✍️ Real-Time Handwritten Digit Recognizer (MNIST CNN)

A complete end-to-end Deep Learning web application that recognizes handwritten digits in real time. Built with **PyTorch**, a custom Convolutional Neural Network (CNN) trained on the **MNIST** dataset, and an interactive **HTML5 Canvas** web interface served via **Flask**.

---

## 🌟 Key Features

1. **High-Accuracy CNN Architecture**:
   - 2 Convolutional blocks with 32 and 64 feature maps.
   - Batch Normalization (`BatchNorm2d`, `BatchNorm1d`) for fast convergence and training stability.
   - Dropout layers (25% and 40%) for strong regularization against overfitting.
   - Reaches **>99% test accuracy** on the MNIST test set.

2. **Faithful MNIST Preprocessing Pipeline**:
   - **Bounding Box Cropping**: Detects the bounding box of the drawn stroke.
   - **Aspect-Ratio Preserved Resizing**: Resizes the digit to fit a 20×20 box within a 28×28 canvas.
   - **Center of Mass (Centroid) Alignment**: Automatically calculates the intensity-weighted center of gravity and shifts the digit to the exact center (matching the NIST/MNIST normalization standard).
   - **MNIST Standard Normalization**: Normalizes inputs using MNIST dataset statistics ($\mu = 0.1307$, $\sigma = 0.3081$).

3. **Interactive & Responsive Web Interface**:
   - **Real-Time Inference**: Predicts digits instantly (<15ms latency) as you draw on the canvas.
   - **Full Probability Distribution**: Interactive dynamic bar charts showing confidence scores for all 10 digits (0 through 9).
   - **"What the AI Sees" Preview**: Live display of the preprocessed 28×28 pixel tensor fed into the network.
   - **Canvas Tools**: Adjustable stroke thickness, Clear (`C`), Undo (`Ctrl+Z`), and 1-click Preset Digits (0–9) for instant testing.
   - **Touch & Mouse Support**: Smooth drawing on desktops, tablets, and smartphones.

---

## 📁 Project Structure

```
├── app.py              # Flask server and prediction API
├── model.py            # PyTorch MNISTNet CNN architecture definition
├── train.py            # Training script with data augmentation & metric tracking
├── preprocess.py       # Center-of-mass preprocessing pipeline
├── requirements.txt    # Python dependencies
├── mnist_cnn.pth       # Trained model checkpoint weights
├── model_meta.json     # Saved training statistics and test accuracy
├── static/
│   ├── css/
│   │   └── style.css   # Dark-mode styling, glassmorphism, responsive grid
│   └── js/
│       └── app.js      # Canvas drawing engine, live debounce, API client
└── templates/
    └── index.html      # Frontend interface
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. (Optional) Re-train the Model
To re-train the neural network from scratch:
```bash
python train.py
```
This downloads the MNIST dataset (if not already present), trains for 5 epochs with data augmentation, evaluates test set accuracy, and saves the best model checkpoint to `mnist_cnn.pth`.

---

## 🔬 Model Architecture Details

| Layer | Operation | Output Shape | Details |
|---|---|---|---|
| **Input** | Grayscale Image | `[1, 28, 28]` | Normalized ($\mu=0.1307, \sigma=0.3081$) |
| **Conv Block 1** | Conv2D + BN + ReLU | `[32, 28, 28]` | 32 filters, 3×3, pad 1 |
| | Conv2D + BN + ReLU | `[32, 28, 28]` | 32 filters, 3×3, pad 1 |
| | MaxPool2D + Dropout | `[32, 14, 14]` | 2×2 pool, 25% dropout |
| **Conv Block 2** | Conv2D + BN + ReLU | `[64, 14, 14]` | 64 filters, 3×3, pad 1 |
| | Conv2D + BN + ReLU | `[64, 14, 14]` | 64 filters, 3×3, pad 1 |
| | MaxPool2D + Dropout | `[64, 7, 7]` | 2×2 pool, 25% dropout |
| **Dense Block** | Linear + BN + ReLU | `[128]` | 3,136 $\to$ 128 |
| | Dropout | `[128]` | 40% dropout |
| **Output** | Linear (Logits) | `[10]` | 10 classes (digits 0–9) |

---

## 🔌 API Reference

### `POST /predict`
Performs real-time digit classification on a canvas drawing.

- **Request Body (JSON)**:
  ```json
  {
    "image": "data:image/png;base64,iVBORw0KGgo..."
  }
  ```

- **Response (JSON)**:
  ```json
  {
    "status": "success",
    "prediction": 7,
    "confidence": 99.4,
    "probabilities": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 99.4, 0.2, 0.3],
    "preview_image": "data:image/png;base64,..."
  }
  ```

### `GET /meta`
Returns the metadata and evaluation accuracy of the trained model checkpoint.

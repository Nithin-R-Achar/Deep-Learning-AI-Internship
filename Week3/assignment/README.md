# Fashion-MNIST Image Classification Using CNN

## 1. Project Title
**Fashion Item Recognition Using Convolutional Neural Network (CNN)**

## 2. Introduction
Fashion-MNIST Image Classification is a deep learning mini project that identifies clothing items from images. It uses a Convolutional Neural Network (CNN) and a fully connected neural network to classify clothing into different categories.

## 3. Objectives
- Load and visualize Fashion-MNIST images.
- Build a fully connected neural network.
- Build a simple CNN model.
- Compare the accuracy of both models.
- Plot training and validation accuracy and loss.
- Display and explain five misclassified images.
- Save the trained CNN model.

## 4. Dataset
The project uses the Fashion-MNIST dataset.

It contains 70,000 grayscale images:
- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Number of categories: 10

### Clothing Categories
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## 5. Technologies Used
- Python
- TensorFlow
- NumPy
- Matplotlib
- Convolutional Neural Network (CNN)

## 6. Installation
Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib
```

## 7. How to Run the Project
1. Install Python on your computer.
2. Install the required libraries.
3. Save the Python code as `fashion_mnist_project.py`.
4. Open the terminal in the project folder.
5. Run the following command:

```bash
python fashion_mnist_project.py
```

The dataset may download automatically the first time, so an internet connection may be required.

## 8. Project Workflow

### Step 1: Load and Visualize Images
The Fashion-MNIST dataset is loaded, and ten sample clothing images are displayed.

### Step 2: Fully Connected Neural Network
A fully connected neural network is created using Flatten and Dense layers. It learns from the image pixel values.

### Step 3: Build CNN
The CNN uses convolution, max-pooling, flattening, and dense layers to learn image patterns and classify clothing.

### Step 4: Compare Models
Both models are evaluated using test images. Their accuracy values are displayed for comparison.

### Step 5: Plot Graphs
Training and validation accuracy and loss are plotted to understand model learning performance.

### Step 6: Analyze Misclassified Images
Five incorrectly classified images are displayed with their actual and predicted clothing categories.

Misclassification may occur because some clothing categories have similar shapes or unclear details.

### Step 7: Save the Model
The trained CNN model is saved as:

`fashion_mnist_cnn.keras`

## 9. Model Comparison

| Model | Test Accuracy |
|---|---|
| Fully Connected Neural Network | Obtained after running the code |
| Convolutional Neural Network (CNN) | Obtained after running the code |

The CNN may perform better because it learns spatial patterns and image features. The actual accuracy depends on training results.

## 10. Output
The project produces:
- Sample clothing images
- Fully connected model accuracy
- CNN model accuracy
- Training and validation accuracy graph
- Training and validation loss graph
- Five misclassified images
- Saved CNN model file

## 11. Advantages
- Simple and easy to understand.
- Uses a standard image classification dataset.
- Compares two deep learning approaches.
- Displays graphs and incorrect predictions.
- Saves the trained model for future use.

## 12. Limitations
- Uses small grayscale images.
- Trains for only three epochs.
- Similar clothing categories can be confused.
- Requires TensorFlow and other libraries.

## 13. Conclusion
This mini project demonstrates how deep learning can recognize clothing images using the Fashion-MNIST dataset. A fully connected neural network and a CNN are trained and compared. Training graphs help evaluate model performance, while five misclassified examples show where the CNN makes errors. Finally, the trained CNN model is saved for future use.

## 14. Future Improvements
- Train for more epochs.
- Improve CNN architecture.
- Use data augmentation.
- Develop a simple interface for uploading and classifying clothing images.

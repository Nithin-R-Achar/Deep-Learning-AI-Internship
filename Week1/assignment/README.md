# Numerical Pattern Predictor

**Skill Nexis – Deep Learning Week 1 Mini Project**

## 1. Project Overview

Numerical Pattern Predictor is a simple machine learning project developed using Python and PyTorch. It learns a mathematical relationship from numerical data and predicts outputs for new inputs.

## 2. Objective

To train a model that learns the pattern:

`y = 3x + 2`

The model predicts output values based on input values and compares actual results with predicted results.

## 3. Technologies Used

* Python
* PyTorch
* Matplotlib

## 4. Features

* Generates numerical data.
* Trains a model using weight and bias.
* Uses gradient descent and backpropagation.
* Compares two learning rates: `0.01` and `0.001`.
* Splits data into training, validation, and testing sets.
* Displays training and validation loss graphs.
* Displays actual versus predicted values.
* Prints prediction results for new inputs.

## 5. Working Principle

1. Generate numerical data using `y = 3x + 2`.
2. Divide the data into training, validation, and testing sets.
3. Initialize weight and bias.
4. Train the model to reduce prediction error.
5. Compare two learning rates.
6. Evaluate predictions using test data.
7. Display graphs and print the results.

## 6. Installation

Install the required libraries:

```bash
pip install torch matplotlib
```

## 7. How to Run

1. Open the Python file or Jupyter Notebook.
2. Run the code from beginning to end.
3. Observe the loss curves and actual-versus-predicted graph.
4. Check the printed prediction values.

## 8. Example

For the pattern `y = 3x + 2`:

* Input: `4`
* Expected output: `14`

The model should predict a value close to 14.

## 9. Expected Output

* Training and validation loss graph.
* Actual versus predicted graph.
* Printed actual and predicted values.
* Prediction for a new input.
* Test loss value.

## 10. Conclusion

The Numerical Pattern Predictor demonstrates how PyTorch can learn a numerical relationship using weight, bias, gradient descent, and backpropagation. The project compares learning rates and evaluates predictions using loss values and graphs.

## Author

Add your name here.

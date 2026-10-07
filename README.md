# Handwritten Digit Recognition Using CNN

A beginner-friendly deep learning mini-project that recognizes handwritten digits from images using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Overview

Flow:

Upload handwritten digit image -> Segment digits -> Preprocess each digit -> CNN model -> Predict all digits -> Display sequence and confidence

The model is trained on MNIST, which contains grayscale images of handwritten digits from 0 to 9.

## Folder Structure

```text
Handwritten Digit Recognition Using CNN/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- digit_model.keras        # Created after training the model
```

The model is trained on MNIST, which contains grayscale images of handwritten digits from 0 to 9. The app can handle images that contain multiple digits and will predict them in reading order.

- Python
- TensorFlow/Keras
- CNN
- MNIST dataset
- NumPy
- Pillow
- Streamlit

## Installation Commands

Create and activate a virtual environment if you want an isolated setup, then install the required packages:

```bash
pip install -r requirements.txt
```

If you prefer installing packages one by one:

```bash
pip install tensorflow streamlit pillow numpy
```

## Commands to Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

- download MNIST automatically through TensorFlow/Keras
- normalize the images
- reshape them for CNN input
- train the CNN
- evaluate it on the test dataset
- save the trained model as `digit_model.keras`

## Commands to Run the Streamlit App

After training finishes, start the web app:

```bash
streamlit run app.py
```

## How the App Works

1. The user uploads a JPG, JPEG, or PNG image.
2. The app displays the uploaded image.
3. The image is converted to grayscale.
4. The app detects the digit shapes using simple image segmentation.
5. Each digit is cropped separately.
6. Every crop is resized to 28 x 28 pixels and normalized to the 0 to 1 range.
7. The app checks whether the image needs inversion so it matches MNIST-style digits.
8. Each digit crop is sent to the CNN.
9. The app shows the full predicted digit sequence and confidence for each digit.

## Important Code Explanation

### 1. Loading MNIST

In `train_model.py`, the dataset is loaded using:

```python
keras.datasets.mnist.load_data()
```

This gives training and testing images of handwritten digits.

### 2. Preprocessing

The images are divided by 255 so the pixel values move from 0 to 1. CNNs usually learn better when inputs are normalized.

The images are also reshaped from 28 x 28 to 28 x 28 x 1 so the CNN can treat them as grayscale images with a channel dimension.

### 3. CNN Architecture

The model contains:

- `Conv2D` layers to detect patterns such as edges and strokes
- `ReLU` activation to add non-linearity
- `MaxPooling2D` layers to reduce the size of the feature maps
- `Flatten` to convert 2D features into a 1D vector
- `Dense` layers to make the final decision
- `Softmax` output with 10 classes for digits 0 to 9

### 4. Compilation

The model uses:

- `Adam` optimizer for efficient learning
- `sparse_categorical_crossentropy` loss for multi-class classification
- `accuracy` as the metric

### 5. Training and Evaluation

The model trains on the MNIST training set, validates on part of the training data, and then evaluates itself on the test set.

### 6. Saving the Model

After training, the model is saved to `digit_model.keras` so the Streamlit app can load it later without retraining.

### 7. Streamlit App

The app uses `st.file_uploader()` for image upload, `PIL` for image handling, and TensorFlow/Keras to make predictions.

## Deep Learning Concepts Explained

### CNN

A Convolutional Neural Network is a deep learning model made for image data. It learns visual patterns automatically.

### Convolution

Convolution uses small filters to scan an image and detect useful patterns like lines, curves, and edges.

### Pooling

Pooling reduces the size of feature maps while keeping the important information. This makes the model faster and helps reduce overfitting.

### ReLU

ReLU stands for Rectified Linear Unit. It keeps positive values and turns negative values into zero. This helps the network learn complex patterns.

### Flattening

Flattening converts the 2D output of convolution layers into a 1D vector so it can be passed into dense layers.

### Softmax

Softmax converts the final outputs into probabilities that add up to 1. The highest probability is the predicted digit.

### Backpropagation

Backpropagation is the process used by neural networks to adjust their weights by calculating how wrong the prediction was and moving the weights in the right direction.

### Adam Optimizer

Adam is a popular optimization algorithm that updates model weights efficiently using adaptive learning rates. It usually works well without much tuning.

## Notes

- If `digit_model.keras` is missing, run `python train_model.py` first.
- The app works best when digits are separated and clearly visible.
- For best results, use an image with a white background and dark handwritten digits.
- If digits touch each other, the segmentation step may not split them perfectly.

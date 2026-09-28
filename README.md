# MLP Models

A learning-focused repository for implementing and training **Multilayer Perceptron (MLP)** neural networks and understanding the fundamental concepts behind neural networks and deep learning.

The primary goal of this repository is not simply to train models, but to understand **how a neural network learns from data**, how its predictions are evaluated, and how training parameters affect its performance.

## 🎯 Objectives

This repository explores the core concepts involved in training an MLP, including:

* Neural network architecture
* Input, hidden, and output layers
* Neurons and weights
* Biases
* Activation functions
* Forward propagation
* Loss functions
* Backpropagation
* Gradient descent
* Learning rate
* Epochs and batches
* Training and validation
* Test accuracy
* Overfitting and underfitting
* Loss and accuracy curves
* Model evaluation

## 🧠 Dataset

The primary dataset used in this project is **MNIST**, a dataset of handwritten digits ranging from `0` to `9`.

Each image is represented as a matrix of pixel values. Before being passed to the MLP, the image data is transformed into a numerical representation that can be processed by the network.

The model learns patterns in the pixel values and uses those patterns to classify images into their corresponding digit classes.

## 🏗️ Model

The main model is a **Multilayer Perceptron (MLP)** consisting of:

```text
Input Layer
     ↓
Hidden Layer(s)
     ↓
Output Layer
```

The input represents the pixels of an image, while the output represents the probability of the image belonging to each of the ten digit classes.

The project will experiment with different architectures and training configurations to understand how they affect learning and performance.

## 📊 Training

During training, the model repeatedly performs the following process:

```text
Input Data
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Update Weights
    ↓
Repeat
```

The training process is monitored using metrics such as:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* Test accuracy

Loss and accuracy graphs are used to visualize how the model learns over time.

## 🔬 Experiments

Experiments will be used to investigate questions such as:

* How does the number of hidden layers affect performance?
* How does the number of neurons affect learning?
* What happens when the learning rate changes?
* How does batch size affect training?
* How does the choice of activation function affect the model?
* What does overfitting look like?
* How can training and validation loss be interpreted?
* How does the model's test accuracy relate to its training performance?

The purpose of these experiments is to connect the mathematical concepts behind neural networks with observable model behavior.

## 🛠️ Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook

## 📁 Repository Structure

```text
MLP-Models/
│
├── notebooks/
│   └── ...
│
├── models/
│   └── ...
│
├── experiments/
│   └── ...
│
├── results/
│   └── ...
│
├── requirements.txt
└── README.md
```

The structure may evolve as additional experiments and models are added.

## 🚀 Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd MLP-Models
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## 📈 Results

Model results, training curves, and experiment outputs will be documented as the project develops.

Example metrics include:

| Metric              | Description                                           |
| ------------------- | ----------------------------------------------------- |
| Training Loss       | Measures how well the model fits the training data    |
| Validation Loss     | Measures performance on unseen validation data        |
| Training Accuracy   | Percentage of correctly classified training samples   |
| Validation Accuracy | Percentage of correctly classified validation samples |
| Test Accuracy       | Final performance on the test dataset                 |

## 📚 Learning Focus

This repository is part of my ongoing study of **machine learning and deep learning fundamentals**.

The emphasis is on understanding the complete training process rather than treating neural networks as black boxes.

In particular, I am focusing on being able to explain:

> **What happens to the input as it moves through the network, how the model produces a prediction, how the loss is calculated, and how the weights are updated so the model can improve.**

## 🔮 Future Work

Planned improvements and experiments include:

* Experimenting with different MLP architectures
* Comparing activation functions
* Visualizing training and validation curves
* Investigating over-fitting and regularization
* Experimenting with different optimizers
* Comparing different learning rates and batch sizes
* Implementing additional datasets
* Exploring convolutional neural networks after establishing a strong understanding of MLPs

## 📌 Status

**In development**

This repository will evolve as new concepts, experiments, and implementations are added.

# Neural Network Implementation for Fashion-MNIST Classification

This project implements a custom neural network for classifying Fashion-MNIST images. It includes multiple optimization algorithms, hyperparameter tuning, and visualization tools.

## Features
- **Custom Neural Network Implementation**
- **Multiple Optimization Algorithms:**
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Nesterov Accelerated Gradient (NAG)
  - RMSprop
  - Adam
  - NAdam
- **Flexible Network Architecture**
- **Hyperparameter Tuning with Wandb**
- **Comprehensive Visualization and Analysis**

## Project Structure
```
DA6401_assignment1/
│
├── models/
│   ├── neural_network.py   # Neural network implementation
│   └── optimizers.py       # Optimization algorithms
│
├── utils/
│   ├── data_loader.py      # Data preprocessing
│   ├── activations.py      # Activation functions
│   └── losses.py           # Loss function implementations
│
├── train.py                # Main training script
├── confusion_matrix.py      # Confusion matrix visualization
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Training the Model
- **Standard Training:**
  ```bash
  python train.py
  ```
- **Custom Configuration:**
  ```bash
  python train.py --optimizer adam --learning_rate 0.001 --num_layers 3 --hidden_size 128
  ```
- **Hyperparameter Sweep:**
  ```bash
  python train.py sweep
  ```
- **Confusion Matrix Visualization:**
  ```bash
  python confusion_matrix.py
  ```

## Hyperparameter Configuration
The project supports extensive hyperparameter tuning:
- **Epochs:** 5, 10
- **Hidden Layers:** 3, 4, 5
- **Hidden Layer Size:** 32, 64, 128
- **Weight Decay:** 0, 0.0005, 0.5
- **Learning Rate:** 1e-3, 1e-4
- **Optimizers:** SGD, Momentum, Nesterov Accelerated Gradient, RMSprop, Adam, NAdam
- **Batch Size:** 16, 32, 64
- **Weight Initialization:** Random, Xavier
- **Activation Functions:** Sigmoid, Tanh, ReLU

## Wandb Integration
The project uses **Weights & Biases (Wandb)** for:
- Experiment tracking
- Hyperparameter optimization
- Result visualization
- Performance comparison

## Key Implementations
### Neural Network
- Flexible architecture
- Manual backpropagation
- Multiple activation functions

### Optimizers
- Implemented from scratch
- Support for various optimization algorithms

### Visualization
- Confusion matrix
- Performance metrics
- Hyperparameter importance

## Requirements
The project depends on the following libraries:
- Fashion-MNIST Dataset
- Weights & Biases (wandb)
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Seaborn
- Keras

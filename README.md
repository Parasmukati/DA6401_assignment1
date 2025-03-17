# Neural Network Implementation for Fashion-MNIST Classification

## Features

- Custom Neural Network Implementation
- Multiple Optimization Algorithms
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Nesterov Accelerated Gradient (NAG)
  - RMSprop
  - Adam
  - NAdam
- Flexible Network Architecture
- Hyperparameter Tuning with Wandb
- Comprehensive Visualization and Analysis

## Project Structure
neural_network_assignment/
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


## Training the Model
1. Standard Training:
    python train.py

2. Custom Configuration:
    python train.py --optimizer adam --learning_rate 0.001 --num_layers 3 --hidden_size 128

3. Hyperparameter Sweep
    python train.py sweep

4. Confusion Matrix Visualization
    python confusion_matrix.py

## Hyperparameter Configuration
The project supports extensive hyperparameter tuning:

Epochs: 5, 10
Hidden Layers: 3, 4, 5
Hidden Layer Size: 32, 64, 128
Weight Decay: 0, 0.0005, 0.5
Learning Rate: 1e-3, 1e-4
Optimizers:
SGD
Momentum
Nesterov Accelerated Gradient
RMSprop
Adam
NAdam
Batch Size: 16, 32, 64
Weight Initialization: Random, Xavier
Activation Functions: Sigmoid, Tanh, ReLU

## Wandb Integration
The project uses Weights & Biases (Wandb) for:

Experiment tracking
Hyperparameter optimization
Result visualization
Performance comparison

## Key Implementations
Neural Network

Flexible architecture
Manual backpropagation
Multiple activation functions
Optimizers

Implemented from scratch
Support for various optimization algorithms
Visualization

Confusion matrix
Performance metrics
Hyperparameter importance

## Requirement
Fashion-MNIST Dataset
Weights & Biases
NumPy
Matplotlib
Scikit-learn
numpy
pandas
matplotlib
seaborn
scikit-learn
wandb
keras

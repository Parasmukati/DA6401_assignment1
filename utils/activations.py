import numpy as np

def sigmoid(Z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)

def tanh(Z):
    """Tanh activation function"""
    return np.tanh(Z)

def softmax(Z):
    """Softmax activation function"""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
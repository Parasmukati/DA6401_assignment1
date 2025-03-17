import numpy as np
from utils.activations import sigmoid, relu, tanh, softmax

class FeedForwardNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation, weight_init="random"):
        """
        Initialize neural network with flexible hidden layers
        
        Args:
            input_size (int): Size of input layer (784 for Fashion-MNIST)
            hidden_sizes (list): List of sizes for each hidden layer [64, 32, 16] for 3 hidden layers
            output_size (int): Size of output layer (10 for Fashion-MNIST)
            activation (str): Activation function type
            weight_init (str): Weight initialization method
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.num_layers = len(hidden_sizes)
        
        # Initialize weights and biases
        self.parameters = self.initialize_parameters()
    
    def activation_function(self, Z, activation):
        """Apply activation function"""
        if activation == "sigmoid":
            return sigmoid(Z)
        elif activation == "tanh":
            return tanh(Z)
        elif activation == "ReLU":
            return relu(Z)
        else:  # identity
            return Z

    def initialize_parameters(self):
        """
        Initialize network parameters for all layers
        """
        parameters = {}
        
        # Create list of layer sizes including input and output
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            if self.weight_init == "Xavier":
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
                parameters[f'W{i+1}'] = np.random.normal(0, scale, (layer_sizes[i+1], layer_sizes[i]))
            else:  # Random initialization
                parameters[f'W{i+1}'] = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            
            parameters[f'b{i+1}'] = np.zeros((layer_sizes[i+1], 1))
            
        return parameters
    
    def forward(self, X):
        """
        Forward propagation through all layers
        
        Args:
            X: Input data of shape (input_size, batch_size)
        Returns:
            cache: Dictionary containing intermediate values
            Y_hat: Output probabilities
        """
        cache = {'A0': X}
        A = X
        
        # Hidden layers
        for i in range(self.num_layers):
            Z = np.dot(self.parameters[f'W{i+1}'], A) + self.parameters[f'b{i+1}']
            A = self.activation_function(Z, self.activation)
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A
        
        # Output layer
        Z_out = np.dot(self.parameters[f'W{self.num_layers+1}'], A) + self.parameters[f'b{self.num_layers+1}']
        Y_hat = softmax(Z_out)
        cache[f'Z{self.num_layers+1}'] = Z_out
        cache[f'A{self.num_layers+1}'] = Y_hat
        
        return cache, Y_hat
    


    def backward(self, cache, Y):
        """
        Backpropagation algorithm
        
        Args:
            cache: Dictionary containing intermediate values from forward pass
            Y: True labels (one-hot encoded)
        
        Returns:
            gradients: Dictionary containing gradients for all parameters
        """
        gradients = {}
        m = Y.shape[1]  # batch size
        
        # Output layer gradient
        dZ = cache[f'A{self.num_layers+1}'] - Y
        
        # Gradients for output layer
        gradients[f'W{self.num_layers+1}'] = (1/m) * np.dot(dZ, cache[f'A{self.num_layers}'].T)
        gradients[f'b{self.num_layers+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Backpropagate through hidden layers
        for l in range(self.num_layers, 0, -1):
            # Compute gradient of the previous layer
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            
            # Compute activation derivative
            if self.activation == "sigmoid":
                dZ = dA * (cache[f'A{l}'] * (1 - cache[f'A{l}']))
            elif self.activation == "tanh":
                dZ = dA * (1 - np.power(cache[f'A{l}'], 2))
            elif self.activation == "ReLU":
                dZ = dA * (cache[f'Z{l}'] > 0)
            else:  # identity
                dZ = dA
            
            # Compute gradients
            gradients[f'W{l}'] = (1/m) * np.dot(dZ, cache[f'A{l-1}'].T)
            gradients[f'b{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return gradients

    def predict(self, X):
        """Make predictions"""
        _, Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def print_network_structure(self):
        """
        Print the structure of the network
        """
        print("\nNetwork Structure:")
        print(f"Input Layer: {self.input_size} neurons")
        for i, size in enumerate(self.hidden_sizes):
            print(f"Hidden Layer {i+1}: {size} neurons")
        print(f"Output Layer: {self.output_size} neurons")
        print(f"Activation Function: {self.activation}")
        print(f"Weight Initialization: {self.weight_init}\n")
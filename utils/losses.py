import numpy as np

class LossFunctions:
    @staticmethod
    def cross_entropy(y_pred, y_true, epsilon=1e-15):
        """
        Cross-entropy loss with numerical stability
        
        Args:
            y_pred (np.array): Predicted probabilities
            y_true (np.array): True labels (one-hot encoded)
            epsilon (float): Small value to prevent log(0)
        
        Returns:
            float: Average cross-entropy loss
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[1]
        return loss
    
    @staticmethod
    def mean_squared_error(y_pred, y_true):
        """
        Mean Squared Error loss
        
        Args:
            y_pred (np.array): Predicted probabilities
            y_true (np.array): True labels (one-hot encoded)
        
        Returns:
            float: Average Mean Squared Error
        """
        return np.mean(np.sum(np.square(y_pred - y_true), axis=0))

    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        """
        Derivative of cross-entropy loss
        
        Args:
            y_pred (np.array): Predicted probabilities
            y_true (np.array): True labels (one-hot encoded)
        
        Returns:
            np.array: Gradient of loss with respect to predictions
        """
        return y_pred - y_true
    
    @staticmethod
    def mse_derivative(y_pred, y_true):
        """
        Derivative of Mean Squared Error loss
        
        Args:
            y_pred (np.array): Predicted probabilities
            y_true (np.array): True labels (one-hot encoded)
        
        Returns:
            np.array: Gradient of loss with respect to predictions
        """
        return 2 * (y_pred - y_true) / y_pred.shape[1]
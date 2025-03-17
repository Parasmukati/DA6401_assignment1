import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

def load_and_visualize_fashion_mnist():
    # Load the Fashion-MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Define class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create figure with 2x5 subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Plot one image for each class
    for i in range(10):
        # Find first instance of class i
        idx = np.where(y_train == i)[0][0]
        
        # Plot the image
        axes[i].imshow(X_train[idx], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{i}. {class_names[i]}')
    
    plt.tight_layout()
    
    # Initialize wandb
    wandb.init(project="fashion_mnist_sweep")
    
    # Log the plot to wandb
    wandb.log({"fashion_mnist_samples": wandb.Image(plt)})
    
    # Display dataset information
    print("Dataset shapes:")
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # Test the function
    (X_train, y_train), (X_test, y_test) = load_and_visualize_fashion_mnist()
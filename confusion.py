# In your train.py or create a new file confusion_matrix.py
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# Robust import handling
try:
    from neural_network import FeedForwardNN
    from optimizers import (
        SGD, Momentum, NAG, RMSprop, Adam, NAdam
    )
    from utils.losses import LossFunctions
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure your project structure is correct.")
    sys.exit(1)

def initialize_wandb(args):
    """
    Robust Wandb initialization
    
    Args:
        args: Argument parser namespace
    
    Returns:
        bool: Whether Wandb was successfully initialized
    """
    try:
        # Use environment variables as fallback
        project = args.wandb_project or os.environ.get('WANDB_PROJECT', 'neural-network-default')
        entity = args.wandb_entity or os.environ.get('WANDB_ENTITY')
        
        # Initialize Wandb
        wandb.init(
            project=project,
            entity=entity,
            config=vars(args)
        )
        return True
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without Wandb logging")
        return False

def train_epoch(model, X, y, optimizer, loss_fn='cross_entropy'):
    """
    Train model for one epoch with detailed error handling and logging
    """
    try:
        batch_size = 32  # Default batch size
        num_batches = X.shape[1] // batch_size
        total_loss = 0
        total_correct = 0
        
        for i in range(num_batches):
            # Get batch
            start = i * batch_size
            end = start + batch_size
            batch_X = X[:, start:end]
            batch_y = y[:, start:end]
            
            # Forward pass
            cache, y_pred = model.forward(batch_X)
            
            # Compute loss
            if loss_fn == 'cross_entropy':
                loss = LossFunctions.cross_entropy(y_pred, batch_y)
            else:
                loss = LossFunctions.mean_squared_error(y_pred, batch_y)
            
            # Backward pass
            gradients = model.backward(cache, batch_y)
            
            # Update parameters
            model.parameters = optimizer.update(model.parameters, gradients)
            
            # Compute accuracy
            predictions = np.argmax(y_pred, axis=0)
            true_labels = np.argmax(batch_y, axis=0)
            correct = np.sum(predictions == true_labels)
            
            total_loss += loss
            total_correct += correct
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / X.shape[1]
        
        return avg_loss, avg_accuracy
    
    except Exception as e:
        print(f"Error during training epoch: {e}")
        raise

def train(model, X_train, y_train, X_val, y_val, args):
    """
    Main training loop with comprehensive logging and error handling
    """
    # Initialize Wandb
    wandb_active = initialize_wandb(args)
    
    # Get optimizer
    optimizer = get_optimizer(args)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    try:
        for epoch in range(args.epochs):
            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model, X_train, y_train, 
                optimizer, 
                loss_fn=args.loss
            )
            
            # Validate
            val_loss, val_acc = train_epoch(
                model, X_val, y_val, 
                optimizer, 
                loss_fn=args.loss
            )
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Log to wandb if active
            if wandb_active:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        
        return history
    
    except Exception as e:
        print(f"Training failed: {e}")
        raise

def prepare_data(dataset='fashion_mnist', test_size=0.1, random_state=42):
    """
    Prepare dataset for training with robust error handling
    
    Args:
        dataset (str): Dataset to use
        test_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        Tuple of preprocessed training, validation, and test data
    """
    try:
        # Load dataset
        if dataset == 'fashion_mnist':
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        elif dataset == 'mnist':
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Preprocess data
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

def create_one_hot(y, num_classes=10):
    """
    Convert labels to one-hot encoding with error handling
    
    Args:
        y (np.array): Original labels
        num_classes (int): Number of classes
    
    Returns:
        np.array: One-hot encoded labels
    """
    try:
        one_hot = np.zeros((num_classes, len(y)))
        one_hot[y, np.arange(len(y))] = 1
        return one_hot
    except Exception as e:
        print(f"Error creating one-hot encoding: {e}")
        raise

def get_optimizer(args):
    """
    Select optimizer based on arguments with comprehensive error handling
    
    Args:
        args: Argument parser namespace
    
    Returns:
        Optimizer instance
    """
    optimizer_map = {
        'sgd': lambda: SGD(learning_rate=args.learning_rate),
        'momentum': lambda: Momentum(learning_rate=args.learning_rate, momentum=args.momentum),
        'nag': lambda: NAG(learning_rate=args.learning_rate, momentum=args.momentum),
        'rmsprop': lambda: RMSprop(learning_rate=args.learning_rate, beta=args.beta),
        'adam': lambda: Adam(learning_rate=args.learning_rate, 
                             beta1=args.beta1, 
                             beta2=args.beta2),
        'nadam': lambda: NAdam(learning_rate=args.learning_rate, 
                               beta1=args.beta1, 
                               beta2=args.beta2)
    }
    
    try:
        return optimizer_map[args.optimizer]()
    except KeyError:
        print(f"Unknown optimizer: {args.optimizer}")
        print("Available optimizers:", list(optimizer_map.keys()))
        sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments with comprehensive options
    """
    parser = argparse.ArgumentParser(description='Neural Network Training')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', 
                        default='neural-network-implementation', 
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', 
                        default=None, 
                        help='Wandb entity name (optional)')
    
    # Model arguments
    parser.add_argument('--dataset', 
                        default='fashion_mnist', 
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=32, 
                        help='Batch size for training')
    parser.add_argument('--loss', 
                        default='cross_entropy', 
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', 
                        default='adam', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimization algorithm')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001, 
                        help='Learning rate')
    parser.add_argument('--momentum', 
                        type=float, 
                        default=0.9, 
                        help='Momentum for momentum-based optimizers')
    parser.add_argument('--beta', 
                        type=float, 
                        default=0.9, 
                        help='Beta parameter for RMSprop')
    parser.add_argument('--beta1', 
                        type=float, 
                        default=0.9, 
                        help='Beta1 parameter for Adam/NAdam')
    parser.add_argument('--beta2', 
                        type=float, 
                        default=0.999, 
                        help='Beta2 parameter for Adam/NAdam')
    
    # Network architecture arguments
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=2, 
                        help='Number of hidden layers')
    parser.add_argument('--hidden_size', 
                        type=int, 
                        default=128, 
                        help='Number of neurons in each hidden layer')
    parser.add_argument('--activation', 
                        default='ReLU', 
                        choices=['sigmoid', 'tanh', 'ReLU'],
                        help='Activation function')
    parser.add_argument('--weight_init', 
                        default='Xavier', 
                        choices=['random', 'Xavier'],
                        help='Weight initialization method')
    
    return parser.parse_args()

def create_advanced_confusion_matrix(model, X_test, y_test):
    """
    Create an advanced confusion matrix visualization
    
    Args:
        model (FeedForwardNN): Trained neural network
        X_test (np.array): Test features
        y_test (np.array): Test labels
    
    Returns:
        Figure with confusion matrix visualization
    """
    # Prepare test data
    X_test_processed = X_test.reshape(X_test.shape[0], -1).T
    
    # Debugging: Check data preparation
    print("X_test_processed shape:", X_test_processed.shape)
    print("Sample X_test_processed:", X_test_processed[:, 0])
    
    # Make predictions
    _, y_pred_prob = model.forward(X_test_processed)
    
    # Debugging: Check model predictions
    print("y_pred_prob shape:", y_pred_prob.shape)
    print("Sample y_pred_prob:", y_pred_prob[:, 0])
    
    y_pred = np.argmax(y_pred_prob, axis=0)
    
    # Debugging: Check computed predictions
    print("y_pred shape:", y_pred.shape)
    print("Sample y_pred:", y_pred[0])
    
    # Fashion-MNIST class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    ]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Debugging: Check confusion matrix
    print("Confusion Matrix:\n", cm)
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Absolute Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Percentage Confusion Matrix
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=16)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    # Display the figure
    plt.show()
    
    return fig, cm, cm_percent

def analyze_model_performance(model, X_test, y_test):
    """
    Comprehensive model performance analysis
    
    Args:
        model (FeedForwardNN): Trained neural network
        X_test (np.array): Test features
        y_test (np.array): Test labels
    
    Returns:
        dict: Performance metrics
    """
    # Prepare test data
    X_test_processed = X_test.reshape(X_test.shape[0], -1).T
    
    # Make predictions
    _, y_pred_prob = model.forward(X_test_processed)
    y_pred = np.argmax(y_pred_prob, axis=0)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Class-wise metrics
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    ]
    
    # Compute per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Overall accuracy
    overall_accuracy = np.mean(y_pred == y_test)
    
    # Most confused class pairs
    most_confused = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                confused_count = cm[i, j]
                most_confused.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': confused_count
                })
    
    # Sort most confused pairs
    most_confused.sort(key=lambda x: x['count'], reverse=True)
    
    # Prepare performance report
    performance_report = {
        'overall_accuracy': overall_accuracy,
        'class_accuracy': dict(zip(class_names, class_accuracy)),
        'most_confused_pairs': most_confused[:5]  # Top 5 most confused pairs
    }
    
    return performance_report

def log_confusion_matrix(model, args):
    """
    Log confusion matrix to Wandb
    
    Args:
        model (FeedForwardNN): Trained neural network
        args (Namespace): Training arguments
    """
    # Load test data
    # (_, _), (X_test, y_test) = prepare_data(dataset=args.dataset)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(dataset=args.dataset)

    # Create confusion matrix (figure, counts, percentages)
    cm_fig, cm, cm_percent = create_advanced_confusion_matrix(model, X_test, y_test)

    # Analyze performance
    performance_report = analyze_model_performance(model, X_test, y_test)

    # Log confusion matrix figure and other metrics to the existing Wandb run
    wandb.log({
        "confusion_matrix": wandb.Image(cm_fig),
        "overall_accuracy": performance_report['overall_accuracy'],
        "class_accuracy": performance_report['class_accuracy'],
        "most_confused_pairs": performance_report['most_confused_pairs']
    })

    # Optionally close the figure
    plt.close(cm_fig)

    # Print performance report to console (for debugging)
    print("\nPerformance Report:")
    print(f"Overall Accuracy: {performance_report['overall_accuracy']:.4f}")
    print("\nClass-wise Accuracy:")
    for cls, acc in performance_report['class_accuracy'].items():
        print(f"{cls}: {acc:.4f}")

    print("\nMost Confused Pairs:")
    for pair in performance_report['most_confused_pairs']:
        print(f"{pair['true_class']} often misclassified as {pair['predicted_class']}: {pair['count']} times")

    return performance_report 

def main():
    """
    Main training function for standard training
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
            dataset=args.dataset
        )
        
        # Convert labels to one-hot
        y_train_oh = create_one_hot(y_train)
        y_val_oh = create_one_hot(y_val)
        y_test_oh = create_one_hot(y_test)
        
        # Initialize model
        model = FeedForwardNN(
            input_size=784,
            hidden_sizes=[args.hidden_size] * args.num_layers,
            output_size=10,
            activation=args.activation,
            weight_init=args.weight_init
        )
        
        # Train model
        history = train(
            model, 
            X_train.T, y_train_oh, 
            X_val.T, y_val_oh, 
            args
        )
        
        # Final evaluation on test set
        _, test_acc = train_epoch(
            model, 
            X_test.T, y_test_oh, 
            get_optimizer(args), 
            loss_fn=args.loss
        )
        print(f"Test Accuracy: {test_acc:.4f}")
        log_confusion_matrix(model, args)
    
    except Exception as e:
        # Print the error message and line number
        print(f"An error occurred during training: {e}")
        print("Error details:")
        traceback.print_exc()  # This will print the full traceback, including the line number
        sys.exit(1)

def sweep_train():
    """
    Specialized training function for Wandb sweeps
    """
    # Initialize wandb with current sweep configuration
    wandb.init()
    
    # Get configuration from wandb
    config = wandb.config
    
    # Prepare arguments for existing training functions
    class SweepArgs:
        def __init__(self, config):
            self.wandb_project = 'fashion_mnist_sweep'
            self.wandb_entity = None
            self.dataset = 'fashion_mnist'
            self.epochs = config.epochs
            self.batch_size = config.batch_size
            self.loss = 'cross_entropy'
            self.optimizer = config.optimizer
            self.learning_rate = config.learning_rate
            self.momentum = 0.9
            self.beta = 0.9
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.num_layers = config.num_layers
            self.hidden_size = config.hidden_size
            self.activation = config.activation
            self.weight_init = config.weight_init
    
    # Create arguments object
    args = SweepArgs(config)
    
    # Prepare data with 10% validation split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
        dataset='fashion_mnist', 
        test_size=0.1, 
        random_state=42
    )
    
    # Convert labels to one-hot
    y_train_oh = create_one_hot(y_train)
    y_val_oh = create_one_hot(y_val)
    y_test_oh = create_one_hot(y_test)
    
    # Initialize model
    model = FeedForwardNN(
        input_size=784,
        hidden_sizes=[args.hidden_size] * args.num_layers,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init
    )
    
    # Create meaningful run name
    run_name = f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}"
    wandb.run.name = run_name
    
    # Train model
    history = train(
        model, 
        X_train.T, y_train_oh, 
        X_val.T, y_val_oh, 
        args
    )
    
    # Final evaluation on test set
    _, test_acc = train_epoch(
        model, 
        X_test.T, y_test_oh, 
        get_optimizer(args), 
        loss_fn=args.loss
    )
    
    # Log final metrics
    wandb.log({
        'test_accuracy': test_acc,
        'best_val_accuracy': max(history['val_accuracy'])
    })
    
    return max(history['val_accuracy'])

def run_sweep():
    """
    Run Wandb sweep
    """
    # Initialize sweep
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {'values': [5, 10]},
            'num_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [16, 32, 64]},
            'weight_init': {'values': ['random', 'Xavier']},
            'activation': {'values': ['sigmoid', 'tanh', 'ReLU']}
        }
    }
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project='fashion_mnist_sweep'
    )
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=50)

main()
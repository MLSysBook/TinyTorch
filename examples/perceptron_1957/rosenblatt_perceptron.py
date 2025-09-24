"""
The Perceptron (1957) - Frank Rosenblatt
=========================================

Historical Context:
Frank Rosenblatt's Perceptron was the first trainable artificial neural network.
It could learn to classify linearly separable patterns, sparking the first wave
of neural network research and dreams of artificial intelligence.

What You're Building:
The same perceptron that started it all - a single-layer network that can
learn simple classification tasks through iterative weight updates.

Required Modules (can run after Module 4):
- Module 2 (Tensor): Core data structure
- Module 3 (Activations): Step function for binary output
- Module 4 (Layers): Dense layer for linear transformation

This Example Demonstrates:
- The original perceptron architecture
- Why it could only solve linearly separable problems
- The foundation that all modern neural networks build upon
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import Sigmoid  # Using sigmoid as step function approximation


class Perceptron:
    """
    Rosenblatt's Perceptron - the network that started it all.
    
    Historical note: The original used a step function, but we'll use
    sigmoid for smooth gradients (a later innovation).
    """
    
    def __init__(self, input_size=2, output_size=1):
        # Single layer - just like the original!
        self.linear = Dense(input_size, output_size)
        self.activation = Sigmoid()  # Original used step function
        
    def forward(self, x):
        """Forward pass through the perceptron."""
        x = self.linear(x)
        x = self.activation(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def predict(self, x):
        """Binary classification prediction."""
        output = self.forward(x)
        return (output.data > 0.5).astype(int)


def generate_linear_data(n_samples=100):
    """
    Generate linearly separable data - the kind perceptron can solve.
    This represents the AND logic gate that Rosenblatt demonstrated.
    """
    np.random.seed(42)
    
    # Generate random points
    X = np.random.randn(n_samples, 2)
    
    # Linearly separable rule: points above the line y = -x + 0.5
    y = (X[:, 1] > -X[:, 0] + 0.5).astype(int).reshape(-1, 1)
    
    return X, y


def demonstrate_perceptron():
    """Demonstrate the historic perceptron."""
    
    print("="*60)
    print("THE PERCEPTRON (1957) - The First Trainable Neural Network")
    print("="*60)
    print()
    print("Historical Context:")
    print("Frank Rosenblatt's perceptron proved machines could learn from data.")
    print("It could classify patterns that were linearly separable.")
    print()
    
    # Generate linearly separable data
    X_train, y_train = generate_linear_data(100)
    
    # Create the historic perceptron
    perceptron = Perceptron(input_size=2, output_size=1)
    
    print("Architecture: Input(2) → Linear → Sigmoid → Output(1)")
    print(f"Parameters: {perceptron.linear.weights.size + perceptron.linear.bias.size}")
    print()
    
    # Test on some samples (without training - random weights)
    test_samples = np.array([
        [0.0, 1.0],   # Should be class 1 (above line)
        [1.0, 0.0],   # Should be class 0 (below line)
        [-1.0, 1.0],  # Should be class 1 (above line)
        [1.0, -1.0]   # Should be class 0 (below line)
    ])
    
    print("Testing on sample points (before training):")
    print("Point        → Expected → Predicted")
    
    for i, point in enumerate(test_samples):
        expected = 1 if point[1] > -point[0] + 0.5 else 0
        predicted = perceptron.predict(Tensor(point.reshape(1, -1)))[0, 0]
        print(f"{point} → {expected}        → {predicted}")
    
    print()
    print("Classification accuracy (random weights): ~50%")
    print()
    print("Historical Impact:")
    print("✓ Proved machines could learn from examples")
    print("✓ Inspired decades of neural network research")
    print("✓ Foundation for deep learning revolution")
    print()
    print("Limitation: Could only solve linearly separable problems")
    print("Next breakthrough needed: Hidden layers (see xor_1969 example)")
    print()
    print("After Module 6 (Autograd), you can train this perceptron to converge!")
    print("="*60)


if __name__ == "__main__":
    demonstrate_perceptron()
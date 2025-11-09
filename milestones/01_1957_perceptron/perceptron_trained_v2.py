#!/usr/bin/env python3
"""
The Perceptron (1957) - Frank Rosenblatt [WITH STANDARDIZED DASHBOARD]
=======================================================================

This is a REFACTORED version showing how the standardized dashboard system
keeps milestone code clean and focused on the ML task.

✅ Compare this to perceptron_trained.py to see the improvement!
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, Sigmoid, BinaryCrossEntropyLoss, SGD

# Import standardized dashboard
sys.path.insert(0, os.path.join(os.getcwd(), 'milestones'))
from milestone_dashboard import MilestoneRunner


# ============================================================================
# MODEL DEFINITION - Your code, clean and focused!
# ============================================================================

class Perceptron:
    """Simple perceptron: Linear + Sigmoid"""
    
    def __init__(self, input_size=2, output_size=1):
        self.linear = Linear(input_size, output_size)
        self.activation = Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self.linear.parameters()


# ============================================================================
# DATA GENERATION - Simple and clean
# ============================================================================

def generate_data(n_samples=100, seed=None):
    """Generate linearly separable data."""
    if seed is not None:
        np.random.seed(seed)
    
    # Class 1: Top-right cluster
    class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([3, 3])
    labels1 = np.ones((n_samples // 2, 1))
    
    # Class 0: Bottom-left cluster
    class0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([1, 1])
    labels0 = np.zeros((n_samples // 2, 1))
    
    # Combine and shuffle
    X = np.vstack([class1, class0])
    y = np.vstack([labels1, labels0])
    
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return Tensor(X), Tensor(y)


# ============================================================================
# TRAINING - Focus on the ML, dashboard handles the rest!
# ============================================================================

def train_perceptron(model, X, y, runner, epochs=100, lr=0.1):
    """Train the perceptron - dashboard shows the drama!"""
    
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    # Start training with live dashboard
    runner.start_training(total_epochs=epochs)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = (pred_classes == y.data).mean() * 100
        
        # Update dashboard (it handles all the display magic!)
        runner.update(epoch, loss.data.item(), accuracy)
        
        # Dashboard automatically detects and announces breakthroughs!
    
    return predictions


# ============================================================================
# MAIN - Clean and focused on the story!
# ============================================================================

def main():
    """Train perceptron with beautiful dashboard."""
    
    # Prepare data
    X, y = generate_data(n_samples=100, seed=42)
    
    # Create model
    model = Perceptron(input_size=2, output_size=1)
    
    # Model info for dashboard
    model_info = {
        "architecture": "Linear(2→1) + Sigmoid",
        "params": "3 (2 weights + 1 bias)"
    }
    
    dataset_info = {
        "name": "Linearly Separable 2D",
        "samples": "100 (50 per class)"
    }
    
    # Run milestone with standardized dashboard!
    with MilestoneRunner("1957 Perceptron", model_info, dataset_info) as runner:
        
        # Train with live dashboard
        predictions = train_perceptron(model, X, y, runner, epochs=100, lr=0.1)
        
        # Calculate final metrics
        pred_classes = (predictions.data > 0.5).astype(int)
        final_accuracy = (pred_classes == y.data).mean() * 100
        
        # Record completion (triggers achievement checks!)
        runner.record_completion({
            "accuracy": final_accuracy,
            "epochs": 100,
        })


if __name__ == "__main__":
    main()




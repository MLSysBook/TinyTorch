#!/usr/bin/env python3
"""
XOR Network Training with TinyTorch Universal Dashboard

This example demonstrates training a neural network to solve the classic XOR problem
using the beautiful TinyTorch training dashboard with real-time plots.

Expected Result: 100% accuracy on XOR truth table with gorgeous visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tinytorch as tt
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import MeanSquaredError as MSELoss
from tinytorch.core.autograd import Variable

# Import the universal dashboard
from examples.common.training_dashboard import create_xor_dashboard

def create_dataset():
    """Create the XOR dataset."""
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    
    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ], dtype=np.float32)
    
    return X, y

def create_model():
    """Create and initialize the XOR network."""
    fc1 = Dense(2, 8)   # Slightly larger for better learning
    fc2 = Dense(8, 1)
    
    # Initialize with reasonable values
    for layer in [fc1, fc2]:
        fan_in = layer.weights.shape[0]
        std = np.sqrt(2.0 / fan_in)
        layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
        layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
        
        layer.weights = Variable(layer.weights, requires_grad=True)
        layer.bias = Variable(layer.bias, requires_grad=True)
    
    return fc1, fc2

def forward_pass(fc1, fc2, X, requires_grad=True):
    """Forward pass through the network."""
    relu = ReLU()
    
    x_var = Variable(Tensor(X), requires_grad=requires_grad)
    h = fc1(x_var)
    h = relu(h)
    out = fc2(h)
    return out

def evaluate_accuracy(fc1, fc2, X, y):
    """Evaluate model accuracy."""
    predictions = forward_pass(fc1, fc2, X, requires_grad=False)
    pred_data = predictions.data._data
    
    predicted_classes = (pred_data > 0.5).astype(int)
    correct = np.sum(predicted_classes == y)
    return correct / y.shape[0]

def main():
    """Main training with dashboard"""
    
    # Create dashboard
    dashboard = create_xor_dashboard()
    
    # Show welcome screen
    dashboard.show_welcome(
        model_info={
            "Architecture": "2 → 8 → 1",
            "Activation": "ReLU + Linear output",
            "Parameters": "~30",
            "Task": "Non-linear XOR function"
        },
        config={
            "Optimizer": "SGD",
            "Learning Rate": "0.1",
            "Batch Size": "4 (full dataset)",
            "Loss Function": "Mean Squared Error"
        }
    )
    
    # Create dataset and model
    X, y = create_dataset()
    fc1, fc2 = create_model()
    
    # Setup training
    params = [fc1.weights, fc1.bias, fc2.weights, fc2.bias]
    optimizer = SGD(params, learning_rate=0.1)
    loss_fn = MSELoss()
    
    # Start training
    epochs = 100
    dashboard.start_training(num_epochs=epochs, target_accuracy=1.0)
    
    # Training loop
    for epoch in range(epochs):
        
        # Forward pass
        predictions = forward_pass(fc1, fc2, X)
        
        # Loss
        y_var = Variable(Tensor(y), requires_grad=False)
        loss = loss_fn(predictions, y_var)
        
        # Extract loss value
        if hasattr(loss.data, 'data'):
            loss_val = float(loss.data.data)
        else:
            loss_val = float(loss.data._data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Fix bias gradients if needed
        for layer in [fc1, fc2]:
            if layer.bias.grad is not None:
                if hasattr(layer.bias.grad.data, 'data'):
                    grad = layer.bias.grad.data.data
                else:
                    grad = layer.bias.grad.data
                
                if len(grad.shape) == 2:
                    layer.bias.grad = Variable(Tensor(np.sum(grad, axis=0)))
        
        # Update parameters
        optimizer.step()
        
        # Calculate accuracies
        train_accuracy = evaluate_accuracy(fc1, fc2, X, y)
        test_accuracy = train_accuracy  # Same data for XOR
        
        # Add convergence metric
        extra_metrics = {
            "Convergence": max(0, 1.0 - loss_val)  # How close to converged
        }
        
        # Update dashboard
        dashboard.update_epoch(
            epoch + 1, 
            train_accuracy, 
            test_accuracy, 
            loss_val,
            extra_metrics
        )
        
        # Early stopping if perfect accuracy
        if train_accuracy >= 0.99:
            break
    
    # Final evaluation
    final_accuracy = evaluate_accuracy(fc1, fc2, X, y)
    
    # Show final predictions
    predictions = forward_pass(fc1, fc2, X, requires_grad=False)
    pred_data = predictions.data._data
    
    print("\n" + "="*50)
    print("🔍 FINAL PREDICTIONS:")
    print("-"*50)
    print("Input  | Target | Prediction | Correct")
    print("-"*50)
    
    for i in range(X.shape[0]):
        x_input = X[i]
        target = y[i, 0]
        pred = pred_data[i, 0]
        correct = "✅" if abs(pred - target) < 0.5 else "❌"
        print(f"{x_input} |   {target}    |   {pred:.3f}    |  {correct}")
    
    print("-"*50)
    
    # Finish training
    results = dashboard.finish_training(final_accuracy)
    
    if final_accuracy >= 0.99:
        print("\n🎉 [bold green]XOR PROBLEM SOLVED![/bold green]")
        print("🧠 Your neural network learned the non-linear XOR function!")
    
    return results

if __name__ == "__main__":
    results = main()
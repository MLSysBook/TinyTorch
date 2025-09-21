#!/usr/bin/env python3
"""
XOR Network Training with TinyTorch

This example demonstrates training a neural network to solve the classic XOR problem,
proving that multi-layer networks can learn non-linear functions.

Just like in PyTorch, we:
1. Create a dataset
2. Build a model
3. Train with gradient descent
4. Evaluate performance

Architecture: 2 → 4 → 1 with ReLU and Sigmoid
Expected Result: 100% accuracy on XOR truth table
"""

import numpy as np
import tinytorch as tt
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import MeanSquaredError as MSELoss
from tinytorch.core.autograd import Variable


def create_dataset():
    """Create the XOR dataset."""
    # XOR truth table
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
    # Simple model: 2 → 4 → 1
    fc1 = Dense(2, 4)  # 2 inputs -> 4 hidden
    fc2 = Dense(4, 1)  # 4 hidden -> 1 output
    
    # Initialize with reasonable values (He initialization)
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


def train_network(fc1, fc2, X, y, epochs=500, lr=0.1):
    """Train the network using gradient descent."""
    # Optimizer
    params = [fc1.weights, fc1.bias, fc2.weights, fc2.bias]
    optimizer = SGD(params, learning_rate=lr)
    
    print("Training XOR Network...")
    print("-" * 40)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = forward_pass(fc1, fc2, X)
        
        # Loss
        y_var = Variable(Tensor(y), requires_grad=False)
        loss_fn = MSELoss()
        loss = loss_fn(predictions, y_var)
        
        if hasattr(loss.data, 'data'):
            loss_val = float(loss.data.data)
        else:
            loss_val = float(loss.data._data)
        losses.append(loss_val)
        
        # Backward
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
                    # Sum over batch dimension
                    layer.bias.grad = Variable(Tensor(np.sum(grad, axis=0)))
        
        # Update
        optimizer.step()
        
        # Log progress
        if epoch % 100 == 0:
            accuracy = evaluate_model(fc1, fc2, X, y)
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {accuracy:.1%}")
    
    return losses


def evaluate_model(fc1, fc2, X, y):
    """Evaluate model accuracy."""
    predictions = forward_pass(fc1, fc2, X, requires_grad=False)
    pred_data = predictions.data._data
    
    predicted_classes = (pred_data > 0.5).astype(int)
    correct = np.sum(predicted_classes == y)
    return correct / y.shape[0]


def main():
    print("=" * 50)
    print("🧠 XOR Network with TinyTorch")
    print("=" * 50)
    print()
    
    # Create dataset
    X, y = create_dataset()
    
    # Build model
    fc1, fc2 = create_model()
    
    # Train model
    losses = train_network(fc1, fc2, X, y, epochs=500)
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("📊 Final Results:")
    print("-" * 40)
    
    predictions = forward_pass(fc1, fc2, X, requires_grad=False)
    pred_data = predictions.data._data
    
    print("Input  | Target | Prediction | Correct")
    print("-" * 40)
    
    for i in range(X.shape[0]):
        x_input = X[i]
        target = y[i, 0]
        pred = pred_data[i, 0]
        correct = "✅" if abs(pred - target) < 0.5 else "❌"
        print(f"{x_input} |   {target}    |   {pred:.3f}    |  {correct}")
    
    accuracy = evaluate_model(fc1, fc2, X, y)
    print("-" * 40)
    print(f"Final Accuracy: {accuracy:.1%}")
    
    if accuracy == 1.0:
        print("\n🎉 SUCCESS! XOR problem solved!")
        print("Your TinyTorch framework can learn non-linear functions!")
    
    # Show learning progress
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\nLearning Progress:")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Improvement:  {initial_loss - final_loss:.4f}")
    
    return accuracy


if __name__ == "__main__":
    accuracy = main()
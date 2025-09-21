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
"""

import numpy as np
import tinytorch as tt
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import MeanSquaredError as MSELoss


def create_dataset():
    """Create the XOR dataset."""
    # XOR truth table
    X = Tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = Tensor([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ])
    
    return X, y


class XORNetwork:
    """A simple 2-layer network for solving XOR."""
    
    def __init__(self):
        from tinytorch.core.autograd import Variable
        
        # Architecture: 2 -> 4 -> 1
        self.hidden = Dense(2, 4)
        self.output = Dense(4, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
        # Convert parameters to Variables for training
        self.hidden.weights = Variable(self.hidden.weights, requires_grad=True)
        self.hidden.bias = Variable(self.hidden.bias, requires_grad=True)
        self.output.weights = Variable(self.output.weights, requires_grad=True)
        self.output.bias = Variable(self.output.bias, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Convert input to Variable if it isn't already
        from tinytorch.core.autograd import Variable
        if not hasattr(x, 'requires_grad'):
            x = Variable(x, requires_grad=True)
            
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        return [self.hidden.weights, self.hidden.bias, 
                self.output.weights, self.output.bias]


def train(model, X, y, epochs=1000, lr=0.5):
    """Train the model using gradient descent."""
    optimizer = SGD(model.parameters(), learning_rate=lr)
    loss_fn = MSELoss()
    
    print("Training XOR Network...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        
        # Compute loss
        loss = loss_fn(predictions, y)
        
        # Backward pass (if autograd is available)
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Manual weight update for demonstration
            # (In real TinyTorch with autograd, the above would work)
            pass
        
        # Log progress
        if epoch % 100 == 0:
            accuracy = evaluate(model, X, y)
            # Handle both Variable and Tensor loss types
            loss_val = loss.data.data if hasattr(loss.data, 'data') else float(loss.data)
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {accuracy:.1%}")
    
    return model


def evaluate(model, X, y):
    """Evaluate model accuracy."""
    predictions = model.forward(X)
    # Handle Variable data extraction
    pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
    y_data = y.data.data if hasattr(y.data, 'data') else y.data
    
    predicted_classes = (pred_data > 0.5).astype(int)
    correct = np.sum(predicted_classes == y_data)
    return correct / y.shape[0]


def main():
    print("=" * 50)
    print("🧠 XOR Network with TinyTorch")
    print("=" * 50)
    print()
    
    # Create dataset
    X, y = create_dataset()
    
    # Build model
    model = XORNetwork()
    
    # Train model
    model = train(model, X, y, epochs=500)
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("📊 Final Results:")
    print("-" * 40)
    
    predictions = model.forward(X)
    
    print("Input  | Target | Prediction | Correct")
    print("-" * 40)
    # Handle Variable data extraction for printing
    X_data = X.data.data if hasattr(X.data, 'data') else X.data
    y_data = y.data.data if hasattr(y.data, 'data') else y.data
    pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
    
    for i in range(X.shape[0]):
        # Convert to numpy for indexing
        X_np = np.array(X_data)
        y_np = np.array(y_data)
        pred_np = np.array(pred_data)
        
        x_input = X_np[i]
        target = y_np[i, 0]
        pred = pred_np[i, 0]
        correct = "✅" if abs(pred - target) < 0.5 else "❌"
        print(f"{x_input} |   {target}    |   {pred:.3f}    |  {correct}")
    
    accuracy = evaluate(model, X, y)
    print("-" * 40)
    print(f"Final Accuracy: {accuracy:.1%}")
    
    if accuracy == 1.0:
        print("\n🎉 SUCCESS! XOR problem solved!")
        print("Your TinyTorch framework can learn non-linear functions!")
    
    return accuracy


if __name__ == "__main__":
    accuracy = main()
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
    # XOR truth table as numpy arrays (matches working pattern)
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


class XORNetwork:
    """A simple 2-layer network for solving XOR."""
    
    def __init__(self):
        from tinytorch.core.autograd import Variable
        
        # Architecture: 2 -> 4 -> 1
        self.hidden = Dense(2, 4)
        self.output = Dense(4, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
        # Initialize with better values (He initialization)
        for layer in [self.hidden, self.output]:
            fan_in = layer.weights.shape[0]
            std = np.sqrt(2.0 / fan_in)
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
        
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


def train(model, X, y, epochs=1000, lr=0.1):
    """Train the model using gradient descent."""
    from tinytorch.core.autograd import Variable
    
    optimizer = SGD(model.parameters(), learning_rate=lr)
    loss_fn = MSELoss()
    
    print("Training XOR Network...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Forward pass (exact pattern from working test)
        x_var = Variable(Tensor(X), requires_grad=True)
        h = model.hidden(x_var)
        h = model.relu(h)
        predictions = model.output(h)
        predictions = model.sigmoid(predictions)
        
        # Compute loss
        y_var = Variable(Tensor(y), requires_grad=False)
        loss = loss_fn(predictions, y_var)
        
        # Backward pass (if autograd is available)
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            
            # Fix bias gradients if needed (from working test)
            for layer in [model.hidden, model.output]:
                if layer.bias.grad is not None:
                    if hasattr(layer.bias.grad.data, 'data'):
                        grad = layer.bias.grad.data.data
                    else:
                        grad = layer.bias.grad.data
                    
                    if len(grad.shape) == 2:
                        # Sum over batch dimension
                        layer.bias.grad = Variable(Tensor(np.sum(grad, axis=0)))
            
            optimizer.step()
        else:
            # Manual weight update for demonstration
            # (In real TinyTorch with autograd, the above would work)
            pass
        
        # Log progress
        if epoch % 100 == 0:
            accuracy = evaluate(model, X, y)
            # Handle both Variable and Tensor loss types
            if hasattr(loss.data, 'data'):
                loss_val = float(loss.data.data)
            else:
                loss_val = float(loss.data._data)
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {accuracy:.1%}")
    
    return model


def evaluate(model, X, y):
    """Evaluate model accuracy."""
    from tinytorch.core.autograd import Variable
    
    # Use the same forward pattern
    x_var = Variable(Tensor(X), requires_grad=False)
    h = model.hidden(x_var)
    h = model.relu(h)
    predictions = model.output(h)
    predictions = model.sigmoid(predictions)
    
    # Handle Variable data extraction
    pred_data = predictions.data._data
    
    predicted_classes = (pred_data > 0.5).astype(int)
    correct = np.sum(predicted_classes == y)
    return correct / y.shape[0]


def main():
    from tinytorch.core.autograd import Variable
    
    print("=" * 50)
    print("🧠 XOR Network with TinyTorch")
    print("=" * 50)
    print()
    
    # Create dataset
    X, y = create_dataset()
    
    # Build model
    model = XORNetwork()
    
    # Train model
    model = train(model, X, y, epochs=200)
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("📊 Final Results:")
    print("-" * 40)
    
    # Final predictions using same pattern
    x_var = Variable(Tensor(X), requires_grad=False)
    h = model.hidden(x_var)
    h = model.relu(h)
    predictions = model.output(h)
    predictions = model.sigmoid(predictions)
    
    print("Input  | Target | Prediction | Correct")
    print("-" * 40)
    
    pred_data = predictions.data._data
    
    for i in range(X.shape[0]):
        x_input = X[i]
        target = y[i, 0]
        pred = pred_data[i, 0]
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
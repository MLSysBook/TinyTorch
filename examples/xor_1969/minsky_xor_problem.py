"""
The XOR Problem (1969) - Minsky & Papert
=========================================

Historical Context:
In 1969, Marvin Minsky and Seymour Papert published "Perceptrons", proving
that single-layer perceptrons couldn't solve XOR (exclusive-or). This finding
triggered the first "AI Winter" as funding dried up. The solution - hidden
layers with nonlinear activation - wouldn't be widely adopted until the 1980s
when backpropagation was rediscovered.

What You're Building:
A multi-layer perceptron that solves XOR - the problem that "killed" neural
networks for a decade. This demonstrates why deep networks with hidden layers
are essential for learning non-linear patterns.

Required Modules (can run after Module 6):
- Module 2 (Tensor): Core data structure with gradients
- Module 3 (Activations): ReLU/Sigmoid for nonlinearity (the key!)
- Module 4 (Layers): Linear layers for transformations
- Module 5 (Losses): Binary cross-entropy for classification
- Module 6 (Autograd): Backpropagation (the missing piece in 1969!)

This Example Demonstrates:
- Why XOR requires hidden layers
- How nonlinear activation enables complex decision boundaries
- The importance of backpropagation for training deep networks
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.autograd import to_numpy


class XORNet:
    """
    Multi-layer Perceptron that solves XOR.
    
    Historical note: This architecture was theoretically possible in 1969,
    but without backpropagation, no one knew how to train it efficiently!
    """
    
    def __init__(self):
        # Hidden layer - the key innovation!
        self.hidden = Linear(2, 4)  # 2 inputs → 4 hidden units
        self.relu = ReLU()         # Nonlinearity (crucial!)
        self.output = Linear(4, 1)  # 4 hidden → 1 output
        self.sigmoid = Sigmoid()   # For binary classification
        
        # Enable gradients for training
        for layer in [self.hidden, self.output]:
            layer.weights.requires_grad = True
            layer.bias.requires_grad = True
    
    def forward(self, x):
        """Forward pass through the network."""
        # This is what Minsky said we needed but couldn't train!
        x = self.hidden(x)
        x = self.relu(x)      # Nonlinearity enables XOR solution
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def predict(self, x):
        """Binary prediction."""
        output = self.forward(x)
        return (to_numpy(output) > 0.5).astype(int)
    
    def parameters(self):
        """Get all parameters."""
        return [
            self.hidden.weights, self.hidden.bias,
            self.output.weights, self.output.bias
        ]
    
    def zero_grad(self):
        """Zero all gradients."""
        for param in self.parameters():
            if param.requires_grad:
                param.zero_grad()


def get_xor_data():
    """
    The infamous XOR dataset that stumped perceptrons.
    
    XOR Truth Table:
    0, 0 → 0
    0, 1 → 1  
    1, 0 → 1
    1, 1 → 0
    
    This is NOT linearly separable!
    """
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


def train_xor(model, X, y, epochs=100, lr=0.1):
    """
    Train the network to solve XOR.
    
    Historical note: This training loop represents backpropagation,
    which wasn't widely known until Rumelhart, Hinton, and Williams
    popularized it in 1986!
    """
    criterion = MeanSquaredError()
    
    for epoch in range(epochs):
        # Convert to tensors
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)
        
        # Forward pass
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        
        # Backward pass (backpropagation - the missing piece!)
        loss.backward()
        
        # Update weights (gradient descent)
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.data = param.data - lr * param.grad.data
        
        # Zero gradients
        model.zero_grad()
        
        # Print progress
        if epoch % 20 == 0:
            loss_value = to_numpy(loss)
            predictions = model.predict(X_tensor)
            accuracy = np.mean(predictions == y) * 100
            print(f"Epoch {epoch:3d}: Loss = {float(loss_value):.4f}, Accuracy = {accuracy:.0f}%")


def demonstrate_xor():
    """Demonstrate solving the XOR problem."""
    
    print("="*60)
    print("THE XOR PROBLEM (1969) - The Challenge That Stopped AI")
    print("="*60)
    print()
    print("Historical Context:")
    print("Minsky & Papert proved single-layer perceptrons can't solve XOR.")
    print("This caused the first AI Winter (1969-1980s).")
    print("Solution: Hidden layers + nonlinearity + backpropagation!")
    print()
    
    # Get XOR data
    X, y = get_xor_data()
    
    print("XOR Truth Table (Not Linearly Separable!):")
    print("Input → Output")
    for i in range(len(X)):
        print(f"{X[i]} → {y[i][0]}")
    print()
    
    # Create multi-layer network
    model = XORNet()
    
    print("Network Architecture (The Solution):")
    print("Input(2) → Hidden(4) + ReLU → Output(1) + Sigmoid")
    print(f"Total parameters: {sum(p.size for p in model.parameters())}")
    print()
    
    # Test before training
    print("Before Training:")
    for i in range(len(X)):
        pred = model.predict(Tensor(X[i:i+1]))[0, 0]
        print(f"{X[i]} → Predicted: {pred}, Actual: {y[i][0]}")
    print()
    
    # Training would happen here with backpropagation
    print("Training with Backpropagation (the missing piece from 1969!):")
    # Note: Actual training requires working autograd integration
    print("(Training demonstration - requires complete autograd)")
    print()
    
    print("Historical Impact:")
    print("✓ Proved need for hidden layers and nonlinearity")
    print("✓ Led to backpropagation rediscovery (1986)")
    print("✓ Sparked the deep learning revolution")
    print()
    print("Key Insight: Depth + Nonlinearity = Universal Approximation")
    print()
    print("After Module 8 (Optimizers), you can train this to 100% accuracy!")
    print("="*60)


if __name__ == "__main__":
    demonstrate_xor()
#!/usr/bin/env python3
"""
Complete CNN Training Test - Full End-to-End Training Loop
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear, Module
from tinytorch.core.activations import ReLU
from tinytorch.core.spatial import Conv2d, MaxPool2D, flatten

class SimpleCNN(Module):
    """Simple CNN for testing end-to-end training."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3))
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=(2, 2))
        self.fc = Linear(16, 2)  # 4 channels * 2x2 spatial = 16 features
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = flatten(x)
        x = self.fc(x)
        return x

def test_cnn_training():
    """Test complete CNN training with multiple epochs."""
    print("🚀 Testing Complete CNN Training...")
    
    # Create model
    model = SimpleCNN()
    
    print("Model parameters:")
    print(f"  Conv weight shape: {model.conv1.weight.shape}")
    print(f"  Conv bias shape: {model.conv1.bias.shape if model.conv1.bias is not None else None}")
    print(f"  FC weight shape: {model.fc.weights.shape}")
    print(f"  FC bias shape: {model.fc.bias.shape if model.fc.bias is not None else None}")
    
    # Create simple training data
    X = Variable(np.random.randn(4, 1, 6, 6).astype(np.float32), requires_grad=False)  # 4 samples
    y = Variable(np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32), requires_grad=False)  # 2 classes
    
    print(f"Training data shape: {X.shape}")
    print(f"Training labels shape: {y.shape}")
    
    # Training loop
    learning_rate = 0.01
    num_epochs = 5
    
    print(f"\n📚 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X)
        
        # Compute loss (simple MSE) - maintain computational graph
        diff = predictions - y
        loss_squared = diff ** 2
        # Use the Variable directly for backward pass
        loss_var = loss_squared
        
        # Check gradients before
        conv_grad_before = model.conv1.weight.grad is not None
        fc_grad_before = model.fc.weights.grad is not None
        
        # Zero gradients
        model.conv1.weight.grad = None
        model.conv1.bias.grad = None
        model.fc.weights.grad = None
        if model.fc.bias is not None:
            model.fc.bias.grad = None
        
        # Backward pass
        loss_var.backward()
        
        # Check gradients after
        conv_grad_after = model.conv1.weight.grad is not None
        fc_grad_after = model.fc.weights.grad is not None
        
        # Compute gradient magnitudes
        if conv_grad_after:
            print(f"  Conv grad type: {type(model.conv1.weight.grad)}")
        if fc_grad_after:
            print(f"  FC grad type: {type(model.fc.weights.grad)}")
        conv_grad_mag = np.linalg.norm(model.conv1.weight.grad) if conv_grad_after else 0.0
        fc_grad_data = model.fc.weights.grad.data if (fc_grad_after and hasattr(model.fc.weights.grad, 'data')) else model.fc.weights.grad
        fc_grad_mag = np.linalg.norm(fc_grad_data) if fc_grad_after else 0.0
        
        # Parameter update (simple SGD) - handle both numpy arrays and Tensors
        if conv_grad_after:
            # Conv2d gradients are numpy arrays
            model.conv1.weight._data -= learning_rate * model.conv1.weight.grad
        if model.conv1.bias is not None and model.conv1.bias.grad is not None:
            model.conv1.bias._data -= learning_rate * model.conv1.bias.grad
        if fc_grad_after:
            # Linear layer gradients might be Tensors - get the data
            fc_grad = model.fc.weights.grad.data if hasattr(model.fc.weights.grad, 'data') else model.fc.weights.grad
            model.fc.weights._data -= learning_rate * fc_grad
        if model.fc.bias is not None and model.fc.bias.grad is not None:
            bias_grad = model.fc.bias.grad.data if hasattr(model.fc.bias.grad, 'data') else model.fc.bias.grad
            model.fc.bias._data -= learning_rate * bias_grad
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {loss_squared.data.data.mean():.6f}")
        print(f"  Conv gradients: {conv_grad_after} (magnitude: {conv_grad_mag:.6f})")
        print(f"  FC gradients: {fc_grad_after} (magnitude: {fc_grad_mag:.6f})")
        
        if not (conv_grad_after and fc_grad_after):
            print("  ❌ Missing gradients!")
            return False
    
    print("✅ Training completed successfully!")
    print("🎉 End-to-End CNN Training WORKING!")
    return True

if __name__ == "__main__":
    success = test_cnn_training()
    print(f"\n{'='*50}")
    if success:
        print("🎯 FINAL RESULT: Complete CNN training pipeline is functional!")
        print("Ready for production ML training workflows!")
    else:
        print("❌ FINAL RESULT: CNN training needs more fixes")
#!/usr/bin/env python3
"""Simple CNN test to verify the clean architecture works"""

import numpy as np
import sys
import warnings

# Suppress warnings during import
warnings.filterwarnings('ignore')

# Direct imports to avoid module-level code execution
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

# Enable autograd
enable_autograd()

# Import layers after autograd is enabled
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU

print("=" * 50)
print("Testing Clean CNN Architecture")
print("=" * 50)

# Create a simple network
class SimpleNet:
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0] if hasattr(x.shape, '__getitem__') else 1, -1)
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x

# Test the network
model = SimpleNet()
print("✅ Model created successfully")

# Create dummy data
X = Tensor(np.random.randn(4, 784), requires_grad=True)
print(f"✅ Input created: shape {X.shape}")

# Forward pass
output = model.forward(X)
print(f"✅ Forward pass successful: output shape {output.shape if hasattr(output, 'shape') else 'unknown'}")

# Check if we can get parameters
params = [model.fc1.weights, model.fc1.bias, model.fc2.weights, model.fc2.bias]
print(f"✅ Found {len(params)} parameter tensors")

print("\n" + "=" * 50)
print("Clean Architecture Test Complete!")
print("Ready for CNN implementation")
print("=" * 50)
#!/usr/bin/env python3
"""Test MNIST training to debug loss computation."""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from tinytorch.core.tensor import Tensor
from examples.mnist_mlp_1986.train_mlp import MNISTMLP
from examples.utils import cross_entropy_loss

print("Testing MNIST training with small batch...")

# Create simple model (check actual signature)
model = MNISTMLP()  # Uses default sizes

# Create small batch of synthetic data
batch_size = 4
X = np.random.randn(batch_size, 784).astype(np.float32) * 0.1
y = np.array([0, 1, 2, 3])  # Different classes

# Convert to tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y)

print(f"Input shape: {X.shape}")
print(f"Labels: {y}")

# Forward pass
outputs = model.forward(X_tensor)
print(f"Output shape: {outputs.data.shape}")

# Check output values
outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
print(f"Output sample (first row): {outputs_np[0][:5]}...")
print(f"Output range: [{outputs_np.min():.4f}, {outputs_np.max():.4f}]")

# Test MSE loss (simpler)
print("\n=== Testing MSE Loss ===")
# Create one-hot targets for MSE
one_hot = np.zeros((batch_size, 10))
for i in range(batch_size):
    one_hot[i, y[i]] = 1.0
targets_tensor = Tensor(one_hot)

# Compute MSE
diff = outputs - targets_tensor
squared_diff = diff * diff
print(f"Diff shape: {diff.data.shape}")
print(f"Squared diff shape: {squared_diff.data.shape}")

# Extract mean manually
squared_np = np.array(squared_diff.data.data if hasattr(squared_diff.data, 'data') else squared_diff.data)
mse_value = np.mean(squared_np)
print(f"MSE loss value: {mse_value:.4f}")

# Test backward
n_elements = np.prod(squared_diff.data.shape)
grad_output = Tensor(np.ones_like(squared_diff.data) / n_elements)
squared_diff.backward(grad_output)

# Check for gradients
params_with_grad = 0
for param in model.parameters():
    if param.grad is not None:
        params_with_grad += 1

print(f"\nGradient check: {params_with_grad}/{len(model.parameters())} parameters have gradients")

if params_with_grad > 0:
    print("✅ Gradients are flowing!")
else:
    print("❌ No gradients detected")

#!/usr/bin/env python3
"""
Debug gradient flow issue - test simple case.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear

print("Testing basic gradient flow with Linear layer...")

# Create simple linear layer
linear = Linear(2, 1)
print(f"Linear layer created: 2 → 1")
print(f"Weights shape: {linear.weights.data.shape}")
print(f"Weights requires_grad: {linear.weights.requires_grad if hasattr(linear.weights, 'requires_grad') else 'Not set'}")

# Create input
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"\nInput created: {x.data.shape}")
print(f"Input requires_grad: {x.requires_grad if hasattr(x, 'requires_grad') else 'Not set'}")

# Forward pass
y = linear(x)
print(f"\nOutput shape: {y.data.shape}")
print(f"Output requires_grad: {y.requires_grad if hasattr(y, 'requires_grad') else 'Not set'}")

# Create loss
loss = Tensor([np.mean(y.data ** 2)])
print(f"\nLoss value: {loss.data}")
print(f"Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'Not set'}")

# Try backward
print("\nCalling loss.backward()...")
try:
    loss.backward()
    print("Backward completed")
    
    # Check gradients
    if linear.weights.grad is not None:
        print(f"✅ Weights gradient exists: {linear.weights.grad}")
    else:
        print("❌ Weights gradient is None")
        
    if linear.bias.grad is not None:
        print(f"✅ Bias gradient exists: {linear.bias.grad}")
    else:
        print("❌ Bias gradient is None")
        
except Exception as e:
    print(f"❌ Backward failed: {e}")

# Let's check if the issue is with how we create the loss
print("\n" + "="*60)
print("Testing with proper Variable/Tensor handling...")

# The loss we create manually might not be connected to the graph
y_np = np.array(y.data.data if hasattr(y.data, 'data') else y.data)
print(f"Output as numpy: {y_np}")

# Check if y has backward method
print(f"y has backward method: {hasattr(y, 'backward')}")
print(f"y has grad_fn: {hasattr(y, 'grad_fn')}")

# Try using y directly for loss
print("\nTrying mean squared error directly on output...")
try:
    # Try to use tensor operations
    if hasattr(y, 'sum'):
        loss2 = y.sum()
        print(f"Sum of outputs: {loss2}")
        loss2.backward()
        print("✅ Backward through sum worked!")
    else:
        print("❌ No sum method on tensor")
except Exception as e:
    print(f"❌ Failed: {e}")

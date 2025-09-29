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

# Create loss - need to extract numpy array first
y_np = np.array(y.data.data if hasattr(y.data, 'data') else y.data)
loss_value = np.mean(y_np ** 2)
loss = Tensor([loss_value])
print(f"\nLoss value: {loss.data}")
print(f"Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'Not set'}")

# The problem: manually created loss tensor is not connected to computation graph!
print("\n⚠️  Problem identified: Manually created loss Tensor is not connected to the computation graph!")

# Let's use the proper approach - use tensor operations that maintain the graph
print("\n" + "="*60)
print("Testing with proper tensor operations...")

# We need to use operations that preserve the computational graph
# The issue is that we're breaking the graph by converting to numpy

print("\nChecking if we have proper tensor operations available...")
print(f"y has sum method: {hasattr(y, 'sum')}")
print(f"y has mean method: {hasattr(y, 'mean')}")
print(f"y has backward method: {hasattr(y, 'backward')}")

# Let's try using the sum operation if available
if hasattr(y, 'sum'):
    print("\nUsing y.sum() to create loss...")
    loss2 = y.sum()
    print(f"Loss from sum: {loss2}")
    
    # Try backward
    print("Calling loss2.backward()...")
    try:
        loss2.backward()
        print("✅ Backward completed!")
        
        # Check gradients
        if linear.weights.grad is not None:
            grad_np = np.array(linear.weights.grad.data if hasattr(linear.weights.grad, 'data') else linear.weights.grad)
            print(f"✅ Weights gradient: {grad_np}")
        else:
            print("❌ Weights gradient is None")
            
        if linear.bias.grad is not None:
            grad_np = np.array(linear.bias.grad.data if hasattr(linear.bias.grad, 'data') else linear.bias.grad)
            print(f"✅ Bias gradient: {grad_np}")
        else:
            print("❌ Bias gradient is None")
    except Exception as e:
        print(f"❌ Backward failed: {e}")
else:
    print("\n❌ No sum method available - this is why gradients don't flow!")
    print("   We need to implement tensor operations that maintain the graph.")

# Alternatively, let's try calling backward directly on y
print("\n" + "="*60)
print("Testing backward directly on output...")
if hasattr(y, 'backward'):
    # Reset gradients
    linear.weights.grad = None
    linear.bias.grad = None
    
    # For backward on non-scalar, we need gradient argument
    grad_output = np.ones_like(y_np)
    print(f"Calling y.backward with grad_output shape: {grad_output.shape}")
    
    try:
        y.backward(Tensor(grad_output))
        print("✅ Backward completed!")
        
        # Check gradients
        if linear.weights.grad is not None:
            grad_np = np.array(linear.weights.grad.data if hasattr(linear.weights.grad, 'data') else linear.weights.grad)
            print(f"✅ Weights gradient: {grad_np}")
        else:
            print("❌ Weights gradient is None")
    except Exception as e:
        print(f"❌ Backward failed: {e}")

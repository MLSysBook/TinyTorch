#!/usr/bin/env python3
"""
Test if we can get gradients by calling backward directly on output.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear

print("Testing direct backward on Linear layer output...")

# Create simple linear layer
linear = Linear(2, 1)
print(f"Linear layer: 2 → 1")

# Create input
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Input shape: {x.data.shape}")

# Forward pass
y = linear(x)
print(f"Output shape: {y.data.shape}")
print(f"Output has backward: {hasattr(y, 'backward')}")

# Call backward directly on output with gradient
if hasattr(y, 'backward'):
    print("\nCalling y.backward() with gradient...")
    
    # For non-scalar backward, we need to provide gradient
    grad_output = Tensor(np.ones_like(y.data))
    
    try:
        y.backward(grad_output)
        print("✅ Backward call succeeded")
        
        # Check for gradients
        if hasattr(linear.weights, 'grad') and linear.weights.grad is not None:
            print(f"✅ Weights have gradient!")
            grad_data = linear.weights.grad.data if hasattr(linear.weights.grad, 'data') else linear.weights.grad
            print(f"   Gradient shape: {np.array(grad_data).shape}")
            print(f"   Gradient values: {grad_data}")
        else:
            print("❌ No gradient on weights")
            
        if hasattr(linear.bias, 'grad') and linear.bias.grad is not None:
            print(f"✅ Bias has gradient!")
            grad_data = linear.bias.grad.data if hasattr(linear.bias.grad, 'data') else linear.bias.grad
            print(f"   Gradient values: {grad_data}")
        else:
            print("❌ No gradient on bias")
            
    except TypeError as e:
        if "missing" in str(e):
            # Try without argument
            print("Trying backward without gradient argument...")
            y.backward()
            print("✅ Backward succeeded without argument")
            
            # Check gradients again
            if hasattr(linear.weights, 'grad') and linear.weights.grad is not None:
                print(f"✅ Weights have gradient!")
            else:
                print("❌ No gradient on weights")
    except Exception as e:
        print(f"❌ Backward failed: {e}")
        
print("\n" + "="*60)
print("CONCLUSION:")
print("The gradient flow issue is at the framework level.")
print("The backward pass needs to be properly implemented in the autograd system.")

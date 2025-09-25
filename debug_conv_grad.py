#!/usr/bin/env python3
"""
Debug Conv2d gradient flow
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable
from tinytorch.core.spatial import Conv2d, conv2d_vars

def test_conv_gradient():
    """Test convolution gradient computation in isolation."""
    print("🔍 Debugging Conv2d Gradient Flow...")
    
    # Create a simple Conv2d layer
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), bias=False)
    
    print(f"Conv weight shape: {conv.weight.shape}")
    print(f"Conv weight type: {type(conv.weight)}")
    print(f"Conv weight requires_grad: {conv.weight.requires_grad}")
    print(f"Conv weight grad before: {conv.weight.grad is not None}")
    
    # Create simple input
    x = Variable(np.random.randn(1, 2, 2).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    print(f"Input type: {type(x)}")
    
    # Forward pass
    print("\n--- Forward Pass ---")
    y = conv(x)
    print(f"Output shape: {y.shape}")
    print(f"Output type: {type(y)}")
    print(f"Output has grad_fn: {hasattr(y, 'grad_fn') and y.grad_fn is not None}")
    
    # Create loss
    loss = y ** 2
    print(f"Loss variable: {loss}")
    print(f"Loss data: {loss.data.data}")
    
    # Backward pass
    print("\n--- Backward Pass ---")
    loss.backward()
    
    print(f"Conv weight grad after: {conv.weight.grad is not None}")
    if conv.weight.grad is not None:
        print(f"Conv weight grad shape: {conv.weight.grad.shape}")
        print(f"Conv weight grad values: {conv.weight.grad}")
    
    # Test conv2d_vars directly
    print("\n--- Testing conv2d_vars directly ---")
    # Reset gradients
    conv.weight.grad = None
    
    # Create Variables manually
    input_var = Variable(x.data, requires_grad=True)
    weight_var = Variable(conv.weight.data, requires_grad=True) 
    weight_var._source_tensor = conv.weight  # Reference to original Parameter
    
    print(f"Weight var source tensor: {weight_var._source_tensor is conv.weight}")
    
    # Call conv2d_vars directly
    result = conv2d_vars(input_var, weight_var, None, (2, 2))
    print(f"Direct conv2d_vars result shape: {result.shape}")
    
    # Create loss and backward
    loss2 = result ** 2
    loss2.backward()
    
    print(f"After direct conv2d_vars backward:")
    print(f"Conv weight grad: {conv.weight.grad is not None}")
    if conv.weight.grad is not None:
        print(f"Conv weight grad shape: {conv.weight.grad.shape}")

if __name__ == "__main__":
    test_conv_gradient()
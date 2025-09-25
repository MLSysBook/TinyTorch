#!/usr/bin/env python3
"""
Quick test for Conv2d gradient flow.
Tests if gradients are properly computed for Conv2d parameters.
"""

import numpy as np
import sys
import os

# Add modules to path
sys.path.append('modules/09_spatial')
sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')
sys.path.append('modules/04_layers')

from spatial_dev import Conv2d
from tensor_dev import Tensor
from autograd_dev import Variable

def test_conv2d_gradients():
    """Test that Conv2d produces gradients for its parameters."""
    print("🔬 Testing Conv2d Gradient Flow...")
    
    # Create small Conv2d layer
    conv = Conv2d(in_channels=2, out_channels=3, kernel_size=(2, 2))
    print(f"Conv2d created: {conv.in_channels} -> {conv.out_channels}, kernel {conv.kernel_size}")
    
    # Create small input
    x_data = np.random.randn(2, 4, 4)  # 2 channels, 4x4 image
    x = Variable(Tensor(x_data), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = conv(x)
    print(f"Output shape: {y.shape}")
    print(f"Output type: {type(y)}")
    
    # Check if output is Variable
    assert isinstance(y, Variable), f"Expected Variable, got {type(y)}"
    
    # Create fake loss (sum all outputs)
    loss = Variable(Tensor(np.sum(y.data.data)), requires_grad=True)
    print(f"Loss: {loss.data.data}")
    
    # Check parameter gradients before backward
    print("\nBefore backward pass:")
    print(f"Conv weight grad: {hasattr(conv.weight, 'grad') and conv.weight.grad is not None}")
    if conv.bias is not None:
        print(f"Conv bias grad: {hasattr(conv.bias, 'grad') and conv.bias.grad is not None}")
    
    # Backward pass
    print("\n🔥 Running backward pass...")
    try:
        # Create gradient for output
        grad_output = Variable(Tensor(np.ones_like(y.data.data)), requires_grad=False)
        
        # Call the gradient function manually (simulating backward)
        if hasattr(y, 'grad_fn') and y.grad_fn is not None:
            print("Calling grad_fn...")
            y.grad_fn(grad_output)
        else:
            print("❌ No grad_fn found on output Variable")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check parameter gradients after backward
    print("\nAfter backward pass:")
    weight_has_grad = hasattr(conv.weight, 'grad') and conv.weight.grad is not None
    print(f"Conv weight grad: {weight_has_grad}")
    if weight_has_grad:
        print(f"  Weight grad shape: {conv.weight.grad.shape if hasattr(conv.weight.grad, 'shape') else 'No shape'}")
        print(f"  Weight grad type: {type(conv.weight.grad)}")
        if hasattr(conv.weight.grad, 'data'):
            grad_magnitude = np.abs(conv.weight.grad.data).mean()
        else:
            grad_magnitude = np.abs(conv.weight.grad).mean()
        print(f"  Weight grad magnitude: {grad_magnitude}")
    
    if conv.bias is not None:
        bias_has_grad = hasattr(conv.bias, 'grad') and conv.bias.grad is not None
        print(f"Conv bias grad: {bias_has_grad}")
        if bias_has_grad:
            print(f"  Bias grad shape: {conv.bias.grad.shape if hasattr(conv.bias.grad, 'shape') else 'No shape'}")
            if hasattr(conv.bias.grad, 'data'):
                grad_magnitude = np.abs(conv.bias.grad.data).mean()
            else:
                grad_magnitude = np.abs(conv.bias.grad).mean()
            print(f"  Bias grad magnitude: {grad_magnitude}")
    
    # Test result
    if weight_has_grad:
        print("\n✅ Conv2d gradient test PASSED! Gradients are flowing properly.")
        return True
    else:
        print("\n❌ Conv2d gradient test FAILED! No gradients found.")
        return False

if __name__ == "__main__":
    success = test_conv2d_gradients()
    sys.exit(0 if success else 1)
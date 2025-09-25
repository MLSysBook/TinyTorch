#!/usr/bin/env python3
"""
Simplified CNN Test - Focus on gradient flow without module import tests
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

# Import only the needed classes without triggering module tests
from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear, Module
from tinytorch.core.activations import ReLU

# Import spatial classes directly
from tinytorch.core.spatial import Conv2d, MaxPool2D, flatten

def test_simple_cnn_gradient():
    """Test CNN gradient flow with minimal setup."""
    print("🔄 Testing Simple CNN Gradient Flow...")
    
    # Create simple inputs
    x = Variable(np.random.randn(1, 8, 8).astype(np.float32), requires_grad=True)
    print(f"  Input shape: {x.shape}")
    
    # Test Conv2d
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3))
    conv_out = conv(x)
    print(f"  Conv output shape: {conv_out.shape}")
    print(f"  Conv output is Variable: {isinstance(conv_out, Variable)}")
    print(f"  Conv output has grad_fn: {conv_out.grad_fn is not None if isinstance(conv_out, Variable) else 'N/A'}")
    
    # Test ReLU
    relu = ReLU()
    relu_out = relu(conv_out)
    print(f"  ReLU output shape: {relu_out.shape}")
    print(f"  ReLU output is Variable: {isinstance(relu_out, Variable)}")
    print(f"  ReLU output has grad_fn: {relu_out.grad_fn is not None if isinstance(relu_out, Variable) else 'N/A'}")
    
    # Test MaxPool2D
    pool = MaxPool2D(pool_size=(2, 2))
    pool_out = pool(relu_out)
    print(f"  Pool output shape: {pool_out.shape}")
    print(f"  Pool output is Variable: {isinstance(pool_out, Variable)}")
    print(f"  Pool output has grad_fn: {pool_out.grad_fn is not None if isinstance(pool_out, Variable) else 'N/A'}")
    
    # Test flatten
    flat_out = flatten(pool_out)
    print(f"  Flatten output shape: {flat_out.shape}")
    print(f"  Flatten output is Variable: {isinstance(flat_out, Variable)}")
    print(f"  Flatten output has grad_fn: {flat_out.grad_fn is not None if isinstance(flat_out, Variable) else 'N/A'}")
    
    # Test Linear layer
    fc = Linear(flat_out.shape[1], 1)  # Use actual flattened size
    final_out = fc(flat_out)
    print(f"  FC output shape: {final_out.shape}")
    print(f"  FC output is Variable: {isinstance(final_out, Variable)}")
    print(f"  FC output has grad_fn: {final_out.grad_fn is not None if isinstance(final_out, Variable) else 'N/A'}")
    print(f"  Final prediction: {final_out.data}")
    
    # Test backward pass
    print("  Testing backward pass...")
    
    # Check parameter gradients before
    conv_weight_grad_before = conv.weight.grad
    fc_weight_grad_before = fc.weights.grad
    print(f"    Conv weight grad before: {conv_weight_grad_before is not None}")
    print(f"    FC weight grad before: {fc_weight_grad_before is not None}")
    
    # Create loss and backward
    target = Variable(np.array([[0.5]], dtype=np.float32), requires_grad=False)
    loss = (final_out - target) ** 2
    print(f"  Loss: {loss.data}")
    
    # Reset gradients
    conv.weight.grad = None
    fc.weights.grad = None
    if conv.bias is not None:
        conv.bias.grad = None
    if fc.bias is not None:
        fc.bias.grad = None
    
    # Backward pass
    loss.backward()
    
    # Check parameter gradients after
    conv_weight_grad_after = conv.weight.grad
    fc_weight_grad_after = fc.weights.grad
    print(f"    Conv weight grad after: {conv_weight_grad_after is not None}")
    print(f"    FC weight grad after: {fc_weight_grad_after is not None}")
    
    if conv_weight_grad_after is not None:
        print(f"      Conv grad shape: {conv_weight_grad_after.shape}")
        print(f"      Conv grad magnitude: {np.linalg.norm(conv_weight_grad_after.data):.6f}")
    
    if fc_weight_grad_after is not None:
        print(f"      FC grad shape: {fc_weight_grad_after.shape}")
        print(f"      FC grad magnitude: {np.linalg.norm(fc_weight_grad_after.data):.6f}")
    
    # Success check
    gradients_working = (conv_weight_grad_after is not None) and (fc_weight_grad_after is not None)
    
    if gradients_working:
        print("  ✅ CNN gradient flow WORKING!")
    else:
        print("  ❌ CNN gradient flow BROKEN!")
    
    return gradients_working

if __name__ == "__main__":
    print("🔥 Simple CNN Gradient Test")
    print("=" * 40)
    
    try:
        success = test_simple_cnn_gradient()
        print("\n" + "=" * 40)
        if success:
            print("🎉 SUCCESS: CNN gradient flow is working!")
            print("Ready for full CNN training!")
        else:
            print("❌ FAILED: CNN gradient flow needs more fixes")
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
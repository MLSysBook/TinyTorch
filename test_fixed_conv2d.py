#!/usr/bin/env python3
"""
Test the fixed Conv2d implementation from spatial module.
Imports just Conv2d to avoid pooling issues.
"""

import numpy as np
import sys
import os

# Add modules to path
sys.path.append('modules/09_spatial')
sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')
sys.path.append('modules/04_layers')

# Import directly from source files
from tensor_dev import Tensor
from autograd_dev import Variable
from layers_dev import Parameter, Module

# Load just the Conv2d class from spatial_dev without executing the module
import importlib.util

def load_conv2d_class():
    """Load just the Conv2d class without executing the full module"""
    spec = importlib.util.spec_from_file_location("spatial_partial", "modules/09_spatial/spatial_dev.py")
    module = importlib.util.module_from_spec(spec)
    
    # Execute only the class definition part
    with open("modules/09_spatial/spatial_dev.py", 'r') as f:
        content = f.read()
    
    # Extract just the Conv2d class definition
    lines = content.split('\n')
    conv2d_lines = []
    in_conv2d_class = False
    indent_level = 0
    
    for line in lines:
        if 'class Conv2d(Module):' in line:
            in_conv2d_class = True
            indent_level = len(line) - len(line.lstrip())
            conv2d_lines.append(line)
        elif in_conv2d_class:
            if line.strip() == '':
                conv2d_lines.append(line)
            elif len(line) - len(line.lstrip()) > indent_level:
                # Still inside the class
                conv2d_lines.append(line)
            elif line.strip().startswith('#'):
                # Comment line
                conv2d_lines.append(line)
            else:
                # End of class
                break
    
    # Create namespace with dependencies
    namespace = {
        'Module': Module,
        'Parameter': Parameter,
        'Variable': Variable,
        'Tensor': Tensor,
        'np': np,
        'Tuple': tuple,  # For type hints
        'Union': object  # For type hints
    }
    
    # Execute the class definition
    exec('\n'.join(conv2d_lines), namespace)
    return namespace['Conv2d']

def test_conv2d_gradients():
    """Test that the fixed Conv2d produces gradients for its parameters."""
    print("🔬 Testing Fixed Conv2d Gradient Flow...")
    
    # Load Conv2d class
    Conv2d = load_conv2d_class()
    
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
    
    # Check parameter gradients before backward
    print("\nBefore backward pass:")
    print(f"Conv weight grad exists: {hasattr(conv.weight, 'grad') and conv.weight.grad is not None}")
    if conv.bias is not None:
        print(f"Conv bias grad exists: {hasattr(conv.bias, 'grad') and conv.bias.grad is not None}")
    
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
            return False
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check parameter gradients after backward
    print("\nAfter backward pass:")
    weight_has_grad = hasattr(conv.weight, 'grad') and conv.weight.grad is not None
    print(f"Conv weight grad exists: {weight_has_grad}")
    if weight_has_grad:
        print(f"  Weight grad shape: {conv.weight.grad.shape if hasattr(conv.weight.grad, 'shape') else 'No shape'}")
        print(f"  Weight grad type: {type(conv.weight.grad)}")
        grad_magnitude = np.abs(conv.weight.grad).mean()
        print(f"  Weight grad magnitude: {grad_magnitude}")
    
    if conv.bias is not None:
        bias_has_grad = hasattr(conv.bias, 'grad') and conv.bias.grad is not None
        print(f"Conv bias grad exists: {bias_has_grad}")
        if bias_has_grad:
            print(f"  Bias grad shape: {conv.bias.grad.shape if hasattr(conv.bias.grad, 'shape') else 'No shape'}")
            grad_magnitude = np.abs(conv.bias.grad).mean()
            print(f"  Bias grad magnitude: {grad_magnitude}")
    
    # Test result
    if weight_has_grad:
        print("\n✅ FIXED Conv2d gradient test PASSED! Gradients are flowing properly.")
        return True
    else:
        print("\n❌ FIXED Conv2d gradient test FAILED! No gradients found.")
        return False

if __name__ == "__main__":
    success = test_conv2d_gradients()
    sys.exit(0 if success else 1)
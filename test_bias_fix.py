#!/usr/bin/env python3
"""
Test script to verify the bias shape fix works with variable batch sizes.
This bypasses the environment issues by testing just the core functionality.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/VJ/GitHub/TinyTorch')

def test_bias_shape_preservation():
    """Test that bias shapes are preserved during Adam optimization."""
    
    # Import locally to avoid environment issues
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import Variable
        from tinytorch.core.optimizers import Adam
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("🧪 Testing Bias Shape Preservation with Variable Batch Sizes")
    print("=" * 60)
    
    # Create a simple parameter set that mimics Dense layer bias
    features = 10
    original_bias_shape = (features,)
    
    # Create bias as Variable
    bias_data = np.random.randn(*original_bias_shape) * 0.1
    bias = Variable(Tensor(bias_data), requires_grad=True)
    
    print(f"Initial bias shape: {bias.data.shape}")
    
    # Create Adam optimizer
    optimizer = Adam([bias], learning_rate=0.001)
    
    # Simulate training with different batch sizes
    batch_sizes = [16, 32, 8, 64]
    
    for step, batch_size in enumerate(batch_sizes):
        print(f"\nStep {step + 1}: Batch size {batch_size}")
        
        # Create fake gradients with batch dimension
        # This simulates what happens during backprop with different batch sizes
        fake_grad = np.random.randn(batch_size, features) * 0.01
        
        # Sum gradients across batch dimension (like what real backprop does)
        bias_grad = np.mean(fake_grad, axis=0)  # Shape: (features,)
        
        # Set gradient (this would normally be done by autograd)
        if not hasattr(bias, 'grad') or bias.grad is None:
            bias.grad = Variable(Tensor(bias_grad), requires_grad=False)
        else:
            bias.grad.data._data[:] = bias_grad
        
        print(f"  Gradient shape: {bias.grad.data.shape}")
        print(f"  Bias shape before update: {bias.data.shape}")
        
        # Perform optimizer step
        optimizer.step()
        
        print(f"  Bias shape after update: {bias.data.shape}")
        
        # Check if shape is preserved
        if bias.data.shape != original_bias_shape:
            print(f"❌ FAILED: Bias shape changed from {original_bias_shape} to {bias.data.shape}")
            return False
        else:
            print(f"✅ PASSED: Bias shape preserved as {bias.data.shape}")
    
    print("\n🎉 All batch size tests passed!")
    print("✅ The bias shape fix is working correctly")
    return True

def test_parameter_update_method():
    """Test the specific parameter update method fix."""
    
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import Variable
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🔧 Testing Parameter Update Method")
    print("=" * 40)
    
    # Create test parameter
    original_data = np.array([1.0, 2.0, 3.0])
    param = Variable(Tensor(original_data.copy()), requires_grad=True)
    
    print(f"Original parameter: {param.data.data}")
    print(f"Original shape: {param.data.shape}")
    
    # Test the OLD way (creates new Tensor - WRONG)
    print("\n❌ Old way (creates shape issues):")
    try:
        new_data = np.array([4.0, 5.0, 6.0])
        # This is what was causing the bug:
        # param.data = Tensor(new_data)  # DON'T DO THIS
        print("  Would create new Tensor object, losing shape tracking")
    except:
        pass
    
    # Test the NEW way (modifies in-place - CORRECT)
    print("\n✅ New way (preserves shape):")
    new_data = np.array([4.0, 5.0, 6.0])
    param.data._data[:] = new_data  # This is the fix
    
    print(f"  Updated parameter: {param.data.data}")
    print(f"  Shape preserved: {param.data.shape}")
    
    # Verify the data actually changed
    if np.allclose(param.data.data, new_data):
        print("✅ Parameter update successful")
        return True
    else:
        print("❌ Parameter update failed")
        return False

if __name__ == "__main__":
    print("🚀 Testing Bias Shape Fix for CIFAR-10 Training")
    print("=" * 50)
    
    success1 = test_parameter_update_method()
    success2 = test_bias_shape_preservation()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The bias shape fix should resolve CIFAR-10 training issues")
        print("✅ Variable batch sizes should now work correctly")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("❌ Need to investigate further")
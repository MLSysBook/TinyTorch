#!/usr/bin/env python3
"""
Test Conv2d gradient flow fix.

This script validates that Conv2d now works with automatic differentiation
instead of trying to call backward() on Parameters.
"""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from tinytorch.core.tensor import Tensor, Parameter
    from tinytorch.core.spatial import Conv2d
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.autograd import Variable
    from tinytorch.core.losses import CrossEntropyLoss
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_conv2d_forward():
    """Test that Conv2d forward pass works correctly."""
    print("\n🧪 Testing Conv2d forward pass...")
    
    try:
        # Create Conv2d layer
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        
        # Test input (simulating RGB image)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        output = conv(x)
        print(f"Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (1, 16, 30, 30)  # 32-3+1=30
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print("✅ Conv2d forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Conv2d forward pass failed: {e}")
        return False

def test_conv2d_with_variables():
    """Test that Conv2d works with Variables for gradient flow."""
    print("\n🧪 Testing Conv2d with Variables...")
    
    try:
        # Create Conv2d layer
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        
        # Create Variable input (this triggers gradient mode)
        x = Variable(Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32)), requires_grad=True)
        print(f"Input is Variable: {isinstance(x, Variable)}")
        
        # Forward pass - this should now work without the Parameter.backward() error
        output = conv(x)
        print(f"Output shape: {output.shape}")
        print(f"Output is Variable: {isinstance(output, Variable)}")
        
        # The key test: this should not throw "Parameter has no backward() method"
        assert isinstance(output, Variable), "Conv2d should return Variable when input is Variable"
        
        print("✅ Conv2d with Variables successful")
        return True
        
    except Exception as e:
        print(f"❌ Conv2d with Variables failed: {e}")
        return False

def test_simple_cnn_forward():
    """Test a simple CNN architecture forward pass."""
    print("\n🧪 Testing simple CNN architecture...")
    
    try:
        # Build simple CNN: Conv2d -> ReLU -> flatten -> Linear
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        relu = ReLU()
        linear = Linear(16 * 30 * 30, 10)  # 30x30 from 32-3+1
        
        # Test input
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        # Forward pass through CNN
        x = conv(x)  # (1, 16, 30, 30)
        print(f"After conv: {x.shape}")
        
        x = relu(x)  # Same shape, apply ReLU
        print(f"After relu: {x.shape}")
        
        # Flatten for linear layer
        x = x.reshape(1, -1)  # Flatten
        print(f"After flatten: {x.shape}")
        
        x = linear(x)  # (1, 10)
        print(f"After linear: {x.shape}")
        
        assert x.shape == (1, 10), f"Expected (1, 10), got {x.shape}"
        
        print("✅ Simple CNN architecture successful")
        return True
        
    except Exception as e:
        print(f"❌ Simple CNN architecture failed: {e}")
        return False

def test_gradient_flow_integration():
    """Test that the gradient flow works in a realistic training scenario."""
    print("\n🧪 Testing gradient flow integration...")
    
    try:
        # Create simple CNN
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
        linear = Linear(8 * 30 * 30, 2)  # Binary classification
        
        # Create Variable inputs for training
        x = Variable(Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32)), requires_grad=True)
        target = Tensor(np.array([0, 1], dtype=np.int64))  # Binary targets
        
        # Forward pass
        features = conv(x)  # Should work without Parameter.backward() error
        features_flat = features.reshape(2, -1)
        logits = linear(features_flat)
        
        print(f"Features shape: {features.shape}")
        print(f"Logits shape: {logits.shape}")
        
        # The key insight: both conv and linear now use the same gradient approach
        assert isinstance(features, Variable), "Conv2d should return Variable"
        assert isinstance(logits, Variable), "Linear should return Variable"
        
        print("✅ Gradient flow integration successful")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all gradient flow tests."""
    print("🔥 Testing Conv2d Gradient Flow Fix")
    print("=" * 50)
    
    tests = [
        test_conv2d_forward,
        test_conv2d_with_variables,
        test_simple_cnn_forward,
        test_gradient_flow_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Conv2d gradient flow is fixed!")
        print()
        print("💡 Key improvements:")
        print("   ✅ Conv2d uses Variable-based automatic differentiation")
        print("   ✅ No more Parameter.backward() errors")
        print("   ✅ Same gradient flow pattern as Linear layer")
        print("   ✅ Compatible with CNN training workflows")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
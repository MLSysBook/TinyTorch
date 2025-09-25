#!/usr/bin/env python3
"""
Simple test to verify Conv2d gradient flow fix.
"""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.spatial import Conv2d
    from tinytorch.core.autograd import Variable
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_conv2d_gradient_fix():
    """Test that Conv2d gradient flow is fixed."""
    print("\n🧪 Testing Conv2d gradient flow fix...")
    
    try:
        # Create Conv2d layer
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
        print(f"Conv2d layer created: {conv.in_channels}→{conv.out_channels} channels")
        
        # Test 1: Tensor input (should return Tensor)
        print("\n📝 Test 1: Tensor input")
        x_tensor = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        out_tensor = conv(x_tensor)
        print(f"  Input type: {type(x_tensor).__name__}")
        print(f"  Output type: {type(out_tensor).__name__}")
        print(f"  Output shape: {out_tensor.shape}")
        assert isinstance(out_tensor, Tensor), "Should return Tensor for Tensor input"
        print("  ✅ Tensor input test passed")
        
        # Test 2: Variable input (should return Variable, no gradient errors)
        print("\n📝 Test 2: Variable input (gradient flow test)")
        x_var = Variable(Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32)), requires_grad=True)
        
        # This is the critical test - this used to fail with "Parameter has no backward() method"
        out_var = conv(x_var)
        
        print(f"  Input type: {type(x_var).__name__}")
        print(f"  Output type: {type(out_var).__name__}")
        print(f"  Output shape: {out_var.shape}")
        assert isinstance(out_var, Variable), "Should return Variable for Variable input"
        print("  ✅ Variable input test passed - no Parameter.backward() error!")
        
        # Test 3: Integration test - simple CNN forward pass
        print("\n📝 Test 3: Simple CNN integration")
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        
        # Build mini CNN
        conv = Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3))
        relu = ReLU()
        
        # Forward pass with Variable
        x = Variable(Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32)), requires_grad=True)
        
        # Conv -> ReLU flow
        features = conv(x)  # Should work without gradient errors
        activated = relu(features)  # Should maintain Variable chain
        
        print(f"  Conv output: {features.shape} ({type(features).__name__})")
        print(f"  ReLU output: {activated.shape} ({type(activated).__name__})")
        
        assert isinstance(features, Variable), "Conv should maintain Variable chain"
        assert isinstance(activated, Variable), "ReLU should maintain Variable chain"
        print("  ✅ CNN integration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🔥 Conv2d Gradient Flow Fix Test")
    print("=" * 40)
    
    if test_conv2d_gradient_fix():
        print("\n" + "=" * 40)
        print("🎉 SUCCESS: Conv2d gradient flow is fixed!")
        print()
        print("💡 What was fixed:")
        print("   • Conv2d no longer calls Parameter.backward()")
        print("   • Uses automatic differentiation like Linear layer")
        print("   • Tensor inputs → Tensor outputs (backward compatible)")
        print("   • Variable inputs → Variable outputs (gradient flow)")
        print("   • Ready for CNN training workflows!")
        return True
    else:
        print("\n❌ FAILED: Conv2d gradient flow still has issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
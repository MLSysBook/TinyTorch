#!/usr/bin/env python3
"""
Test Matrix Multiplication Gradient Flow
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.autograd import Variable
from tinytorch.core.layers import matmul

print("🔍 TESTING MATMUL GRADIENT FLOW")
print("=" * 40)

def test_matmul_gradients():
    """Test matmul gradient computation step by step."""
    print("\n1. Testing Matrix Multiplication Gradients...")

    # Create Variables
    x = Variable([[1.0, 2.0]], requires_grad=True)
    w = Variable([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

    print(f"x: {x}")
    print(f"w: {w}")
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"w.requires_grad: {w.requires_grad}")

    # Matrix multiplication
    result = matmul(x, w)
    print(f"result: {result}")
    print(f"result.requires_grad: {result.requires_grad}")
    print(f"result.grad_fn exists: {result.grad_fn is not None}")

    if result.grad_fn is not None:
        print(f"result.grad_fn: {result.grad_fn}")

    # Manual backward test with known gradient
    print("\n2. Manual backward test...")
    x.grad = None
    w.grad = None

    # Create a gradient to pass back
    output_grad = Variable([[1.0, 1.0]], requires_grad=False)
    print(f"Passing gradient: {output_grad}")

    # Call the gradient function directly
    if result.grad_fn is not None:
        try:
            print("Calling grad_fn directly...")
            result.grad_fn(output_grad)
            print(f"x.grad after direct call: {x.grad}")
            print(f"w.grad after direct call: {w.grad}")
        except Exception as e:
            print(f"❌ Direct grad_fn call failed: {e}")
            import traceback
            traceback.print_exc()

    # Reset and test with backward()
    print("\n3. Testing with result.backward()...")
    x.grad = None
    w.grad = None

    try:
        result.backward()
        print(f"x.grad after backward: {x.grad}")
        print(f"w.grad after backward: {w.grad}")
    except Exception as e:
        print(f"❌ result.backward() failed: {e}")
        import traceback
        traceback.print_exc()

def test_scalar_matmul():
    """Test matmul with scalar loss."""
    print("\n4. Testing Matmul with Scalar Loss...")

    x = Variable([[2.0, 3.0]], requires_grad=True)
    w = Variable([[0.1], [0.2]], requires_grad=True)

    print(f"x: {x}")
    print(f"w: {w}")

    # Matrix multiplication
    result = matmul(x, w)  # Should be [[2*0.1 + 3*0.2]] = [[0.8]]
    print(f"result: {result}")

    # Convert to scalar and create loss
    scalar_value = np.sum(result.data.data if hasattr(result.data, 'data') else result.data)
    print(f"scalar_value: {scalar_value}")

    loss = Variable(scalar_value)
    print(f"loss: {loss}")

    # Reset gradients
    x.grad = None
    w.grad = None

    # Backward pass
    try:
        loss.backward()
        print(f"After loss.backward():")
        print(f"  x.grad: {x.grad}")
        print(f"  w.grad: {w.grad}")
    except Exception as e:
        print(f"❌ loss.backward() failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_matmul_gradients()
    test_scalar_matmul()
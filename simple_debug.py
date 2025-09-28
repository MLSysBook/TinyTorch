#!/usr/bin/env python3
"""
Simple Debug Test - Isolate the gradient flow issue
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.autograd import Variable

print("🔍 SIMPLE GRADIENT FLOW DEBUG")
print("=" * 40)

def test_basic_backward():
    """Test the most basic backward pass."""
    print("\n1. Testing Basic Backward Pass...")

    # Create Variables
    x = Variable([2.0], requires_grad=True)
    y = Variable([3.0], requires_grad=True)

    print(f"x: {x}")
    print(f"y: {y}")

    # Simple multiplication
    z = x * y
    print(f"z = x * y: {z}")
    print(f"z.grad_fn: {z.grad_fn}")

    # Test backward
    print("Calling z.backward()...")
    try:
        z.backward()
        print(f"x.grad after backward: {x.grad}")
        print(f"y.grad after backward: {y.grad}")
        print("✅ Basic backward worked!")
    except Exception as e:
        print(f"❌ Basic backward failed: {e}")
        import traceback
        traceback.print_exc()

def test_scalar_creation():
    """Test creating Variables from scalar losses."""
    print("\n2. Testing Scalar Variable Creation...")

    x = Variable([2.0], requires_grad=True)
    y = Variable([3.0], requires_grad=True)
    z = x * y

    # Extract scalar value and create new Variable (like in the training code)
    print(f"z.data type: {type(z.data)}")
    print(f"z.data.data type: {type(z.data.data)}")
    print(f"z.data.data value: {z.data.data}")

    # This is what the training code does
    scalar_value = np.sum(z.data.data if hasattr(z.data, 'data') else z.data)
    print(f"scalar_value: {scalar_value} (type: {type(scalar_value)})")

    # Create loss Variable
    loss = Variable(scalar_value)
    print(f"loss: {loss}")

    # Try backward
    print("Calling loss.backward()...")
    try:
        loss.backward()
        print("✅ Loss backward completed")
    except Exception as e:
        print(f"❌ Loss backward failed: {e}")

def test_gradient_propagation():
    """Test if gradients actually propagate through operations."""
    print("\n3. Testing Gradient Propagation...")

    # Track what happens step by step
    x = Variable([2.0], requires_grad=True)
    print(f"Step 1 - x created: {x}")

    y = Variable([3.0], requires_grad=True)
    print(f"Step 2 - y created: {y}")

    # Monitor the multiplication operation
    print("Step 3 - Performing x * y...")
    z = x * y
    print(f"z result: {z}")
    print(f"z.requires_grad: {z.requires_grad}")
    print(f"z.grad_fn exists: {z.grad_fn is not None}")

    # Check if grad_fn is callable
    if z.grad_fn is not None:
        print("Step 4 - grad_fn is present, testing manual call...")
        try:
            # Create a dummy gradient and test the function
            dummy_grad = Variable([1.0])
            print(f"Calling grad_fn with dummy gradient: {dummy_grad}")
            z.grad_fn(dummy_grad)
            print(f"After manual grad_fn call - x.grad: {x.grad}, y.grad: {y.grad}")
        except Exception as e:
            print(f"❌ Manual grad_fn call failed: {e}")
            import traceback
            traceback.print_exc()

    # Now test regular backward
    print("Step 5 - Testing regular z.backward()...")
    try:
        # Reset gradients first
        x.grad = None
        y.grad = None

        z.backward()
        print(f"After z.backward() - x.grad: {x.grad}, y.grad: {y.grad}")
    except Exception as e:
        print(f"❌ z.backward() failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_backward()
    test_scalar_creation()
    test_gradient_propagation()
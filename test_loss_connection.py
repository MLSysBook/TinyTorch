#!/usr/bin/env python3
"""
Test Loss Connection to Computational Graph
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.autograd import Variable
from tinytorch.core.layers import matmul

print("🔍 TESTING LOSS CONNECTION TO COMPUTATIONAL GRAPH")
print("=" * 50)

def test_loss_connection():
    """Test how to properly create scalar loss that maintains graph."""
    print("\n1. Testing Different Loss Creation Methods...")

    x = Variable([[2.0, 3.0]], requires_grad=True)
    w = Variable([[0.1], [0.2]], requires_grad=True)

    # Matrix multiplication
    result = matmul(x, w)  # [[0.8]]
    print(f"result: {result}")
    print(f"result.grad_fn: {result.grad_fn}")

    # Method 1: Extract scalar and create new Variable (BROKEN)
    print("\n--- Method 1: Extract scalar and create new Variable ---")
    x.grad = None
    w.grad = None

    scalar_value = np.sum(result.data.data if hasattr(result.data, 'data') else result.data)
    loss1 = Variable(scalar_value)
    print(f"loss1: {loss1}")
    print(f"loss1.grad_fn: {loss1.grad_fn}")

    loss1.backward()
    print(f"Method 1 - x.grad: {x.grad}, w.grad: {w.grad}")

    # Method 2: Use Variable operations to create scalar (SHOULD WORK)
    print("\n--- Method 2: Use Variable operations to create scalar ---")
    x.grad = None
    w.grad = None

    # Sum using Variable operations (maintains graph)
    if hasattr(Variable, 'sum'):
        loss2 = Variable.sum(result)
    else:
        # Create our own sum operation
        loss2 = result  # If result is already scalar, use it directly
        if result.data.data.size > 1:
            # If result has multiple elements, sum manually while preserving graph
            # For now, just use the first element if it's scalar
            pass

    print(f"loss2: {loss2}")
    print(f"loss2.grad_fn: {loss2.grad_fn}")

    loss2.backward()
    print(f"Method 2 - x.grad: {x.grad}, w.grad: {w.grad}")

    # Method 3: Direct backward on result (KNOWN TO WORK)
    print("\n--- Method 3: Direct backward on result ---")
    x.grad = None
    w.grad = None

    result.backward()
    print(f"Method 3 - x.grad: {x.grad}, w.grad: {w.grad}")

def test_sum_operation():
    """Test if Variable has sum operation."""
    print("\n2. Testing Variable Sum Operation...")

    # Check if Variable class has sum method
    print(f"Variable has sum method: {hasattr(Variable, 'sum')}")

    # Try to create a Variable and see what methods it has
    v = Variable([[1.0, 2.0, 3.0]])
    print(f"Variable methods: {[method for method in dir(v) if not method.startswith('_')]}")

if __name__ == "__main__":
    test_loss_connection()
    test_sum_operation()
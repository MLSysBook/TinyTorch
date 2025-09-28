#!/usr/bin/env python3
"""
Test Fixed Sum Operation That Maintains Computational Graph
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.autograd import Variable
from tinytorch.core.layers import matmul

print("🔍 TESTING FIXED SUM OPERATION")
print("=" * 40)

def test_fixed_sum():
    """Test the new Variable.sum method that preserves gradients."""
    print("\n1. Testing Variable.sum with Computational Graph...")

    x = Variable([[2.0, 3.0]], requires_grad=True)
    w = Variable([[0.1], [0.2]], requires_grad=True)

    # Matrix multiplication
    result = matmul(x, w)  # [[0.8]]
    print(f"result: {result}")
    print(f"result.grad_fn: {result.grad_fn}")

    # Use Variable.sum to create scalar loss while preserving graph
    loss = Variable.sum(result)
    print(f"loss: {loss}")
    print(f"loss.grad_fn: {loss.grad_fn}")

    # Test backward pass
    x.grad = None
    w.grad = None

    loss.backward()
    print(f"After loss.backward():")
    print(f"  x.grad: {x.grad}")
    print(f"  w.grad: {w.grad}")

    if x.grad is not None and w.grad is not None:
        print("✅ SUCCESS: Variable.sum preserves gradients!")
    else:
        print("❌ FAILURE: Variable.sum doesn't preserve gradients")

def test_linear_layer_with_sum():
    """Test complete Linear layer with Variable.sum loss."""
    print("\n2. Testing Linear Layer with Variable.sum Loss...")

    from tinytorch.core.layers import Linear

    # Create Linear layer
    layer = Linear(1, 1)

    # Simple training data
    x = Variable([[2.0]], requires_grad=False)  # Input doesn't need gradients
    y_true = Variable([[5.0]], requires_grad=False)  # Target

    print(f"Initial weight: {layer.weights.data.data if hasattr(layer.weights.data, 'data') else layer.weights.data}")
    print(f"Initial bias: {layer.bias.data.data if hasattr(layer.bias.data, 'data') else layer.bias.data}")

    # Forward pass
    y_pred = layer(x)
    print(f"Prediction: {y_pred.data.data if hasattr(y_pred.data, 'data') else y_pred.data}")

    # Loss using Variable.sum (preserves graph)
    diff = y_pred - y_true
    squared_diff = diff * diff
    loss = Variable.sum(squared_diff) / 2.0
    print(f"Loss: {loss.data.data if hasattr(loss.data, 'data') else loss.data}")

    # Clear gradients
    if hasattr(layer.weights, 'grad'):
        layer.weights.grad = None
    if hasattr(layer.bias, 'grad'):
        layer.bias.grad = None

    # Backward pass
    loss.backward()

    print(f"Weight gradient: {getattr(layer.weights, 'grad', 'MISSING')}")
    print(f"Bias gradient: {getattr(layer.bias, 'grad', 'MISSING')}")

    # Check if gradients exist
    weight_grad_exists = hasattr(layer.weights, 'grad') and layer.weights.grad is not None
    bias_grad_exists = hasattr(layer.bias, 'grad') and layer.bias.grad is not None

    if weight_grad_exists and bias_grad_exists:
        print("✅ SUCCESS: Linear layer gradients computed!")
    else:
        print("❌ FAILURE: Linear layer gradients missing")

if __name__ == "__main__":
    test_fixed_sum()
    test_linear_layer_with_sum()
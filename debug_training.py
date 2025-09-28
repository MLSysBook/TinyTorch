#!/usr/bin/env python3
"""
Debug Training Pipeline Issues

This script isolates the training problems to understand exactly what's failing.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear, Parameter
from tinytorch.core.optimizers import SGD

print("🔍 DEBUGGING TRAINING PIPELINE ISSUES")
print("=" * 50)

def test_parameter_type():
    """Test if Parameter is properly Variable-based."""
    print("\n1. Testing Parameter Type...")

    # Create a Parameter
    param = Parameter([1.0, 2.0])
    print(f"Parameter type: {type(param)}")
    print(f"Has requires_grad: {hasattr(param, 'requires_grad')}")
    print(f"requires_grad value: {getattr(param, 'requires_grad', 'MISSING')}")
    print(f"Has grad: {hasattr(param, 'grad')}")
    print(f"grad value: {getattr(param, 'grad', 'MISSING')}")

    if hasattr(param, '_variable'):
        print(f"Internal Variable type: {type(param._variable)}")
        print(f"Internal requires_grad: {param._variable.requires_grad}")

    print("✅ Parameter type test complete")

def test_linear_layer_parameters():
    """Test Linear layer parameter types."""
    print("\n2. Testing Linear Layer Parameters...")

    layer = Linear(2, 3)
    print(f"Weights type: {type(layer.weights)}")
    print(f"Bias type: {type(layer.bias)}")

    print(f"Weights requires_grad: {getattr(layer.weights, 'requires_grad', 'MISSING')}")
    print(f"Bias requires_grad: {getattr(layer.bias, 'requires_grad', 'MISSING')}")

    print("✅ Linear layer parameter test complete")

def test_variable_operations():
    """Test basic Variable operations."""
    print("\n3. Testing Variable Operations...")

    # Test Variable creation
    a = Variable([2.0], requires_grad=True)
    b = Variable([3.0], requires_grad=True)

    print(f"a.requires_grad: {a.requires_grad}")
    print(f"b.requires_grad: {b.requires_grad}")

    # Test multiplication
    c = a * b
    print(f"c type: {type(c)}")
    print(f"c.requires_grad: {getattr(c, 'requires_grad', 'MISSING')}")

    # Test backward
    c.backward()
    print(f"a.grad after backward: {a.grad}")
    print(f"b.grad after backward: {b.grad}")

    print("✅ Variable operations test complete")

def test_matmul_gradient_flow():
    """Test matrix multiplication gradient flow."""
    print("\n4. Testing Matrix Multiplication Gradient Flow...")

    # Create input Variable
    x = Variable([[1.0, 2.0]], requires_grad=True)

    # Create weight Variable (similar to Parameter)
    w = Variable([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)

    print(f"x.requires_grad: {x.requires_grad}")
    print(f"w.requires_grad: {w.requires_grad}")

    # Matrix multiplication
    from tinytorch.core.layers import matmul
    result = matmul(x, w)

    print(f"result type: {type(result)}")
    print(f"result.requires_grad: {getattr(result, 'requires_grad', 'MISSING')}")

    # Create scalar loss
    loss = Variable(np.sum(result.data.data))
    print(f"loss value: {loss.data.data}")

    # Backward pass
    loss.backward()

    print(f"x.grad after backward: {x.grad}")
    print(f"w.grad after backward: {w.grad}")

    print("✅ Matrix multiplication gradient flow test complete")

def test_linear_layer_gradient_flow():
    """Test Linear layer gradient flow with Variables."""
    print("\n5. Testing Linear Layer Gradient Flow...")

    # Create Linear layer
    layer = Linear(2, 1)

    # Create input Variable
    x = Variable([[1.0, 2.0]], requires_grad=True)

    print(f"Input x.requires_grad: {x.requires_grad}")
    print(f"Layer weights requires_grad: {getattr(layer.weights, 'requires_grad', 'MISSING')}")
    print(f"Layer bias requires_grad: {getattr(layer.bias, 'requires_grad', 'MISSING')}")

    # Forward pass
    output = layer(x)

    print(f"Output type: {type(output)}")
    print(f"Output requires_grad: {getattr(output, 'requires_grad', 'MISSING')}")
    print(f"Output value: {output.data.data if hasattr(output.data, 'data') else output.data}")

    # Create scalar loss
    loss = Variable(np.sum(output.data.data if hasattr(output.data, 'data') else output.data))

    # Backward pass
    print("Starting backward pass...")
    loss.backward()

    print(f"Input x.grad: {x.grad}")
    print(f"Weights grad: {getattr(layer.weights, 'grad', 'MISSING')}")
    print(f"Bias grad: {getattr(layer.bias, 'grad', 'MISSING')}")

    print("✅ Linear layer gradient flow test complete")

def test_simple_training():
    """Test simple training step."""
    print("\n6. Testing Simple Training Step...")

    # Create simple model: y = w*x + b
    layer = Linear(1, 1)

    # Simple data: y = 2*x + 1
    x = Variable([[1.0]], requires_grad=False)  # Input doesn't need gradients
    y_true = Variable([[3.0]], requires_grad=False)  # Target: 2*1 + 1 = 3

    print(f"Initial weight: {layer.weights.data.data}")
    print(f"Initial bias: {layer.bias.data.data}")

    # Forward pass
    y_pred = layer(x)
    print(f"Prediction: {y_pred.data.data if hasattr(y_pred.data, 'data') else y_pred.data}")

    # Loss (simple MSE)
    diff = y_pred - y_true
    loss = Variable(0.5 * np.sum((diff.data.data if hasattr(diff.data, 'data') else diff.data) ** 2))
    print(f"Loss: {loss.data.data}")

    # Backward pass
    print("Computing gradients...")
    loss.backward()

    print(f"Weight gradient: {getattr(layer.weights, 'grad', 'MISSING')}")
    print(f"Bias gradient: {getattr(layer.bias, 'grad', 'MISSING')}")

    # Check if we can create optimizer
    try:
        optimizer = SGD([layer.weights, layer.bias], learning_rate=0.1)
        print("✅ Optimizer created successfully")

        # Try one optimization step
        optimizer.step()
        print("✅ Optimization step completed")

        print(f"Updated weight: {layer.weights.data.data}")
        print(f"Updated bias: {layer.bias.data.data}")

    except Exception as e:
        print(f"❌ Optimizer failed: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Simple training test complete")

if __name__ == "__main__":
    try:
        test_parameter_type()
        test_linear_layer_parameters()
        test_variable_operations()
        test_matmul_gradient_flow()
        test_linear_layer_gradient_flow()
        test_simple_training()

        print("\n" + "=" * 50)
        print("🎯 DEBUG SUMMARY: All tests completed!")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
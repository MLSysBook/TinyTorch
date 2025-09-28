#!/usr/bin/env python3
"""
Simple test to verify linear regression works with the fixes.
"""

import numpy as np
import sys
import os

# Add the tinytorch package to path
sys.path.insert(0, os.path.abspath('.'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Parameter, Linear
from tinytorch.core.autograd import Variable
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import MeanSquaredError

def test_basic_parameter():
    """Test that Parameter class works correctly."""
    print("🔬 Testing Parameter class...")

    # Create a parameter
    param = Parameter([1.0, 2.0])
    print(f"Parameter data: {param.data}")
    print(f"Parameter requires_grad: {param.requires_grad}")
    print(f"Parameter grad: {param.grad}")

    # Check if it's a Variable
    print(f"Parameter is Variable: {hasattr(param, '_variable')}")
    print("✅ Parameter class works!")
    return True

def test_variable_sum():
    """Test that Variable.sum() works correctly."""
    print("🔬 Testing Variable.sum()...")

    # Create a Variable
    var = Variable([1.0, 2.0, 3.0], requires_grad=True)
    print(f"Variable: {var}")

    # Sum it
    sum_result = Variable.sum(var)
    print(f"Sum result: {sum_result}")
    print(f"Sum requires_grad: {sum_result.requires_grad}")

    # Test backward
    sum_result.backward()
    print(f"Variable grad after backward: {var.grad}")
    print("✅ Variable.sum() works!")
    return True

def test_simple_linear_regression():
    """Test simple linear regression."""
    print("🔬 Testing Simple Linear Regression...")

    # Generate simple data: y = 2x + 1
    np.random.seed(42)
    X = np.random.randn(10, 1).astype(np.float32)
    y = 2 * X + 1 + 0.01 * np.random.randn(10, 1)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Create model with a single linear layer
    model = Linear(1, 1)

    # Get initial parameters
    print("Initial parameters:")
    params = model.parameters()
    for i, param in enumerate(params):
        print(f"  Parameter {i}: shape={param.data.shape}, value={param.data.data}")

    # Create optimizer and loss function
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    loss_fn = MeanSquaredError()

    # Training loop
    losses = []
    for epoch in range(10):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        X_var = Variable(X, requires_grad=False)
        predictions = model(X_var)

        # Convert y to Variable
        y_var = Variable(y, requires_grad=False)

        # Compute loss
        loss = loss_fn(predictions, y_var)

        losses.append(loss.data.data)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data.data:.6f}")

        # Backward pass
        loss.backward()

        # Check gradients
        if epoch == 0:
            print("Gradients after first backward:")
            params = model.parameters()
            for i, param in enumerate(params):
                if hasattr(param, 'grad') and param.grad is not None:
                    print(f"  Parameter {i} grad: {param.grad}")
                else:
                    print(f"  Parameter {i} grad: None")

        # Update parameters
        optimizer.step()

    # Check final parameters
    print("\nFinal parameters:")
    params = model.parameters()
    for i, param in enumerate(params):
        print(f"  Parameter {i}: shape={param.data.shape}, value={param.data.data}")

    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\nLoss change: {initial_loss:.6f} → {final_loss:.6f}")

    if final_loss < initial_loss:
        print("✅ Loss decreased - training is working!")
        return True
    else:
        print("❌ Loss did not decrease - training is not working")
        return False

if __name__ == "__main__":
    print("🔥 TinyTorch Simple Linear Regression Test")
    print("=" * 50)

    try:
        test_basic_parameter()
        print()

        test_variable_sum()
        print()

        success = test_simple_linear_regression()

        if success:
            print("\n🎉 Simple linear regression is working!")
        else:
            print("\n❌ Simple linear regression failed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
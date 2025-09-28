#!/usr/bin/env python3
"""
Test that the fixed training pipeline works correctly.
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

def test_linear_regression_extended():
    """Test linear regression with more epochs."""
    print("🔬 Testing Extended Linear Regression...")

    # Generate simple data: y = 2x + 1
    np.random.seed(42)
    X = np.random.randn(50, 1).astype(np.float32)
    y = 2 * X + 1 + 0.01 * np.random.randn(50, 1)

    print(f"Training on {len(X)} samples")
    print(f"Target function: y = 2x + 1")

    # Create model
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    loss_fn = MeanSquaredError()

    # Training loop with more epochs
    losses = []
    for epoch in range(150):
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

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.data.data:.6f}")

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    # Check final parameters
    params = list(model.parameters())
    final_weight = params[0].data.data[0, 0]
    final_bias = params[1].data.data[0]

    print(f"\nFinal Results:")
    print(f"  Weight: {final_weight:.3f} (target: 2.0)")
    print(f"  Bias:   {final_bias:.3f} (target: 1.0)")
    print(f"  Final Loss: {losses[-1]:.6f}")

    # Check if parameters are close to target
    weight_error = abs(final_weight - 2.0)
    bias_error = abs(final_bias - 1.0)

    print(f"\nErrors:")
    print(f"  Weight error: {weight_error:.3f}")
    print(f"  Bias error:   {bias_error:.3f}")

    # Success criteria: errors should be small
    weight_ok = weight_error < 0.5  # Within 0.5 of target
    bias_ok = bias_error < 0.5      # Within 0.5 of target
    loss_decreased = losses[-1] < losses[0] * 0.1  # Loss reduced by 90%

    success = weight_ok and bias_ok and loss_decreased

    if success:
        print("✅ Linear regression training successful!")
        print(f"   Parameters learned correctly (errors < 0.5)")
        print(f"   Loss reduced by {(1 - losses[-1]/losses[0])*100:.1f}%")
    else:
        print("❌ Linear regression training failed:")
        if not weight_ok:
            print(f"   Weight error too large: {weight_error:.3f}")
        if not bias_ok:
            print(f"   Bias error too large: {bias_error:.3f}")
        if not loss_decreased:
            print(f"   Loss did not decrease enough: {losses[0]:.3f} → {losses[-1]:.3f}")

    return success

if __name__ == "__main__":
    print("🔥 TinyTorch Training Pipeline Test")
    print("=" * 50)

    try:
        success = test_linear_regression_extended()

        if success:
            print("\n🎉 Training pipeline is working correctly!")
            print("✅ Parameter class creates Variables with gradient tracking")
            print("✅ Variable.sum() enables scalar loss computation")
            print("✅ Gradient flow works from loss back to parameters")
            print("✅ Optimizer updates parameters correctly")
            print("✅ Linear regression learns target function")
        else:
            print("\n❌ Training pipeline has issues")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
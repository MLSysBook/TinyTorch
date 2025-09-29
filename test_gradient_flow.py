#!/usr/bin/env python3
"""Test gradient flow through the system."""

import sys
import os
import numpy as np

# Add to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Suppress module test outputs
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()):
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.losses import MSELoss
    from tinytorch.core.optimizers import SGD

print("Testing gradient flow...")

# Create a simple network
class SimpleNet:
    def __init__(self):
        self.fc1 = Linear(2, 3)
        self.relu = ReLU()
        self.fc2 = Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return [self.fc1.weights, self.fc1.bias,
                self.fc2.weights, self.fc2.bias]

# Test forward pass
print("\n1. Testing forward pass...")
net = SimpleNet()
x = Variable(np.array([[1.0, 2.0]]), requires_grad=False)
y_true = Variable(np.array([[0.5]]), requires_grad=False)

try:
    # Forward pass
    y_pred = net.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   ✅ Forward pass successful")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test loss computation
print("\n2. Testing loss computation...")
try:
    # Use simple manual loss for testing
    diff = y_pred - y_true
    loss = diff * diff  # Simple squared error

    # Get loss value
    if hasattr(loss, 'data'):
        loss_data = loss.data
        if hasattr(loss_data, 'item'):
            loss_value = loss_data.item()
        elif hasattr(loss_data, '__float__'):
            loss_value = float(loss_data)
        else:
            loss_value = np.mean(loss_data)
    else:
        loss_value = float(loss)

    print(f"   Loss value: {loss_value}")
    print(f"   ✅ Loss computation successful")
except Exception as e:
    print(f"   ❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()

# Test backward pass
print("\n3. Testing backward pass...")
try:
    # Check if loss has backward method
    if hasattr(loss, 'backward'):
        loss.backward()
        print(f"   ✅ Backward pass triggered")

        # Check gradients
        for i, param in enumerate(net.parameters()):
            if hasattr(param, 'grad'):
                grad_exists = param.grad is not None
                if grad_exists:
                    grad_norm = np.linalg.norm(param.grad.data) if hasattr(param.grad, 'data') else np.linalg.norm(param.grad)
                    print(f"   Parameter {i}: grad norm = {grad_norm:.6f}")
                else:
                    print(f"   Parameter {i}: No gradient")
            else:
                print(f"   Parameter {i}: No grad attribute")
    else:
        print(f"   ❌ Loss doesn't have backward method")
except Exception as e:
    print(f"   ❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test optimizer step
print("\n4. Testing optimizer update...")
try:
    optimizer = SGD(net.parameters(), learning_rate=0.01)

    # Store initial weights
    if hasattr(net.fc1.weights, 'data'):
        initial_weight = np.copy(net.fc1.weights.data.data) if hasattr(net.fc1.weights.data, 'data') else np.copy(net.fc1.weights.data)
    else:
        initial_weight = np.copy(net.fc1.weights)

    # Update
    optimizer.step()

    # Check if weights changed
    if hasattr(net.fc1.weights, 'data'):
        current_weight = net.fc1.weights.data.data if hasattr(net.fc1.weights.data, 'data') else net.fc1.weights.data
    else:
        current_weight = net.fc1.weights

    # Convert to numpy if needed
    if hasattr(current_weight, 'data'):
        current_weight = current_weight.data

    weight_changed = not np.allclose(initial_weight, current_weight)

    if weight_changed:
        print(f"   ✅ Weights updated successfully")
    else:
        print(f"   ❌ Weights did not change after optimizer step")

except Exception as e:
    print(f"   ❌ Optimizer update failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Gradient flow test complete!")
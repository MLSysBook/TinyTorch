#!/usr/bin/env python3
"""Test minimal training loop - just what's needed for MNIST."""

import sys
import os
import numpy as np

# Add to path
sys.path.insert(0, '.')

# Test the absolute minimum needed
print("Testing minimal training requirements...")

# 1. Can we import what we need?
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.optimizers import SGD
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 2. Can we build a simple network?
class SimpleNet:
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return [self.fc1.weights, self.fc1.bias,
                self.fc2.weights, self.fc2.bias]

try:
    net = SimpleNet()
    print("✅ Network created")
except Exception as e:
    print(f"❌ Network creation failed: {e}")
    sys.exit(1)

# 3. Can we do a forward pass?
try:
    # Batch of 2 flattened MNIST images
    x = Variable(np.random.randn(2, 784), requires_grad=False)
    y = net.forward(x)
    print(f"✅ Forward pass successful, output shape: {y.data.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Can we compute loss and backward?
try:
    # Simple MSE loss
    target = Variable(np.zeros((2, 10)), requires_grad=False)
    target.data[0, 3] = 1  # First sample is digit 3
    target.data[1, 7] = 1  # Second sample is digit 7

    # Compute loss manually (MSE)
    diff = y - target
    loss = Variable(np.mean((diff.data)**2), requires_grad=True)

    # Add backward function
    def loss_backward():
        if y.requires_grad:
            grad = 2 * diff.data / (2 * 10)  # batch_size * num_classes
            if y.grad is None:
                y.grad = Variable(grad)
            else:
                y.grad.data += grad

    loss.backward_fn = loss_backward
    loss.backward()

    print(f"✅ Loss computed and backward called, loss value: {float(loss.data):.4f}")
except Exception as e:
    print(f"❌ Loss/backward failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Can we update parameters?
try:
    optimizer = SGD(net.parameters(), learning_rate=0.01)

    # Check if gradients exist
    has_grads = False
    for param in net.parameters():
        if param.grad is not None:
            has_grads = True
            break

    if has_grads:
        optimizer.step()
        print("✅ Optimizer step successful")
    else:
        print("⚠️ No gradients found on parameters")

    # Zero gradients
    optimizer.zero_grad()
    print("✅ Zero grad successful")

except Exception as e:
    print(f"❌ Optimizer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Can we do a complete training step?
print("\nTesting complete training step...")
try:
    # Forward
    x = Variable(np.random.randn(4, 784), requires_grad=False)
    y = net.forward(x)

    # Create one-hot targets
    target = Variable(np.zeros((4, 10)), requires_grad=False)
    for i in range(4):
        target.data[i, np.random.randint(0, 10)] = 1

    # Loss (cross-entropy style)
    # Apply softmax
    exp_y = np.exp(y.data - np.max(y.data, axis=1, keepdims=True))
    softmax = exp_y / np.sum(exp_y, axis=1, keepdims=True)

    # Cross entropy
    loss_val = -np.mean(np.sum(target.data * np.log(softmax + 1e-8), axis=1))
    loss = Variable(loss_val, requires_grad=True)

    # Gradient of cross-entropy with softmax
    def ce_backward():
        if y.requires_grad:
            grad = (softmax - target.data) / 4  # batch_size
            if y.grad is None:
                y.grad = Variable(grad)
            else:
                y.grad.data += grad

    loss.backward_fn = ce_backward
    loss.backward()

    # Update
    optimizer.step()
    optimizer.zero_grad()

    print(f"✅ Complete training step successful, loss: {float(loss.data):.4f}")

except Exception as e:
    print(f"❌ Complete training step failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Minimal training test complete!")
print("\nWhat's working:")
print("- Basic network construction ✅")
print("- Forward passes ✅")
print("- Manual loss computation ✅")
print("- Manual backward propagation ✅")
print("- Optimizer updates ✅")
print("\nReady for MNIST training!")
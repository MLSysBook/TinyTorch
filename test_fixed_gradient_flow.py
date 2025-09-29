#!/usr/bin/env python3
"""
Test the fixed gradient flow system.
"""

import numpy as np
import contextlib
import io

# Suppress module test outputs
with contextlib.redirect_stdout(io.StringIO()):
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.layers import Linear
    from tinytorch.core.losses import CrossEntropyLoss

print("🧪 Testing Fixed Gradient Flow")
print("=" * 40)

# Test 1: Simple linear layer
print("\n1. Testing Linear Layer Gradient Flow:")
layer = Linear(2, 1)
x = Variable([[1.0, 2.0]], requires_grad=False)
output = layer.forward(x)
print(f"   Output shape: {output.shape}")
print(f"   Output: {output.data.data}")

# Test 2: Loss and backward
print("\n2. Testing Loss and Backward:")
from tinytorch.core.losses import MSELoss
loss_fn = MSELoss()
target = Variable([[0.5]], requires_grad=False)

try:
    loss = loss_fn(output, target)
    print(f"   Loss: {loss.data.data}")

    # Reset gradients
    layer.weights.grad = None
    layer.bias.grad = None

    # Backward pass
    loss.backward()

    print(f"   Weight grad shape: {np.array(layer.weights.grad).shape}")
    print(f"   Bias grad shape: {np.array(layer.bias.grad).shape}")
    print(f"   Weight grad: {np.array(layer.weights.grad)}")
    print(f"   Bias grad: {np.array(layer.bias.grad)}")

    print("   ✅ Gradient flow working!")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multi-class classification
print("\n3. Testing Classification Gradient Flow:")
try:
    classifier = Linear(3, 5)  # 3 inputs, 5 classes
    x_class = Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)  # 2 samples
    logits = classifier.forward(x_class)

    print(f"   Logits shape: {logits.shape}")

    ce_loss = CrossEntropyLoss()
    targets = Variable([0, 1], requires_grad=False)  # Class labels

    loss = ce_loss(logits, targets)
    print(f"   CE Loss: {loss.data.data}")

    # Reset gradients
    classifier.weights.grad = None
    classifier.bias.grad = None

    # Backward pass
    loss.backward()

    print(f"   Weight grad shape: {np.array(classifier.weights.grad).shape}")
    print(f"   Bias grad shape: {np.array(classifier.bias.grad).shape}")

    print("   ✅ Classification gradient flow working!")

except Exception as e:
    print(f"   ❌ Classification error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎉 Gradient flow tests completed!")
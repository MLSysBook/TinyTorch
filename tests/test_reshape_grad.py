#!/usr/bin/env python3
"""Test if reshape preserves computation graph."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

enable_autograd()

print("Testing reshape gradient flow...")

# Create tensor with grad_fn
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = a * 2.0  # Has MulBackward

print(f"b: requires_grad={b.requires_grad}, has_grad_fn={hasattr(b, '_grad_fn') and b._grad_fn is not None}")

# Reshape
c = b.reshape(4)
print(f"c (reshaped): requires_grad={c.requires_grad}, has_grad_fn={hasattr(c, '_grad_fn') and c._grad_fn is not None}")

# Loss
loss = c.sum()
print(f"loss: requires_grad={loss.requires_grad}, has_grad_fn={hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")

# Backward
print("\nCalling backward...")
loss.backward(np.ones_like(loss.data))

print(f"a.grad: {a.grad}")

if a.grad is None:
    print("❌ reshape() breaks gradient flow!")
else:
    print("✅ reshape() preserves gradient flow")


#!/usr/bin/env python3
"""Check if loss has _grad_fn."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss

enable_autograd()

print("Testing CrossEntropyLoss _grad_fn...")

# Simple test
logits = Tensor(np.random.randn(2, 5), requires_grad=True)
targets = Tensor([1, 3], requires_grad=False)

print(f"Logits: requires_grad={logits.requires_grad}")
print(f"Targets: requires_grad={targets.requires_grad}")

loss_fn = CrossEntropyLoss()
loss = loss_fn.forward(logits, targets)

print(f"\nLoss: {loss.data:.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss has _grad_fn: {hasattr(loss, '_grad_fn')}")
if hasattr(loss, '_grad_fn'):
    print(f"Loss _grad_fn: {loss._grad_fn}")
    print(f"Loss _grad_fn type: {type(loss._grad_fn).__name__ if loss._grad_fn else None}")

# Try backward
print(f"\nCalling loss.backward()...")
try:
    loss.backward(np.ones_like(loss.data))
    print("Backward completed")
    print(f"Logits gradient: {logits.grad}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


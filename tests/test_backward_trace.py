#!/usr/bin/env python3
"""Trace backward calls to see what's happening."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd, Function
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.models.transformer import GPT

# Enable autograd
enable_autograd()

# Monkey-patch Function.apply to log calls
original_apply = Function.apply

def traced_apply(self, grad_output):
    class_name = type(self).__name__
    print(f"    → Calling {class_name}.apply() with grad shape {grad_output.shape if hasattr(grad_output, 'shape') else type(grad_output)}")
    result = original_apply(self, grad_output)
    if isinstance(result, tuple):
        print(f"      Returns {len(result)} gradients")
    else:
        print(f"      Returns {type(result)}")
    return result

Function.apply = traced_apply

print("="*70)
print("BACKWARD TRACE")
print("="*70)

# Small GPT model
vocab_size = 10
embed_dim = 16
num_layers = 1
num_heads = 2
seq_length = 4
batch_size = 1

model = GPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_heads=num_heads
)

# Set requires_grad
params = model.parameters()
for param in params:
    param.requires_grad = True

print(f"\nModel has {len(params)} parameters")

# Forward pass
x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)

print(f"\nForward pass...")
logits = model.forward(x)
loss_fn = CrossEntropyLoss()
logits_flat = logits.reshape(batch_size * seq_length, vocab_size)
targets_flat = targets.reshape(batch_size * seq_length)
loss = loss_fn.forward(logits_flat, targets_flat)

print(f"Loss: {loss.data:.4f}")

# Backward pass with tracing
print(f"\nBackward pass (tracing calls)...")
loss.backward(np.ones_like(loss.data))

print(f"\nChecking gradients...")
grads_count = sum(1 for p in params if p.grad is not None)
print(f"Parameters with gradients: {grads_count}/{len(params)}")

if grads_count == 0:
    print("\n❌ NO GRADIENTS - backward() didn't propagate!")
else:
    print(f"\n✅ Some gradients computed")


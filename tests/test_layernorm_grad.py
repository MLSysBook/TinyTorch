#!/usr/bin/env python3
"""Test LayerNorm gradient flow in isolation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.models.transformer import LayerNorm

enable_autograd()

print("="*70)
print("LAYERNORM GRADIENT TEST")
print("="*70)

embed_dim = 8
ln = LayerNorm(embed_dim)

print(f"\ngamma type: {type(ln.gamma)}")
print(f"gamma requires_grad: {ln.gamma.requires_grad}")
print(f"beta type: {type(ln.beta)}")
print(f"beta requires_grad: {ln.beta.requires_grad}")

# Input
x = Tensor(np.random.randn(2, 4, embed_dim), requires_grad=True)
print(f"\nInput x: shape={x.shape}, requires_grad={x.requires_grad}")

# Forward
output = ln.forward(x)
print(f"Output: shape={output.shape}, requires_grad={output.requires_grad}")
print(f"Output has _grad_fn: {hasattr(output, '_grad_fn') and output._grad_fn is not None}")

# Loss
loss = output.sum()
print(f"\nLoss: {loss.data:.4f}")

# Backward
print("\nCalling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print("\nGradient check:")
print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
print(f"  gamma.grad: {'✅' if ln.gamma.grad is not None else '❌'}")
print(f"  beta.grad: {'✅' if ln.beta.grad is not None else '❌'}")

if ln.gamma.grad is not None:
    print(f"    gamma grad norm: {np.linalg.norm(ln.gamma.grad):.6f}")
if ln.beta.grad is not None:
    print(f"    beta grad norm: {np.linalg.norm(ln.beta.grad):.6f}")

print("\n" + "="*70)
if ln.gamma.grad is None or ln.beta.grad is None:
    print("❌ LayerNorm parameters don't receive gradients")
    print("   Issue: Gamma/beta aren't part of computation graph")
else:
    print("✅ LayerNorm gradients work!")


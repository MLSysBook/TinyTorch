#!/usr/bin/env python3
"""Test if residual connections preserve gradients."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.models.transformer import LayerNorm
from tinytorch.core.layers import Linear

enable_autograd()

print("="*70)
print("RESIDUAL CONNECTION GRADIENT TEST")
print("="*70)

# Simple setup: x → LayerNorm → Linear → Residual (x + linear_out)
embed_dim = 8

ln = LayerNorm(embed_dim)
linear = Linear(embed_dim, embed_dim)

print(f"\nLayerNorm gamma requires_grad: {ln.gamma.requires_grad}")
print(f"LayerNorm beta requires_grad: {ln.beta.requires_grad}")
print(f"Linear weight requires_grad: {linear.weight.requires_grad}")

# Input
x = Tensor(np.random.randn(2, 4, embed_dim), requires_grad=True)

# Forward: x → ln → linear → residual
print(f"\n1. LayerNorm...")
ln_out = ln.forward(x)
print(f"   ln_out has _grad_fn: {hasattr(ln_out, '_grad_fn') and ln_out._grad_fn is not None}")

print(f"\n2. Linear...")
linear_out = linear.forward(ln_out)
print(f"   linear_out has _grad_fn: {hasattr(linear_out, '_grad_fn') and linear_out._grad_fn is not None}")

print(f"\n3. Residual (x + linear_out)...")
residual_out = x + linear_out
print(f"   residual_out has _grad_fn: {hasattr(residual_out, '_grad_fn') and residual_out._grad_fn is not None}")

# Loss
loss = residual_out.sum()
print(f"\n4. Loss: {loss.data:.4f}")

print(f"\n5. Backward...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print(f"\nGradient check:")
print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
print(f"  ln.gamma.grad: {'✅' if ln.gamma.grad is not None else '❌'}")
print(f"  ln.beta.grad: {'✅' if ln.beta.grad is not None else '❌'}")
print(f"  linear.weight.grad: {'✅' if linear.weight.grad is not None else '❌'}")

print("\n" + "="*70)
if ln.gamma.grad is None:
    print("❌ Residual connections break gradient flow to LayerNorm")
    print("   Gradients bypass the LayerNorm path")
else:
    print("✅ Residual connections preserve gradients!")


#!/usr/bin/env python3
"""Test x2 → ln2 → mlp → residual gradient flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.models.transformer import LayerNorm, MLP

enable_autograd()

print("="*70)
print("LAYERNORM → MLP → RESIDUAL TEST")
print("="*70)

embed_dim = 16

ln = LayerNorm(embed_dim)
mlp = MLP(embed_dim)

print(f"\nInitial requires_grad:")
print(f"  ln.gamma: {ln.gamma.requires_grad}")
print(f"  mlp.linear1.weight: {mlp.linear1.weight.requires_grad}")
print(f"  mlp.linear2.weight: {mlp.linear2.weight.requires_grad}")

# Input
x = Tensor(np.random.randn(1, 4, embed_dim), requires_grad=True)

# Forward: x → ln → mlp → residual
print(f"\n1. LayerNorm(x)...")
ln_out = ln.forward(x)
print(f"   ln_out has _grad_fn: {hasattr(ln_out, '_grad_fn') and ln_out._grad_fn is not None}")

print(f"\n2. MLP(ln_out)...")
mlp_out = mlp.forward(ln_out)
print(f"   mlp_out has _grad_fn: {hasattr(mlp_out, '_grad_fn') and mlp_out._grad_fn is not None}")

print(f"\n3. Residual (x + mlp_out)...")
residual_out = x + mlp_out
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
print(f"  mlp.linear1.weight.grad: {'✅' if mlp.linear1.weight.grad is not None else '❌'}")
print(f"  mlp.linear2.weight.grad: {'✅' if mlp.linear2.weight.grad is not None else '❌'}")

print("\n" + "="*70)
if ln.gamma.grad is None:
    print("❌ LayerNorm doesn't get gradients before MLP")
if mlp.linear1.weight.grad is None:
    print("❌ MLP first layer doesn't get gradients")
if mlp.linear2.weight.grad is None:
    print("❌ MLP second layer doesn't get gradients")


#!/usr/bin/env python3
"""Test if attention Q/K/V projections get gradients."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.attention import MultiHeadAttention

enable_autograd()

print("="*70)
print("ATTENTION Q/K/V GRADIENT TEST")
print("="*70)

embed_dim = 16
num_heads = 2

attn = MultiHeadAttention(embed_dim, num_heads)

# Check requires_grad on projections
print("\nProjection requires_grad:")
print(f"  q_proj.weight: {attn.q_proj.weight.requires_grad}")
print(f"  k_proj.weight: {attn.k_proj.weight.requires_grad}")
print(f"  v_proj.weight: {attn.v_proj.weight.requires_grad}")
print(f"  out_proj.weight: {attn.out_proj.weight.requires_grad}")

# Forward pass
x = Tensor(np.random.randn(1, 4, embed_dim), requires_grad=True)
mask = Tensor(np.tril(np.ones((4, 4))))

print("\nForward pass...")
output = attn.forward(x, mask)
print(f"  Output has _grad_fn: {hasattr(output, '_grad_fn') and output._grad_fn is not None}")

# Loss and backward
loss = output.sum()
print(f"\nLoss: {loss.data:.4f}")

print("\nCalling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print("\nGradient check:")
print(f"  q_proj.weight: {'✅' if attn.q_proj.weight.grad is not None else '❌'}")
print(f"  k_proj.weight: {'✅' if attn.k_proj.weight.grad is not None else '❌'}")
print(f"  v_proj.weight: {'✅' if attn.v_proj.weight.grad is not None else '❌'}")
print(f"  out_proj.weight: {'✅' if attn.out_proj.weight.grad is not None else '❌'}")

print("\n" + "="*70)
if attn.q_proj.weight.grad is None:
    print("❌ Q/K/V projections don't receive gradients")
    print("   Likely: Attention forward pass breaks computation graph")
else:
    print("✅ All attention projections receive gradients!")


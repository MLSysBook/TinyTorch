#!/usr/bin/env python3
"""Test LayerNorm → Attention → Residual gradient flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.models.transformer import LayerNorm
from tinytorch.core.attention import MultiHeadAttention

enable_autograd()

print("="*70)
print("LAYERNORM → ATTENTION → RESIDUAL TEST")
print("="*70)

embed_dim = 16
num_heads = 2
seq_len = 4

ln = LayerNorm(embed_dim)
attn = MultiHeadAttention(embed_dim, num_heads)

print(f"\nInitial requires_grad:")
print(f"  ln.gamma: {ln.gamma.requires_grad}")
print(f"  attn.q_proj.weight: {attn.q_proj.weight.requires_grad}")
print(f"  attn.out_proj.weight: {attn.out_proj.weight.requires_grad}")

# Input
x = Tensor(np.random.randn(1, seq_len, embed_dim), requires_grad=True)
mask = Tensor(np.tril(np.ones((seq_len, seq_len))))

# Forward: x → ln → attn → residual
print(f"\n1. x...")
print(f"   x has _grad_fn: {hasattr(x, '_grad_fn') and x._grad_fn is not None}")

print(f"\n2. LayerNorm(x)...")
ln_out = ln.forward(x)
print(f"   ln_out has _grad_fn: {hasattr(ln_out, '_grad_fn') and ln_out._grad_fn is not None}")

print(f"\n3. Attention(ln_out)...")
attn_out = attn.forward(ln_out, mask)
print(f"   attn_out has _grad_fn: {hasattr(attn_out, '_grad_fn') and attn_out._grad_fn is not None}")

print(f"\n4. Residual (x + attn_out)...")
residual_out = x + attn_out
print(f"   residual_out has _grad_fn: {hasattr(residual_out, '_grad_fn') and residual_out._grad_fn is not None}")

# Loss
loss = residual_out.sum()
print(f"\n5. Loss: {loss.data:.4f}")

print(f"\n6. Backward...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print(f"\nGradient check:")
print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
print(f"  ln.gamma.grad: {'✅' if ln.gamma.grad is not None else '❌'}")
print(f"  ln.beta.grad: {'✅' if ln.beta.grad is not None else '❌'}")
print(f"  attn.q_proj.weight.grad: {'✅' if attn.q_proj.weight.grad is not None else '❌'}")
print(f"  attn.k_proj.weight.grad: {'✅' if attn.k_proj.weight.grad is not None else '❌'}")
print(f"  attn.v_proj.weight.grad: {'✅' if attn.v_proj.weight.grad is not None else '❌'}")
print(f"  attn.out_proj.weight.grad: {'✅' if attn.out_proj.weight.grad is not None else '❌'}")

print("\n" + "="*70)
if ln.gamma.grad is None:
    print("❌ LayerNorm doesn't get gradients before attention")
if attn.q_proj.weight.grad is None:
    print("❌ Q/K/V projections don't get gradients through attention")
if attn.out_proj.weight.grad is None:
    print("❌ Out projection doesn't get gradients")

if ln.gamma.grad and attn.q_proj.weight.grad and attn.out_proj.weight.grad:
    print("✅ Full path works: LayerNorm → Attention → Residual!")


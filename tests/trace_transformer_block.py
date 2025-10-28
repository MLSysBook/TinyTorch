#!/usr/bin/env python3
"""Trace computation graph through TransformerBlock."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.models.transformer import TransformerBlock

enable_autograd()

print("="*70)
print("TRANSFORMER BLOCK TRACE")
print("="*70)

embed_dim = 16
num_heads = 2
seq_len = 4

block = TransformerBlock(embed_dim, num_heads)

# Set requires_grad
for param in block.parameters():
    param.requires_grad = True

# Input
x = Tensor(np.random.randn(1, seq_len, embed_dim), requires_grad=True)
print(f"\n1. Input x: shape={x.shape}, has_grad_fn={hasattr(x, '_grad_fn') and x._grad_fn is not None}")

# Create mask
mask = Tensor(np.tril(np.ones((seq_len, seq_len))))

# Manually trace through block.forward()
print("\n2. LayerNorm 1...")
ln1_out = block.ln1.forward(x)
print(f"   ln1_out: shape={ln1_out.shape}, has_grad_fn={hasattr(ln1_out, '_grad_fn') and ln1_out._grad_fn is not None}")

print("\n3. Attention...")
attn_out = block.attention.forward(ln1_out, mask)
print(f"   attn_out: shape={attn_out.shape}, has_grad_fn={hasattr(attn_out, '_grad_fn') and attn_out._grad_fn is not None}")

print("\n4. Residual connection 1 (x + attn_out)...")
x2 = x + attn_out
print(f"   x2: shape={x2.shape}, has_grad_fn={hasattr(x2, '_grad_fn') and x2._grad_fn is not None}")

print("\n5. LayerNorm 2...")
ln2_out = block.ln2.forward(x2)
print(f"   ln2_out: shape={ln2_out.shape}, has_grad_fn={hasattr(ln2_out, '_grad_fn') and ln2_out._grad_fn is not None}")

print("\n6. MLP...")
mlp_out = block.mlp.forward(ln2_out)
print(f"   mlp_out: shape={mlp_out.shape}, has_grad_fn={hasattr(mlp_out, '_grad_fn') and mlp_out._grad_fn is not None}")

print("\n7. Residual connection 2 (x2 + mlp_out)...")
output = x2 + mlp_out
print(f"   output: shape={output.shape}, has_grad_fn={hasattr(output, '_grad_fn') and output._grad_fn is not None}")

# Loss
loss = output.sum()
print(f"\n8. Loss: {loss.data:.4f}, has_grad_fn={hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")

# Backward
print("\n9. Calling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print("\n10. Checking parameter gradients...")
params_with_labels = [
    ("ln1.gamma", block.ln1.gamma),
    ("ln1.beta", block.ln1.beta),
    ("attn.q_proj.weight", block.attention.q_proj.weight),
    ("attn.out_proj.weight", block.attention.out_proj.weight),
    ("ln2.gamma", block.ln2.gamma),
    ("ln2.beta", block.ln2.beta),
    ("mlp.linear1.weight", block.mlp.linear1.weight),
    ("mlp.linear2.weight", block.mlp.linear2.weight),
]

for name, param in params_with_labels:
    has_grad = param.grad is not None
    status = "✅" if has_grad else "❌"
    print(f"    {status} {name}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check specific issues
if not (hasattr(ln1_out, '_grad_fn') and ln1_out._grad_fn):
    print("❌ LayerNorm doesn't set _grad_fn")
elif block.ln1.gamma.grad is None:
    print("❌ LayerNorm has _grad_fn but gradients don't reach parameters")
    print("   Likely issue: LayerNorm backward function not working")


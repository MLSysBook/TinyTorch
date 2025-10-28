#!/usr/bin/env python3
"""Test scaled_dot_product_attention gradient flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.attention import scaled_dot_product_attention

enable_autograd()

print("="*70)
print("SCALED DOT PRODUCT ATTENTION GRADIENT TEST")
print("="*70)

batch_size = 2
seq_len = 4
d_k = 8

# Create Q, K, V
Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
V = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

print(f"\nInputs:")
print(f"  Q: requires_grad={Q.requires_grad}")
print(f"  K: requires_grad={K.requires_grad}")
print(f"  V: requires_grad={V.requires_grad}")

# Create mask
mask = Tensor(np.tril(np.ones((seq_len, seq_len))))

print(f"\nForward pass...")
output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

print(f"  Output: shape={output.shape}, has_grad_fn={hasattr(output, '_grad_fn') and output._grad_fn is not None}")
print(f"  Attn weights: shape={attn_weights.shape}, has_grad_fn={hasattr(attn_weights, '_grad_fn') and attn_weights._grad_fn is not None}")

# Loss and backward
loss = output.sum()
print(f"\nLoss: {loss.data:.4f}")

print("\nCalling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print("\nGradient check:")
print(f"  Q.grad: {'✅' if Q.grad is not None else '❌'}")
print(f"  K.grad: {'✅' if K.grad is not None else '❌'}")
print(f"  V.grad: {'✅' if V.grad is not None else '❌'}")

if Q.grad is not None:
    print(f"    Q grad norm: {np.linalg.norm(Q.grad):.6f}")
if K.grad is not None:
    print(f"    K grad norm: {np.linalg.norm(K.grad):.6f}")
if V.grad is not None:
    print(f"    V grad norm: {np.linalg.norm(V.grad):.6f}")

print("\n" + "="*70)
if Q.grad is None or K.grad is None or V.grad is None:
    print("❌ scaled_dot_product_attention breaks gradient flow")
    print("   Need to check the implementation")
else:
    print("✅ scaled_dot_product_attention gradients work!")


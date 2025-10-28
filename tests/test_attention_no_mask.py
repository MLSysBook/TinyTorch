#!/usr/bin/env python3
"""Test attention without mask to isolate issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.attention import scaled_dot_product_attention

enable_autograd()

print("="*70)
print("ATTENTION WITHOUT MASK")
print("="*70)

batch_size = 2
seq_len = 4
d_k = 8

Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
V = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

print(f"\nForward pass WITHOUT mask...")
output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=None)

loss = output.sum()
print(f"Loss: {loss.data:.4f}")

print("\nCalling backward()...")
loss.backward(np.ones_like(loss.data))

print("\nGradient check:")
print(f"  Q.grad: {'✅' if Q.grad is not None else '❌'}")
print(f"  K.grad: {'✅' if K.grad is not None else '❌'}")
print(f"  V.grad: {'✅' if V.grad is not None else '❌'}")

print("\n" + "="*70)

# Now test each step manually
print("\nMANUAL STEP-BY-STEP TEST")
print("="*70)

Q2 = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
K2 = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
V2 = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

print("\n1. K.transpose()...")
K_T = K2.transpose()
print(f"   K_T has _grad_fn: {hasattr(K_T, '_grad_fn') and K_T._grad_fn is not None}")

print("\n2. Q @ K^T...")
scores = Q2.matmul(K_T)
print(f"   scores has _grad_fn: {hasattr(scores, '_grad_fn') and scores._grad_fn is not None}")

print("\n3. scores * scale...")
import math
scale_factor = 1.0 / math.sqrt(d_k)
scores_scaled = scores * scale_factor
print(f"   scores_scaled has _grad_fn: {hasattr(scores_scaled, '_grad_fn') and scores_scaled._grad_fn is not None}")

print("\n4. softmax(scores)...")
from tinytorch.core.activations import Softmax
softmax = Softmax()
attn_w = softmax.forward(scores_scaled, dim=-1)
print(f"   attn_w has _grad_fn: {hasattr(attn_w, '_grad_fn') and attn_w._grad_fn is not None}")

print("\n5. attn_w @ V...")
output2 = attn_w.matmul(V2)
print(f"   output has _grad_fn: {hasattr(output2, '_grad_fn') and output2._grad_fn is not None}")

print("\n6. Backward...")
loss2 = output2.sum()
loss2.backward(np.ones_like(loss2.data))

print("\nGradient check:")
print(f"  Q2.grad: {'✅' if Q2.grad is not None else '❌'}")
print(f"  K2.grad: {'✅' if K2.grad is not None else '❌'}")
print(f"  V2.grad: {'✅' if V2.grad is not None else '❌'}")


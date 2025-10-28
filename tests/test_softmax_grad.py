#!/usr/bin/env python3
"""Test if Softmax breaks gradient flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.activations import Softmax

enable_autograd()

print("="*70)
print("SOFTMAX GRADIENT TEST")
print("="*70)

batch_size = 2
seq_len = 4
d_k = 8

Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

print("\n1. Q @ K^T...")
K_T = K.transpose()
scores = Q.matmul(K_T)
print(f"   scores: shape={scores.shape}, has_grad_fn={hasattr(scores, '_grad_fn') and scores._grad_fn}")

print("\n2. Softmax(scores)...")
softmax = Softmax()
attn_weights = softmax.forward(scores, dim=-1)
print(f"   attn_weights: shape={attn_weights.shape}, has_grad_fn={hasattr(attn_weights, '_grad_fn') and attn_weights._grad_fn}")
if hasattr(attn_weights, '_grad_fn') and attn_weights._grad_fn:
    print(f"   attn_weights._grad_fn type: {type(attn_weights._grad_fn).__name__}")

print("\n3. attn_weights.sum()...")
loss = attn_weights.sum()
print(f"   loss: {loss.data:.4f}")

print("\n4. Backward...")
loss.backward(np.ones_like(loss.data))

print("\nGradient check:")
print(f"  Q.grad: {'✅' if Q.grad is not None else '❌'}")
if Q.grad is not None:
    print(f"    norm: {np.linalg.norm(Q.grad):.6f}")

print(f"  K.grad: {'✅' if K.grad is not None else '❌'}")
if K.grad is not None:
    print(f"    norm: {np.linalg.norm(K.grad):.6f}")

print(f"  scores.grad: {'✅' if scores.grad is not None else '❌'}")
if scores.grad is not None:
    print(f"    norm: {np.linalg.norm(scores.grad):.6f}")

print("\n" + "="*70)
if Q.grad is None or K.grad is None:
    print("❌ Softmax breaks gradient flow to Q and K")
    print("   Issue: SoftmaxBackward not working correctly")
else:
    print("✅ Softmax preserves gradients!")


#!/usr/bin/env python3
"""Test if Q @ K^T computes gradients for both Q and K."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

enable_autograd()

print("="*70)
print("Q @ K^T GRADIENT TEST")
print("="*70)

batch_size = 2
seq_len = 4
d_k = 8

Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

print("\n1. K.transpose()...")
K_T = K.transpose()
print(f"   K_T: shape={K_T.shape}, has_grad_fn={hasattr(K_T, '_grad_fn') and K_T._grad_fn is not None}")
if hasattr(K_T, '_grad_fn') and K_T._grad_fn:
    print(f"   K_T._grad_fn type: {type(K_T._grad_fn).__name__}")

print("\n2. Q @ K_T...")
scores = Q.matmul(K_T)
print(f"   scores: shape={scores.shape}, has_grad_fn={hasattr(scores, '_grad_fn') and scores._grad_fn is not None}")
if hasattr(scores, '_grad_fn') and scores._grad_fn:
    print(f"   scores._grad_fn type: {type(scores._grad_fn).__name__}")

print("\n3. scores.sum()...")
loss = scores.sum()
print(f"   loss: {loss.data:.4f}, has_grad_fn={hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")

print("\n4. Backward...")
loss.backward(np.ones_like(loss.data))

print("\nGradient check:")
print(f"  Q.grad: {'✅' if Q.grad is not None else '❌'}")
if Q.grad is not None:
    print(f"    shape: {Q.grad.shape}, norm: {np.linalg.norm(Q.grad):.6f}")

print(f"  K.grad: {'✅' if K.grad is not None else '❌'}")
if K.grad is not None:
    print(f"    shape: {K.grad.shape}, norm: {np.linalg.norm(K.grad):.6f}")

print(f"  K_T.grad: {'✅' if K_T.grad is not None else '❌'}")
if K_T.grad is not None:
    print(f"    shape: {K_T.grad.shape}, norm: {np.linalg.norm(K_T.grad):.6f}")

print("\n" + "="*70)
if Q.grad is None:
    print("❌ Q doesn't get gradients from Q @ K^T")
if K.grad is None:
    print("❌ K doesn't get gradients from Q @ K^T")
if Q.grad is None or K.grad is None:
    print("   Issue: MatmulBackward or TransposeBackward not working")


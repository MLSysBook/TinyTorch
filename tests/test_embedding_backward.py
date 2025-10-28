#!/usr/bin/env python3
"""Test Embedding backward specifically."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.text.embeddings import Embedding

# Enable autograd
enable_autograd()

print("="*70)
print("EMBEDDING BACKWARD TEST")
print("="*70)

# Create embedding
vocab_size = 10
embed_dim = 8
embedding = Embedding(vocab_size, embed_dim)
embedding.weight.requires_grad = True

print(f"\nEmbedding weight: shape={embedding.weight.shape}, requires_grad={embedding.weight.requires_grad}")

# Forward pass
indices = Tensor([[1, 2, 3]])
print(f"Indices: {indices.data}")

embedded = embedding.forward(indices)
print(f"Embedded: shape={embedded.shape}, requires_grad={embedded.requires_grad}")
print(f"Has _grad_fn: {hasattr(embedded, '_grad_fn') and embedded._grad_fn is not None}")
if hasattr(embedded, '_grad_fn') and embedded._grad_fn:
    print(f"_grad_fn type: {type(embedded._grad_fn).__name__}")

# Simple loss
loss = embedded.sum()
print(f"\nLoss: {loss.data}, requires_grad={loss.requires_grad}")
print(f"Loss has _grad_fn: {hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")

# Backward
print("\nCalling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradient
print(f"\nEmbedding weight gradient:")
if embedding.weight.grad is None:
    print("  ❌ NO GRADIENT")
else:
    print(f"  ✅ Has gradient: shape={embedding.weight.grad.shape}")
    print(f"     Non-zero entries: {np.count_nonzero(embedding.weight.grad)}")
    print(f"     Gradient norm: {np.linalg.norm(embedding.weight.grad):.6f}")
    
    # Check which indices got gradients
    for i in range(vocab_size):
        grad_norm = np.linalg.norm(embedding.weight.grad[i])
        if grad_norm > 0:
            print(f"     Index {i}: grad_norm={grad_norm:.6f}")


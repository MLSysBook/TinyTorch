#!/usr/bin/env python3
"""Minimal GPT test to isolate the gradient flow issue."""

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
print("MINIMAL GPT GRADIENT FLOW TEST")
print("="*70)

# Simplified version of what GPT does
vocab_size = 10
embed_dim = 8
seq_len = 3

# Token embedding
token_emb_layer = Embedding(vocab_size, embed_dim)
token_emb_layer.weight.requires_grad = True

# Position embedding  
pos_emb_layer = Embedding(seq_len, embed_dim)
pos_emb_layer.weight.requires_grad = True

# Input
tokens = Tensor([[1, 2, 3]])  # batch=1, seq=3
print(f"Tokens: {tokens.data}")

# Forward pass - mimicking GPT
print("\n1. Token embedding...")
token_emb = token_emb_layer.forward(tokens)
print(f"   shape={token_emb.shape}, has_grad_fn={hasattr(token_emb, '_grad_fn') and token_emb._grad_fn is not None}")

print("\n2. Position embedding...")
positions = Tensor([[0, 1, 2]])
pos_emb = pos_emb_layer.forward(positions)
print(f"   shape={pos_emb.shape}, has_grad_fn={hasattr(pos_emb, '_grad_fn') and pos_emb._grad_fn is not None}")

print("\n3. Add embeddings...")
combined = token_emb + pos_emb
print(f"   shape={combined.shape}, has_grad_fn={hasattr(combined, '_grad_fn') and combined._grad_fn is not None}")

print("\n4. Simple loss (sum)...")
loss = combined.sum()
print(f"   loss={loss.data:.4f}, has_grad_fn={hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")

# Backward
print("\n5. Calling backward()...")
loss.backward(np.ones_like(loss.data))

# Check gradients
print("\n6. Checking gradients...")
if token_emb_layer.weight.grad is None:
    print("   ❌ Token embedding: NO GRADIENT")
else:
    print(f"   ✅ Token embedding: gradient norm={np.linalg.norm(token_emb_layer.weight.grad):.4f}")

if pos_emb_layer.weight.grad is None:
    print("   ❌ Position embedding: NO GRADIENT")
else:
    print(f"   ✅ Position embedding: gradient norm={np.linalg.norm(pos_emb_layer.weight.grad):.4f}")

print("\n" + "="*70)
if token_emb_layer.weight.grad is not None and pos_emb_layer.weight.grad is not None:
    print("✅ SUCCESS: Both embeddings receive gradients!")
else:
    print("❌ FAILURE: Gradients not flowing through addition")
print("="*70)


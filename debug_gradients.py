#!/usr/bin/env python3
"""
Debug script to trace gradient propagation through the TinyTorch stack.
Tests each component step-by-step to find where gradients stop flowing.
"""

import numpy as np
from tinytorch import Tensor, Linear, Sigmoid, BinaryCrossEntropyLoss, SGD

print("=" * 70)
print("🔍 GRADIENT FLOW DEBUGGING")
print("=" * 70)

# ============================================================================
# TEST 1: Basic Tensor Operations
# ============================================================================
print("\n[TEST 1] Basic Tensor Operations")
print("-" * 70)

x = Tensor([[1.0, 2.0]], requires_grad=True)
print(f"✓ Created tensor x: {x.data}")
print(f"  requires_grad: {x.requires_grad}")
print(f"  grad: {x.grad}")

y = x * 2
print(f"\n✓ Created y = x * 2: {y.data}")
print(f"  requires_grad: {y.requires_grad}")
print(f"  grad: {y.grad}")

loss = y.sum()
print(f"\n✓ Created loss = y.sum(): {loss.data}")
print(f"  requires_grad: {loss.requires_grad}")

print("\n📊 Before backward:")
print(f"  x.grad: {x.grad}")

loss.backward()

print("\n📊 After backward:")
print(f"  x.grad: {x.grad}")

if x.grad is not None and np.allclose(x.grad, [[2.0, 2.0]]):
    print("✅ TEST 1 PASSED: Basic gradients work!")
else:
    print("❌ TEST 1 FAILED: Basic gradients don't work!")
    print(f"   Expected: [[2.0, 2.0]], Got: {x.grad}")

# ============================================================================
# TEST 2: Linear Layer Forward Pass
# ============================================================================
print("\n\n[TEST 2] Linear Layer Forward Pass")
print("-" * 70)

layer = Linear(2, 1)
print(f"✓ Created Linear(2, 1)")
print(f"  weight.data: {layer.weight.data}")
print(f"  weight.requires_grad: {layer.weight.requires_grad}")
print(f"  bias.data: {layer.bias.data}")
print(f"  bias.requires_grad: {layer.bias.requires_grad}")

x = Tensor([[1.0, 2.0]], requires_grad=True)
out = layer(x)
print(f"\n✓ Forward pass output: {out.data}")
print(f"  out.requires_grad: {out.requires_grad}")

# ============================================================================
# TEST 3: Linear Layer Backward Pass
# ============================================================================
print("\n\n[TEST 3] Linear Layer Backward Pass")
print("-" * 70)

layer = Linear(2, 1)
w_before = layer.weight.data.copy()
b_before = layer.bias.data.copy()

print(f"Before backward:")
print(f"  weight: {w_before}")
print(f"  bias: {b_before}")
print(f"  weight.grad: {layer.weight.grad}")
print(f"  bias.grad: {layer.bias.grad}")

x = Tensor([[1.0, 2.0]], requires_grad=True)
out = layer(x)
loss = out.sum()

print(f"\n✓ Created loss: {loss.data}")

loss.backward()

print(f"\nAfter backward:")
print(f"  weight.grad: {layer.weight.grad}")
print(f"  bias.grad: {layer.bias.grad}")
print(f"  x.grad: {x.grad}")

if layer.weight.grad is not None and layer.bias.grad is not None:
    print("✅ TEST 3 PASSED: Linear layer gradients computed!")
else:
    print("❌ TEST 3 FAILED: Linear layer gradients missing!")

# ============================================================================
# TEST 4: Optimizer Step
# ============================================================================
print("\n\n[TEST 4] Optimizer Step")
print("-" * 70)

layer = Linear(2, 1)
optimizer = SGD(layer.parameters(), lr=0.1)

print(f"✓ Created optimizer with lr=0.1")
print(f"  Num parameters: {len(optimizer.params)}")

w_before = layer.weight.data.copy()
b_before = layer.bias.data.copy()

print(f"\nBefore training step:")
print(f"  weight: {w_before}")
print(f"  bias: {b_before}")

# Forward
x = Tensor([[1.0, 2.0]], requires_grad=True)
out = layer(x)
loss = out.sum()

print(f"\n✓ Forward pass, loss: {loss.data}")

# Backward
loss.backward()

print(f"\nAfter backward:")
print(f"  weight.grad: {layer.weight.grad}")
print(f"  bias.grad: {layer.bias.grad}")

# Step
optimizer.step()

w_after = layer.weight.data.copy()
b_after = layer.bias.data.copy()

print(f"\nAfter optimizer.step():")
print(f"  weight: {w_after}")
print(f"  bias: {b_after}")
print(f"  weight changed: {not np.allclose(w_before, w_after)}")
print(f"  bias changed: {not np.allclose(b_before, b_after)}")

if not np.allclose(w_before, w_after) or not np.allclose(b_before, b_after):
    print("✅ TEST 4 PASSED: Optimizer updates parameters!")
else:
    print("❌ TEST 4 FAILED: Optimizer didn't update parameters!")

# ============================================================================
# TEST 5: Full Training Step with Sigmoid + BCE
# ============================================================================
print("\n\n[TEST 5] Full Training Step (Linear + Sigmoid + BCE)")
print("-" * 70)

layer = Linear(2, 1)
sigmoid = Sigmoid()
loss_fn = BinaryCrossEntropyLoss()
optimizer = SGD(layer.parameters(), lr=0.1)

w_before = layer.weight.data.copy()
b_before = layer.bias.data.copy()

print(f"Before training:")
print(f"  weight: {w_before}")
print(f"  bias: {b_before}")

# Data
x = Tensor([[1.0, 2.0]], requires_grad=True)
y_true = Tensor([[1.0]])

# Forward
logits = layer(x)
print(f"\n✓ Logits: {logits.data}")

probs = sigmoid(logits)
print(f"✓ Probs: {probs.data}")

loss = loss_fn(probs, y_true)
print(f"✓ Loss: {loss.data}")

# Backward
print("\n📊 Calling loss.backward()...")
loss.backward()

print(f"\nAfter backward:")
print(f"  loss.grad: {loss.grad}")
print(f"  probs.grad: {probs.grad}")
print(f"  logits.grad: {logits.grad}")
print(f"  weight.grad: {layer.weight.grad}")
print(f"  bias.grad: {layer.bias.grad}")

# Update
optimizer.step()

w_after = layer.weight.data.copy()
b_after = layer.bias.data.copy()

print(f"\nAfter optimizer.step():")
print(f"  weight: {w_after}")
print(f"  bias: {b_after}")
print(f"  weight changed: {not np.allclose(w_before, w_after)}")
print(f"  bias changed: {not np.allclose(b_before, b_after)}")

if not np.allclose(w_before, w_after) or not np.allclose(b_before, b_after):
    print("✅ TEST 5 PASSED: Full training step works!")
else:
    print("❌ TEST 5 FAILED: Full training step doesn't update weights!")

# ============================================================================
# TEST 6: Parameters Function
# ============================================================================
print("\n\n[TEST 6] Layer parameters() method")
print("-" * 70)

layer = Linear(2, 1)
params = layer.parameters()

print(f"✓ layer.parameters() returned {len(params)} parameters")
for i, p in enumerate(params):
    print(f"  param[{i}]: shape={p.shape}, requires_grad={p.requires_grad}, type={type(p)}")

if len(params) == 2:
    print("✅ TEST 6 PASSED: parameters() returns weight and bias!")
else:
    print("❌ TEST 6 FAILED: parameters() should return 2 tensors!")

print("\n" + "=" * 70)
print("🏁 DEBUGGING COMPLETE")
print("=" * 70)

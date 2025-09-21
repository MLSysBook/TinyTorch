# Automatic Differentiation Demo

See the magic of automatic differentiation - the engine that powers deep learning!

## What This Demonstrates

- **Gradient computation** through computational graphs
- **Chain rule** in action for composite functions
- **Backpropagation** through neural network layers
- **Gradient descent** optimization

## Examples Included

1. **Simple Gradients** - Basic derivative computation
2. **Computational Graphs** - Gradients flowing through operations
3. **Neural Network Layer** - Weight gradients for learning
4. **Gradient Descent** - Optimization in action
5. **Chain Rule** - How gradients propagate through deep networks

## Running the Demo

```bash
python demo.py
```

Expected output:
```
🔬 Automatic Differentiation with TinyTorch
==================================================

📐 Example 1: Simple Gradient
----------------------------------------
x = 2.0
y = 3.0
z = x*y + x^2 = 10.0

Gradients after backward():
∂z/∂x = 7.0  (should be y + 2x = 3 + 4 = 7)
∂z/∂y = 2.0  (should be x = 2)

🌳 Example 2: Computational Graph
----------------------------------------
Computational graph:
  a = 1.0
  b = 2.0
  c = 3.0
  d = a + b = 3.0
  e = d * c = 9.0

Gradients:
∂e/∂a = 3.0  (flows through d)
∂e/∂b = 3.0  (flows through d)
∂e/∂c = 3.0  (direct multiplication)

[... more examples ...]
```

## Key Concepts

### Computational Graph
```
    x ──┐
         ├─→ * ──┐
    y ──┘        ├─→ + ──→ loss
           x ──→ ^2 ──┘
```

### Chain Rule
```
∂loss/∂x = ∂loss/∂output · ∂output/∂hidden · ∂hidden/∂x
```

### Gradient Flow
```
Forward:  Input → Operations → Output → Loss
Backward: Loss → Gradients → Weight Updates
```

## Why This Matters

Without automatic differentiation:
- You'd compute gradients by hand (error-prone)
- Training deep networks would be impossible
- No PyTorch, TensorFlow, or modern AI

With YOUR autograd:
- Exact gradients automatically
- Can train networks of any depth
- Same capability as major frameworks!

## Requirements

- Module 09 (Autograd) completed
- TinyTorch package exported

## Mathematical Foundation

The autograd system implements:
- Forward-mode differentiation
- Reverse-mode differentiation (backpropagation)
- Dynamic computational graph construction
- Automatic chain rule application

You built this from scratch! 🎉
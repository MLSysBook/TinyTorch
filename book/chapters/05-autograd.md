---
title: "Autograd"
description: "Automatic differentiation engine for gradient computation"
difficulty: "⭐⭐⭐⭐"
time_estimate: "8-10 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# 05. Autograd

**🏗️ FOUNDATION TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 6-8 hours

## Overview

Build the automatic differentiation engine that makes neural network training possible. This module implements the mathematical foundation that enables backpropagation—transforming TinyTorch from a static computation library into a dynamic, trainable ML framework.

## Learning Objectives

By the end of this module, you will be able to:

- **Master automatic differentiation theory**: Understand computational graphs, chain rule application, and gradient flow
- **Implement gradient tracking systems**: Build the Variable class that automatically computes and accumulates gradients
- **Create differentiable operations**: Extend all mathematical operations to support backward propagation
- **Apply backpropagation algorithms**: Implement the gradient computation that enables neural network optimization
- **Integrate with ML systems**: Connect automatic differentiation with layers, networks, and training algorithms

## Build → Use → Analyze

This module follows TinyTorch's **Build → Use → Analyze** framework:

1. **Build**: Implement Variable class and gradient computation system using mathematical differentiation rules
2. **Use**: Apply automatic differentiation to complex expressions and neural network forward passes
3. **Analyze**: Understand computational graph construction, memory usage, and performance characteristics of autodiff systems

## Implementation Guide

### Automatic Differentiation System
```python
# Variables track gradients automatically
x = Variable(5.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)

# Complex mathematical expressions
z = x**2 + 2*x*y + y**3
print(f"f(x,y) = {z.data}")  # Forward pass result

# Automatic gradient computation
z.backward()
print(f"df/dx = {x.grad}")  # ∂f/∂x = 2x + 2y = 16
print(f"df/dy = {y.grad}")  # ∂f/∂y = 2x + 3y² = 37
```

### Neural Network Integration
```python
# Seamless integration with existing TinyTorch components
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU

# Create differentiable network
x = Variable([[1.0, 2.0, 3.0]], requires_grad=True)
layer1 = Dense(3, 4)  # Weights automatically become Variables
layer2 = Dense(4, 1)
relu = ReLU()

# Forward pass builds computational graph
h1 = relu(layer1(x))
output = layer2(h1)
loss = output.sum()

# Backward pass computes all gradients
loss.backward()

# All parameters now have gradients
print(f"Layer 1 weight gradients: {layer1.weights.grad.shape}")
print(f"Layer 2 bias gradients: {layer2.bias.grad.shape}")
print(f"Input gradients: {x.grad.shape}")
```

### Computational Graph Construction
```python
# Automatic graph building for complex operations
def complex_function(x, y):
    a = x * y          # Multiplication node
    b = x + y          # Addition node  
    c = a / b          # Division node
    return c.sin()     # Trigonometric node

x = Variable(2.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)
result = complex_function(x, y)

# Chain rule applied automatically through entire graph
result.backward()
print(f"Complex gradient dx: {x.grad}")
print(f"Complex gradient dy: {y.grad}")
```

## Getting Started

### Prerequisites
Ensure you understand the mathematical building blocks:

   ```bash
# Activate TinyTorch environment
   source bin/activate-tinytorch.sh

# Verify prerequisite modules
tito test --module tensor
tito test --module activations
tito test --module layers
   ```

### Development Workflow
1. **Open the development file**: `modules/source/08_autograd/autograd_dev.py`
2. **Implement Variable class**: Create gradient tracking wrapper around Tensors
3. **Add basic operations**: Implement differentiable arithmetic (add, multiply, power)
4. **Build backward propagation**: Implement chain rule for gradient computation
5. **Extend to all operations**: Add gradients for activations, matrix operations, etc.
6. **Export and verify**: `tito export --module autograd && tito test --module autograd`

## Testing

### Comprehensive Test Suite
Run the full test suite to verify mathematical correctness:

```bash
# TinyTorch CLI (recommended)
tito test --module autograd

# Direct pytest execution
python -m pytest tests/ -k autograd -v
```

### Test Coverage Areas
- ✅ **Variable Creation**: Test gradient tracking initialization and properties
- ✅ **Basic Operations**: Verify arithmetic operations compute correct gradients
- ✅ **Chain Rule**: Ensure composite functions apply chain rule correctly
- ✅ **Backpropagation**: Test gradient flow through complex computational graphs
- ✅ **Neural Network Integration**: Verify seamless operation with layers and activations

### Inline Testing & Mathematical Verification
The module includes comprehensive mathematical validation:
```python
# Example inline test output
🔬 Unit Test: Variable gradient tracking...
✅ Variable creation with gradient tracking
✅ Leaf variables correctly identified
✅ Gradient accumulation works correctly
📈 Progress: Variable System ✓

# Mathematical verification
🔬 Unit Test: Chain rule implementation...
✅ f(x) = x² → df/dx = 2x ✓
✅ f(x,y) = xy → df/dx = y, df/dy = x ✓
✅ Complex compositions follow chain rule ✓
📈 Progress: Differentiation Rules ✓
```

### Manual Testing Examples
```python
from autograd_dev import Variable
import math

# Test basic differentiation rules
x = Variable(3.0, requires_grad=True)
y = x**2
y.backward()
print(f"d(x²)/dx at x=3: {x.grad}")  # Should be 6

# Test chain rule
x = Variable(2.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)
z = (x + y) * (x - y)  # Difference of squares
z.backward()
print(f"d/dx = {x.grad}")  # Should be 2x = 4
print(f"d/dy = {y.grad}")  # Should be -2y = -6

# Test with transcendental functions
x = Variable(1.0, requires_grad=True)
y = x.exp().log()  # Should equal x
y.backward()
print(f"d(exp(log(x)))/dx: {x.grad}")  # Should be 1
```

## Systems Thinking Questions

### Real-World Applications
- **Deep Learning Frameworks**: PyTorch, TensorFlow, JAX all use automatic differentiation for training
- **Scientific Computing**: Automatic differentiation enables gradient-based optimization in physics, chemistry, engineering
- **Financial Modeling**: Risk analysis and portfolio optimization use autodiff for sensitivity analysis
- **Robotics**: Control systems use gradients for trajectory optimization and inverse kinematics

### Mathematical Foundations
- **Chain Rule**: ∂f/∂x = (∂f/∂u)(∂u/∂x) for composite functions f(u(x))
- **Computational Graphs**: Directed acyclic graphs representing function composition
- **Forward Mode vs Reverse Mode**: Different autodiff strategies with different computational complexities
- **Gradient Accumulation**: Handling multiple computational paths to same variable

### Automatic Differentiation Theory
- **Dual Numbers**: Mathematical foundation using infinitesimals for forward-mode AD
- **Reverse Accumulation**: Backpropagation as reverse-mode automatic differentiation
- **Higher-Order Derivatives**: Computing gradients of gradients for advanced optimization
- **Jacobian Computation**: Efficient computation of vector-valued function gradients

### Implementation Patterns
- **Gradient Function Storage**: Each operation stores its backward function in the computational graph
- **Topological Sorting**: Ordering gradient computation to respect dependencies
- **Memory Management**: Efficient storage and cleanup of intermediate values
- **Numerical Stability**: Handling edge cases in gradient computation

## 🎉 Ready to Build?

You're about to implement the mathematical foundation that makes modern AI possible! Automatic differentiation is the invisible engine that powers every neural network, from simple classifiers to GPT and beyond.

Understanding autodiff from first principles—implementing the Variable class and chain rule yourself—will give you deep insight into how deep learning really works. This is where mathematics meets software engineering to create something truly powerful. Take your time, understand each gradient rule, and enjoy building the heart of machine learning!

 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/09_autograd/autograd_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/09_autograd/autograd_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/09_autograd/autograd_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/08_dataloader.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/10_autograd.html" title="next page">Next Module →</a>
</div>

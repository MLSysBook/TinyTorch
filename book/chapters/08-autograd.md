---
title: "Autograd"
description: "Automatic differentiation engine for gradient computation"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# Module: Autograd
---
**Course Navigation:** [Home](../intro.html) → [Autograd](#)

---



## 📊 Module Info
- **Difficulty**: ⭐⭐⭐⭐ Advanced
- **Time Estimate**: 6-8 hours
- **Prerequisites**: Tensor, Activations, Layers modules
- **Next Steps**: Training, Optimizers modules

**Build the automatic differentiation engine that makes neural network training possible**

## 🎯 Learning Objectives

After completing this module, you will:
- Understand how automatic differentiation works through computational graphs
- Implement the Variable class that tracks gradients and operations
- Build backward propagation for gradient computation
- Create differentiable versions of all mathematical operations
- Master the mathematical foundations of backpropagation

## 🧠 Build → Use → Analyze

This module follows the TinyTorch pedagogical framework:

1. **Build**: Create the Variable class and gradient computation system
2. **Use**: Perform automatic differentiation on complex expressions
3. **Analyze**: Understand how gradients flow through computational graphs and optimize performance

## 📚 What You'll Build

### **Variable Class**
```python
# Gradient-tracking wrapper around Tensors
x = Variable(5.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)
z = x * y + x**2
z.backward()
print(x.grad)  # Gradient of z with respect to x
print(y.grad)  # Gradient of z with respect to y
```

### **Differentiable Operations**
```python
# All operations track gradients automatically
def f(x, y):
    return x**2 + 2*x*y + y**2

x = Variable(2.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)
result = f(x, y)
result.backward()
print(f"df/dx = {x.grad}")  # Should be 2x + 2y = 10
print(f"df/dy = {y.grad}")  # Should be 2x + 2y = 10
```

### **Neural Network Integration**
```python
# Works seamlessly with existing TinyTorch components
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Dense

# Create differentiable network
x = Variable([[1.0, 2.0, 3.0]])
layer = Dense(3, 2)
relu = ReLU()

# Forward pass with gradient tracking
output = relu(layer(x))
loss = output.sum()
loss.backward()

# Gradients available for all parameters
print(layer.weights.grad)  # Weight gradients
print(layer.bias.grad)     # Bias gradients
```

## 🚀 Getting Started

### Prerequisites

1. **Activate the virtual environment**:
   ```bash
   source bin/activate-tinytorch.sh
   ```

2. **Start development environment**:
   ```bash
   tito jupyter
   ```

### Development Workflow

1. **Open the development file**:
   ```bash
   # Then open modules/source/07_autograd/autograd_dev.py
   ```

2. **Implement the core components**:
   - Start with Variable class (gradient tracking)
   - Add basic operations (add, multiply, etc.)
   - Implement backward propagation
   - Add activation function gradients

3. **Test your implementation**:
   ```bash
   tito test --module 07_autograd
   ```

## 📊 Understanding Automatic Differentiation

### The Chain Rule in Action

Automatic differentiation is based on the chain rule:
```
If z = f(g(x)), then dz/dx = (dz/df) * (df/dx)
```

### Computational Graph Example
```
Expression: f(x, y) = (x + y) * (x - y)

Forward Pass:
x = 2, y = 3
a = x + y = 5
b = x - y = -1  
f = a * b = -5

Backward Pass:
df/df = 1
df/da = b = -1, df/db = a = 5
da/dx = 1, da/dy = 1
db/dx = 1, db/dy = -1
df/dx = df/da * da/dx + df/db * db/dx = (-1)(1) + (5)(1) = 4
df/dy = df/da * da/dy + df/db * db/dy = (-1)(1) + (5)(-1) = -6
```

### Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Variable** | Tensor wrapper with gradient tracking | `Variable(5.0, requires_grad=True)` |
| **Computational Graph** | DAG representing operations | `z = x * y` creates graph |
| **Forward Pass** | Computing function values | `z.data` contains result |
| **Backward Pass** | Computing gradients | `z.backward()` fills gradients |
| **Leaf Node** | Variable created by user | `x = Variable(5.0)` |
| **Gradient Function** | How to compute gradients | `grad_fn` for each operation |

## 🧪 Testing Your Implementation

### Unit Tests
```bash
tito test --module 07_autograd
```

**Test Coverage**:
- ✅ Variable creation and properties
- ✅ Basic arithmetic operations
- ✅ Gradient computation correctness
- ✅ Chain rule implementation
- ✅ Integration with existing modules

### Manual Testing
```python
# Test basic gradients
x = Variable(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # Should be 2*2 + 3 = 7

# Test chain rule
x = Variable(2.0, requires_grad=True)
y = Variable(3.0, requires_grad=True)
z = x * y
w = z + x
w.backward()
print(x.grad)  # Should be y + 1 = 4
print(y.grad)  # Should be x = 2
```

## 📊 Mathematical Foundations

### Gradient Computation Rules

| Operation | Forward | Backward (Gradient) |
|-----------|---------|-------------------|
| Addition | `z = x + y` | `dx = dz, dy = dz` |
| Multiplication | `z = x * y` | `dx = y * dz, dy = x * dz` |
| Power | `z = x^n` | `dx = n * x^(n-1) * dz` |
| Exp | `z = exp(x)` | `dx = exp(x) * dz` |
| Log | `z = log(x)` | `dx = (1/x) * dz` |
| ReLU | `z = max(0, x)` | `dx = (x > 0) * dz` |
| Sigmoid | `z = 1/(1+exp(-x))` | `dx = z * (1-z) * dz` |

### Advanced Concepts
- **Higher-order gradients**: Gradients of gradients
- **Jacobian matrices**: Gradients for vector functions
- **Hessian matrices**: Second-order derivatives
- **Gradient checkpointing**: Memory optimization

## 🔧 Integration with TinyTorch

After implementation, your autograd system will enable:

```python
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU

# Create a simple neural network
x = Variable([[1.0, 2.0, 3.0]])
layer1 = Dense(3, 4)
layer2 = Dense(4, 1)
relu = ReLU()

# Forward pass
h = relu(layer1(x))
output = layer2(h)
loss = output.sum()

# Backward pass
loss.backward()

# All gradients computed automatically!
print(layer1.weights.grad)
print(layer2.weights.grad)
```

## 🎯 Success Criteria

Your autograd module is complete when:

1. **All tests pass**: `tito test --module 07_autograd`
2. **Variable imports correctly**: `from tinytorch.core.autograd import Variable`
3. **Basic operations work**: Can create Variables and do arithmetic
4. **Gradients compute correctly**: Backward pass produces correct gradients
5. **Integration works**: Seamlessly works with existing TinyTorch modules

## 💡 Implementation Tips

### Start with the Basics
1. **Variable class** - Wrap Tensors with gradient tracking
2. **Simple operations** - Start with addition and multiplication
3. **Backward method** - Implement gradient computation
4. **Test frequently** - Verify gradients match analytical solutions

### Design Patterns
```python
class Variable:
    def __init__(self, data, requires_grad=True, grad_fn=None):
        # Store data, gradient state, and computation history
        
    def backward(self, gradient=None):
        # Implement backpropagation using chain rule
        
def add(a, b):
    # Create new Variable with grad_fn that knows how to backprop
    def backward_fn(grad):
        # Distribute gradient to inputs
    return Variable(result, grad_fn=backward_fn)
```

### Common Challenges
- **Gradient accumulation** - Handle multiple paths to same Variable
- **Memory management** - Store intermediate values efficiently
- **Numerical stability** - Handle edge cases in gradient computation
- **Graph construction** - Build computation graph correctly

## 🔧 Advanced Features (Optional)

If you finish early, try implementing:
- **Higher-order gradients** - Gradients of gradients
- **Gradient checkpointing** - Memory optimization
- **Custom operations** - Define your own differentiable functions
- **Gradient clipping** - Prevent exploding gradients

## 🚀 Next Steps

Once you complete the autograd module:

1. **Move to Training**: `cd modules/source/08_training/`
2. **Build optimization algorithms**: Implement SGD, Adam, etc.
3. **Create training loops**: Put it all together
4. **Train real models**: Use your autograd system for actual ML!

## 🔗 Why Autograd Matters

Automatic differentiation is the foundation of modern ML:
- **Neural networks** require gradients for backpropagation
- **Optimization** needs gradients for parameter updates
- **Research** benefits from easy gradient computation
- **Production** systems rely on efficient autodiff

This module transforms TinyTorch from a static computation library into a dynamic, trainable ML framework! 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/08_autograd/autograd_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/08_autograd/autograd_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/08_autograd/autograd_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/07_dataloader.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/09_optimizers.html" title="next page">Next Module →</a>
</div>

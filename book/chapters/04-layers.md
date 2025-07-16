---
title: "Layers"
description: "Neural network layers (Linear, activation layers)"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# Module: Layers
---
**Course Navigation:** [Home](../intro.html) → [Layers](#)

---



## 📊 Module Info
- **Difficulty**: ⭐⭐ Intermediate
- **Time Estimate**: 4-5 hours
- **Prerequisites**: Tensor, Activations modules
- **Next Steps**: Networks module

**Build the fundamental transformations that compose into neural networks**

## 🎯 Learning Objectives

After completing this module, you will:
- Understand layers as functions that transform tensors: `y = f(x)`
- Implement Dense layers with linear transformations: `y = Wx + b`
- Add activation functions for nonlinearity (ReLU, Sigmoid, Tanh)
- See how neural networks are just function composition
- Build intuition for neural network architecture before diving into training

## 🧱 Build → Use → Understand

This module follows the TinyTorch pedagogical framework:

1. **Build**: Dense layers and activation functions from scratch
2. **Use**: Transform tensors and see immediate results
3. **Understand**: How neural networks transform information

## 📚 What You'll Build

### **Dense Layer**
```python
layer = Dense(input_size=3, output_size=2)
x = Tensor([[1.0, 2.0, 3.0]])
y = layer(x)  # Shape: (1, 2)
```

### **Activation Functions**
```python
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()

x = Tensor([[-1.0, 0.0, 1.0]])
y_relu = relu(x)      # [0.0, 0.0, 1.0]
y_sigmoid = sigmoid(x)  # [0.27, 0.5, 0.73]
y_tanh = tanh(x)      # [-0.76, 0.0, 0.76]
```

### **Neural Networks**
```python
# 3 → 4 → 2 network
layer1 = Dense(input_size=3, output_size=4)
activation1 = ReLU()
layer2 = Dense(input_size=4, output_size=2)
activation2 = Sigmoid()

# Forward pass
x = Tensor([[1.0, 2.0, 3.0]])
h1 = layer1(x)
h1_activated = activation1(h1)
h2 = layer2(h1_activated)
output = activation2(h2)
```

## 🚀 Getting Started

### Prerequisites
- Complete Module 1: Tensor ✅
- Understand basic linear algebra (matrix multiplication)
- Familiar with Python classes and methods

### Quick Start
```bash
# Navigate to the layers module
cd modules/layers

# Work in the development notebook
jupyter notebook layers_dev.ipynb

# Or work in the Python file
code layers_dev.py
```

## 📖 Module Structure

```
modules/layers/
├── layers_dev.py           # Main development file (work here!)
├── layers_dev.ipynb        # Jupyter notebook version
├── tests/
│   └── test_layers.py      # Comprehensive tests
├── README.md              # This file
└── solutions/             # Reference implementations (if stuck)
```

## 🎓 Learning Path

### Step 1: Dense Layer (Linear Transformation)
- Understand `y = Wx + b`
- Implement weight initialization
- Handle matrix multiplication and bias addition
- Test with single examples and batches

### Step 2: Activation Functions
- Implement ReLU: `max(0, x)`
- Implement Sigmoid: `1 / (1 + e^(-x))`
- Implement Tanh: `tanh(x)`
- Understand why nonlinearity is crucial

### Step 3: Layer Composition
- Chain layers together
- Build complete neural networks
- See how simple layers create complex functions

### Step 4: Real-World Application
- Build an image classification network
- Understand how architecture affects capability

## 🧪 Testing Your Implementation

### Module-Level Tests
```bash
# Run comprehensive tests
python -m pytest tests/test_layers.py -v

# Quick test
python -c "from layers_dev import Dense, ReLU; print('✅ Layers working!')"
```

### Package-Level Tests
```bash
# Export to package
python ../../bin/tito.py sync

# Test integration
python ../../bin/tito.py test --module layers
```

## 🎯 Key Concepts

### **Layers as Functions**
- Input: Tensor with some shape
- Transformation: Mathematical operation
- Output: Tensor with possibly different shape

### **Linear vs Nonlinear**
- Dense layers: Linear transformations
- Activation functions: Nonlinear transformations
- Composition: Linear + Nonlinear = Complex functions

### **Neural Networks = Function Composition**
```
Input → Dense → ReLU → Dense → Sigmoid → Output
```

### **Why This Matters**
- **Modularity**: Build complex networks from simple parts
- **Reusability**: Same layers work for different problems
- **Understanding**: Know how each part contributes to the whole

## 🔍 Common Issues

### **Import Errors**
```python
# Make sure you're in the right directory
import sys
sys.path.append('../../')
from modules.tensor.tensor_dev import Tensor
```

### **Shape Mismatches**
```python
# Check input/output sizes match
layer1 = Dense(input_size=3, output_size=4)
layer2 = Dense(input_size=4, output_size=2)  # 4 matches output of layer1
```

### **Gradient Issues (Later)**
```python
# Use proper weight initialization
limit = math.sqrt(6.0 / (input_size + output_size))
weights = np.random.uniform(-limit, limit, (input_size, output_size))
```

## 🎉 Success Criteria

You've successfully completed this module when:
- ✅ All tests pass (`pytest tests/test_layers.py`)
- ✅ You can build a 2-layer neural network
- ✅ You understand how layers transform tensors
- ✅ You see the connection between layers and neural networks
- ✅ Package export works (`tito test --module layers`)

## 🚀 What's Next

After completing this module, you're ready for:
- **Module 3: Networks** - Compose layers into common architectures
- **Module 4: Training** - Learn how networks improve through experience
- **Module 5: Applications** - Use networks for real problems

## 🤝 Getting Help

- Check the tests for examples of expected behavior
- Look at the solutions/ directory if you're stuck
- Review the pedagogical principles in `docs/pedagogy/`
- Remember: Build → Use → Understand!

---

**Great job building the foundation of neural networks!** 🎉

*This module implements the core insight: neural networks are just function composition of simple building blocks.* 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/04_layers/layers_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/04_layers/layers_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/04_layers/layers_dev.py
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
<a class="left-prev" href="../chapters/03_activations.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/05_networks.html" title="next page">Next Module →</a>
</div>

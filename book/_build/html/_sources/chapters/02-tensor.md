---
title: "Tensor"
description: "Core tensor data structure and operations"
difficulty: "⭐⭐"
time_estimate: "4-6 hours"
prerequisites: []
next_steps: []
learning_objectives: ["**Understand what tensors are** and why they're essential for ML", '**Implement a complete Tensor class** with core operations', '**Handle tensor shapes, data types, and memory management** efficiently', '**Implement element-wise operations and reductions** with proper broadcasting', '**Have a solid foundation** for building neural networks']
---

# Module: Tensor

```{div} badges
⭐⭐ | ⏱️ 4-6 hours
```


## 📊 Module Info
- **Difficulty**: ⭐⭐ Intermediate
- **Time Estimate**: 4-6 hours
- **Prerequisites**: Setup module
- **Next Steps**: Activations, Layers

Build the foundation of TinyTorch! This module implements the core Tensor class - the fundamental data structure that powers all neural networks and machine learning operations.

## 🎯 Learning Objectives

By the end of this module, you will:
- **Understand what tensors are** and why they're essential for ML
- **Implement a complete Tensor class** with core operations
- **Handle tensor shapes, data types, and memory management** efficiently
- **Implement element-wise operations and reductions** with proper broadcasting
- **Have a solid foundation** for building neural networks

## 🧠 Build → Use → Understand

1. **Build**: Complete Tensor class with arithmetic operations, shape management, and reductions
2. **Use**: Create tensors, perform operations, and validate with real data
3. **Understand**: How tensors serve as the foundation for all neural network computations

## 📚 What You'll Build

### Core Tensor Class
```python
# Creating tensors
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([[0.5, 1.5], [2.5, 3.5]])

# Properties
print(x.shape)    # (2, 2)
print(x.size)     # 4
print(x.dtype)    # float64

# Element-wise operations
z = x + y         # Addition
w = x * y         # Multiplication
p = x ** 2        # Exponentiation

# Shape manipulation
reshaped = x.reshape(4, 1)  # (4, 1)
transposed = x.T            # (2, 2) transposed

# Reductions
total = x.sum()             # Scalar sum
means = x.mean(axis=0)      # Mean along axis
```

### Essential Operations
- **Arithmetic**: Addition, subtraction, multiplication, division, powers
- **Shape management**: Reshape, transpose, broadcasting rules
- **Reductions**: Sum, mean, min, max along any axis
- **Memory handling**: Efficient data storage and copying

## 🚀 Getting Started

### Prerequisites Check
```bash
tito test --module setup  # Should pass ✅
```

### Development Workflow
```bash
# Navigate to tensor module
cd modules/source/02_tensor

# Open development file
jupyter notebook tensor_dev.ipynb
# OR edit directly: code tensor_dev.py
```

### Step-by-Step Implementation
1. **Basic Tensor class** - Constructor and properties
2. **Shape management** - Understanding tensor dimensions
3. **Arithmetic operations** - Addition, multiplication, etc.
4. **Utility methods** - Reshape, transpose, sum, mean
5. **Error handling** - Robust edge case management

## 🧪 Testing Your Implementation

### Inline Testing
```python
# Test in the notebook or Python REPL
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Shape: {x.shape}")  # Should be (2, 2)
print(f"Sum: {x.sum()}")    # Should be 10.0
```

### Module Tests
```bash
# Export your tensor implementation
tito export

# Test your implementation
tito test --module tensor
```

### Manual Verification
```python
# Create and test tensors
from tinytorch.core.tensor import Tensor

x = Tensor([1, 2, 3, 4, 5])
y = Tensor([2, 4, 6, 8, 10])

# Test operations
assert (x + y).data.tolist() == [3, 6, 9, 12, 15]
assert (x * 2).data.tolist() == [2, 4, 6, 8, 10]
print("✅ Basic operations working!")
```

## 🎯 Key Concepts

### **Tensors as Universal Data Structures**
- **Scalars**: 0-dimensional tensors (single numbers)
- **Vectors**: 1-dimensional tensors (arrays) 
- **Matrices**: 2-dimensional tensors (common in ML)
- **Higher dimensions**: Images (3D), video (4D), etc.

### **Why Tensors Matter in ML**
- **Neural networks**: All computations operate on tensors
- **GPU acceleration**: operates on tensor primitives
- **Broadcasting**: Efficient operations across different shapes
- **Vectorization**: Process entire datasets simultaneously

### **Real-World Connections**
- **PyTorch/TensorFlow**: Your implementation mirrors production frameworks
- **NumPy**: Foundation for scientific computing (we build similar abstractions)
- **Production systems**: Understanding tensors is essential for ML engineering

### **Memory and Performance**
- **Data layout**: How tensors store data efficiently
- **Broadcasting**: Smart operations without data copying
- **View vs Copy**: Understanding memory management

## 🎉 Ready to Build?

The tensor module is where TinyTorch really begins. You're about to create the fundamental building block that will power neural networks, training loops, and production ML systems.

Take your time, test thoroughly, and enjoy building something that really works! 🔥 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/02_tensor/tensor_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/02_tensor/tensor_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/02_tensor/tensor_dev.py
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
<a class="left-prev" href="../chapters/01_setup.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/03_activations.html" title="next page">Next Module →</a>
</div>

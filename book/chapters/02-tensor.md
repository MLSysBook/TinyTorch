---
title: "Tensor"
description: "Core tensor data structure and operations"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: ["✅ Understand what tensors are and why they're essential for ML", '✅ Implement a complete Tensor class with core operations', '✅ Handle tensor shapes, data types, and memory management', '✅ Implement element-wise operations and reductions', '✅ Have a solid foundation for building neural networks']
---

# 🔥 Module: Tensor
---
**Course Navigation:** [Home](../intro.html) → [Module 2: 02 Tensor](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐⭐ | <strong>Time:</strong> 4-6 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐⭐ Intermediate
- **Time Estimate**: 4-6 hours
- **Prerequisites**: Setup module
- **Next Steps**: Activations, Layers

Build the foundation of TinyTorch! This module implements the core Tensor class - the fundamental data structure that powers all neural networks and machine learning operations.

## 🎯 Learning Objectives

By the end of this module, you will:
- ✅ Understand what tensors are and why they're essential for ML
- ✅ Implement a complete Tensor class with core operations
- ✅ Handle tensor shapes, data types, and memory management
- ✅ Implement element-wise operations and reductions
- ✅ Have a solid foundation for building neural networks

## 📋 Module Structure

```
modules/tensor/
├── README.md           # 📖 This file - Module overview
├── tensor_dev.ipynb        # 📓 Main development notebook
├── test_tensor.py      # 🧪 Automated tests  
└── check_tensor.py     # ✅ Manual verification (coming soon)
```

## 🚀 Getting Started

### Step 1: Complete Prerequisites
Make sure you've completed the setup module:
```bash
python bin/tito.py test --module setup  # Should pass
```

### Step 2: Open the Tensor Notebook
```bash
# Start from the tensor module directory
cd modules/tensor/

# Open the development notebook
jupyter lab tensor_dev.ipynb
```

### Step 3: Work Through the Implementation
The notebook guides you through building:
1. **Basic Tensor class** - Constructor and properties
2. **Shape management** - Understanding tensor dimensions
3. **Arithmetic operations** - Addition, multiplication, etc.
4. **Utility methods** - Reshape, transpose, sum, mean
5. **Error handling** - Robust edge case management

### Step 4: Export and Test
```bash
# Export your tensor implementation
python bin/tito.py sync

# Test your implementation
python bin/tito.py test --module tensor
```

## 📚 What You'll Implement

### Core Tensor Class
You'll build a complete `Tensor` class that supports:

#### 1. Construction and Properties
```python
# Creating tensors
a = Tensor([1, 2, 3])              # 1D tensor
b = Tensor([[1, 2], [3, 4]])       # 2D tensor
c = Tensor(5.0)                     # Scalar tensor

# Properties
print(a.shape)      # (3,)
print(b.size)       # 4  
print(c.dtype)      # float32
```

#### 2. Arithmetic Operations
```python
# Element-wise operations
result = a + b        # Addition
result = a * 2        # Scalar multiplication
result = a @ b        # Matrix multiplication (bonus)
```

#### 3. Utility Methods
```python
# Shape manipulation
reshaped = b.reshape(1, 4)    # Change shape
transposed = b.transpose()     # Swap dimensions

# Reductions
total = a.sum()               # Sum all elements
mean_val = a.mean()           # Average value
max_val = a.max()             # Maximum value
```

### Technical Requirements
Your Tensor class must:
- Handle multiple data types (int, float)
- Support N-dimensional arrays
- Implement proper error checking
- Work with NumPy arrays internally
- Export to `tinytorch.core.tensor`

## 🧪 Testing Your Implementation

### Automated Tests
```bash
python bin/tito.py test --module tensor
```

Tests verify:
- ✅ Tensor creation (scalars, vectors, matrices)
- ✅ Property access (shape, size, dtype)
- ✅ Arithmetic operations (all combinations)
- ✅ Utility methods (reshape, transpose, reductions)
- ✅ Error handling (invalid operations)

### Interactive Testing
```python
# Test in the notebook or Python REPL
from tinytorch.core.tensor import Tensor

# Create and test tensors
a = Tensor([1, 2, 3])
b = Tensor([[1, 2], [3, 4]])
print(a + 5)        # Should work
print(a.sum())      # Should return scalar
```

## 🎯 Success Criteria

Your tensor module is complete when:

1. **All tests pass**: `python bin/tito.py test --module tensor`
2. **Tensor imports correctly**: `from tinytorch.core.tensor import Tensor`
3. **Basic operations work**: Can create tensors and do arithmetic
4. **Properties work**: Shape, size, dtype return correct values
5. **Utilities work**: Reshape, transpose, reductions function properly

## 💡 Implementation Tips

### Start with the Basics
1. **Simple constructor** - Handle lists and NumPy arrays
2. **Basic properties** - Shape, size, dtype
3. **One operation** - Start with addition
4. **Test frequently** - Verify each feature works

### Design Patterns
```python
class Tensor:
    def __init__(self, data, dtype=None):
        # Convert input to numpy array
        # Store shape, size, dtype
        
    def __add__(self, other):
        # Handle tensor + tensor
        # Handle tensor + scalar
        # Return new Tensor
        
    def sum(self, axis=None):
        # Reduce along specified axis
        # Return scalar or tensor
```

### Common Challenges
- **Shape compatibility** - Check dimensions for operations
- **Data type handling** - Convert inputs consistently  
- **Memory efficiency** - Don't create unnecessary copies
- **Error messages** - Provide helpful debugging info

## 🔧 Advanced Features (Optional)

If you finish early, try implementing:
- **Broadcasting** - Operations on different-shaped tensors
- **Slicing** - `tensor[1:3, :]` syntax
- **In-place operations** - `tensor += other`
- **Matrix multiplication** - `tensor @ other`

## 🚀 Next Steps

Once you complete the tensor module:

1. **Move to Autograd**: `cd modules/autograd/`
2. **Build automatic differentiation**: Enable gradient computation
3. **Combine with tensors**: Make tensors differentiable
4. **Prepare for neural networks**: Ready for the MLP module

## 🔗 Why Tensors Matter

Tensors are the foundation of all ML systems:
- **Neural networks** store weights and activations as tensors
- **Training** computes gradients on tensors
- **Data processing** represents batches as tensors
- **GPU acceleration** operates on tensor primitives

Your tensor implementation will power everything else in TinyTorch!

## 🎉 Ready to Build?

The tensor module is where TinyTorch really begins. You're about to create the fundamental building block that will power neural networks, training loops, and production ML systems.

Take your time, test thoroughly, and enjoy building something that really works! 🔥 

---

## 🚀 Ready to Build?

```{admonition} Choose Your Environment
:class: tip
**Quick Start:** [🚀 Launch Builder](https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/02_tensor/tensor_dev.ipynb) *(Jump directly into implementation)*

**Full Development:** [📓 Open Jupyter](https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/02_tensor/tensor_dev.ipynb) *(Complete development environment)*

**Cloud Environment:** [☁️ Open in Colab](https://colab.research.google.com/github/MLSysBook/TinyTorch/blob/main/modules/source/02_tensor/tensor_dev.ipynb) *(Google's notebook environment)*
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/01_setup.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/03_activations.html" title="next page">Next Module →</a>
</div>

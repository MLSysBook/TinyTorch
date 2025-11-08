---
title: "Tensor"
description: "Core tensor data structure and operations"
module_number: 1
tier: "foundation"
difficulty: "beginner"
time_estimate: "4-6 hours"
prerequisites: ["Environment Setup"]
next_module: "02. Activations"
learning_objectives:
  - "Understand tensors as N-dimensional arrays and their role in ML systems"
  - "Implement a complete Tensor class with arithmetic and shape operations"
  - "Handle memory management, data types, and broadcasting efficiently"
  - "Recognize how tensor operations form the foundation of PyTorch/TensorFlow"
  - "Analyze computational complexity and memory usage of tensor operations"
---

# 01. Tensor

**🏗️ FOUNDATION TIER** | Difficulty: ⭐ (1/4) | Time: 4-6 hours

**Build N-dimensional arrays from scratch - the foundation of all ML computations.**

---

## What You'll Build

The **Tensor** class is the fundamental data structure of machine learning. It represents N-dimensional arrays and provides operations for manipulation, computation, and transformation.

By the end of this module, you'll have a working Tensor implementation that handles:

- Creating and initializing N-dimensional arrays
- Arithmetic operations (addition, multiplication, division, powers)
- Shape manipulation (reshape, transpose, broadcasting)
- Reductions (sum, mean, min, max along any axis)
- Memory-efficient data storage and copying

### Example Usage

```python
from tinytorch.core.tensor import Tensor

# Create tensors
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([[0.5, 1.5], [2.5, 3.5]])

# Properties
print(x.shape)    # (2, 2)
print(x.size)     # 4
print(x.dtype)    # float64

# Operations
z = x + y         # Addition
w = x * y         # Element-wise multiplication
p = x ** 2        # Exponentiation

# Shape manipulation
reshaped = x.reshape(4, 1)
transposed = x.T

# Reductions
total = x.sum()             # Scalar sum
means = x.mean(axis=0)      # Mean along axis
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement the Tensor class from scratch using NumPy as the underlying array library. You'll create constructors, operator overloading, shape manipulation methods, and reduction operations.

### 2. Use
Apply your Tensor implementation to real problems: matrix multiplication, data normalization, statistical computations. Test with various shapes and data types.

### 3. Understand
Grasp the systems-level implications: why tensor operations dominate compute time, how memory layout affects performance, and how broadcasting enables efficient computations without data copying.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize tensors as the universal data structure in ML frameworks, understanding how all neural network operations decompose into tensor primitives

2. **Core Implementation**: Build a complete Tensor class supporting arithmetic, shape manipulation, and reductions with proper error handling

3. **Pattern Recognition**: Understand broadcasting rules and how they enable efficient computations across different tensor shapes

4. **Framework Connection**: See how your implementation mirrors PyTorch's `torch.Tensor` and TensorFlow's `tf.Tensor` design

5. **Performance Trade-offs**: Analyze memory usage vs computation speed, understanding when to copy data vs create views

---

## Why This Matters

### Production Context

Every modern ML framework is built on tensors:

- **PyTorch**: `torch.Tensor` is the core class - all operations work with tensors
- **TensorFlow**: `tf.Tensor` represents data flowing through computation graphs  
- **JAX**: `jax.numpy.ndarray` extends NumPy with automatic differentiation
- **NumPy**: The foundation - understanding tensors starts here

By building your own Tensor class, you'll understand what happens when you call `torch.matmul()` or `tf.reduce_sum()` - not just the API, but the actual computation.

### Systems Reality Check

**Performance Note**: Tensor operations dominate training time. A single matrix multiplication in a linear layer might take 90% of forward pass time. Understanding tensor internals is essential for optimization.

**Memory Note**: Large models store billions of parameters as tensors. A GPT-3 scale model requires 350GB of memory just for weights (175B parameters × 2 bytes for FP16). Efficient tensor memory management is critical.

---

## Implementation Guide

### Prerequisites Check

Verify your environment is ready:

```bash
tito system doctor
```

All checks should pass before starting implementation.

### Development Workflow

```bash
# Navigate to tensor module
cd modules/source/01_tensor/

# Open development file (choose your preferred method)
jupyter lab tensor_dev.py          # Jupytext (recommended)
# OR
code tensor_dev.py                 # Direct Python editing
```

### Step-by-Step Build

#### Step 1: Tensor Class Foundation

Create the basic Tensor class with initialization and properties:

```python
class Tensor:
    def __init__(self, data, dtype=None):
        """Initialize tensor from Python list or NumPy array"""
        self.data = np.array(data, dtype=dtype)
    
    @property
    def shape(self):
        """Return tensor shape"""
        return self.data.shape
    
    @property
    def size(self):
        """Return total number of elements"""
        return self.data.size
```

**Why this matters**: Properties enable clean API design - users can write `x.shape` instead of `x.get_shape()`, matching PyTorch conventions.

#### Step 2: Arithmetic Operations

Implement operator overloading for element-wise operations:

```python
def __add__(self, other):
    """Element-wise addition"""
    return Tensor(self.data + other.data)

def __mul__(self, other):
    """Element-wise multiplication"""
    return Tensor(self.data * other.data)
```

**Systems insight**: These operations vectorize automatically via NumPy, achieving ~100x speedup over Python loops. This is why frameworks use tensors.

#### Step 3: Shape Manipulation

Implement reshape, transpose, and broadcasting:

```python
def reshape(self, *shape):
    """Return tensor with new shape"""
    return Tensor(self.data.reshape(*shape))

@property
def T(self):
    """Return transposed tensor"""
    return Tensor(self.data.T)
```

**Memory consideration**: Reshape and transpose often return *views* (no data copying) for efficiency. Understanding views vs copies is crucial for memory optimization.

#### Step 4: Reductions

Implement aggregation operations along axes:

```python
def sum(self, axis=None):
    """Sum tensor elements along axis"""
    return Tensor(self.data.sum(axis=axis))

def mean(self, axis=None):
    """Mean of tensor elements along axis"""
    return Tensor(self.data.mean(axis=axis))
```

**Production pattern**: Reductions are fundamental - every loss function uses them. Understanding axis semantics prevents bugs in multi-dimensional operations.

---

## Testing Your Implementation

### Inline Tests

Test within your development file:

```python
# Create test tensors
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])

# Test operations
assert x.shape == (2, 2)
assert (x + y).data.tolist() == [[6, 8], [10, 12]]
assert x.sum().data == 10
print("✓ Basic operations working")
```

### Module Export & Validation

```bash
# Export your implementation to TinyTorch package
tito export 01

# Run comprehensive test suite
tito test 01
```

**Expected output**:
```
✓ All tests passed! [25/25]
✓ Module 01 complete!
```

---

## Where This Code Lives

After export, your Tensor implementation becomes part of the TinyTorch package:

```python
# Other modules and future code can now import YOUR implementation:
from tinytorch.core.tensor import Tensor

# Used throughout TinyTorch:
from tinytorch.core.layers import Linear      # Uses Tensor for weights
from tinytorch.core.activations import ReLU   # Operates on Tensors
from tinytorch.core.autograd import backward  # Computes Tensor gradients
```

**Package structure**:
```
tinytorch/
├── core/
│   ├── tensor.py  ← YOUR implementation exports here
│   ├── activations.py
│   ├── layers.py
│   └── ...
```

---

## Systems Thinking Questions

Reflect on these questions as you build (no right/wrong answers):

1. **Complexity Analysis**: Why is matrix multiplication O(n³) for n×n matrices? How does this affect training time for large models?

2. **Memory Trade-offs**: When should reshape create a view vs copy data? What are the performance implications?

3. **Production Scaling**: A GPT-3 scale model has 175 billion parameters. How much memory is required to store these as FP32 tensors? As FP16?

4. **Design Decisions**: Why do frameworks like PyTorch store data as NumPy arrays internally? What are alternatives?

5. **Framework Comparison**: How does your Tensor class differ from `torch.Tensor`? What features are missing? Why might those features matter?

---

## Real-World Connections

### Industry Applications

- **Deep Learning Training**: All neural network layers operate on tensors (Linear, Conv2d, Attention all perform tensor operations)
- **Scientific Computing**: Tensors represent multidimensional data (climate models, molecular simulations)
- **Computer Vision**: Images are 3D tensors (height × width × channels)
- **NLP**: Text embeddings are 2D tensors (sequence_length × embedding_dim)

### Research Applications

- **Automatic Differentiation**: Frameworks like PyTorch track tensor operations to compute gradients
- **Distributed Training**: Large models split tensors across GPUs using tensor parallelism
- **Quantization**: Tensors can be stored in reduced precision (INT8 instead of FP32) for efficiency

---

## What's Next?

**Congratulations!** You've built the foundation of TinyTorch. Your Tensor class will power everything that follows - from activation functions to complete neural networks.

Next, you'll add nonlinearity to enable networks to learn complex patterns.

**Module 02: Activations** - Implement ReLU, Sigmoid, Tanh, and other activation functions that transform tensor values

[Continue to Module 02: Activations →](02-activations.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Tensor API Reference](../appendices/api-reference.html#tensor)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)

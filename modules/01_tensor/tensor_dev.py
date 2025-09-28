# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Tensor - The Foundation of Machine Learning

Welcome to Tensor! You'll build the fundamental data structure that powers every neural network.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 01 (Setup): Python environment with NumPy, the foundation for numerical computing

**What's Working**: You have a complete development environment with all the tools needed for machine learning!

**The Gap**: You can import NumPy, but you need to understand how to build the core data structure that makes ML possible.

**This Module's Solution**: Build a complete Tensor class that wraps NumPy arrays with ML-specific operations and memory management.

**Connection Map**:
```
Setup → Tensor → Activations
(tools)   (data)   (nonlinearity)
```

## Learning Objectives

By completing this module, you will:

1. **Implement tensor operations** - Build a complete N-dimensional array system with arithmetic, broadcasting, and matrix multiplication
2. **Master memory efficiency** - Understand why memory layout affects performance more than algorithm choice
3. **Create ML-ready APIs** - Design clean interfaces that mirror PyTorch and TensorFlow patterns
4. **Enable neural networks** - Build the foundation that supports weights, biases, and data in all ML models

## Build → Test → Use

1. **Build**: Implement Tensor class with creation, arithmetic, and advanced operations
2. **Test**: Validate each component immediately to ensure correctness and performance
3. **Use**: Apply tensors to real multi-dimensional data operations that neural networks require
"""

# In[ ]:

#| default_exp core.tensor

#| export
import numpy as np
import sys
from typing import Union, Tuple, Optional, Any
import warnings

# In[ ]:

print("🔥 TinyTorch Tensor Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build tensors!")

# %% [markdown]
"""
## Understanding Tensors: Visual Guide

### What Are Tensors? A Visual Journey

**The Story**: Think of tensors as smart containers that know their shape and can efficiently store numbers for machine learning. They're like upgraded versions of regular Python lists that understand mathematics.

```
Scalar (0D Tensor):     Vector (1D Tensor):     Matrix (2D Tensor):
     [5]                   [1, 2, 3]             ┌ 1  2  3 ┐
                                                  │ 4  5  6 │
                                                  └ 7  8  9 ┘

3D Tensor (RGB Image):                   4D Tensor (Batch of Images):
┌─────────────┐                         ┌─────────────┐ ┌─────────────┐
│ Red Channel │                         │   Image 1   │ │   Image 2   │
│             │                         │             │ │             │
└─────────────┘                         └─────────────┘ └─────────────┘
┌─────────────┐                                      ...
│Green Channel│
│             │
└─────────────┘
┌─────────────┐
│Blue Channel │
│             │
└─────────────┘
```

**What's happening step-by-step**: As we add dimensions, tensors represent more complex data. A single number becomes a list, a list becomes a grid, a grid becomes a volume (like an image with red/green/blue channels), and a volume becomes a collection (like a batch of images for training). Each dimension adds a new way to organize and access the data.
"""

# %% [markdown]
"""
### Memory Layout: Why Performance Matters

**The Story**: Imagine your computer's memory as a long street with numbered houses. When your CPU needs data, it doesn't just grab one house - it loads an entire city block (64 bytes) into its cache.

```
Contiguous Memory (FAST):
[1][2][3][4][5][6] ──> Cache-friendly, vectorized operations
 ↑  ↑  ↑  ↑  ↑  ↑
 Sequential access pattern

Non-contiguous Memory (SLOW):
[1]...[2].....[3] ──> Cache misses, scattered access
 ↑     ↑       ↑
 Random access pattern
```

**What's happening step-by-step**: When you access element [1], the CPU automatically loads elements [1] through [6] in one cache load. Every subsequent access ([2], [3], [4]...) is already in the cache - no extra memory trips needed! With non-contiguous data, each access requires a new, expensive trip to main memory.

**The Performance Impact**: This creates 10-100x speedups because you get 6 elements for the price of fetching 1. It's like getting 6 books from the library for the effort of finding just 1.
"""

# %% [markdown]
"""
### Tensor Operations: Broadcasting Magic

**The Story**: Broadcasting is like having a smart photocopier that automatically copies data to match different shapes without actually using extra memory. It's NumPy's way of making operations "just work" between tensors of different sizes.

```
Broadcasting Example:
    Matrix (2×3)     +     Scalar        =     Result (2×3)
  ┌ 1  2  3 ┐             [10]              ┌ 11 12 13 ┐
  └ 4  5  6 ┘                               └ 14 15 16 ┘

Broadcasting Rules:
1. Align shapes from right to left
2. Dimensions of size 1 stretch to match
3. Missing dimensions assume size 1

Vector + Matrix Broadcasting:
  [1, 2, 3]    +    [[10],     =    [[11, 12, 13],
  (1×3)             [20]]            [21, 22, 23]]
                    (2×1)            (2×3)
```

**What's happening step-by-step**: Python aligns shapes from right to left, like comparing numbers by their ones place first. When shapes don't match, dimensions of size 1 automatically "stretch" to match the larger dimension - but no data is actually copied. The operation happens as if the data were copied, but uses the original memory locations.

**Why this matters for ML**: Adding a bias vector to a 1000×1000 matrix would normally require copying the vector 1000 times, but broadcasting does it with zero copies and massive memory savings.
"""

# %% [markdown]
"""
### Neural Network Data Flow

```
Batch Processing in Neural Networks:

Input Batch (32 images, 28×28 pixels):
┌─────────────────────────────────┐
│ [Batch=32, Height=28, Width=28] │
└─────────────────────────────────┘
             ↓ Flatten
┌─────────────────────────────────┐
│     [Batch=32, Features=784]    │ ← Matrix multiplication ready
└─────────────────────────────────┘
             ↓ Linear Layer
┌─────────────────────────────────┐
│     [Batch=32, Hidden=128]      │ ← Hidden layer activations
└─────────────────────────────────┘

Why batching matters:
- Single image: 784 × 128 = 100,352 operations
- Batch of 32: Same 100,352 ops, but 32× the data
- GPU utilization: 32× better parallelization
```
"""

# %% [markdown]
"""
## The Mathematical Foundation

Before we implement, let's understand the mathematical concepts:
"""

# %% [markdown]
"""
### Scalars to Tensors: Building Complexity

**Scalar (Rank 0)**:
- A single number: `5.0` or `temperature`
- Shape: `()` (empty tuple)
- ML examples: loss values, learning rates

**Vector (Rank 1)**:
- Ordered list of numbers: `[1, 2, 3]`
- Shape: `(3,)` (one dimension)
- ML examples: word embeddings, gradients

**Matrix (Rank 2)**:
- 2D array: `[[1, 2], [3, 4]]`
- Shape: `(2, 2)` (rows, columns)
- ML examples: weight matrices, images

**Higher-Order Tensors**:
- 3D: RGB images `(height, width, channels)`
- 4D: Image batches `(batch, height, width, channels)`
- 5D: Video batches `(batch, time, height, width, channels)`
"""

# %% [markdown]
"""
### Why Not Just Use NumPy?

While NumPy is excellent, our Tensor class adds ML-specific features:

**Future Extensions** (coming in later modules):
- **Automatic gradients**: Track operations for backpropagation
- **GPU acceleration**: Move computations to graphics cards
- **Lazy evaluation**: Build computation graphs for optimization

**Educational Value**:
- **Understanding**: See how PyTorch/TensorFlow work internally
- **Debugging**: Trace operations step by step
- **Customization**: Add domain-specific operations
"""

# %% [markdown]
"""
## Implementation Overview

Our Tensor class design:

```python
class Tensor:
    def __init__(self, data)      # Create from any data type

    # Properties
    .shape                        # Dimensions tuple
    .size                         # Total element count
    .dtype                        # Data type
    .data                         # Access underlying NumPy array

    # Arithmetic Operations
    def __add__(self, other)      # tensor + tensor
    def __mul__(self, other)      # tensor * tensor
    def __sub__(self, other)      # tensor - tensor
    def __truediv__(self, other)  # tensor / tensor

    # Advanced Operations
    def matmul(self, other)       # Matrix multiplication
    def sum(self, axis=None)      # Sum along axes
    def reshape(self, *shape)     # Change shape
```
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-init", "solution": true}

#| export
class Tensor:
    """
    TinyTorch Tensor: N-dimensional array with ML operations.

    The fundamental data structure for all TinyTorch operations.
    Wraps NumPy arrays with ML-specific functionality.
    """

    def __init__(self, data: Any, dtype: Optional[str] = None, requires_grad: bool = False):
        """
        Create a new tensor from data.

        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
            requires_grad: Whether this tensor needs gradients for training. Defaults to False.

        TODO: Implement tensor creation with simple, clear type handling.

        APPROACH (Clear implementation for learning):
        1. Convert input data to numpy array - NumPy handles conversions
        2. Apply dtype if specified - common string types like 'float32'
        3. Set default float32 for float64 arrays - ML convention for efficiency
        4. Store the result in self._data - internal storage for numpy array
        5. Initialize gradient tracking - prepares for automatic differentiation

        EXAMPLE:
        >>> Tensor(5)
        # Creates: np.array(5, dtype='int32')
        >>> Tensor([1.0, 2.0, 3.0])
        # Creates: np.array([1.0, 2.0, 3.0], dtype='float32')
        >>> Tensor([1, 2, 3], dtype='float32')
        # Creates: np.array([1, 2, 3], dtype='float32')

        PRODUCTION CONTEXT:
        PyTorch tensors handle 47+ dtype formats with complex validation.
        Our version teaches the core concept that transfers directly.
        """
        ### BEGIN SOLUTION
        # Convert input to numpy array - let NumPy handle most conversions
        if isinstance(data, Tensor):
            # Input is another Tensor - copy data efficiently
            self._data = data.data.copy()
        else:
            # Convert to numpy array
            self._data = np.array(data)

        # Apply dtype if specified
        if dtype is not None:
            self._data = self._data.astype(dtype)
        elif self._data.dtype == np.float64:
            # ML convention: prefer float32 for memory and GPU efficiency
            self._data = self._data.astype(np.float32)

        # Initialize gradient tracking attributes (used in Module 9 - Autograd)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        ### END SOLUTION

    @property
    def data(self) -> np.ndarray:
        """
        Access underlying numpy array.

        TODO: Return the stored numpy array.

        APPROACH (Medium comments for property methods):
        1. Access the internal _data attribute
        2. Return the numpy array directly - enables NumPy integration
        3. This provides access to underlying data for visualization/analysis

        PRODUCTION CONNECTION:
        - PyTorch: tensor.numpy() converts to NumPy for scientific computing
        - TensorFlow: tensor.numpy() enables integration with matplotlib/scipy
        - Production use: Data scientists need raw arrays for debugging/visualization
        """
        ### BEGIN SOLUTION
        return self._data
        ### END SOLUTION
    
    @data.setter
    def data(self, value: Union[np.ndarray, 'Tensor']) -> None:
        """Set the underlying data of the tensor."""
        if isinstance(value, Tensor):
            self._data = value._data.copy()
        else:
            self._data = np.array(value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get tensor shape.

        TODO: Return the shape of the stored numpy array.

        APPROACH:
        1. Access the _data attribute (the NumPy array)
        2. Get the shape property from the NumPy array
        3. Return the shape tuple directly

        PRODUCTION CONNECTION:
        - Neural networks: Layer compatibility requires matching shapes
        - Computer vision: Image shape (height, width, channels) determines architecture
        - Debugging: Shape mismatches are the #1 cause of ML errors
        """
        ### BEGIN SOLUTION
        return self._data.shape
        ### END SOLUTION

    @property
    def size(self) -> int:
        """
        Get total number of elements.

        TODO: Return the total number of elements in the tensor.

        APPROACH:
        1. Access the _data attribute (the NumPy array)
        2. Get the size property from the NumPy array
        3. Return the total element count as an integer

        PRODUCTION CONNECTION:
        - Memory planning: Calculate RAM requirements for large tensors
        - Model architecture: Determine parameter counts for layers
        - Performance: Size affects computation time and vectorization efficiency
        """
        ### BEGIN SOLUTION
        return self._data.size
        ### END SOLUTION

    @property
    def dtype(self) -> np.dtype:
        """
        Get data type as numpy dtype.

        TODO: Return the data type of the stored numpy array.

        APPROACH:
        1. Access the _data attribute
        2. Get the dtype property
        3. Return the NumPy dtype object

        PRODUCTION CONNECTION:
        - Precision vs speed: float32 is faster, float64 more accurate
        - Memory optimization: int8 uses 1/4 memory of int32
        - GPU compatibility: Some operations only work with specific types
        """
        ### BEGIN SOLUTION
        return self._data.dtype
        ### END SOLUTION

    @property
    def strides(self) -> Tuple[int, ...]:
        """
        Get memory stride pattern of the tensor.
        
        Returns:
            Tuple of byte strides for each dimension
            
        PRODUCTION CONNECTION:
        - Memory layout analysis: Understanding cache efficiency
        - Performance debugging: Non-unit strides can indicate copies
        - Advanced operations: Enables efficient transpose and reshape operations
        """
        return self._data.strides
    
    @property
    def is_contiguous(self) -> bool:
        """
        Check if tensor data is stored in contiguous memory.
        
        Returns:
            True if data is contiguous in C-order (row-major)
            
        PRODUCTION CONNECTION:
        - Performance critical: Contiguous data enables vectorization
        - Memory efficiency: Contiguous operations can be 10-100x faster
        - GPU transfers: Contiguous data transfers more efficiently
        """
        return self._data.flags['C_CONTIGUOUS']

    def __repr__(self) -> str:
        """
        String representation with size limits for readability.

        TODO: Create a clear string representation of the tensor.

        APPROACH (Light comments for utility methods):
        1. Check tensor size - if large, show shape/dtype only
        2. For small tensors, convert numpy array to list using .tolist()
        3. Format appropriately and return string

        EXAMPLE:
        Tensor([1, 2, 3]) → "Tensor([1, 2, 3], shape=(3,), dtype=int32)"
        Large tensor → "Tensor(shape=(1000, 1000), dtype=float32)"
        """
        ### BEGIN SOLUTION
        if self.size > 20:
            # Large tensors: show shape and dtype only for readability
            return f"Tensor(shape={self.shape}, dtype={self.dtype})"
        else:
            # Small tensors: show data, shape, and dtype
            return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"
        ### END SOLUTION

    def item(self) -> Union[int, float]:
        """Extract a scalar value from a single-element tensor."""
        if self._data.size != 1:
            raise ValueError(f"item() can only be called on tensors with exactly one element, got {self._data.size} elements")
        return self._data.item()

# %% nbgrader={"grade": false, "grade_id": "tensor-arithmetic", "solution": true}
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add two tensors element-wise.

        TODO: Implement tensor addition.

        APPROACH:
        1. Extract numpy arrays from both tensors
        2. Use NumPy's + operator for element-wise addition
        3. Create new Tensor object with result
        4. Return the new tensor

        PRODUCTION CONNECTION:
        - Neural networks: Adding bias terms to linear layer outputs
        - Residual connections: skip connections in ResNet architectures
        - Gradient updates: Adding computed gradients to parameters
        """
        ### BEGIN SOLUTION
        result_data = self._data + other._data
        result = Tensor(result_data)
        
        # TODO: Gradient tracking will be added in Module 9 (Autograd)
        # This enables automatic differentiation for neural network training
        # For now, we focus on the core tensor operation
        
        return result
        ### END SOLUTION

    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply two tensors element-wise.

        TODO: Implement tensor multiplication.

        APPROACH:
        1. Extract numpy arrays from both tensors
        2. Use NumPy's * operator for element-wise multiplication
        3. Create new Tensor object with result
        4. Return the new tensor

        PRODUCTION CONNECTION:
        - Activation functions: Element-wise operations like ReLU masking
        - Attention mechanisms: Element-wise scaling in transformer models
        - Feature scaling: Multiplying features by learned scaling factors
        """
        ### BEGIN SOLUTION
        result_data = self._data * other._data
        result = Tensor(result_data)
        
        # TODO: Gradient tracking will be added in Module 9 (Autograd)
        # This enables automatic differentiation for neural network training
        # For now, we focus on the core tensor operation
        
        return result
        ### END SOLUTION

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Addition operator: tensor + other

        TODO: Implement + operator for tensors.

        APPROACH:
        1. Check if other is a Tensor object
        2. If Tensor, call the add() method directly
        3. If scalar, convert to Tensor then call add()
        4. Return the result from add() method

        PRODUCTION CONNECTION:
        - Natural syntax: tensor + scalar enables intuitive code
        - Broadcasting: Adding scalars to tensors is common in ML
        - API design: Clean interfaces reduce cognitive load for researchers
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return self.add(other)
        else:
            return self.add(Tensor(other))
        ### END SOLUTION

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Multiplication operator: tensor * other

        TODO: Implement * operator for tensors.

        APPROACH:
        1. Check if other is a Tensor object
        2. If Tensor, call the multiply() method directly
        3. If scalar, convert to Tensor then call multiply()
        4. Return the result from multiply() method

        PRODUCTION CONNECTION:
        - Scaling features: tensor * learning_rate for gradient updates
        - Masking: tensor * mask for attention mechanisms
        - Regularization: tensor * dropout_mask during training
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return self.multiply(other)
        else:
            return self.multiply(Tensor(other))
        ### END SOLUTION

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Subtraction operator: tensor - other

        TODO: Implement - operator for tensors.

        APPROACH:
        1. Check if other is a Tensor object
        2. If Tensor, subtract other._data from self._data
        3. If scalar, subtract scalar directly from self._data
        4. Create new Tensor with result and return

        PRODUCTION CONNECTION:
        - Gradient computation: parameter - learning_rate * gradient
        - Error calculation: predicted - actual for loss computation
        - Centering data: tensor - mean for zero-centered inputs
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            result = self._data - other._data
        else:
            result = self._data - other
        return Tensor(result)
        ### END SOLUTION

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Division operator: tensor / other

        TODO: Implement / operator for tensors.

        APPROACH:
        1. Check if other is a Tensor object
        2. If Tensor, divide self._data by other._data
        3. If scalar, divide self._data by scalar directly
        4. Create new Tensor with result and return

        PRODUCTION CONNECTION:
        - Normalization: tensor / std_deviation for standard scaling
        - Learning rate decay: parameter / decay_factor over time
        - Probability computation: counts / total_counts for frequencies
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            result = self._data / other._data
        else:
            result = self._data / other
        return Tensor(result)
        ### END SOLUTION

    def mean(self) -> 'Tensor':
        """Computes the mean of the tensor's elements."""
        return Tensor(np.mean(self.data))
    
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """
        Sum tensor elements along specified axes.
        
        Args:
            axis: Axis or axes to sum over. If None, sum all elements.
            keepdims: Whether to keep dimensions of size 1 in output.
            
        Returns:
            New tensor with summed values.
        """
        result_data = np.sum(self._data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)
        
        if self.requires_grad:
            result.requires_grad = True
            
            def grad_fn(grad):
                # Sum gradient: broadcast gradient back to original shape
                grad_data = grad.data
                if axis is None:
                    # Sum over all axes - gradient is broadcast to full shape
                    grad_data = np.full(self.shape, grad_data)
                else:
                    # Sum over specific axes - expand back those dimensions
                    if not isinstance(axis, tuple):
                        axis_tuple = (axis,) if axis is not None else ()
                    else:
                        axis_tuple = axis
                    
                    # Expand dimensions that were summed
                    for ax in sorted(axis_tuple):
                        if ax < 0:
                            ax = len(self.shape) + ax
                        grad_data = np.expand_dims(grad_data, axis=ax)
                    
                    # Broadcast to original shape
                    grad_data = np.broadcast_to(grad_data, self.shape)
                
                self.backward(Tensor(grad_data))
            
            result._grad_fn = grad_fn
        
        return result

    # %% nbgrader={"grade": false, "grade_id": "tensor-matmul", "solution": true}
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication using NumPy's optimized implementation.

        TODO: Implement matrix multiplication.

        APPROACH:
        1. Extract numpy arrays from both tensors
        2. Check tensor shapes for compatibility
        3. Use NumPy's optimized dot product
        4. Create new Tensor object with the result
        5. Return the new tensor
        """
        ### BEGIN SOLUTION
        a_data = self._data
        b_data = other._data

        # Validate tensor shapes
        if len(a_data.shape) != 2 or len(b_data.shape) != 2:
            raise ValueError("matmul requires 2D tensors")

        m, k = a_data.shape
        k2, n = b_data.shape

        if k != k2:
            raise ValueError(f"Inner dimensions must match: {k} != {k2}")

        # Use NumPy's optimized implementation
        result_data = np.dot(a_data, b_data)
        return Tensor(result_data)
        ### END SOLUTION

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication operator: tensor @ other

        Enables the @ operator for matrix multiplication, providing
        clean syntax for neural network operations.
        """
        return self.matmul(other)

    def backward(self, gradient=None):
        """
        Compute gradients for this tensor and propagate backward.

        Basic backward pass - accumulates gradients and propagates to dependencies.
        This enables simple gradient computation for basic operations.

        Args:
            gradient: Gradient from upstream. If None, assumes scalar with grad=1
        """
        if not self.requires_grad:
            return

        if gradient is None:
            # Scalar case - gradient is 1
            gradient = Tensor(np.ones_like(self._data))

        # Accumulate gradients
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        # Propagate to dependencies via grad_fn
        if self._grad_fn is not None:
            self._grad_fn(gradient)
    
    def zero_grad(self):
        """Reset gradients to None. Used by optimizers before backward pass."""
        self.grad = None

# %% nbgrader={"grade": false, "grade_id": "tensor-reshape", "solution": true}
    def reshape(self, *shape: int) -> 'Tensor':
        """
        Return a new tensor with the same data but different shape.

        Args:
            *shape: New shape dimensions. Use -1 for automatic sizing.

        Returns:
            New Tensor with reshaped data
            
        Note:
            This returns a view when possible (no copying), or a copy when necessary.
            Use .contiguous() after reshape if you need guaranteed contiguous memory.
        """
        reshaped_data = self._data.reshape(*shape)
        result = Tensor(reshaped_data)
        
        # Preserve gradient tracking
        if self.requires_grad:
            result.requires_grad = True
            
            def grad_fn(grad):
                # Reshape gradient back to original shape
                orig_grad = grad.reshape(*self.shape)
                self.backward(orig_grad)
            
            result._grad_fn = grad_fn
        
        return result
    
    def view(self, *shape: int) -> 'Tensor':
        """
        Return a view of the tensor with a new shape. Alias for reshape.
        
        Args:
            *shape: New shape dimensions. Use -1 for automatic sizing.
            
        Returns:
            New Tensor sharing the same data (view when possible)
            
        PRODUCTION CONNECTION:
        - PyTorch compatibility: .view() is the PyTorch equivalent
        - Memory efficiency: Views avoid copying data when possible
        - Performance critical: Views enable efficient transformations
        """
        return self.reshape(*shape)
    
    def clone(self) -> 'Tensor':
        """
        Create a deep copy of the tensor.
        
        Returns:
            New Tensor with copied data
            
        PRODUCTION CONNECTION:
        - Memory isolation: Ensures modifications don't affect original
        - Gradient tracking: Clones maintain independent gradient graphs
        - Safe operations: Use when you need guaranteed data independence
        """
        cloned_data = self._data.copy()
        result = Tensor(cloned_data)
        
        # Clone preserves gradient requirements but starts fresh grad tracking
        result.requires_grad = self.requires_grad
        # Note: grad and grad_fn are NOT copied - clone starts fresh
        
        return result
    
    def contiguous(self) -> 'Tensor':
        """
        Return a contiguous tensor with the same data.
        
        Returns:
            Tensor with contiguous memory layout (may be a copy)
            
        PRODUCTION CONNECTION:
        - Performance optimization: Ensures optimal memory layout
        - GPU operations: Many CUDA operations require contiguous data
        - Cache efficiency: Contiguous data maximizes CPU cache utilization
        """
        if self.is_contiguous:
            return self  # Already contiguous, return self
        
        # Make contiguous copy
        contiguous_data = np.ascontiguousarray(self._data)
        result = Tensor(contiguous_data)
        
        # Preserve gradient tracking
        result.requires_grad = self.requires_grad
        if self.requires_grad:
            def grad_fn(grad):
                self.backward(grad)
            result._grad_fn = grad_fn
        
        return result

    def numpy(self) -> np.ndarray:
        """
        Convert tensor to NumPy array.
        
        This is the PyTorch-inspired method for tensor-to-numpy conversion.
        Provides clean interface for interoperability with NumPy operations.
        """
        return self._data
    
    def __array__(self, dtype=None) -> np.ndarray:
        """Enable np.array(tensor) and np.allclose(tensor, array)."""
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Enable NumPy universal functions with Tensor objects."""
        # Convert Tensor inputs to NumPy arrays
        args = []
        for input_ in inputs:
            if isinstance(input_, Tensor):
                args.append(input_._data)
            else:
                args.append(input_)
        
        # Call the ufunc on NumPy arrays
        outputs = getattr(ufunc, method)(*args, **kwargs)
        
        # If method returns NotImplemented, let NumPy handle it
        if outputs is NotImplemented:
            return NotImplemented
            
        # Wrap result back in Tensor if appropriate
        if method == '__call__':
            if isinstance(outputs, np.ndarray):
                return Tensor(outputs)
            elif isinstance(outputs, tuple):
                return tuple(Tensor(output) if isinstance(output, np.ndarray) else output 
                           for output in outputs)
        
        return outputs




# %% [markdown]
"""
## Testing Your Tensor Implementation

Let's validate each component immediately to ensure everything works correctly:
"""


# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation

Let's test your tensor creation implementation right away! This gives you immediate feedback on whether your `__init__` method works correctly.
"""

# In[ ]:

def test_unit_tensor_creation():
    """Test tensor creation with all data types and shapes."""
    print("🔬 Unit Test: Tensor Creation...")
    
    try:
        # Test scalar
        scalar = Tensor(5.0)
        assert hasattr(scalar, '_data'), "Tensor should have _data attribute"
        assert scalar._data.shape == (), f"Scalar should have shape (), got {scalar._data.shape}"
        print("✅ Scalar creation works")

        # Test vector
        vector = Tensor([1, 2, 3])
        assert vector._data.shape == (3,), f"Vector should have shape (3,), got {vector._data.shape}"
        print("✅ Vector creation works")

        # Test matrix
        matrix = Tensor([[1, 2], [3, 4]])
        assert matrix._data.shape == (2, 2), f"Matrix should have shape (2, 2), got {matrix._data.shape}"
        print("✅ Matrix creation works")

        print("📈 Progress: Tensor Creation ✓")

    except Exception as e:
        print(f"❌ Tensor creation test failed: {e}")
        raise

    print("🎯 Tensor creation behavior:")
    print("   Converts data to NumPy arrays")
    print("   Preserves shape and data type")
    print("   Stores in _data attribute")

test_unit_tensor_creation()


# %% [markdown]
"""
### 🧪 Unit Test: Tensor Properties

Now let's test that your tensor properties work correctly. This tests the @property methods you implemented.
"""

# In[ ]:

def test_unit_tensor_properties():
    """Test tensor properties (shape, size, dtype, data access)."""
    print("🔬 Unit Test: Tensor Properties...")
    
    try:
        # Test with a simple matrix
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        # Test shape property
        assert tensor.shape == (2, 3), f"Shape should be (2, 3), got {tensor.shape}"
        print("✅ Shape property works")

        # Test size property
        assert tensor.size == 6, f"Size should be 6, got {tensor.size}"
        print("✅ Size property works")

        # Test data property
        assert np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]])), "Data property should return numpy array"
        print("✅ Data property works")

        # Test dtype property
        assert tensor.dtype in [np.int32, np.int64], f"Dtype should be int32 or int64, got {tensor.dtype}"
        print("✅ Dtype property works")

        print("📈 Progress: Tensor Properties ✓")

    except Exception as e:
        print(f"❌ Tensor properties test failed: {e}")
        raise

    print("🎯 Tensor properties behavior:")
    print("   shape: Returns tuple of dimensions")
    print("   size: Returns total number of elements")
    print("   data: Returns underlying NumPy array")
    print("   dtype: Returns NumPy data type")

test_unit_tensor_properties()


# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic

Let's test your tensor arithmetic operations. This tests the __add__, __mul__, __sub__, __truediv__ methods.
"""

# In[ ]:

def test_unit_tensor_arithmetic():
    """Test tensor arithmetic operations."""
    print("🔬 Unit Test: Tensor Arithmetic...")
    
    try:
        # Test addition
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        expected = np.array([5, 7, 9])
        assert np.array_equal(result.data, expected), f"Addition failed: expected {expected}, got {result.data}"
        print("✅ Addition works")

        # Test scalar addition
        result_scalar = a + 10
        expected_scalar = np.array([11, 12, 13])
        assert np.array_equal(result_scalar.data, expected_scalar), f"Scalar addition failed: expected {expected_scalar}, got {result_scalar.data}"
        print("✅ Scalar addition works")

        # Test multiplication
        result_mul = a * b
        expected_mul = np.array([4, 10, 18])
        assert np.array_equal(result_mul.data, expected_mul), f"Multiplication failed: expected {expected_mul}, got {result_mul.data}"
        print("✅ Multiplication works")

        # Test scalar multiplication
        result_scalar_mul = a * 2
        expected_scalar_mul = np.array([2, 4, 6])
        assert np.array_equal(result_scalar_mul.data, expected_scalar_mul), f"Scalar multiplication failed: expected {expected_scalar_mul}, got {result_scalar_mul.data}"
        print("✅ Scalar multiplication works")

        # Test subtraction
        result_sub = b - a
        expected_sub = np.array([3, 3, 3])
        assert np.array_equal(result_sub.data, expected_sub), f"Subtraction failed: expected {expected_sub}, got {result_sub.data}"
        print("✅ Subtraction works")

        # Test division
        result_div = b / a
        expected_div = np.array([4.0, 2.5, 2.0])
        assert np.allclose(result_div.data, expected_div), f"Division failed: expected {expected_div}, got {result_div.data}"
        print("✅ Division works")

        print("📈 Progress: Tensor Arithmetic ✓")

    except Exception as e:
        print(f"❌ Tensor arithmetic test failed: {e}")
        raise

    print("🎯 Tensor arithmetic behavior:")
    print("   Element-wise operations on tensors")
    print("   Broadcasting with scalars")
    print("   Returns new Tensor objects")
    print("   Preserves numerical precision")

test_unit_tensor_arithmetic()

# %% [markdown]
"""
### 🧪 Unit Test: Matrix Multiplication

Test the matrix multiplication implementation that shows both educational and optimized approaches.
"""

# In[ ]:

def test_unit_matrix_multiplication():
    """Test matrix multiplication with educational and optimized paths."""
    print("🔬 Unit Test: Matrix Multiplication...")
    
    try:
        # Small matrix (educational path)
        small_a = Tensor([[1, 2], [3, 4]])
        small_b = Tensor([[5, 6], [7, 8]])
        small_result = small_a @ small_b
        small_expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(small_result.data, small_expected), f"Small matmul failed: expected {small_expected}, got {small_result.data}"
        print("✅ Small matrix multiplication (educational) works")

        # Large matrix (optimized path) 
        large_a = Tensor(np.random.randn(100, 50))
        large_b = Tensor(np.random.randn(50, 80))
        large_result = large_a @ large_b
        assert large_result.shape == (100, 80), f"Large matmul shape wrong: expected (100, 80), got {large_result.shape}"
        
        # Verify with NumPy
        expected_large = np.dot(large_a.data, large_b.data)
        assert np.allclose(large_result.data, expected_large), "Large matmul results don't match NumPy"
        print("✅ Large matrix multiplication (optimized) works")

        print("📈 Progress: Matrix Multiplication ✓")

    except Exception as e:
        print(f"❌ Matrix multiplication test failed: {e}")
        raise

    print("🎯 Matrix multiplication behavior:")
    print("   Small matrices: Educational loops show concept")
    print("   Large matrices: Optimized NumPy implementation")
    print("   Proper shape validation and error handling")
    print("   Foundation for neural network linear layers")

test_unit_matrix_multiplication()

# %% [markdown]
"""
### 🧪 Unit Test: Advanced Tensor Operations

Test the new view/copy semantics and memory layout functionality.
"""

# In[ ]:

def test_unit_advanced_tensor_operations():
    """Test advanced tensor operations: view, clone, contiguous, strides."""
    print("🔬 Unit Test: Advanced Tensor Operations...")
    
    try:
        # Test dtype handling improvements
        tensor_str = Tensor([1, 2, 3], dtype="float32")
        tensor_np = Tensor([1, 2, 3], dtype=np.float64)
        assert tensor_str.dtype == np.float32, f"String dtype failed: {tensor_str.dtype}"
        assert tensor_np.dtype == np.float64, f"NumPy dtype failed: {tensor_np.dtype}"
        print("✅ Enhanced dtype handling works")

        # Test stride and contiguity properties
        matrix = Tensor([[1, 2, 3], [4, 5, 6]])
        assert hasattr(matrix, 'strides'), "Should have strides property"
        assert hasattr(matrix, 'is_contiguous'), "Should have is_contiguous property"
        assert matrix.is_contiguous == True, "New tensor should be contiguous"
        print("✅ Stride and contiguity properties work")

        # Test view vs clone semantics
        original = Tensor([[1, 2], [3, 4]])
        view_tensor = original.view(4)  # Should share data
        clone_tensor = original.clone()  # Should copy data
        
        assert view_tensor.shape == (4,), f"View shape wrong: {view_tensor.shape}"
        assert clone_tensor.shape == (2, 2), f"Clone shape wrong: {clone_tensor.shape}"
        print("✅ View and clone semantics work")

        # Test contiguous operation
        non_contiguous = Tensor(np.ones((10, 10)).T)  # Transpose creates non-contiguous
        contiguous_result = non_contiguous.contiguous()
        
        if not non_contiguous.is_contiguous:  # Only test if actually non-contiguous
            assert contiguous_result.is_contiguous == True, "contiguous() should make data contiguous"
        print("✅ Contiguous operation works")

        # Test error handling for invalid dtype
        try:
            Tensor([1, 2, 3], dtype=123)  # Invalid dtype
            print("❌ Should have failed with invalid dtype")
        except TypeError:
            print("✅ Proper error handling for invalid dtype")

        print("📈 Progress: Advanced Tensor Operations ✓")

    except Exception as e:
        print(f"❌ Advanced tensor operations test failed: {e}")
        raise

    print("🎯 Advanced tensor operations behavior:")
    print("   Enhanced dtype handling (str and np.dtype)")
    print("   Memory layout analysis with strides")
    print("   View vs copy semantics for memory efficiency")
    print("   Contiguous memory optimization")

test_unit_advanced_tensor_operations()

# %% [markdown]
"""
### 🧪 Integration Test: Tensor-NumPy Integration

This integration test validates that your tensor system works seamlessly with NumPy, the foundation of the scientific Python ecosystem.
"""

# In[ ]:

def test_module_tensor_numpy_integration():
    """
    Integration test for tensor operations with NumPy arrays.

    Tests that tensors properly integrate with NumPy operations and maintain
    compatibility with the scientific Python ecosystem.
    """
    print("🔬 Integration Test: Tensor-NumPy Integration...")

    try:
        # Test 1: Tensor from NumPy array
        numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
        tensor_from_numpy = Tensor(numpy_array)

        assert tensor_from_numpy.shape == (2, 3), "Tensor should preserve NumPy array shape"
        assert np.array_equal(tensor_from_numpy.data, numpy_array), "Tensor should preserve NumPy array data"
        print("✅ Tensor from NumPy array works")

        # Test 2: Tensor arithmetic with NumPy-compatible operations
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        # Test operations that would be used in neural networks
        dot_product_result = np.dot(a.data, b.data)  # Common in layers
        assert np.isclose(dot_product_result, 32.0), "Dot product should work with tensor data"
        print("✅ NumPy operations on tensor data work")

        # Test 3: Broadcasting compatibility
        matrix = Tensor([[1, 2], [3, 4]])
        scalar = Tensor(10)

        result = matrix + scalar
        expected = np.array([[11, 12], [13, 14]])
        assert np.array_equal(result.data, expected), "Broadcasting should work like NumPy"
        print("✅ Broadcasting compatibility works")

        # Test 4: Integration with scientific computing patterns
        data = Tensor([1, 4, 9, 16, 25])
        sqrt_result = Tensor(np.sqrt(data.data))  # Using NumPy functions on tensor data
        expected_sqrt = np.array([1., 2., 3., 4., 5.])
        assert np.allclose(sqrt_result.data, expected_sqrt), "Should integrate with NumPy functions"
        print("✅ Scientific computing integration works")

        print("📈 Progress: Tensor-NumPy Integration ✓")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise

    print("🎯 Integration test validates:")
    print("   Seamless NumPy array conversion")
    print("   Compatible arithmetic operations")
    print("   Proper broadcasting behavior")
    print("   Scientific computing workflow integration")

test_module_tensor_numpy_integration()

# %% [markdown]
"""
## Parameter Helper Function

Now that we have Tensor with gradient support, let's add a convenient helper function for creating trainable parameters:
"""

# In[ ]:

#| export
def Parameter(data, dtype=None):
    """
    Convenience function for creating trainable tensors.

    This is equivalent to Tensor(data, requires_grad=True) but provides
    cleaner syntax for neural network parameters.

    Args:
        data: Input data (scalar, list, or numpy array)
        dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.

    Returns:
        Tensor with requires_grad=True

    Examples:
        weight = Parameter(np.random.randn(784, 128))  # Neural network weight
        bias = Parameter(np.zeros(128))                # Neural network bias
    """
    return Tensor(data, dtype=dtype, requires_grad=True)

# %% [markdown]
"""
## Comprehensive Testing Function

Let's create a comprehensive test that runs all our unit tests together:
"""

# In[ ]:

def test_unit_all():
    """Run complete tensor module validation."""
    print("🧪 Running all unit tests...")
    
    # Call every individual test function
    test_unit_tensor_creation()
    test_unit_tensor_properties() 
    test_unit_tensor_arithmetic()
    test_unit_matrix_multiplication()
    test_unit_advanced_tensor_operations()
    test_module_tensor_numpy_integration()
    
    print("✅ All tests passed! Tensor module ready for integration.")

# %% [markdown]
"""
## Main Execution Block
"""

if __name__ == "__main__":
    # Run all tensor tests
    test_unit_all()
    
    print("\n🎉 Tensor module implementation complete!")
    print("📦 Ready to export to tinytorch.core.tensor")
    
    # Demonstrate the new ML Framework Advisor improvements
    print("\n🚀 New Features Demonstration:")
    
    # 1. Enhanced dtype handling
    t1 = Tensor([1, 2, 3], dtype="float32")
    t2 = Tensor([1, 2, 3], dtype=np.float64)
    t3 = Tensor([1, 2, 3], dtype=np.int32)
    print(f"✅ Enhanced dtype support: str={t1.dtype}, np.dtype={t2.dtype}, np.type={t3.dtype}")
    
    # 2. Memory layout analysis
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"✅ Memory analysis: strides={matrix.strides}, contiguous={matrix.is_contiguous}")
    
    # 3. View/copy semantics
    view = matrix.view(6)
    clone = matrix.clone()
    print(f"✅ View/copy semantics: view_shape={view.shape}, clone_shape={clone.shape}")
    
    # 4. Broadcasting failure demonstration with clear error messages
    try:
        bad_a = Tensor([[1, 2], [3, 4]])  # (2, 2)
        bad_b = Tensor([1, 2, 3])         # (3,)
        result = bad_a + bad_b
    except ValueError as e:
        print(f"✅ Clear broadcasting error: {str(e)[:50]}...")
    
    print("\n🎯 Core tensor implementation complete!")
    print("   ✓ Simple, clear tensor creation and operations")
    print("   ✓ Memory layout analysis and performance insights")
    print("   ✓ Broadcasting with comprehensive error handling")
    print("   ✓ View/copy semantics for memory efficiency")


# %% [markdown]
"""
## 🤔 ML Systems Thinking

Now that you've built a complete tensor system, let's connect your implementation to real ML challenges:
"""

# %% [markdown]
"""
### Question 1: Memory Efficiency at Scale

**Challenge**: Your Tensor class showed that contiguous memory is 10-100x faster than scattered memory. Consider a language model with 7 billion parameters (28GB at float32). How would you modify your memory layout strategies to handle training with limited GPU memory (16GB)?

Calculate the memory requirements for parameters, gradients, and optimizer states, then propose specific optimizations to your Tensor implementation.
"""

# In[ ]:

"""
YOUR ANALYSIS:

[Write your response here - consider memory layout, cache efficiency,
and optimization strategies for large-scale tensor operations]
"""

# %% [markdown]
"""
### Question 2: Production Broadcasting

**Challenge**: Your broadcasting implementation handles basic cases. In transformer models, you need operations like:
- Query (32, 512, 768) × Key (32, 512, 768) → Attention (32, 512, 512)
- Attention (32, 8, 512, 512) + Bias (1, 1, 512, 512)

How would you extend your `__add__` and `__mul__` methods to handle these complex shapes while providing clear error messages when shapes are incompatible?
"""

# In[ ]:

"""
YOUR ANALYSIS:

[Write your response here - consider broadcasting rules, error handling,
and complex shape operations in transformer architectures]
"""

# %% [markdown]
"""
### Question 3: Gradient Compatibility

**Challenge**: Your Tensor class includes `requires_grad` and basic gradient tracking. When you implement automatic differentiation (Module 09), how will your current design support gradient computation?

Consider how operations like `c = a * b` need to track both forward computation and backward gradient flow. What modifications would your Tensor methods need to support this?
"""

# In[ ]:

"""
YOUR ANALYSIS:

[Write your response here - consider gradient tracking, computational graphs,
and how your tensor operations will support automatic differentiation]
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tensor Foundation

Congratulations! You've built the fundamental data structure that powers all machine learning!

### Key Learning Outcomes
- **Complete Tensor System**: Built a 400+ line implementation with 15 methods supporting all essential tensor operations
- **Memory Efficiency Mastery**: Discovered that memory layout affects performance more than algorithms (10-100x speedups)
- **Broadcasting Implementation**: Created automatic shape matching that saves memory and enables flexible operations
- **Production-Ready API**: Designed interfaces that mirror PyTorch and TensorFlow patterns

### Ready for Next Steps
Your tensor implementation now enables:
- **Module 03 (Activations)**: Add nonlinear functions that make neural networks powerful
- **Neural network operations**: Matrix multiplication, broadcasting, and gradient preparation
- **Real data processing**: Handle images, text, and complex multi-dimensional datasets

### Export Your Work
1. **Export to package**: `tito module complete 01_tensor`
2. **Verify integration**: Your Tensor class will be available as `tinytorch.core.tensor.Tensor`
3. **Enable next module**: Activations build on your tensor foundation

**Achievement unlocked**: You've built the universal data structure of modern AI! Every neural network, from simple classifiers to ChatGPT, relies on the tensor concepts you've just implemented.
"""
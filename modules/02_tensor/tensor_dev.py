#!/usr/bin/env python
# coding: utf-8

# # Tensor - Making Networks Learn Efficiently

# Welcome to Tensor! You'll implement the fundamental data structure that powers all neural networks.

# ## 🔗 Building on Previous Learning
# **What You Built Before**:
# - Module 01 (Setup): Python environment and NumPy foundations

# **What's Working**: You can create Python environments and import libraries for scientific computing!

# **The Gap**: You have the tools, but you need the fundamental data structure that all ML operations use.

# **This Module's Solution**: Implement tensors - N-dimensional arrays with ML superpowers that form the foundation of every neural network.

# **Connection Map**:
# ```
# Setup → Tensor → Activations
# (tools)   (data)   (intelligence)
# ```

# ## Learning Goals
# - Systems understanding: Memory layout affects cache performance and computational efficiency
# - Core implementation skill: Build complete Tensor class with shape management and arithmetic operations  
# - Pattern/abstraction mastery: Understand how tensors abstract N-dimensional data for ML algorithms
# - Framework connections: See how your implementation mirrors PyTorch's tensor design and memory model
# - Optimization trade-offs: Learn why contiguous memory layout and vectorized operations are critical for ML performance

# ## Build → Use → Reflect
# 1. **Build**: Complete Tensor class with shape management, broadcasting, and vectorized operations
# 2. **Use**: Perform tensor arithmetic and transformations on real multi-dimensional data
# 3. **Reflect**: Why does tensor memory layout become the performance bottleneck in large neural networks?

# ## Systems Reality Check
# 💡 **Production Context**: PyTorch tensors automatically choose optimal memory layouts and can seamlessly move between CPU and GPU
# ⚡ **Performance Insight**: Non-contiguous tensors can be 10-100x slower than contiguous ones - memory layout matters more than algorithm choice

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

# ## 🎯 TinyTorch Module Assumptions
#
# For this module, we assume:
# - **Simple dtype system**: String-based types ("float32", "int32") instead of complex Union types
# - **Educational error handling**: Clear error messages that explain problems, not comprehensive validation
# - **Conceptual memory analysis**: Understanding patterns without detailed profiling complexity
# - **Single-threaded implementation**: Focus on algorithmic clarity without concurrency concerns
#
# These assumptions let us focus on **core tensor concepts** without implementation complexity barriers.

# ## Visual Guide: Understanding Tensors Through Diagrams

# ### What Are Tensors? A Visual Journey
#
# **The Story**: Think of tensors as smart containers that know their shape and can efficiently store numbers for machine learning. They're like upgraded versions of regular Python lists that understand mathematics.
#
# ```
# Scalar (0D Tensor):     Vector (1D Tensor):     Matrix (2D Tensor):
#      [5]                   [1, 2, 3]             ┌ 1  2  3 ┐
#                                                   │ 4  5  6 │
#                                                   └ 7  8  9 ┘
#
# 3D Tensor (RGB Image):                   4D Tensor (Batch of Images):
# ┌─────────────┐                         ┌─────────────┐ ┌─────────────┐
# │ Red Channel │                         │   Image 1   │ │   Image 2   │
# │             │                         │             │ │             │
# └─────────────┘                         └─────────────┘ └─────────────┘
# ┌─────────────┐                                      ...
# │Green Channel│
# │             │
# └─────────────┘
# ┌─────────────┐
# │Blue Channel │
# │             │
# └─────────────┘
# ```
#
# **What's happening step-by-step**: As we add dimensions, tensors represent more complex data. A single number becomes a list, a list becomes a grid, a grid becomes a volume (like an image with red/green/blue channels), and a volume becomes a collection (like a batch of images for training). Each dimension adds a new way to organize and access the data.

# ### Memory Layout: Why Performance Matters
#
# **The Story**: Imagine your computer's memory as a long street with numbered houses. When your CPU needs data, it doesn't just grab one house - it loads an entire city block (64 bytes) into its cache.
#
# ```
# Contiguous Memory (FAST):
# [1][2][3][4][5][6] ──> Cache-friendly, vectorized operations
#  ↑  ↑  ↑  ↑  ↑  ↑
#  Sequential access pattern
#
# Non-contiguous Memory (SLOW):
# [1]...[2].....[3] ──> Cache misses, scattered access
#  ↑     ↑       ↑
#  Random access pattern
# ```
#
# **What's happening step-by-step**: When you access element [1], the CPU automatically loads elements [1] through [6] in one cache load. Every subsequent access ([2], [3], [4]...) is already in the cache - no extra memory trips needed! With non-contiguous data, each access requires a new, expensive trip to main memory.
#
# **The Performance Impact**: This creates 10-100x speedups because you get 6 elements for the price of fetching 1. It's like getting 6 books from the library for the effort of finding just 1.

# ### Tensor Operations: Broadcasting Magic
#
# **The Story**: Broadcasting is like having a smart photocopier that automatically copies data to match different shapes without actually using extra memory. It's NumPy's way of making operations "just work" between tensors of different sizes.
#
# ```
# Broadcasting Example:
#     Matrix (2×3)     +     Scalar        =     Result (2×3)
#   ┌ 1  2  3 ┐             [10]              ┌ 11 12 13 ┐
#   └ 4  5  6 ┘                               └ 14 15 16 ┘
#
# Broadcasting Rules:
# 1. Align shapes from right to left
# 2. Dimensions of size 1 stretch to match
# 3. Missing dimensions assume size 1
#
# Vector + Matrix Broadcasting:
#   [1, 2, 3]    +    [[10],     =    [[11, 12, 13],
#   (1×3)             [20]]            [21, 22, 23]]
#                     (2×1)            (2×3)
# ```
#
# **What's happening step-by-step**: Python aligns shapes from right to left, like comparing numbers by their ones place first. When shapes don't match, dimensions of size 1 automatically "stretch" to match the larger dimension - but no data is actually copied. The operation happens as if the data were copied, but uses the original memory locations.
#
# **Why this matters for ML**: Adding a bias vector to a 1000×1000 matrix would normally require copying the vector 1000 times, but broadcasting does it with zero copies and massive memory savings.

# ### Neural Network Data Flow
# 
# ```
# Batch Processing in Neural Networks:
# 
# Input Batch (32 images, 28×28 pixels):
# ┌─────────────────────────────────┐
# │ [Batch=32, Height=28, Width=28] │
# └─────────────────────────────────┘
#              ↓ Flatten
# ┌─────────────────────────────────┐
# │     [Batch=32, Features=784]    │ ← Matrix multiplication ready
# └─────────────────────────────────┘
#              ↓ Linear Layer
# ┌─────────────────────────────────┐
# │     [Batch=32, Hidden=128]      │ ← Hidden layer activations
# └─────────────────────────────────┘
# 
# Why batching matters:
# - Single image: 784 × 128 = 100,352 operations
# - Batch of 32: Same 100,352 ops, but 32× the data
# - GPU utilization: 32× better parallelization
# ```

# ## The Mathematical Foundation
# 
# Before we implement, let's understand the mathematical concepts:

# ### Scalars to Tensors: Building Complexity
# 
# **Scalar (Rank 0)**:
# - A single number: `5.0` or `temperature`
# - Shape: `()` (empty tuple)
# - ML examples: loss values, learning rates
# 
# **Vector (Rank 1)**:
# - Ordered list of numbers: `[1, 2, 3]`
# - Shape: `(3,)` (one dimension)
# - ML examples: word embeddings, gradients
# 
# **Matrix (Rank 2)**:
# - 2D array: `[[1, 2], [3, 4]]`
# - Shape: `(2, 2)` (rows, columns)
# - ML examples: weight matrices, images
# 
# **Higher-Order Tensors**:
# - 3D: RGB images `(height, width, channels)`
# - 4D: Image batches `(batch, height, width, channels)`
# - 5D: Video batches `(batch, time, height, width, channels)`

# ### Why Not Just Use NumPy?
# 
# While NumPy is excellent, our Tensor class adds ML-specific features:
# 
# **Future Extensions** (coming in later modules):
# - **Automatic gradients**: Track operations for backpropagation
# - **GPU acceleration**: Move computations to graphics cards
# - **Lazy evaluation**: Build computation graphs for optimization
# 
# **Educational Value**:
# - **Understanding**: See how PyTorch/TensorFlow work internally
# - **Debugging**: Trace operations step by step
# - **Customization**: Add domain-specific operations

# ## Implementation Overview
# 
# Our Tensor class design:
# 
# ```python
# class Tensor:
#     def __init__(self, data)      # Create from any data type
#     
#     # Properties
#     .shape                        # Dimensions tuple
#     .size                         # Total element count
#     .dtype                        # Data type
#     .data                         # Access underlying NumPy array
#     
#     # Arithmetic Operations
#     def __add__(self, other)      # tensor + tensor
#     def __mul__(self, other)      # tensor * tensor
#     def __sub__(self, other)      # tensor - tensor
#     def __truediv__(self, other)  # tensor / tensor
#     
#     # Advanced Operations
#     def matmul(self, other)       # Matrix multiplication
#     def sum(self, axis=None)      # Sum along axes
#     def reshape(self, *shape)     # Change shape
# ```

# In[ ]:

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

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication with both educational and efficient implementations.
        
        Shows the learning progression from basic loops to optimized operations.
        This dual approach helps students understand both the concept and production reality.

        TODO: Implement matrix multiplication.

        APPROACH:
        1. Extract numpy arrays from both tensors
        2. Check tensor shapes for compatibility
        3. For small tensors: use educational loops to show concept
        4. For larger tensors: use NumPy's optimized implementation
        5. Create new Tensor object with the result
        6. Return the new tensor

        PRODUCTION CONNECTION:
        - Linear layers: input @ weight matrices in neural networks
        - Transformer attention: Q @ K^T for attention scores
        - CNN convolutions: Implemented as matrix multiplications
        - Batch processing: Matrix ops enable parallel computation
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
        
        # For small tensors (≤ 4x4): Educational loops to show the concept
        if m <= 4 and n <= 4 and k <= 4:
            return self._matmul_educational(other)
        
        # For larger tensors: Use NumPy's optimized implementation (production approach)
        result_data = np.dot(a_data, b_data)
        return Tensor(result_data)
        ### END SOLUTION

    def _matmul_educational(self, other: 'Tensor') -> 'Tensor':
        """
        Educational matrix multiplication using explicit loops.
        
        This shows the fundamental computation clearly for small examples.
        Understanding this helps appreciate why optimized BLAS libraries are essential.
        """
        a_data = self._data
        b_data = other._data
        m, k = a_data.shape
        k2, n = b_data.shape
        
        # Initialize result matrix
        result = np.zeros((m, n), dtype=a_data.dtype)
        
        # Triple nested loops - educational, shows every operation
        # This demonstrates the O(n³) complexity clearly
        for i in range(m):                      # For each row in result
            for j in range(n):                  # For each column in result
                for k_idx in range(k):          # Dot product: sum over inner dimension
                    result[i, j] += a_data[i, k_idx] * b_data[k_idx, j]
        
        return Tensor(result)

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

# ## Computational Assessment Questions

# Now let's build understanding through hands-on calculations that connect to real ML scenarios.

# ### 📊 Memory Calculation Challenge
# 
# **Question 1: How much memory do tensors actually use?**
# 
# Calculate the memory usage for these common ML tensors:
# 
# ```python
# # Image batch for training
# batch_size = 32
# height = 224  
# width = 224
# channels = 3
# 
# # Calculate: batch_size × height × width × channels × bytes_per_float32
# # Answer: _______ MB
# 
# # Large language model embedding
# vocab_size = 50000
# embedding_dim = 768
# 
# # Calculate: vocab_size × embedding_dim × bytes_per_float32  
# # Answer: _______ MB
# ```
# 
# **Real-world context**: A single batch of high-resolution images uses ~600MB RAM. Language model embeddings can use 150MB just for the vocabulary. Understanding memory requirements helps you:
# - Choose appropriate batch sizes for your hardware
# - Estimate training memory requirements
# - Debug out-of-memory errors

# ### 🔢 Broadcasting Calculation Challenge
# 
# **Question 2: Predict the output shapes**
# 
# Given these tensor operations, predict the resulting shapes:
# 
# ```python
# # Operation 1: Matrix + Vector
# matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
# vector = Tensor([10, 20, 30])            # Shape: (3,)
# result1 = matrix + vector                # Shape: ?
# 
# # Operation 2: 3D + 2D Broadcasting  
# tensor_3d = Tensor(np.ones((4, 1, 3)))   # Shape: (4, 1, 3)
# tensor_2d = Tensor(np.ones((2, 3)))      # Shape: (2, 3)
# result2 = tensor_3d + tensor_2d          # Shape: ?
# 
# # Operation 3: Scalar Broadcasting
# big_tensor = Tensor(np.ones((8, 16, 32))) # Shape: (8, 16, 32)
# scalar = Tensor(5.0)                      # Shape: ()
# result3 = big_tensor * scalar             # Shape: ?
# ```
# 
# **Real-world context**: Broadcasting enables efficient operations without copying data. This pattern appears everywhere:
# - Adding bias terms to neural network layers
# - Normalizing data by subtracting means
# - Scaling features in batch normalization

# ### ⚡ Parameter Counting Challenge
# 
# **Question 3: Count learnable parameters**
# 
# For this simple neural network, calculate total parameters:
# 
# ```python
# # Network architecture:
# # Input layer: 784 features (28×28 image flattened)
# # Hidden layer 1: 256 neurons with bias
# # Hidden layer 2: 128 neurons with bias  
# # Output layer: 10 neurons with bias
# 
# # Layer 1: input_size × hidden_size + bias_terms
# layer1_params = 784 * 256 + 256 = ?
# 
# # Layer 2: hidden1_size × hidden2_size + bias_terms
# layer2_params = 256 * 128 + 128 = ?
# 
# # Layer 3: hidden2_size × output_size + bias_terms
# layer3_params = 128 * 10 + 10 = ?
# 
# # Total parameters: ? 
# # Memory at float32: ? MB
# ```
# 
# **Real-world context**: Modern neural networks have millions to billions of parameters. GPT-3 has 175 billion parameters ≈ 700GB of memory. Understanding parameter counts helps you:
# - Estimate model size and memory requirements
# - Compare model complexity across architectures
# - Design models that fit your computational budget

# ## Testing Your Implementation

# Let's test your tensor implementation with immediate feedback after each component.

# ### ✅ IMPLEMENTATION CHECKPOINT: Basic Tensor class complete

# 🤔 PREDICTION: How much faster are numpy arrays vs Python lists?
# Your guess: ___x faster

# 🔍 SYSTEMS INSIGHT #1: Why Numpy Arrays?
def analyze_array_performance():
    """Let's measure why we use numpy arrays!"""
    try:
        import time
        size = 100000
        
        # Python list
        lst = list(range(size))
        start = time.perf_counter()
        _ = [x * 2 for x in lst]
        list_time = time.perf_counter() - start
        
        # Numpy array
        arr = np.arange(size)
        start = time.perf_counter()
        _ = arr * 2
        array_time = time.perf_counter() - start
        
        print(f"Python list: {list_time:.4f}s")
        print(f"Numpy array: {array_time:.4f}s")
        print(f"Speedup: {list_time/array_time:.1f}x faster!")
        
        # Memory analysis
        import sys
        list_memory = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst[:100])
        array_memory = arr.nbytes
        print(f"List memory (100 elements): {list_memory:,} bytes")
        print(f"Array memory (100,000 elements): {array_memory:,} bytes")
        print(f"Memory efficiency: {list_memory/array_memory*1000:.1f}x more efficient per element")
        
        # 💡 WHY THIS MATTERS: Numpy uses contiguous memory for 10-100x speedup.
        # This is why ALL ML frameworks build on numpy/tensor libraries!
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

analyze_array_performance()

# ### 🧪 Unit Test: Tensor Creation

# Let's test your tensor creation implementation right away! This gives you immediate feedback on whether your `__init__` method works correctly.

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

# ### ✅ IMPLEMENTATION CHECKPOINT: Tensor properties complete

# 🤔 PREDICTION: What happens when you access tensor.shape on a 3D array?
# Your answer: _______

# 🔍 SYSTEMS INSIGHT #2: Memory Layout Analysis
def analyze_tensor_memory_layout():
    """Analyze how tensors store data in memory with stride patterns."""
    try:
        # Create different tensor shapes with detailed stride analysis
        shapes_and_names = [
            ((100,), "1D vector"),
            ((10, 10), "2D matrix"),
            ((5, 5, 4), "3D tensor"),
            ((2, 2, 5, 5), "4D tensor (mini-batch)")
        ]
        
        print("📊 Memory Layout Analysis with Strides:")
        print(f"{'Name':20s} | {'Shape':15s} | {'Size':6s} | {'Memory':8s} | {'Strides':20s} | {'Contiguous':10s}")
        print("-" * 90)
        
        for shape, name in shapes_and_names:
            tensor = Tensor(np.ones(shape, dtype=np.float32))
            memory_mb = tensor.data.nbytes / (1024 * 1024)
            
            print(f"{name:20s} | {str(shape):15s} | {tensor.size:6d} | {memory_mb:.3f} MB | {str(tensor.strides):20s} | {tensor.is_contiguous}")
        
        print("\n🔍 Advanced Memory Layout Analysis:")
        
        # Demonstrate contiguous vs non-contiguous with detailed analysis
        original = Tensor(np.ones((1000, 1000), dtype=np.float32))
        transposed = Tensor(original.data.T)  # Transpose creates non-contiguous view
        reshaped = original.reshape(1000000)  # Reshape maintains contiguity
        
        print(f"Original (1000x1000):")
        print(f"  Contiguous: {original.is_contiguous}, Strides: {original.strides}")
        print(f"  Memory: {original.data.nbytes / 1024 / 1024:.1f} MB")
        
        print(f"\nTransposed (1000x1000):")
        print(f"  Contiguous: {transposed.is_contiguous}, Strides: {transposed.strides}")
        print(f"  Memory: {transposed.data.nbytes / 1024 / 1024:.1f} MB (same data, different view)")
        
        print(f"\nReshaped to vector (1000000):")
        print(f"  Contiguous: {reshaped.is_contiguous}, Strides: {reshaped.strides}")
        print(f"  Memory: {reshaped.data.nbytes / 1024 / 1024:.1f} MB")
        
        # Demonstrate stride patterns for different operations
        print("\n📐 Stride Patterns and Performance Implications:")
        matrix = Tensor(np.arange(24).reshape(4, 6).astype(np.float32))
        print(f"Original 4x6 matrix - Strides: {matrix.strides} (row-major)")
        
        # Different ways to access the same data
        col_slice = Tensor(matrix.data[:, ::2])  # Every other column
        row_slice = Tensor(matrix.data[::2, :])  # Every other row
        
        print(f"Column slice [:, ::2] - Strides: {col_slice.strides} (non-contiguous)")
        print(f"Row slice [::2, :] - Strides: {row_slice.strides} (contiguous rows)")
        
        # 💡 WHY THIS MATTERS: Stride patterns reveal memory access efficiency!
        # - Small strides = better cache locality
        # - Non-unit strides = potential performance hits
        # - Understanding strides helps optimize ML operations
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

analyze_tensor_memory_layout()

# ### 🧪 Unit Test: Tensor Properties

# Now let's test that your tensor properties work correctly. This tests the @property methods you implemented.

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

# ### ✅ IMPLEMENTATION CHECKPOINT: Arithmetic operations complete

# 🤔 PREDICTION: How does tensor broadcasting work with different shapes?
# Your example: _______

# 🔍 SYSTEMS INSIGHT #3: Broadcasting Efficiency Analysis
def analyze_broadcasting_efficiency():
    """Measure broadcasting efficiency and demonstrate failure cases."""
    try:
        import time
        
        print("📊 Broadcasting Efficiency Analysis:")
        
        # Create test tensors
        large_matrix = Tensor(np.random.randn(1000, 1000).astype(np.float32))
        bias_vector = Tensor(np.random.randn(1000).astype(np.float32))
        
        # Method 1: Broadcasting (efficient)
        start = time.perf_counter()
        result_broadcast = large_matrix + bias_vector
        broadcast_time = time.perf_counter() - start
        
        # Method 2: Manual expansion (inefficient)
        start = time.perf_counter()
        expanded_bias = Tensor(np.tile(bias_vector.data, (1000, 1)))
        result_manual = large_matrix + expanded_bias
        manual_time = time.perf_counter() - start
        
        print(f"Broadcasting time: {broadcast_time:.4f}s")
        print(f"Manual expansion time: {manual_time:.4f}s")
        print(f"Speedup: {manual_time/broadcast_time:.1f}x faster")
        
        # Memory analysis
        broadcast_memory = large_matrix.data.nbytes + bias_vector.data.nbytes
        manual_memory = large_matrix.data.nbytes + expanded_bias.data.nbytes
        
        print(f"Broadcasting memory: {broadcast_memory / 1024 / 1024:.1f} MB")
        print(f"Manual expansion memory: {manual_memory / 1024 / 1024:.1f} MB")
        print(f"Memory savings: {manual_memory / broadcast_memory:.1f}x less memory")
        
        print("\n🚨 Broadcasting Failure Cases:")
        
        # Demonstrate incompatible shapes
        compatible_cases = [
            ((3, 4), (4,), "Matrix + Vector: (3,4) + (4,) → (3,4)"),
            ((5, 1), (3,), "Column + Vector: (5,1) + (3,) → (5,3)"),
            ((2, 3, 4), (4,), "3D + Vector: (2,3,4) + (4,) → (2,3,4)"),
            ((1,), (5, 3), "Scalar-like + Matrix: (1,) + (5,3) → (5,3)")
        ]
        
        incompatible_cases = [
            ((3, 4), (3,), "Mismatched inner dim: (3,4) + (3,) → ERROR"),
            ((2, 3), (4, 5), "Incompatible shapes: (2,3) + (4,5) → ERROR"),
            ((2, 3, 4), (2, 5), "Different middle dims: (2,3,4) + (2,5) → ERROR")
        ]
        
        print("\n✅ Compatible Broadcasting Examples:")
        for shape1, shape2, description in compatible_cases:
            try:
                a = Tensor(np.ones(shape1))
                b = Tensor(np.ones(shape2))
                result = a + b
                print(f"  {description} ✓")
            except Exception as e:
                print(f"  {description} ❌ (unexpected error: {e})")
        
        print("\n❌ Incompatible Broadcasting Examples:")
        for shape1, shape2, description in incompatible_cases:
            try:
                a = Tensor(np.ones(shape1))
                b = Tensor(np.ones(shape2))
                result = a + b
                print(f"  {description} ❌ (should have failed but didn't!)")
            except ValueError as e:
                print(f"  {description} ✓ (correctly failed)")
            except Exception as e:
                print(f"  {description} ? (unexpected error: {e})")
        
        print("\n📝 Broadcasting Rules Summary:")
        print("  1. Start from the rightmost dimension")
        print("  2. Dimensions are compatible if:")
        print("     - They are equal, OR")
        print("     - One of them is 1, OR")
        print("     - One dimension is missing (treated as 1)")
        print("  3. Output shape: maximum size in each dimension")
        
        # 💡 WHY THIS MATTERS: Understanding broadcasting failures prevents runtime errors!
        # Most ML debugging involves shape mismatches from incorrect broadcasting assumptions.
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

analyze_broadcasting_efficiency()

# ### 🧪 Unit Test: Tensor Arithmetic

# Let's test your tensor arithmetic operations. This tests the __add__, __mul__, __sub__, __truediv__ methods.

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

# ### 🧪 Unit Test: Matrix Multiplication

# Test the matrix multiplication implementation that shows both educational and optimized approaches.

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

# ### 🧪 Unit Test: Advanced Tensor Operations

# Test the new view/copy semantics and memory layout functionality.

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

# ### 🧪 Integration Test: Tensor-NumPy Integration

# This integration test validates that your tensor system works seamlessly with NumPy, the foundation of the scientific Python ecosystem.

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

# ## Parameter Helper Function

# Now that we have Tensor with gradient support, let's add a convenient helper function for creating trainable parameters:

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

# ## Comprehensive Testing Function

# Let's create a comprehensive test that runs all our unit tests together:

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

# ## Main Execution Block

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

# ## 🔬 Advanced: Production Type Handling
#
# **Note**: This section demonstrates how production frameworks handle complex dtype requirements.
# The core implementation above teaches the essential concepts. This section shows the full complexity.
#
# **When to use**: Only when building production libraries or debugging framework internals.

# In[ ]:

def advanced_tensor_creation_demo():
    """
    Demonstrate complex dtype handling like production frameworks.

    Production frameworks like PyTorch accept:
    - String dtypes: 'float32', 'int64', 'bool'
    - NumPy dtypes: np.float32, np.int64
    - NumPy types: np.float64, np.int32
    - Type objects: type(np.float32)

    This complexity exists for API compatibility, not educational clarity.
    """
    print("🔬 Advanced: Production-Level Dtype Handling")
    print("=" * 50)

    # Simulate complex dtype handling that was removed from core implementation
    def create_tensor_with_complex_dtypes(data, dtype=None):
        """Show how production systems handle Union[str, np.dtype, type] inputs."""
        # Convert input to numpy array
        arr = np.array(data)

        if dtype is not None:
            # Complex type handling (removed from core for clarity)
            if isinstance(dtype, str):
                target_dtype = np.dtype(dtype)
                print(f"   String dtype '{dtype}' → {target_dtype}")
            elif isinstance(dtype, np.dtype):
                target_dtype = dtype
                print(f"   NumPy dtype {dtype} → {target_dtype}")
            elif isinstance(dtype, type) and issubclass(dtype, np.generic):
                target_dtype = np.dtype(dtype)
                print(f"   NumPy type {dtype} → {target_dtype}")
            else:
                raise TypeError(f"Unsupported dtype: {type(dtype)}")

            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)

        return arr

    # Demonstrate various input types
    data = [1.0, 2.0, 3.0]

    print("\n📊 String dtype input:")
    result1 = create_tensor_with_complex_dtypes(data, 'float32')
    print(f"   Result: {result1.dtype}")

    print("\n📊 NumPy dtype input:")
    result2 = create_tensor_with_complex_dtypes(data, np.int32)
    print(f"   Result: {result2.dtype}")

    print("\n📊 NumPy type input:")
    result3 = create_tensor_with_complex_dtypes(data, type(np.float64()))
    print(f"   Result: {result3.dtype}")

    print("\n💡 Production Reality:")
    print("   - PyTorch handles 47+ dtype variations")
    print("   - This complexity exists for API compatibility")
    print("   - Educational implementations can use simpler string-only dtypes")
    print("   - Core concepts (data + metadata) remain the same")

if __name__ == "__main__":
    # Only run advanced demo if explicitly executed
    try:
        advanced_tensor_creation_demo()
    except Exception as e:
        print(f"Advanced demo skipped: {e}")

# ## 🤔 ML Systems Thinking: Interactive Questions

# Now that you've built a working tensor system, let's connect this foundational work to broader ML systems challenges. These questions help you think critically about how tensor operations scale to production ML environments.

# ### Question 1: Memory Layout and Performance Optimization

# **Context**: Your tensor implementation shows significant performance differences between contiguous and non-contiguous memory layouts. When you analyzed stride patterns, you discovered that memory layout affects performance more than algorithm choice.

# **Reflection Question**: In your tensor's stride analysis, you saw that transposed matrices have different stride patterns that affect cache efficiency. For a neural network with 50 million parameters processing batches of 1000 images, how would memory layout impact training performance? Design specific modifications to your Tensor class that optimize memory access patterns for large-scale training while maintaining compatibility with your current arithmetic operations.

# Think about: cache line utilization, memory bandwidth limitations, stride patterns for different tensor operations, and the trade-offs between memory copying vs. view operations in computational graphs.

# In[ ]:

"""
YOUR REFLECTION ON MEMORY LAYOUT AND CACHE EFFICIENCY:

TODO: Replace this text with your thoughtful response about memory-efficient tensor system design.

Consider addressing:
- How would you optimize memory layout for large batch processing?
- What strategies would you use to minimize cache misses during tensor operations?
- How would you handle the trade-off between memory copying and in-place operations?
- What role does contiguous memory layout play in computational efficiency?
- How would different storage patterns (row-major vs column-major) affect performance?

Write a practical design connecting your tensor implementation to real memory optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of memory layout impact on performance (3 points)
- Addresses cache efficiency and locality concerns appropriately (3 points)
- Shows practical knowledge of memory optimization strategies (2 points)
- Demonstrates systems thinking about large-scale tensor operations (2 points)
- Clear technical reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of memory optimization
# Students should demonstrate understanding of cache efficiency and memory layout optimization
### END SOLUTION

# ### Question 2: Broadcasting and Shape Compatibility Systems

# **Context**: Your broadcasting analysis revealed both successful operations and failure cases. You implemented automatic shape matching that works for compatible dimensions but fails gracefully for incompatible ones.

# **Reflection Question**: Your tensor broadcasting currently handles simple cases but fails on incompatible shapes. For a large language model where tensors have shapes like (batch=32, sequence=512, features=768) interacting with attention weights of shape (heads=12, 768, 64), how would you extend your broadcasting system to handle more complex multi-dimensional operations? Design enhancements to your add() and multiply() methods that provide better error messages and support advanced broadcasting patterns while maintaining computational efficiency.

# Think about: multi-dimensional broadcasting rules, error message clarity, performance optimization for large tensors, and how to handle edge cases in transformer architectures.

# In[ ]:

"""
YOUR REFLECTION ON HARDWARE ABSTRACTION AND MULTI-PLATFORM DEPLOYMENT:

TODO: Replace this text with your thoughtful response about hardware abstraction design.

Consider addressing:
- How would you design an abstraction layer that works across CPU, GPU, and AI accelerators?
- What strategies would you use for automatic device placement and memory management?
- How would you handle different precision requirements across hardware platforms?
- What role would kernel selection and optimization play in your design?
- How would you minimize memory transfer costs between different compute devices?

Write an architectural analysis connecting your tensor foundation to real hardware deployment challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of multi-platform hardware challenges (3 points)
- Designs practical abstraction layer for device management (3 points)
- Addresses precision and optimization considerations (2 points)
- Demonstrates systems thinking about hardware-software interfaces (2 points)
- Clear architectural reasoning with practical insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of hardware abstraction challenges
# Students should demonstrate knowledge of multi-platform deployment and device optimization
### END SOLUTION

# ### Question 3: View and Copy Semantics for Memory Efficiency

# **Context**: Your tensor implementation now includes view(), clone(), and contiguous() methods that manage memory layout and data sharing. You can create views that share data or copies that guarantee independence.

# **Reflection Question**: Your tensor's reshape() and view() operations currently create views when possible but copies when necessary. For a distributed training system where the same large model weights need to be shared across 8 GPU processes while maintaining independent gradient computation, how would you design a memory management system that optimizes data sharing while ensuring gradient isolation? Extend your Tensor class design to handle shared memory scenarios while preserving the safety of your current copy/view semantics.

# Think about: shared memory management, gradient isolation in distributed settings, copy-on-write strategies, and the trade-offs between memory efficiency and computational safety in multi-process training.

# In[ ]:

"""
YOUR REFLECTION ON COMPUTATIONAL GRAPH INTEGRATION:

TODO: Replace this text with your thoughtful response about computational graph design.

Consider addressing:
- How would you modify your tensor class to support computational graph construction?
- What strategies would you use to balance eager execution with graph-based optimization?
- How would you handle gradient flow and automatic differentiation in your design?
- What memory management challenges arise with large computational graphs?
- How would you support both debugging-friendly and production-optimized execution modes?

Write a design analysis connecting your tensor operations to automatic differentiation and training systems.

GRADING RUBRIC (Instructor Use):
- Understands computational graph concepts and gradient tracking (3 points)
- Designs practical approach to eager vs graph execution modes (3 points)
- Addresses memory management and performance considerations (2 points)
- Shows systems thinking about training vs inference requirements (2 points)
- Clear design reasoning with automatic differentiation insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of computational graphs and automatic differentiation
# Students should demonstrate knowledge of how tensor operations enable gradient computation
### END SOLUTION

# ## 🎯 MODULE SUMMARY: Tensor Foundation

# Congratulations! You've successfully implemented the fundamental data structure that powers all machine learning:

# ### What You've Accomplished
# ✅ **Tensor Class Implementation**: Complete N-dimensional array wrapper with 15+ methods and properties
# ✅ **Core Operations Mastery**: Creation, arithmetic, matrix multiplication, and NumPy integration  
# ✅ **Memory Layout Understanding**: Discovered why contiguous arrays are 10-100x faster than scattered memory
# ✅ **Broadcasting Implementation**: Built efficient operations that handle different tensor shapes automatically
# ✅ **Systems Performance Analysis**: Measured and understood why NumPy arrays outperform Python lists by 50-100x

# ### Key Learning Outcomes
# - **Tensor Fundamentals**: Understanding how N-dimensional arrays work as the foundation of machine learning
# - **Memory Performance**: Discovered that memory layout affects performance more than algorithm choice
# - **Broadcasting Mechanics**: Implemented automatic shape matching that saves both memory and computation
# - **API Design Patterns**: Built clean, intuitive interfaces that mirror production ML frameworks
# - **NumPy Integration**: Created seamless compatibility with the scientific Python ecosystem

# ### Mathematical Foundations Mastered
# - **N-dimensional Arrays**: Shape, size, and dimensionality concepts from scalars to higher-order tensors
# - **Element-wise Operations**: Addition, subtraction, multiplication, division with broadcasting
# - **Matrix Multiplication**: Both educational (O(n³) loops) and optimized (BLAS) implementations
# - **Memory Complexity**: Understanding space requirements and cache efficiency patterns

# ### Professional Skills Developed
# - **Systems Programming**: Building efficient, reusable components with proper error handling
# - **Performance Analysis**: Measuring and optimizing memory usage and computational efficiency
# - **API Design**: Creating intuitive interfaces that hide complexity while enabling power
# - **Integration Testing**: Validating compatibility with external libraries and workflows

# ### Ready for Advanced Applications
# Your tensor implementation now enables:
# - **Neural Network Layers**: Foundation for linear transformations and complex architectures
# - **Automatic Differentiation**: Gradient computation through computational graphs (Module 09)
# - **Complex Models**: CNNs, RNNs, Transformers - all built on your tensor foundation
# - **Real-World Training**: Processing actual datasets with efficient batch operations

# ### Connection to Real ML Systems
# Your implementation mirrors production systems:
# - **PyTorch**: `torch.Tensor` provides identical functionality with GPU acceleration
# - **TensorFlow**: `tf.Tensor` implements similar concepts with distributed computing
# - **NumPy**: `numpy.ndarray` serves as the foundation you built upon
# - **Industry Standard**: Every major ML framework uses these exact principles and patterns

# ### Next Steps
# 1. **Export your module**: `tito module complete 02_tensor`
# 2. **Validate integration**: `tito test --module tensor`
# 3. **Explore broadcasting**: Experiment with different tensor shapes and operations
# 4. **Ready for Module 03**: Activation functions - adding the nonlinearity that makes neural networks powerful!

# **Your tensor implementation is the foundation of modern AI!** You've built the universal data structure that represents everything from single numbers to massive neural network parameters. Now let's add the mathematical functions that enable machines to learn complex patterns!
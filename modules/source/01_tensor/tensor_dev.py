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
# Module 1: Tensor - Core Data Structure

Welcome to the Tensor module! This is where TinyTorch really begins. You'll implement the fundamental data structure that powers all ML systems.

## Learning Goals
- Understand tensors as N-dimensional arrays with ML-specific operations
- Implement a complete Tensor class with arithmetic operations
- Handle shape management, data types, and memory layout
- Build the foundation for neural networks and automatic differentiation
- Master the NBGrader workflow with comprehensive testing

## Build → Use → Understand
1. **Build**: Create the Tensor class with core operations
2. **Use**: Perform tensor arithmetic and transformations
3. **Understand**: How tensors form the foundation of ML systems
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.tensor

#| export
import numpy as np
import sys
from typing import Union, List, Tuple, Optional, Any

# %% nbgrader={"grade": false, "grade_id": "tensor-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Tensor Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build tensors!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/01_tensor/tensor_dev.py`  
**Building Side:** Code exports to `tinytorch.core.tensor`

```python
# Final package structure:
from tinytorch.core.tensor import Tensor  # The foundation of everything!
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.layers import Dense, Conv2D
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.Tensor`
- **Consistency:** All tensor operations live together in `core.tensor`
- **Foundation:** Every other module depends on Tensor
"""

# %% [markdown]
"""
## Step 1: What is a Tensor?

### Definition
A **tensor** is an N-dimensional array with ML-specific operations. Think of it as a container that can hold data in multiple dimensions:

- **Scalar** (0D): A single number - `5.0`
- **Vector** (1D): A list of numbers - `[1, 2, 3]`  
- **Matrix** (2D): A 2D array - `[[1, 2], [3, 4]]`
- **Higher dimensions**: 3D, 4D, etc. for images, video, batches

### Why Tensors Matter in ML
Tensors are the foundation of all machine learning because:
- **Neural networks** process tensors (images, text, audio)
- **Batch processing** requires multiple samples at once
- **GPU acceleration** works efficiently with tensors
- **Automatic differentiation** needs structured data

### Real-World Examples
- **Image**: 3D tensor `(height, width, channels)` - `(224, 224, 3)` for RGB images
- **Batch of images**: 4D tensor `(batch_size, height, width, channels)` - `(32, 224, 224, 3)`
- **Text**: 2D tensor `(sequence_length, embedding_dim)` - `(100, 768)` for BERT embeddings
- **Audio**: 2D tensor `(time_steps, features)` - `(16000, 1)` for 1 second of audio

### Why Not Just Use NumPy?
We will use NumPy internally, but our Tensor class adds:
- **ML-specific operations** (later: gradients, GPU support)
- **Consistent API** for neural networks
- **Type safety** and error checking
- **Integration** with the rest of TinyTorch

Let's start building!
"""

# %% [markdown]
"""
## 🧠 The Mathematical Foundation

### Linear Algebra Refresher
Tensors are generalizations of scalars, vectors, and matrices:

```
Scalar (0D): 5
Vector (1D): [1, 2, 3]
Matrix (2D): [[1, 2], [3, 4]]
Tensor (3D): [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

### Why This Matters for Neural Networks
- **Forward Pass**: Matrix multiplication between layers
- **Batch Processing**: Multiple samples processed simultaneously
- **Convolutions**: 3D operations on image data
- **Gradients**: Derivatives computed across all dimensions

### Connection to Real ML Systems
Every major ML framework uses tensors:
- **PyTorch**: `torch.Tensor`
- **TensorFlow**: `tf.Tensor`
- **JAX**: `jax.numpy.ndarray`
- **TinyTorch**: `tinytorch.core.tensor.Tensor` (what we're building!)

### Performance Considerations
- **Memory Layout**: Contiguous arrays for cache efficiency
- **Vectorization**: SIMD operations for speed
- **Broadcasting**: Efficient operations on different shapes
- **Type Consistency**: Avoiding unnecessary conversions
"""

# %% [markdown]
"""
## Step 2: The Tensor Class Foundation

### Core Concept
Our Tensor class wraps NumPy arrays with ML-specific functionality. It needs to:
- Handle different input types (scalars, lists, numpy arrays)
- Provide consistent shape and type information
- Support arithmetic operations
- Maintain compatibility with the rest of TinyTorch

### Design Principles
- **Simplicity**: Easy to create and use
- **Consistency**: Predictable behavior across operations
- **Performance**: Efficient NumPy backend
- **Extensibility**: Ready for future features (gradients, GPU)
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Tensor:
    """
    TinyTorch Tensor: N-dimensional array with ML operations.
    
    The fundamental data structure for all TinyTorch operations.
    Wraps NumPy arrays with ML-specific functionality.
    """
    
    def __init__(self, data: Union[int, float, List, np.ndarray], dtype: Optional[str] = None):
        """
        Create a new tensor from data.
        
        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
            
        TODO: Implement tensor creation with proper type handling.
        
        STEP-BY-STEP:
        1. Check if data is a scalar (int/float) - convert to numpy array
        2. Check if data is a list - convert to numpy array  
        3. Check if data is already a numpy array - use as-is
        4. Apply dtype conversion if specified
        5. Store the result in self._data
        
        EXAMPLE:
        Tensor(5) → stores np.array(5)
        Tensor([1, 2, 3]) → stores np.array([1, 2, 3])
        Tensor(np.array([1, 2, 3])) → stores the array directly
        
        HINTS:
        - Use isinstance() to check data types
        - Use np.array() for conversion
        - Handle dtype parameter for type conversion
        - Store the array in self._data
        """
        ### BEGIN SOLUTION
        # Convert input to numpy array
        if isinstance(data, (int, float, np.number)):
            # Handle Python and NumPy scalars
            if dtype is None:
                # Auto-detect type: int for integers, float32 for floats
                if isinstance(data, int) or (isinstance(data, np.number) and np.issubdtype(type(data), np.integer)):
                    dtype = 'int32'
                else:
                    dtype = 'float32'
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, list):
            # Let NumPy auto-detect type, then convert if needed
            temp_array = np.array(data)
            if dtype is None:
                # Use NumPy's auto-detected type, but prefer float32 for floats
                if temp_array.dtype == np.float64:
                    dtype = 'float32'
                else:
                    dtype = str(temp_array.dtype)
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            # Already a numpy array
            if dtype is None:
                # Keep existing dtype, but prefer float32 for float64
                if data.dtype == np.float64:
                    dtype = 'float32'
                else:
                    dtype = str(data.dtype)
            self._data = data.astype(dtype) if dtype != data.dtype else data.copy()
        else:
            # Try to convert unknown types
            self._data = np.array(data, dtype=dtype)
        ### END SOLUTION
    
    @property
    def data(self) -> np.ndarray:
        """
        Access underlying numpy array.
        
        TODO: Return the stored numpy array.
        
        HINT: Return self._data (the array you stored in __init__)
        """
        ### BEGIN SOLUTION
        return self._data
        ### END SOLUTION
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get tensor shape.
        
        TODO: Return the shape of the stored numpy array.
        
        HINT: Use .shape attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).shape should return (3,)
        """
        ### BEGIN SOLUTION
        return self._data.shape
        ### END SOLUTION
    
    @property
    def size(self) -> int:
        """
        Get total number of elements.
        
        TODO: Return the total number of elements in the tensor.
        
        HINT: Use .size attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).size should return 3
        """
        ### BEGIN SOLUTION
        return self._data.size
        ### END SOLUTION
    
    @property
    def dtype(self) -> np.dtype:
        """
        Get data type as numpy dtype.
        
        TODO: Return the data type of the stored numpy array.
        
        HINT: Use .dtype attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).dtype should return dtype('int32')
        """
        ### BEGIN SOLUTION
        return self._data.dtype
        ### END SOLUTION
    
    def __repr__(self) -> str:
        """
        String representation.
        
        TODO: Create a clear string representation of the tensor.
        
        APPROACH:
        1. Convert the numpy array to a list for readable output
        2. Include the shape and dtype information
        3. Format: "Tensor([data], shape=shape, dtype=dtype)"
        
        EXAMPLE:
        Tensor([1, 2, 3]) → "Tensor([1, 2, 3], shape=(3,), dtype=int32)"
        
        HINTS:
        - Use .tolist() to convert numpy array to list
        - Include shape and dtype information
        - Keep format consistent and readable
        """
        ### BEGIN SOLUTION
        return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"
        ### END SOLUTION
    
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add two tensors element-wise.
        
        TODO: Implement tensor addition.
        
        APPROACH:
        1. Add the numpy arrays using +
        2. Return a new Tensor with the result
        3. Handle broadcasting automatically
        
        EXAMPLE:
        Tensor([1, 2]) + Tensor([3, 4]) → Tensor([4, 6])
        
        HINTS:
        - Use self._data + other._data
        - Return Tensor(result)
        - NumPy handles broadcasting automatically
        """
        ### BEGIN SOLUTION
        result = self._data + other._data
        return Tensor(result)
        ### END SOLUTION

    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply two tensors element-wise.
        
        TODO: Implement tensor multiplication.
        
        APPROACH:
        1. Multiply the numpy arrays using *
        2. Return a new Tensor with the result
        3. Handle broadcasting automatically
        
        EXAMPLE:
        Tensor([1, 2]) * Tensor([3, 4]) → Tensor([3, 8])
        
        HINTS:
        - Use self._data * other._data
        - Return Tensor(result)
        - This is element-wise, not matrix multiplication
        """
        ### BEGIN SOLUTION
        result = self._data * other._data
        return Tensor(result)
        ### END SOLUTION

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Addition operator: tensor + other
        
        TODO: Implement + operator for tensors.
        
        APPROACH:
        1. If other is a Tensor, use tensor addition
        2. If other is a scalar, convert to Tensor first
        3. Return the result
        
        EXAMPLE:
        Tensor([1, 2]) + Tensor([3, 4]) → Tensor([4, 6])
        Tensor([1, 2]) + 5 → Tensor([6, 7])
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
        1. If other is a Tensor, use tensor multiplication
        2. If other is a scalar, convert to Tensor first
        3. Return the result
        
        EXAMPLE:
        Tensor([1, 2]) * Tensor([3, 4]) → Tensor([3, 8])
        Tensor([1, 2]) * 3 → Tensor([3, 6])
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
        1. Convert other to Tensor if needed
        2. Subtract using numpy arrays
        3. Return new Tensor with result
        
        EXAMPLE:
        Tensor([5, 6]) - Tensor([1, 2]) → Tensor([4, 4])
        Tensor([5, 6]) - 1 → Tensor([4, 5])
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
        1. Convert other to Tensor if needed
        2. Divide using numpy arrays
        3. Return new Tensor with result
        
        EXAMPLE:
        Tensor([6, 8]) / Tensor([2, 4]) → Tensor([3, 2])
        Tensor([6, 8]) / 2 → Tensor([3, 4])
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            result = self._data / other._data
        else:
            result = self._data / other
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
## Step 3: Tensor Arithmetic Operations

### Why Arithmetic Matters
Tensor arithmetic is the foundation of all neural network operations:
- **Forward pass**: Matrix multiplications and additions
- **Activation functions**: Element-wise operations
- **Loss computation**: Differences and squares
- **Gradient computation**: Chain rule applications

### Operations We'll Implement
- **Addition**: Element-wise addition of tensors
- **Multiplication**: Element-wise multiplication
- **Python operators**: `+`, `-`, `*`, `/` for natural syntax
- **Broadcasting**: Handle different shapes automatically
"""

# %% [markdown]
"""
## Step 3: Tensor Arithmetic Methods

The arithmetic methods are now part of the Tensor class above. Let's test them!
"""

# %% [markdown]
"""
## Step 4: Python Operator Overloading

### Why Operator Overloading?
Python's magic methods allow us to use natural syntax:
- `a + b` instead of `a.add(b)`
- `a * b` instead of `a.multiply(b)`
- `a - b` for subtraction
- `a / b` for division

This makes tensor operations feel natural and readable.
"""

# %% [markdown]
"""
## Step 4: Operator Overloading

The operator methods (__add__, __mul__, __sub__, __truediv__) are now part of the Tensor class above. This enables natural syntax like `a + b` and `a * b`.
"""

# %% [markdown]
"""
### 🧪 Test Your Tensor Implementation

Once you implement the Tensor class above, run these cells to test your implementation:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-creation", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test tensor creation and properties
print("Testing tensor creation...")

# Test scalar creation
scalar = Tensor(5.0)
assert scalar.shape == (), f"Scalar shape should be (), got {scalar.shape}"
assert scalar.size == 1, f"Scalar size should be 1, got {scalar.size}"
assert scalar.data.item() == 5.0, f"Scalar value should be 5.0, got {scalar.data.item()}"

# Test vector creation
vector = Tensor([1, 2, 3])
assert vector.shape == (3,), f"Vector shape should be (3,), got {vector.shape}"
assert vector.size == 3, f"Vector size should be 3, got {vector.size}"
assert np.array_equal(vector.data, np.array([1, 2, 3])), "Vector data mismatch"

# Test matrix creation
matrix = Tensor([[1, 2], [3, 4]])
assert matrix.shape == (2, 2), f"Matrix shape should be (2, 2), got {matrix.shape}"
assert matrix.size == 4, f"Matrix size should be 4, got {matrix.size}"
assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]])), "Matrix data mismatch"

# Test dtype handling
float_tensor = Tensor([1.0, 2.0, 3.0])
assert float_tensor.dtype == np.float32, f"Float tensor dtype should be float32, got {float_tensor.dtype}"

int_tensor = Tensor([1, 2, 3])
# Note: NumPy may default to int64 on some systems, so we check for integer types
assert int_tensor.dtype in [np.int32, np.int64], f"Int tensor dtype should be int32 or int64, got {int_tensor.dtype}"

print("✅ Tensor creation tests passed!")
print(f"✅ Scalar: {scalar}")
print(f"✅ Vector: {vector}")
print(f"✅ Matrix: {matrix}")

# %% nbgrader={"grade": true, "grade_id": "test-tensor-arithmetic", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test tensor arithmetic operations
print("Testing tensor arithmetic...")

# Test addition
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
expected = np.array([5, 7, 9])
assert np.array_equal(c.data, expected), f"Addition failed: expected {expected}, got {c.data}"

# Test multiplication
d = a * b
expected = np.array([4, 10, 18])
assert np.array_equal(d.data, expected), f"Multiplication failed: expected {expected}, got {d.data}"

# Test subtraction
e = b - a
expected = np.array([3, 3, 3])
assert np.array_equal(e.data, expected), f"Subtraction failed: expected {expected}, got {e.data}"

# Test division
f = b / a
expected = np.array([4.0, 2.5, 2.0])
assert np.allclose(f.data, expected), f"Division failed: expected {expected}, got {f.data}"

# Test scalar operations
g = a + 10
expected = np.array([11, 12, 13])
assert np.array_equal(g.data, expected), f"Scalar addition failed: expected {expected}, got {g.data}"

h = a * 2
expected = np.array([2, 4, 6])
assert np.array_equal(h.data, expected), f"Scalar multiplication failed: expected {expected}, got {h.data}"

print("✅ Tensor arithmetic tests passed!")
print(f"✅ Addition: {a} + {b} = {c}")
print(f"✅ Multiplication: {a} * {b} = {d}")
print(f"✅ Subtraction: {b} - {a} = {e}")
print(f"✅ Division: {b} / {a} = {f}")

# %% nbgrader={"grade": true, "grade_id": "test-tensor-broadcasting", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test tensor broadcasting
print("Testing tensor broadcasting...")

# Test scalar broadcasting
matrix = Tensor([[1, 2], [3, 4]])
scalar = Tensor(10)
result = matrix + scalar
expected = np.array([[11, 12], [13, 14]])
assert np.array_equal(result.data, expected), f"Scalar broadcasting failed: expected {expected}, got {result.data}"

# Test vector broadcasting
vector = Tensor([1, 2])
result = matrix + vector
expected = np.array([[2, 4], [4, 6]])
assert np.array_equal(result.data, expected), f"Vector broadcasting failed: expected {expected}, got {result.data}"

# Test different shapes
a = Tensor([[1], [2], [3]])  # (3, 1)
b = Tensor([10, 20])         # (2,)
result = a + b
expected = np.array([[11, 21], [12, 22], [13, 23]])
assert np.array_equal(result.data, expected), f"Shape broadcasting failed: expected {expected}, got {result.data}"

print("✅ Tensor broadcasting tests passed!")
print(f"✅ Matrix + Scalar: {matrix} + {scalar} = {result}")
print(f"✅ Broadcasting works correctly!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the core Tensor class for TinyTorch:

### What You've Accomplished
✅ **Tensor Creation**: Handle scalars, vectors, matrices, and higher-dimensional arrays  
✅ **Data Types**: Proper dtype handling with auto-detection and conversion  
✅ **Properties**: Shape, size, dtype, and data access  
✅ **Arithmetic**: Addition, multiplication, subtraction, division  
✅ **Operators**: Natural Python syntax with `+`, `-`, `*`, `/`  
✅ **Broadcasting**: Automatic shape compatibility like NumPy  

### Key Concepts You've Learned
- **Tensors** are the fundamental data structure for ML systems
- **NumPy backend** provides efficient computation with ML-friendly API
- **Operator overloading** makes tensor operations feel natural
- **Broadcasting** enables flexible operations between different shapes
- **Type safety** ensures consistent behavior across operations

### Next Steps
1. **Export your code**: `tito package nbdev --export 01_tensor`
2. **Test your implementation**: `tito module test 01_tensor`
3. **Use your tensors**: 
   ```python
   from tinytorch.core.tensor import Tensor
   t = Tensor([1, 2, 3])
   print(t + 5)  # Your tensor in action!
   ```
4. **Move to Module 2**: Start building activation functions!

**Ready for the next challenge?** Let's add the mathematical functions that make neural networks powerful!
""" 
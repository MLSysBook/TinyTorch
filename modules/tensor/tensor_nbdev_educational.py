# %% [markdown]
"""
# Module 1: Tensor - Core Data Structure with NBDev Educational Features

Welcome to the Tensor module! This demonstrates NBDev's powerful built-in educational capabilities.

## Learning Goals
- Understand tensors as N-dimensional arrays with ML-specific operations
- Implement a complete Tensor class with arithmetic operations
- Handle shape management, data types, and memory layout
- **See NBDev's educational directives in action**

## NBDev Educational Features Demonstrated
- `#|hide` - Complete solutions hidden by default
- `#|code-fold: show` - Code visible but collapsible
- `#|filter_stream` - Clean output by filtering warnings
- Cell tags for instructor/student modes

This module builds the core data structure that all other TinyTorch components will use.
"""

# %% 
#| default_exp core.tensor

# %% [markdown]
"""
## Setup and Imports

First, let's set up our imports and check our environment.
"""

# %%
#| export
#| filter_stream FutureWarning DeprecationWarning
import numpy as np
import sys
from typing import Union, List, Tuple, Optional, Any

print("🔥 TinyTorch Tensor Module - NBDev Educational Version")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build tensors with educational directives!")

# %% [markdown]
"""
## Step 1: What is a Tensor?

A **tensor** is an N-dimensional array with ML-specific operations. Think of it as:
- **Scalar** (0D): A single number - `5.0`
- **Vector** (1D): A list of numbers - `[1, 2, 3]`  
- **Matrix** (2D): A 2D array - `[[1, 2], [3, 4]]`
- **Higher dimensions**: 3D, 4D, etc. for images, video, batches

### 🎓 Your Task
Implement the Tensor class initialization. The solution is hidden below - try it yourself first!
"""

# %%
#| export
#| code-fold: show
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
        
        TODO: Implement the initialization logic
        - Convert input to numpy array
        - Handle different input types (scalar, list, ndarray)
        - Set appropriate dtype
        """
        # 🚨 SOLUTION BELOW - Try implementing first!
        pass  # Remove this and implement
    
    @property 
    def data(self) -> np.ndarray:
        """Access underlying numpy array."""
        # TODO: Return the internal numpy array
        pass
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        # TODO: Return the shape of the internal array
        pass
    
    @property  
    def size(self) -> int:
        """Get total number of elements."""
        # TODO: Return total number of elements
        pass
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type as numpy dtype."""
        # TODO: Return the data type
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        # TODO: Return a nice string representation
        pass

# %% [markdown]
"""
### 🔍 Complete Solution (Hidden by Default)

Click the button below to see the complete implementation:
"""

# %%
#| hide
#| exports
class TensorComplete:
    """
    COMPLETE SOLUTION - This is hidden from students by default.
    
    Instructors can see this, students see the stub above.
    """
    
    def __init__(self, data: Union[int, float, List, np.ndarray], dtype: Optional[str] = None):
        """Complete initialization implementation."""
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
                # Keep NumPy's auto-detected type, but prefer common ML types
                if np.issubdtype(temp_array.dtype, np.integer):
                    dtype = 'int32'
                elif np.issubdtype(temp_array.dtype, np.floating):
                    dtype = 'float32'
                else:
                    dtype = temp_array.dtype
            self._data = temp_array.astype(dtype)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(dtype or data.dtype)
        else:
            raise TypeError(f"Cannot create tensor from {type(data)}")
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def size(self) -> int:
        return self._data.size
    
    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype
    
    def __repr__(self) -> str:
        return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"

# Update the main Tensor class with complete implementation (for demo purposes)
Tensor.__init__ = TensorComplete.__init__
Tensor.data = TensorComplete.data  
Tensor.shape = TensorComplete.shape
Tensor.size = TensorComplete.size
Tensor.dtype = TensorComplete.dtype
Tensor.__repr__ = TensorComplete.__repr__

# %% [markdown]
"""
### 🧪 Test Your Implementation

Let's test the Tensor class you just implemented:
"""

# %%
#| filter_stream FutureWarning
print("=== Testing Tensor Creation ===")

# Scalar tensor
scalar = Tensor(5.0)
print(f"Scalar: {scalar}")

# Vector tensor  
vector = Tensor([1, 2, 3])
print(f"Vector: {vector}")

# Matrix tensor
matrix = Tensor([[1, 2], [3, 4]])
print(f"Matrix: {matrix}")

print(f"\nProperties:")
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix size: {matrix.size}")
print(f"Matrix dtype: {matrix.dtype}")

# %% [markdown]
"""
## Step 2: Arithmetic Operations - Progressive Learning

Now let's add arithmetic operations step by step. Each operation will be revealed progressively.

### 🎯 Challenge: Addition Operation
Implement the `__add__` method. Think about:
- How to handle `tensor + tensor`
- How to handle `tensor + scalar`
"""

# %%
#| code-fold: true
def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
    """
    Addition: tensor + other
    
    TODO: Implement addition
    - Handle Tensor + Tensor case
    - Handle Tensor + scalar case
    - Return new Tensor with result
    """
    # 🚨 Try implementing this yourself first!
    if isinstance(other, Tensor):
        return Tensor(self._data + other._data)
    else:  # scalar
        return Tensor(self._data + other)

# Add to Tensor class
Tensor.__add__ = __add__

# %% [markdown]
"""
### 🎯 Your Turn: Complete the Other Operations

Now implement subtraction, multiplication, and division following the same pattern:
"""

# %%
#| hide
#| export
def _add_remaining_arithmetic_ops():
    """Complete arithmetic operations - hidden solution."""
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction: tensor - other"""
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:  # scalar
            return Tensor(self._data - other)
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication: tensor * other"""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:  # scalar
            return Tensor(self._data * other)
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division: tensor / other"""
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:  # scalar
            return Tensor(self._data / other)
    
    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse addition: scalar + tensor"""
        return Tensor(other + self._data)
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication: scalar * tensor"""
        return Tensor(other * self._data)
    
    # Add methods to Tensor class
    Tensor.__sub__ = __sub__
    Tensor.__mul__ = __mul__
    Tensor.__truediv__ = __truediv__
    Tensor.__radd__ = __radd__
    Tensor.__rmul__ = __rmul__

# Apply the arithmetic operations
_add_remaining_arithmetic_ops()

# %% [markdown]
"""
### 🧪 Test Arithmetic Operations

Let's verify our arithmetic operations work correctly:
"""

# %%
print("=== Testing Arithmetic Operations ===")

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")
print()

# Tensor + Tensor
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print()

# Tensor + Scalar
print(f"a + 10 = {a + 10}")
print(f"a * 2 = {a * 2}")
print(f"5 * a = {5 * a}")  # Test reverse operations

# %% [markdown]
"""
## Step 3: Advanced Operations - Instructor Mode

The following section demonstrates more advanced features that instructors might want to show/hide dynamically.

### Utility Methods with Progressive Revelation
"""

# %%
#| code-fold: show
def reshape(self, *shape: int) -> 'Tensor':
    """
    Reshape tensor to new dimensions.
    
    This is visible by default but collapsible for students who want to focus on other parts.
    """
    return Tensor(self._data.reshape(shape))

def transpose(self) -> 'Tensor':
    """Transpose the tensor (swap dimensions)."""
    return Tensor(self._data.T)

# Add basic methods that are always visible
Tensor.reshape = reshape
Tensor.transpose = transpose

# %% [markdown]
"""
### Advanced Reductions - Student Exercise
"""

# %%
#| hide
def sum(self, axis: Optional[int] = None) -> 'Tensor':
    """Sum elements along axis (or all elements if axis=None)."""
    result = self._data.sum(axis=axis)
    return Tensor(result)

def mean(self, axis: Optional[int] = None) -> 'Tensor':
    """Mean of elements along axis (or all elements if axis=None)."""
    result = self._data.mean(axis=axis)
    return Tensor(result)

def max(self, axis: Optional[int] = None) -> 'Tensor':
    """Maximum element along axis (or all elements if axis=None)."""
    result = self._data.max(axis=axis)
    return Tensor(result)

def min(self, axis: Optional[int] = None) -> 'Tensor':
    """Minimum element along axis (or all elements if axis=None)."""
    result = self._data.min(axis=axis)
    return Tensor(result)

def item(self) -> Union[int, float]:
    """Convert single-element tensor to Python scalar."""
    if self.size != 1:
        raise ValueError(f"Cannot convert tensor of size {self.size} to scalar")
    return self._data.item()

def numpy(self) -> np.ndarray:
    """Convert to numpy array."""
    return self._data.copy()

# Add methods to Tensor class
Tensor.sum = sum
Tensor.mean = mean
Tensor.max = max
Tensor.min = min
Tensor.item = item
Tensor.numpy = numpy

# %% [markdown]
"""
### 🧪 Comprehensive Testing

Let's test all our utility methods:
"""

# %%
#| filter_stream RuntimeWarning
print("=== Testing Utility Methods ===")

# Create test tensor
matrix = Tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original matrix: {matrix}")
print(f"Shape: {matrix.shape}")
print()

# Shape manipulation
reshaped = matrix.reshape(3, 2)
print(f"Reshaped to (3,2): {reshaped}")

transposed = matrix.transpose()
print(f"Transposed: {transposed}")
print()

# Reductions
print(f"Sum (all): {matrix.sum()}")
print(f"Sum (axis=0): {matrix.sum(axis=0)}")  # Sum columns
print(f"Sum (axis=1): {matrix.sum(axis=1)}")  # Sum rows
print()

print(f"Mean: {matrix.mean()}")
print(f"Max: {matrix.max()}")
print(f"Min: {matrix.min()}")

# %% [markdown]
"""
## Step 4: Neural Network Demo - Advanced Section

This section shows how to use our tensor for actual ML operations.
Advanced students can see this, beginners might skip it.
"""

# %%
#| code-fold: true
print("=== Mini Neural Network Demo ===")
print("This demonstrates real ML usage of our Tensor class")

# Simulate a simple linear layer: y = W @ x + b
print("\nSimulating: y = W @ x + b (linear layer)")

# Input vector (batch_size=1, features=3)
x = Tensor([[1.0, 2.0, 3.0]])
print(f"Input x: {x}")

# Weight matrix (output_features=2, input_features=3)  
W = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
print(f"Weights W: {W}")

# Bias vector (output_features=2)
b = Tensor([0.1, 0.2])
print(f"Bias b: {b}")

# For demonstration (real matrix multiplication comes later in the course)
result = W * 2.0 + b  # Scale weights and add bias as demo
print(f"Demo result: {result}")
print(f"Mean activation: {result.mean()}")

# %% [markdown]
"""
## 🎉 Congratulations!

You've successfully implemented a complete Tensor class using NBDev's educational features!

### What You've Learned
- ✅ Tensor creation and properties
- ✅ Arithmetic operations with proper broadcasting
- ✅ Utility methods for shape manipulation and reductions
- ✅ **NBDev's powerful educational directives**

### NBDev Features Demonstrated
- `#|hide` - Hidden complete solutions
- `#|code-fold: show/true` - Collapsible code sections  
- `#|filter_stream` - Clean output
- `#|export` - Code that goes to the package
- Progressive revelation of complexity

### Next Steps
```bash
# Export to package
python bin/tito.py sync --module tensor

# Run tests  
python bin/tito.py test --module tensor

# Build documentation with NBDev
nbdev_docs
```

The power of NBDev is that instructors can control exactly what students see and when, while maintaining a single source of truth!
"""

# %%
#| hide_line
print("🎓 This line is hidden in student view but visible to instructors") 
print("🔥 Module complete! Ready for the next challenge.") 
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

"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/tensor/tensor_dev.py`  
**Building Side:** Code exports to `tinytorch.core.tensor`

```python
# Final package structure:
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense, Conv2D
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.tensor`
- **Consistency:** Core data structure lives in `core.tensor`
"""
"""

# %%
#| default_exp core.tensor

# Setup and imports
import numpy as np
import sys
from typing import Union, List, Tuple, Optional, Any

print("🔥 TinyTorch Tensor Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build tensors!")

# %% [markdown]
"""
## Step 1: What is a Tensor?

A **tensor** is an N-dimensional array with ML-specific operations. Think of it as:
- **Scalar** (0D): A single number - `5.0`
- **Vector** (1D): A list of numbers - `[1, 2, 3]`  
- **Matrix** (2D): A 2D array - `[[1, 2], [3, 4]]`
- **Higher dimensions**: 3D, 4D, etc. for images, video, batches

**Why not just use NumPy?** We will use NumPy internally, but our Tensor class will add:
- ML-specific operations (later: gradients, GPU support)
- Consistent API for neural networks
- Type safety and error checking
- Integration with the rest of TinyTorch

Let's start building!
"""

# %%
#| export
class Tensor:
    """
    TinyTorch Tensor: N-dimensional array with ML operations.
    
    The fundamental data structure for all TinyTorch operations.
    Wraps NumPy arrays with ML-specific functionality.
    
    TODO: Implement the core Tensor class with data handling and properties.
    """
    
    def __init__(self, data: Union[int, float, List, np.ndarray], dtype: Optional[str] = None):
        """
        Create a new tensor from data.
        
        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
            
        TODO: Implement tensor creation with proper type handling.
        """
        raise NotImplementedError("Student implementation required")
    
    @property
    def data(self) -> np.ndarray:
        """Access underlying numpy array."""
        raise NotImplementedError("Student implementation required")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        raise NotImplementedError("Student implementation required")
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        raise NotImplementedError("Student implementation required")
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type as numpy dtype."""
        raise NotImplementedError("Student implementation required")
    
    def __repr__(self) -> str:
        """String representation."""
        raise NotImplementedError("Student implementation required")

# %%
#| hide
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
        """
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
        """Access underlying numpy array."""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self._data.shape
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._data.size
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type as numpy dtype."""
        return self._data.dtype
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"

# %% [markdown]
"""
### 🧪 Test Your Tensor Class

Once you implement the Tensor class above, run this cell to test it:
"""

# %%
# Test the basic Tensor class
try:
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
    
except NotImplementedError as e:
    print(f"⚠️  {e}")
    print("Implement the Tensor class above first!")

# %% [markdown]
"""
## Step 2: Arithmetic Operations

Now let's add the core arithmetic operations. These are essential for neural networks:
- **Addition**: `tensor + other` 
- **Subtraction**: `tensor - other`
- **Multiplication**: `tensor * other`
- **Division**: `tensor / other`

Each operation should handle both **tensor + tensor** and **tensor + scalar** cases.
"""

# %%
#| export
def _add_arithmetic_methods():
    """
    Add arithmetic operations to Tensor class.
    
    TODO: Implement arithmetic methods (__add__, __sub__, __mul__, __truediv__)
    and their reverse operations (__radd__, __rsub__, etc.)
    """
    
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition: tensor + other"""
        raise NotImplementedError("Student implementation required")
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction: tensor - other"""
        raise NotImplementedError("Student implementation required")
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication: tensor * other"""
        raise NotImplementedError("Student implementation required")
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division: tensor / other"""
        raise NotImplementedError("Student implementation required")
    
    # Add methods to Tensor class
    Tensor.__add__ = __add__
    Tensor.__sub__ = __sub__
    Tensor.__mul__ = __mul__
    Tensor.__truediv__ = __truediv__

# %%
#| hide  
#| export
def _add_arithmetic_methods():
    """Add arithmetic operations to Tensor class."""
    
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition: tensor + other"""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:  # scalar
            return Tensor(self._data + other)
    
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
    
    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse subtraction: scalar - tensor"""
        return Tensor(other - self._data)
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication: scalar * tensor"""
        return Tensor(other * self._data)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse division: scalar / tensor"""
        return Tensor(other / self._data)
    
    # Add methods to Tensor class
    Tensor.__add__ = __add__
    Tensor.__sub__ = __sub__
    Tensor.__mul__ = __mul__
    Tensor.__truediv__ = __truediv__
    Tensor.__radd__ = __radd__
    Tensor.__rsub__ = __rsub__
    Tensor.__rmul__ = __rmul__
    Tensor.__rtruediv__ = __rtruediv__

# Call the function to add arithmetic methods
_add_arithmetic_methods()

# %% [markdown]
"""
### 🧪 Test Your Arithmetic Operations

Once you implement the arithmetic methods above, run this cell to test them:
"""

# %%
# Test arithmetic operations
try:
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
    print()
    
    # Scalar + Tensor (reverse operations)
    print(f"10 + a = {10 + a}")
    print(f"2 * a = {2 * a}")
    
except (NotImplementedError, AttributeError) as e:
    print(f"⚠️  {e}")
    print("Implement the arithmetic methods above first!")

# %% [markdown]
"""
## Step 3: Try the Export Process

Now let's export our tensor code! In your terminal, run:

```bash
python bin/tito.py sync --module tensor
```

This will export the code marked with `#| export` to `tinytorch/core/tensor.py`.

Then test it with:

```bash
python bin/tito.py test --module tensor
```

## Next Steps

🎉 **Congratulations!** You've built the foundation of TinyTorch - the Tensor class. 

In the next modules, you'll add:
- **Automatic differentiation** (gradients)
- **Neural network layers**
- **Optimizers and training loops**
- **GPU acceleration**

Each builds on this tensor foundation!
""" 
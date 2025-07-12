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

### Visual Intuition
```
Scalar (0D):    5.0
Vector (1D):    [1, 2, 3, 4]
Matrix (2D):    [[1, 2, 3],
                 [4, 5, 6]]
3D Tensor:      [[[1, 2], [3, 4]],
                 [[5, 6], [7, 8]]]
```

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
    
    APPROACH:
    1. Store the input data as a NumPy array internally
    2. Handle different input types (scalars, lists, numpy arrays)
    3. Implement properties to access shape, size, and data type
    4. Create a clear string representation
    
    EXAMPLE:
    Input: Tensor([1, 2, 3])
    Expected: Tensor with shape (3,), size 3, dtype int32
    
    HINTS:
    - Use NumPy's np.array() to convert inputs
    - Handle dtype parameter for type conversion
    - Store the array in a private attribute like self._data
    - Properties should return information about the stored array
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
        """
        raise NotImplementedError("Student implementation required")
    
    @property
    def data(self) -> np.ndarray:
        """
        Access underlying numpy array.
        
        TODO: Return the stored numpy array.
        
        HINT: Return self._data (the array you stored in __init__)
        """
        raise NotImplementedError("Student implementation required")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get tensor shape.
        
        TODO: Return the shape of the stored numpy array.
        
        HINT: Use .shape attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).shape should return (3,)
        """
        raise NotImplementedError("Student implementation required")
    
    @property
    def size(self) -> int:
        """
        Get total number of elements.
        
        TODO: Return the total number of elements in the tensor.
        
        HINT: Use .size attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).size should return 3
        """
        raise NotImplementedError("Student implementation required")
    
    @property
    def dtype(self) -> np.dtype:
        """
        Get data type as numpy dtype.
        
        TODO: Return the data type of the stored numpy array.
        
        HINT: Use .dtype attribute of the numpy array
        EXAMPLE: Tensor([1, 2, 3]).dtype should return dtype('int32')
        """
        raise NotImplementedError("Student implementation required")
    
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
        """
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
    
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add another tensor to this tensor.
        
        TODO: Implement tensor addition as a method.
        
        APPROACH:
        1. Use the add_tensors function you already implemented
        2. Or implement the addition directly using self._data + other._data
        3. Return a new Tensor with the result
        
        EXAMPLE:
        Tensor([1, 2, 3]).add(Tensor([4, 5, 6])) → Tensor([5, 7, 9])
        
        HINTS:
        - You can reuse add_tensors(self, other)
        - Or implement directly: Tensor(self._data + other._data)
        """
        raise NotImplementedError("Student implementation required")
    
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply this tensor by another tensor.
        
        TODO: Implement tensor multiplication as a method.
        
        APPROACH:
        1. Use the multiply_tensors function you already implemented
        2. Or implement the multiplication directly using self._data * other._data
        3. Return a new Tensor with the result
        
        EXAMPLE:
        Tensor([1, 2, 3]).multiply(Tensor([4, 5, 6])) → Tensor([4, 10, 18])
        
        HINTS:
        - You can reuse multiply_tensors(self, other)
        - Or implement directly: Tensor(self._data * other._data)
        """
        raise NotImplementedError("Student implementation required")
    
    # Arithmetic operators for natural syntax (a + b, a * b, etc.)
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition: tensor + other"""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:  # scalar
            return Tensor(self._data + other)
    
    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse addition: scalar + tensor"""
        return Tensor(other + self._data)
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction: tensor - other"""
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:  # scalar
            return Tensor(self._data - other)
    
    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse subtraction: scalar - tensor"""
        return Tensor(other - self._data)
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication: tensor * other"""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:  # scalar
            return Tensor(self._data * other)
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication: scalar * tensor"""
        return Tensor(other * self._data)
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division: tensor / other"""
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:  # scalar
            return Tensor(self._data / other)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse division: scalar / tensor"""
        return Tensor(other / self._data)

# %% [markdown]
"""
### 🧪 Test Your Tensor Class

Once you implement the Tensor class above, run this cell to test it:
"""

# %%
# Test basic tensor creation
print("Testing Tensor creation...")

try:
    # Test scalar
    t1 = Tensor(5)
    print(f"✅ Scalar: {t1} (shape: {t1.shape}, size: {t1.size})")
    
    # Test vector
    t2 = Tensor([1, 2, 3, 4])
    print(f"✅ Vector: {t2} (shape: {t2.shape}, size: {t2.size})")
    
    # Test matrix
    t3 = Tensor([[1, 2], [3, 4]])
    print(f"✅ Matrix: {t3} (shape: {t3.shape}, size: {t3.size})")
    
    # Test numpy array
    t4 = Tensor(np.array([1.0, 2.0, 3.0]))
    print(f"✅ Numpy: {t4} (shape: {t4.shape}, size: {t4.size})")
    
    # Test dtype
    t5 = Tensor([1, 2, 3], dtype='float32')
    print(f"✅ Dtype: {t5} (dtype: {t5.dtype})")
    
    print("\n🎉 All basic tests passed! Your Tensor class is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement all the required methods!")

# %% [markdown]
"""
## Step 2: Tensor Arithmetic Operations

Now let's add the ability to perform mathematical operations on tensors. This is where tensors become powerful for ML!

### Why Arithmetic Matters
- **Neural networks** perform millions of arithmetic operations
- **Gradients** require addition, multiplication, and other operations
- **Batch processing** needs element-wise operations
- **GPU acceleration** works with parallel arithmetic

### Types of Operations
1. **Element-wise**: Add, subtract, multiply, divide
2. **Broadcasting**: Operations between different shapes
3. **Matrix operations**: Matrix multiplication (later)
4. **Reduction**: Sum, mean, max, min (later)

Let's start with the basics!
"""

# %%
#| export
def add_tensors(a: Tensor, b: Tensor) -> Tensor:
    """
    Add two tensors element-wise.
    
    TODO: Implement element-wise addition of two tensors.
    
    APPROACH:
    1. Extract the numpy arrays from both tensors
    2. Use NumPy's + operator for element-wise addition
    3. Return a new Tensor with the result
    
    EXAMPLE:
    add_tensors(Tensor([1, 2, 3]), Tensor([4, 5, 6])) 
    → Tensor([5, 7, 9])
    
    HINTS:
    - Use a.data and b.data to get the numpy arrays
    - NumPy handles broadcasting automatically
    - Return Tensor(result) to wrap the result
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def add_tensors(a: Tensor, b: Tensor) -> Tensor:
    """Add two tensors element-wise."""
    return Tensor(a.data + b.data)

# %%
#| export
def multiply_tensors(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two tensors element-wise.
    
    TODO: Implement element-wise multiplication of two tensors.
    
    APPROACH:
    1. Extract the numpy arrays from both tensors
    2. Use NumPy's * operator for element-wise multiplication
    3. Return a new Tensor with the result
    
    EXAMPLE:
    multiply_tensors(Tensor([1, 2, 3]), Tensor([4, 5, 6])) 
    → Tensor([4, 10, 18])
    
    HINTS:
    - Use a.data and b.data to get the numpy arrays
    - NumPy handles broadcasting automatically
    - Return Tensor(result) to wrap the result
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def multiply_tensors(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two tensors element-wise."""
    return Tensor(a.data * b.data)

# %% [markdown]
"""
### 🧪 Test Your Arithmetic Operations
"""

# %%
# Test arithmetic operations
print("Testing tensor arithmetic...")

try:
    # Test addition
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = add_tensors(a, b)
    print(f"✅ Addition: {a} + {b} = {c}")
    
    # Test multiplication
    d = multiply_tensors(a, b)
    print(f"✅ Multiplication: {a} * {b} = {d}")
    
    # Test broadcasting (scalar + tensor)
    scalar = Tensor(10)
    e = add_tensors(scalar, a)
    print(f"✅ Broadcasting: {scalar} + {a} = {e}")
    
    print("\n🎉 All arithmetic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement add_tensors and multiply_tensors!")

# %% [markdown]
"""
## Step 3: Tensor Methods (Object-Oriented Approach)

Now let's add methods to the Tensor class itself. This makes the API more intuitive and similar to PyTorch.

### Why Methods Matter
- **Cleaner API**: `tensor.add(other)` instead of `add_tensors(tensor, other)`
- **Method chaining**: `tensor.add(other).multiply(scalar)`
- **Consistency**: Similar to PyTorch's tensor methods
- **Object-oriented**: Encapsulates operations with data
"""

# %%
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
    
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add another tensor to this tensor.
        
        TODO: Implement tensor addition as a method.
        
        APPROACH:
        1. Use the add_tensors function you already implemented
        2. Or implement the addition directly using self._data + other._data
        3. Return a new Tensor with the result
        
        EXAMPLE:
        Tensor([1, 2, 3]).add(Tensor([4, 5, 6])) → Tensor([5, 7, 9])
        
        HINTS:
        - You can reuse add_tensors(self, other)
        - Or implement directly: Tensor(self._data + other._data)
        """
        raise NotImplementedError("Student implementation required")
    
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply this tensor by another tensor.
        
        TODO: Implement tensor multiplication as a method.
        
        APPROACH:
        1. Use the multiply_tensors function you already implemented
        2. Or implement the multiplication directly using self._data * other._data
        3. Return a new Tensor with the result
        
        EXAMPLE:
        Tensor([1, 2, 3]).multiply(Tensor([4, 5, 6])) → Tensor([4, 10, 18])
        
        HINTS:
        - You can reuse multiply_tensors(self, other)
        - Or implement directly: Tensor(self._data * other._data)
        """
        raise NotImplementedError("Student implementation required")
    
    # Arithmetic operators for natural syntax (a + b, a * b, etc.)
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition: tensor + other"""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:  # scalar
            return Tensor(self._data + other)
    
    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse addition: scalar + tensor"""
        return Tensor(other + self._data)
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction: tensor - other"""
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:  # scalar
            return Tensor(self._data - other)
    
    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse subtraction: scalar - tensor"""
        return Tensor(other - self._data)
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication: tensor * other"""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:  # scalar
            return Tensor(self._data * other)
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication: scalar * tensor"""
        return Tensor(other * self._data)
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division: tensor / other"""
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:  # scalar
            return Tensor(self._data / other)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse division: scalar / tensor"""
        return Tensor(other / self._data)

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
    
    def add(self, other: 'Tensor') -> 'Tensor':
        """Add another tensor to this tensor."""
        return Tensor(self._data + other._data)
    
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """Multiply this tensor by another tensor."""
        return Tensor(self._data * other._data)
    
    # Arithmetic operators for natural syntax (a + b, a * b, etc.)
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition: tensor + other"""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:  # scalar
            return Tensor(self._data + other)
    
    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse addition: scalar + tensor"""
        return Tensor(other + self._data)
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction: tensor - other"""
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:  # scalar
            return Tensor(self._data - other)
    
    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse subtraction: scalar - tensor"""
        return Tensor(other - self._data)
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication: tensor * other"""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:  # scalar
            return Tensor(self._data * other)
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication: scalar * tensor"""
        return Tensor(other * self._data)
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division: tensor / other"""
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:  # scalar
            return Tensor(self._data / other)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse division: scalar / tensor"""
        return Tensor(other / self._data)

# %% [markdown]
"""
### 🧪 Test Your Tensor Methods
"""

# %%
# Test tensor methods
print("Testing tensor methods...")

try:
    # Test method-based operations
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    c = a.add(b)
    print(f"✅ Method addition: {a}.add({b}) = {c}")
    
    d = a.multiply(b)
    print(f"✅ Method multiplication: {a}.multiply({b}) = {d}")
    
    # Test method chaining
    e = a.add(b).multiply(Tensor(2))
    print(f"✅ Method chaining: {a}.add({b}).multiply(2) = {e}")
    
    print("\n🎉 All method tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the add and multiply methods!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've built the foundation of TinyTorch:

### What You've Accomplished
✅ **Tensor Creation**: Handle scalars, lists, and numpy arrays  
✅ **Properties**: Access shape, size, and data type  
✅ **Arithmetic**: Element-wise addition and multiplication  
✅ **Methods**: Object-oriented API for operations  
✅ **Testing**: Immediate feedback on your implementation  

### Key Concepts You've Learned
- **Tensors** are N-dimensional arrays with ML operations
- **NumPy integration** provides efficient computation
- **Element-wise operations** work on corresponding elements
- **Broadcasting** automatically handles different shapes
- **Object-oriented design** makes APIs intuitive

### What's Next
In the next modules, you'll build on this foundation:
- **Layers**: Transform tensors with weights and biases
- **Activations**: Add nonlinearity to your networks
- **Networks**: Compose layers into complete models
- **Training**: Learn parameters with gradients and optimization

### Real-World Connection
Your Tensor class is now ready to:
- Store neural network weights and biases
- Process batches of data efficiently
- Handle different data types (images, text, audio)
- Integrate with the rest of the TinyTorch ecosystem

**Ready for the next challenge?** Let's move on to building layers that can transform your tensors!
"""

# %%
# Final verification
print("\n" + "="*50)
print("🎉 TENSOR MODULE COMPLETE!")
print("="*50)
print("✅ Tensor creation and properties")
print("✅ Arithmetic operations")
print("✅ Method-based API")
print("✅ Comprehensive testing")
print("\n🚀 Ready to build layers in the next module!") 
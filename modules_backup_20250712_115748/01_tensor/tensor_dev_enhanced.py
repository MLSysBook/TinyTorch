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
# Module 1: Tensor - Enhanced with nbgrader Support

This is an enhanced version of the tensor module that demonstrates dual-purpose content creation:
- **Self-learning**: Rich educational content with guided implementation
- **Auto-grading**: nbgrader-compatible assignments with hidden tests

## Dual System Benefits

1. **Single Source**: One file generates both learning and assignment materials
2. **Consistent Quality**: Same instructor solutions in both contexts
3. **Flexible Assessment**: Choose between self-paced learning or formal grading
4. **Scalable**: Handle large courses with automated feedback

## How It Works

- **TinyTorch markers**: `#| exercise_start/end` for educational content
- **nbgrader markers**: `### BEGIN/END SOLUTION` for auto-grading
- **Hidden tests**: `### BEGIN/END HIDDEN TESTS` for automatic verification
- **Dual generation**: One command creates both student notebooks and assignments
"""

# %%
#| default_exp core.tensor

# %%
#| export
import numpy as np
from typing import Union, List, Tuple, Optional

# %% [markdown]
"""
## Enhanced Tensor Class

This implementation shows how to create dual-purpose educational content:

### For Self-Learning Students
- Rich explanations and step-by-step guidance
- Detailed hints and examples
- Progressive difficulty with scaffolding

### For Formal Assessment
- Auto-graded with hidden tests
- Immediate feedback on correctness
- Partial credit for complex methods
"""

# %%
#| export
class Tensor:
    """
    TinyTorch Tensor: N-dimensional array with ML operations.
    
    This enhanced version demonstrates dual-purpose educational content
    suitable for both self-learning and formal assessment.
    """
    
    def __init__(self, data: Union[int, float, List, np.ndarray], dtype: Optional[str] = None):
        """
        Create a new tensor from data.
        
        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
        """
        #| exercise_start
        #| hint: Use np.array() to convert input data to numpy array
        #| solution_test: tensor.shape should match input shape
        #| difficulty: easy
        
        ### BEGIN SOLUTION
        # Convert input to numpy array
        if isinstance(data, (int, float)):
            self._data = np.array(data)
        elif isinstance(data, list):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            self._data = np.array(data)
        
        # Apply dtype conversion if specified
        if dtype is not None:
            self._data = self._data.astype(dtype)
        ### END SOLUTION
        
        #| exercise_end
        
    @property
    def data(self) -> np.ndarray:
        """Access underlying numpy array."""
        #| exercise_start
        #| hint: Return the stored numpy array (_data attribute)
        #| solution_test: tensor.data should return numpy array
        #| difficulty: easy
        
        ### BEGIN SOLUTION
        return self._data
        ### END SOLUTION
        
        #| exercise_end
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        #| exercise_start
        #| hint: Use the .shape attribute of the numpy array
        #| solution_test: tensor.shape should return tuple of dimensions
        #| difficulty: easy
        
        ### BEGIN SOLUTION
        return self._data.shape
        ### END SOLUTION
        
        #| exercise_end
        
    @property
    def size(self) -> int:
        """Get total number of elements."""
        #| exercise_start
        #| hint: Use the .size attribute of the numpy array
        #| solution_test: tensor.size should return total element count
        #| difficulty: easy
        
        ### BEGIN SOLUTION
        return self._data.size
        ### END SOLUTION
        
        #| exercise_end
        
    @property
    def dtype(self) -> np.dtype:
        """Get data type as numpy dtype."""
        #| exercise_start
        #| hint: Use the .dtype attribute of the numpy array
        #| solution_test: tensor.dtype should return numpy dtype
        #| difficulty: easy
        
        ### BEGIN SOLUTION
        return self._data.dtype
        ### END SOLUTION
        
        #| exercise_end
        
    def __repr__(self) -> str:
        """String representation of the tensor."""
        #| exercise_start
        #| hint: Format as "Tensor([data], shape=shape, dtype=dtype)"
        #| solution_test: repr should include data, shape, and dtype
        #| difficulty: medium
        
        ### BEGIN SOLUTION
        data_str = self._data.tolist()
        return f"Tensor({data_str}, shape={self.shape}, dtype={self.dtype})"
        ### END SOLUTION
        
        #| exercise_end
        
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add two tensors element-wise.
        
        Args:
            other: Another tensor to add
            
        Returns:
            New tensor with element-wise sum
        """
        #| exercise_start
        #| hint: Use numpy's + operator for element-wise addition
        #| solution_test: result should be new Tensor with correct values
        #| difficulty: medium
        
        ### BEGIN SOLUTION
        result_data = self._data + other._data
        return Tensor(result_data)
        ### END SOLUTION
        
        #| exercise_end
        
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply two tensors element-wise.
        
        Args:
            other: Another tensor to multiply
            
        Returns:
            New tensor with element-wise product
        """
        #| exercise_start
        #| hint: Use numpy's * operator for element-wise multiplication
        #| solution_test: result should be new Tensor with correct values
        #| difficulty: medium
        
        ### BEGIN SOLUTION
        result_data = self._data * other._data
        return Tensor(result_data)
        ### END SOLUTION
        
        #| exercise_end
        
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication of two tensors.
        
        Args:
            other: Another tensor for matrix multiplication
            
        Returns:
            New tensor with matrix product
            
        Raises:
            ValueError: If shapes are incompatible for matrix multiplication
        """
        #| exercise_start
        #| hint: Use np.dot() for matrix multiplication, check shapes first
        #| solution_test: result should handle shape validation and matrix multiplication
        #| difficulty: hard
        
        ### BEGIN SOLUTION
        # Check shape compatibility
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Cannot multiply shapes {self.shape} and {other.shape}")
        
        result_data = np.dot(self._data, other._data)
        return Tensor(result_data)
        ### END SOLUTION
        
        #| exercise_end

# %% [markdown]
"""
## Hidden Tests for Auto-Grading

These tests are hidden from students but used for automatic grading.
They provide comprehensive coverage and immediate feedback.
"""

# %%
### BEGIN HIDDEN TESTS
def test_tensor_creation_basic():
    """Test basic tensor creation (2 points)"""
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert t.data.tolist() == [1, 2, 3]
    assert t.size == 3

def test_tensor_creation_scalar():
    """Test scalar tensor creation (2 points)"""
    t = Tensor(5)
    assert t.shape == ()
    assert t.data.item() == 5
    assert t.size == 1

def test_tensor_creation_2d():
    """Test 2D tensor creation (2 points)"""
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.data.tolist() == [[1, 2], [3, 4]]
    assert t.size == 4

def test_tensor_dtype():
    """Test dtype handling (2 points)"""
    t = Tensor([1, 2, 3], dtype='float32')
    assert t.dtype == np.float32
    assert t.data.dtype == np.float32

def test_tensor_properties():
    """Test tensor properties (2 points)"""
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.size == 6
    assert isinstance(t.data, np.ndarray)

def test_tensor_repr():
    """Test string representation (2 points)"""
    t = Tensor([1, 2, 3])
    repr_str = repr(t)
    assert "Tensor" in repr_str
    assert "shape" in repr_str
    assert "dtype" in repr_str

def test_tensor_add():
    """Test tensor addition (3 points)"""
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1.add(t2)
    assert result.data.tolist() == [5, 7, 9]
    assert result.shape == (3,)

def test_tensor_multiply():
    """Test tensor multiplication (3 points)"""
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1.multiply(t2)
    assert result.data.tolist() == [4, 10, 18]
    assert result.shape == (3,)

def test_tensor_matmul():
    """Test matrix multiplication (4 points)"""
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    result = t1.matmul(t2)
    expected = [[19, 22], [43, 50]]
    assert result.data.tolist() == expected
    assert result.shape == (2, 2)

def test_tensor_matmul_error():
    """Test matrix multiplication error handling (2 points)"""
    t1 = Tensor([[1, 2, 3]])  # Shape (1, 3)
    t2 = Tensor([[4, 5]])     # Shape (1, 2)
    
    try:
        t1.matmul(t2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot multiply shapes" in str(e)

def test_tensor_immutability():
    """Test that operations create new tensors (2 points)"""
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    original_data = t1.data.copy()
    
    result = t1.add(t2)
    
    # Original tensor should be unchanged
    assert np.array_equal(t1.data, original_data)
    # Result should be different object
    assert result is not t1
    assert result.data is not t1.data

### END HIDDEN TESTS

# %% [markdown]
"""
## Usage Examples

### Self-Learning Mode
Students work through the educational content step by step:

```python
# Create tensors
t1 = Tensor([1, 2, 3])
t2 = Tensor([4, 5, 6])

# Basic operations
result = t1.add(t2)
print(f"Addition: {result}")

# Matrix operations
matrix1 = Tensor([[1, 2], [3, 4]])
matrix2 = Tensor([[5, 6], [7, 8]])
product = matrix1.matmul(matrix2)
print(f"Matrix multiplication: {product}")
```

### Assignment Mode
Students submit implementations that are automatically graded:

1. **Immediate feedback**: Know if implementation is correct
2. **Partial credit**: Earn points for each working method
3. **Hidden tests**: Comprehensive coverage beyond visible examples
4. **Error handling**: Points for proper edge case handling

### Benefits of Dual System

1. **Single source**: One implementation serves both purposes
2. **Consistent quality**: Same instructor solutions everywhere
3. **Flexible assessment**: Choose the right tool for each situation
4. **Scalable**: Handle large courses with automated feedback

This approach transforms TinyTorch from a learning framework into a complete course management solution.
"""

# %%
# Test the implementation
if __name__ == "__main__":
    # Basic testing
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    
    print(f"t1: {t1}")
    print(f"t2: {t2}")
    print(f"t1 + t2: {t1.add(t2)}")
    print(f"t1 * t2: {t1.multiply(t2)}")
    
    # Matrix multiplication
    m1 = Tensor([[1, 2], [3, 4]])
    m2 = Tensor([[5, 6], [7, 8]])
    print(f"Matrix multiplication: {m1.matmul(m2)}")
    
    print("✅ Enhanced tensor module working!") 
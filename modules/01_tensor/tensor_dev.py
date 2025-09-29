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
**What You Built Before**: Module 00 (Setup) gave you a Python environment with NumPy

**What's Working**: You have all the tools needed for numerical computing

**The Gap**: You need to build the core data structure that makes ML possible

**This Module's Solution**: Create a Tensor class that wraps NumPy with clean ML operations

## Learning Objectives
1. **Core Implementation**: Build Tensor class with arithmetic operations
2. **Essential Operations**: Addition, multiplication, matrix operations
3. **Testing Skills**: Validate each function immediately after implementation
4. **Integration Knowledge**: Prepare foundation for neural network modules

## Build → Test → Use
1. **Build**: Implement essential tensor operations
2. **Test**: Verify each component works correctly
3. **Use**: Apply tensors to multi-dimensional data
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
## Understanding Tensors

Tensors are N-dimensional arrays that store and manipulate numerical data. Think of them as generalizations of scalars, vectors, and matrices:

- **Scalar (0D)**: A single number like `5.0`
- **Vector (1D)**: A list like `[1, 2, 3]` with shape `(3,)`
- **Matrix (2D)**: A 2D array like `[[1, 2], [3, 4]]` with shape `(2, 2)`
- **3D Tensor**: Like an RGB image with `(height, width, channels)`

Our Tensor class is a PURE data structure that wraps NumPy arrays with clean mathematical operations. This foundation focuses on data storage and computation - gradient tracking will be added in Module 05.
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-init", "solution": true}

#| export
class Tensor:
    """
    TinyTorch Tensor: N-dimensional array with ML operations.

    The fundamental data structure for all TinyTorch operations.
    Wraps NumPy arrays with ML-specific functionality.
    """

    def __init__(self, data: Any, dtype: Optional[str] = None):
        """
        Create a new tensor from data.

        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.

        TODO: Implement tensor creation with simple, clear type handling.

        APPROACH:
        1. Convert input data to numpy array
        2. Apply dtype if specified
        3. Set default float32 for float64 arrays
        4. Store the result in self._data

        EXAMPLE:
        >>> Tensor(5)
        >>> Tensor([1.0, 2.0, 3.0])
        >>> Tensor([1, 2, 3], dtype='float32')
        """
        ### BEGIN SOLUTION
        if isinstance(data, Tensor):
            self._data = data.data.copy()
        else:
            self._data = np.array(data)

        if dtype is not None:
            self._data = self._data.astype(dtype)
        elif self._data.dtype == np.float64:
            self._data = self._data.astype(np.float32)
        ### END SOLUTION

    @property
    def data(self) -> np.ndarray:
        """
        Access underlying numpy array.

        TODO: Return the stored numpy array.
        """
        ### BEGIN SOLUTION
        return self._data
        ### END SOLUTION
    

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get tensor shape.

        TODO: Return the shape of the stored numpy array.
        """
        ### BEGIN SOLUTION
        return self._data.shape
        ### END SOLUTION

    @property
    def size(self) -> int:
        """
        Get total number of elements.

        TODO: Return the total number of elements in the tensor.
        """
        ### BEGIN SOLUTION
        return self._data.size
        ### END SOLUTION

    @property
    def dtype(self) -> np.dtype:
        """
        Get data type as numpy dtype.

        TODO: Return the data type of the stored numpy array.
        """
        ### BEGIN SOLUTION
        return self._data.dtype
        ### END SOLUTION


    def __repr__(self) -> str:
        """
        String representation with size limits for readability.

        TODO: Create a clear string representation of the tensor.
        """
        ### BEGIN SOLUTION
        if self.size > 20:
            return f"Tensor(shape={self.shape}, dtype={self.dtype})"
        else:
            return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"
        ### END SOLUTION

    def numpy(self) -> np.ndarray:
        """Convert tensor to NumPy array."""
        return self._data

# %% nbgrader={"grade": false, "grade_id": "tensor-arithmetic", "solution": true}

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Addition operator: tensor + other

        TODO: Implement + operator for tensors.
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:
            return Tensor(self._data + other)
        ### END SOLUTION

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Multiplication operator: tensor * other

        TODO: Implement * operator for tensors.
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:
            return Tensor(self._data * other)
        ### END SOLUTION

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Subtraction operator: tensor - other

        TODO: Implement - operator for tensors.
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:
            return Tensor(self._data - other)
        ### END SOLUTION

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Division operator: tensor / other

        TODO: Implement / operator for tensors.
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:
            return Tensor(self._data / other)
        ### END SOLUTION


    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication using NumPy's optimized implementation.

        TODO: Implement matrix multiplication.
        """
        ### BEGIN SOLUTION
        if len(self._data.shape) != 2 or len(other._data.shape) != 2:
            raise ValueError("matmul requires 2D tensors")

        m, k = self._data.shape
        k2, n = other._data.shape

        if k != k2:
            raise ValueError(f"Inner dimensions must match: {k} != {k2}")

        result_data = np.dot(self._data, other._data)
        return Tensor(result_data)
        ### END SOLUTION

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication operator: tensor @ other

        Enables the @ operator for matrix multiplication, providing
        clean syntax for neural network operations.
        """
        return self.matmul(other)

    def __getitem__(self, key):
        """
        Access tensor elements using subscript notation: tensor[key]

        Supports all NumPy indexing patterns:
        - Single index: tensor[0]
        - Multiple indices: tensor[0, 1]
        - Slices: tensor[0:2, 1:3]
        - Fancy indexing: tensor[[0, 2], [1, 3]]

        Args:
            key: Index or slice specification

        Returns:
            Scalar, array value, or new Tensor with subset of data

        Examples:
            tensor = Tensor([[1, 2], [3, 4]])
            tensor[0, 0]  # Returns 1 (scalar)
            tensor[0]     # Returns Tensor([1, 2])
            tensor[0:1, 0:1]  # Returns Tensor([[1]])
        """
        result = self._data[key]

        # If result is a scalar, return the scalar value directly
        if np.isscalar(result):
            return result

        # If result is an array, wrap it in a Tensor
        return Tensor(result)

    def reshape(self, *shape: int) -> 'Tensor':
        """
        Return a new tensor with the same data but different shape.

        TODO: Implement tensor reshaping.
        """
        ### BEGIN SOLUTION
        reshaped_data = self._data.reshape(*shape)
        return Tensor(reshaped_data)
        ### END SOLUTION

    def transpose(self) -> 'Tensor':
        """
        Return the transpose of a 2D tensor.

        TODO: Implement tensor transpose.
        """
        ### BEGIN SOLUTION
        if len(self._data.shape) != 2:
            raise ValueError("transpose() requires 2D tensor")
        return Tensor(self._data.T)
        ### END SOLUTION

    # Note: gradient computation will be added in Module 05 (Autograd)
    # This pure Tensor class focuses only on data structure operations




# %% [markdown]
"""
## Class Methods for Tensor Creation
"""


#| export
@classmethod
def zeros(cls, *shape: int) -> 'Tensor':
    """Create a tensor filled with zeros."""
    return cls(np.zeros(shape))

@classmethod
def ones(cls, *shape: int) -> 'Tensor':
    """Create a tensor filled with ones."""
    return cls(np.ones(shape))

@classmethod
def random(cls, *shape: int) -> 'Tensor':
    """Create a tensor with random values."""
    return cls(np.random.randn(*shape))

# Add class methods to Tensor class
Tensor.zeros = zeros
Tensor.ones = ones
Tensor.random = random

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation
This test validates tensor creation with different data types and shapes.
"""

# %%
def test_unit_tensor_creation():
    """Test tensor creation with all data types and shapes."""
    print("🔬 Unit Test: Tensor Creation...")

    try:
        # Test scalar
        scalar = Tensor(5.0)
        assert scalar.shape == (), f"Scalar should have shape (), got {scalar.shape}"
        print("✅ Scalar creation works")

        # Test vector
        vector = Tensor([1, 2, 3])
        assert vector.shape == (3,), f"Vector should have shape (3,), got {vector.shape}"
        print("✅ Vector creation works")

        # Test matrix
        matrix = Tensor([[1, 2], [3, 4]])
        assert matrix.shape == (2, 2), f"Matrix should have shape (2, 2), got {matrix.shape}"
        print("✅ Matrix creation works")

        # Test class methods
        zeros = Tensor.zeros(2, 3)
        ones = Tensor.ones(2, 3)
        random = Tensor.random(2, 3)
        assert zeros.shape == (2, 3), "Zeros tensor should have correct shape"
        assert ones.shape == (2, 3), "Ones tensor should have correct shape"
        assert random.shape == (2, 3), "Random tensor should have correct shape"
        print("✅ Class methods work")

        print("📈 Progress: Tensor Creation ✓")

    except Exception as e:
        print(f"❌ Tensor creation test failed: {e}")
        raise

test_unit_tensor_creation()


# %% [markdown]
"""
### 🧪 Unit Test: Tensor Properties
This test validates tensor properties like shape, size, and data access.
"""

# %%

def test_unit_tensor_properties():
    """Test tensor properties (shape, size, dtype, data access)."""
    print("🔬 Unit Test: Tensor Properties...")

    try:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        assert tensor.shape == (2, 3), f"Shape should be (2, 3), got {tensor.shape}"
        assert tensor.size == 6, f"Size should be 6, got {tensor.size}"
        assert np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]])), "Data property should return numpy array"
        assert tensor.dtype in [np.int32, np.int64], f"Dtype should be int32 or int64, got {tensor.dtype}"
        print("✅ All properties work correctly")

        print("📈 Progress: Tensor Properties ✓")

    except Exception as e:
        print(f"❌ Tensor properties test failed: {e}")
        raise

test_unit_tensor_properties()


# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic
This test validates all arithmetic operations (+, -, *, /) work correctly.
"""

# %%

def test_unit_tensor_arithmetic():
    """Test tensor arithmetic operations."""
    print("🔬 Unit Test: Tensor Arithmetic...")

    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        # Test all operations
        result_add = a + b
        result_mul = a * b
        result_sub = b - a
        result_div = b / a

        expected_add = np.array([5, 7, 9])
        expected_mul = np.array([4, 10, 18])
        expected_sub = np.array([3, 3, 3])
        expected_div = np.array([4.0, 2.5, 2.0])

        assert np.array_equal(result_add.data, expected_add), "Addition failed"
        assert np.array_equal(result_mul.data, expected_mul), "Multiplication failed"
        assert np.array_equal(result_sub.data, expected_sub), "Subtraction failed"
        assert np.allclose(result_div.data, expected_div), "Division failed"

        # Test scalar operations
        result_scalar = a + 10
        expected_scalar = np.array([11, 12, 13])
        assert np.array_equal(result_scalar.data, expected_scalar), "Scalar addition failed"

        print("✅ All arithmetic operations work")
        print("📈 Progress: Tensor Arithmetic ✓")

    except Exception as e:
        print(f"❌ Tensor arithmetic test failed: {e}")
        raise

test_unit_tensor_arithmetic()

# %% [markdown]
"""
### 🧪 Unit Test: Matrix Multiplication
This test validates matrix multiplication and the @ operator.
"""

# %%

def test_unit_matrix_multiplication():
    """Test matrix multiplication."""
    print("🔬 Unit Test: Matrix Multiplication...")

    try:
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(result.data, expected), f"Matmul failed: expected {expected}, got {result.data}"
        print("✅ Matrix multiplication works")

        # Test shape validation
        try:
            bad_a = Tensor([[1, 2]])
            bad_b = Tensor([[1], [2], [3]])  # Incompatible shapes
            result = bad_a @ bad_b
            print("❌ Should have failed with incompatible shapes")
        except ValueError:
            print("✅ Shape validation works")

        print("📈 Progress: Matrix Multiplication ✓")

    except Exception as e:
        print(f"❌ Matrix multiplication test failed: {e}")
        raise

test_unit_matrix_multiplication()

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Operations
This test validates reshape, transpose, and numpy conversion.
"""

# %%

def test_unit_tensor_operations():
    """Test tensor operations: reshape, transpose."""
    print("🔬 Unit Test: Tensor Operations...")

    try:
        # Test reshape
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        reshaped = tensor.reshape(3, 2)
        assert reshaped.shape == (3, 2), f"Reshape failed: expected (3, 2), got {reshaped.shape}"
        print("✅ Reshape works")

        # Test transpose
        matrix = Tensor([[1, 2], [3, 4]])
        transposed = matrix.transpose()
        expected = np.array([[1, 3], [2, 4]])
        assert np.array_equal(transposed.data, expected), "Transpose failed"
        print("✅ Transpose works")

        # Test numpy conversion
        numpy_array = tensor.numpy()
        assert np.array_equal(numpy_array, tensor.data), "Numpy conversion failed"
        print("✅ NumPy conversion works")

        print("📈 Progress: Tensor Operations ✓")

    except Exception as e:
        print(f"❌ Tensor operations test failed: {e}")
        raise

test_unit_tensor_operations()

# %% [markdown]
"""
### 🧪 Complete Module Test
This runs all tests together to validate the complete tensor implementation.
"""

# %%

def test_module():
    """Final comprehensive test of entire tensor module."""
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_tensor_creation()
    test_unit_tensor_properties()
    test_unit_tensor_arithmetic()
    test_unit_matrix_multiplication()
    test_unit_tensor_operations()

    print("\nRunning integration scenarios...")
    print("🔬 Integration Test: End-to-end tensor workflow...")

    # Test realistic usage pattern
    tensor = Tensor([[1, 2], [3, 4]])
    result = (tensor + tensor) @ tensor.transpose()
    assert result.shape == (2, 2)
    print("✅ End-to-end workflow works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 01")

test_module()

# %% [markdown]
"""
## Basic Performance Check

Let's do a simple check to see how our tensor operations perform:
"""

# %%
def check_tensor_performance():
    """Simple performance check for our tensor operations."""
    print("📊 Basic Performance Check:")

    import time

    # Test with small matrices first
    a = Tensor.random(100, 100)
    b = Tensor.random(100, 100)

    start = time.perf_counter()
    result = a @ b
    elapsed = time.perf_counter() - start

    print(f"100x100 matrix multiplication: {elapsed*1000:.2f}ms")
    print(f"Result shape: {result.shape}")
    print("✅ Tensor operations work efficiently!")


if __name__ == "__main__":
    print("🚀 Running Tensor module...")
    test_module()
    print("✅ Module validation complete!")


# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

### Question 1: Tensor Size and Memory
**Context**: Your Tensor class stores data as NumPy arrays. When you created different sized tensors, you saw how memory usage changes.

**Reflection Question**: If you create a 1000×1000 tensor versus a 100×100 tensor, how does memory usage change? Why does this matter for neural networks with millions of parameters?

### Question 2: Operation Performance
**Context**: Your arithmetic operators (+, -, *, /) use NumPy's vectorized operations instead of Python loops.

**Reflection Question**: Why is `tensor1 + tensor2` much faster than looping through each element? How does this speed advantage become critical in neural network training?

### Question 3: Matrix Multiplication Scaling
**Context**: Your `matmul()` method uses NumPy's optimized `np.dot()` function for matrix multiplication.

**Reflection Question**: Matrix multiplication has O(N³) complexity. If you double the matrix size, how much longer does multiplication take? When does this become a bottleneck in neural networks?
"""


# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tensor Foundation Complete!

Congratulations! You've built the fundamental data structure that powers neural networks.

### What You've Accomplished
✅ **Core Tensor Class**: Complete implementation with creation, properties, and operations
✅ **Essential Arithmetic**: Addition, subtraction, multiplication, division with NumPy integration
✅ **Matrix Operations**: Matrix multiplication with @ operator and shape validation
✅ **Shape Manipulation**: Reshape and transpose for data transformation
✅ **Testing Framework**: Comprehensive unit tests validating all functionality

### Key Learning Outcomes
- **Tensor Fundamentals**: N-dimensional arrays as the foundation of ML
- **NumPy Integration**: Leveraging optimized numerical computing
- **Clean API Design**: Operations that mirror PyTorch and TensorFlow patterns
- **Testing Approach**: Immediate validation after each implementation

### Ready for Next Steps
Your pure tensor implementation enables:
- **Module 02 (Activations)**: Add nonlinear functions using clean tensor operations
- **Modules 03-04**: Build layers and losses with focused tensor operations
- **Module 05 (Autograd)**: Will extend this foundation with gradient tracking
- **Real ML Work**: Handle numerical computations with a clean, extensible foundation

### Export Your Work
1. **Module validation**: Complete with `test_module()` comprehensive testing
2. **Export to package**: `tito module complete 01_tensor`
3. **Integration**: Your code becomes `tinytorch.core.tensor.Tensor`
4. **Next module**: Ready for activation functions!

**Achievement unlocked**: You've built the foundation of modern AI systems!
"""
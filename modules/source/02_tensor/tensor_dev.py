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
# Module 02: Tensor - The Foundation of Machine Learning

Welcome to the heart of TinyTorch! This module implements the fundamental data structure that powers all modern ML systems. Every operation in deep learning flows through tensors - they are the universal language of AI.

## What You'll Build
- **Tensor Class**: N-dimensional array wrapper with ML-specific operations
- **Core Operations**: Creation, arithmetic, and shape management
- **Professional API**: Clean interfaces that mirror PyTorch and TensorFlow
- **Foundation Layer**: The building block for all subsequent modules

## Learning Outcomes
- **Understand**: How tensors form the mathematical foundation of ML
- **Implement**: Professional-grade tensor operations from scratch  
- **Connect**: Real-world applications in neural networks and AI
- **Master**: The patterns used by all major ML frameworks
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
## Tensor Class Creation - The Foundation

### The 5 C's Framework
Before implementing our Tensor class, let's understand what we're building through our systematic approach:

#### Concept
**What is a Tensor?**
A tensor is an N-dimensional array with ML-specific operations. It's the universal data structure of machine learning - every piece of data (scalars, vectors, matrices, higher-dimensional arrays) flows through tensors.

**Key insight**: While we could use NumPy arrays directly, our Tensor class adds ML-specific functionality like gradient tracking (future), GPU support (future), and consistent APIs that make building neural networks intuitive.

#### Code Structure
**What we're building:**
```python
class Tensor:
    def __init__(self, data, dtype=None):  # Create from any input
    # Properties for inspection
    .shape    # Dimensions: (2, 3) for 2x3 matrix
    .size     # Total elements: 6 for 2x3 matrix
    .dtype    # Data type: float32, int32, etc.
    .data     # Access underlying NumPy array
    
    # Arithmetic operations
    def __add__(self, other):     # tensor + tensor
    def __mul__(self, other):     # tensor * tensor
    def __sub__(self, other):     # tensor - tensor
    def __truediv__(self, other): # tensor / tensor
```

#### Connections
**Real-world equivalents:**
- **PyTorch**: `torch.Tensor` - Our design mirrors PyTorch's approach
- **TensorFlow**: `tf.Tensor` - Similar concept for distributed computing
- **NumPy**: `np.ndarray` - We wrap this with ML-specific features
- **JAX**: `jnp.ndarray` - Functional programming approach to tensors

**Why this pattern works**: Every major ML framework uses the same "wrap a numerical library" approach because it provides both performance (via optimized backends) and usability (via clean APIs).

#### Constraints
**Implementation requirements:**
- **Input flexibility**: Handle scalars (5), lists ([1,2,3]), nested lists ([[1,2],[3,4]]), and NumPy arrays
- **Type safety**: Consistent data types with automatic promotion (int + float → float)
- **Broadcasting support**: Operations between different shapes (matrix + scalar)
- **Memory efficiency**: Copy data only when necessary
- **Error handling**: Clear messages for invalid operations

#### Context
**Why this matters in ML systems:**

**Data Flow**: Every piece of information in ML flows through tensors:
```python
# Image processing pipeline
image = Tensor(raw_pixels)          # Input: (224, 224, 3)
features = conv_layer(image)        # Transform: (224, 224, 64)
output = classifier(features)       # Classify: (1000,)
```

**Performance**: Modern hardware (GPUs, TPUs) is optimized for tensor operations:
- **Parallel processing**: Compute thousands of elements simultaneously
- **Vectorization**: Single instruction operates on multiple data points
- **Memory locality**: Contiguous arrays enable cache-efficient access

**Automatic Differentiation**: Tensors will track operations for gradient computation:
```python
# Forward pass creates computation graph
x = Tensor([1, 2, 3])  # Input
y = x * 2              # Operation node
z = y + 1              # Operation node
loss = z.sum()         # Final result
# Backward pass computes gradients automatically
```

**Universal Language**: Tensors enable code that works across domains:
- **Computer Vision**: Images as 4D tensors (batch, height, width, channels)
- **NLP**: Text as 3D tensors (batch, sequence_length, embedding_dim)
- **Audio**: Waveforms as 2D tensors (batch, time_steps)
- **Time Series**: Data as 3D tensors (batch, time_steps, features)

### Implementation Strategy
We'll build incrementally:
1. **Core constructor**: Handle different input types with proper type conversion
2. **Properties**: Expose shape, size, dtype, and data access
3. **Arithmetic operations**: Element-wise math with broadcasting
4. **String representation**: Clear output for debugging

This foundation will support all future TinyTorch modules: activations, layers, optimizers, and complete neural networks.
"""

# %% [markdown]
"""
## Tensor Implementation

Now let's implement the Tensor class with clean, production-ready code:
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
        """Add two tensors element-wise."""
        result = self._data + other._data
        return Tensor(result)

    def multiply(self, other: 'Tensor') -> 'Tensor':
        """Multiply two tensors element-wise."""
        result = self._data * other._data
        return Tensor(result)

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition operator: tensor + other"""
        if isinstance(other, Tensor):
            return self.add(other)
        else:
            return self.add(Tensor(other))

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication operator: tensor * other"""
        if isinstance(other, Tensor):
            return self.multiply(other)
        else:
            return self.multiply(Tensor(other))

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction operator: tensor - other"""
        if isinstance(other, Tensor):
            result = self._data - other._data
        else:
            result = self._data - other
        return Tensor(result)

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division operator: tensor / other"""
        if isinstance(other, Tensor):
            result = self._data / other._data
        else:
            result = self._data / other
        return Tensor(result)

    def mean(self) -> 'Tensor':
        """Computes the mean of the tensor's elements."""
        return Tensor(np.mean(self.data))

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Perform matrix multiplication between two tensors."""
        result = np.matmul(self._data, other._data)
        return Tensor(result)

# %% [markdown]
"""
## Tensor Testing

Let's verify our implementation with comprehensive tests:
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor creation
print("🔬 Testing Tensor Creation...")

# Test basic tensor creation
scalar = Tensor(5.0)
assert hasattr(scalar, '_data'), "Tensor should have _data attribute"
assert scalar._data.shape == (), f"Scalar should have shape (), got {scalar._data.shape}"
print("✅ Scalar creation works")

vector = Tensor([1, 2, 3])
assert vector._data.shape == (3,), f"Vector should have shape (3,), got {vector._data.shape}"
print("✅ Vector creation works")

matrix = Tensor([[1, 2], [3, 4]])
assert matrix._data.shape == (2, 2), f"Matrix should have shape (2, 2), got {matrix._data.shape}"
print("✅ Matrix creation works")

print("📈 Tensor Creation ✓")

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor properties
print("🔬 Testing Tensor Properties...")

tensor = Tensor([[1, 2, 3], [4, 5, 6]])

# Test properties
assert tensor.shape == (2, 3), f"Shape should be (2, 3), got {tensor.shape}"
print("✅ Shape property works")

assert tensor.size == 6, f"Size should be 6, got {tensor.size}"
print("✅ Size property works")

assert np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]])), "Data property should return numpy array"
print("✅ Data property works")

assert tensor.dtype in [np.int32, np.int64], f"Dtype should be int32 or int64, got {tensor.dtype}"
print("✅ Dtype property works")

print("📈 Tensor Properties ✓")

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor arithmetic
print("🔬 Testing Tensor Arithmetic...")

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Test addition
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

print("📈 Tensor Arithmetic ✓")

# %% [markdown]
"""
## Usage Examples and Integration

Let's see our Tensor class in action with practical examples:
"""

# %% 
# Demonstrate tensor usage with real examples
print("🚀 Tensor Usage Examples...")

# Example 1: Basic tensor operations
print("\n1. Basic Operations:")
x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])
print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")

# Example 2: Broadcasting with scalars  
print("\n2. Broadcasting:")
print(f"x + 10 = {x + 10}")
print(f"x * 2 = {x * 2}")

# Example 3: Matrix operations
print("\n3. Matrix Operations:")
A = Tensor([[1, 2], [3, 4]])
B = Tensor([[5, 6], [7, 8]])
print(f"A = {A}")
print(f"B = {B}")
print(f"A + B = {A + B}")
print(f"A @ B = {A.matmul(B)}")  # Matrix multiplication

# Example 4: Different data types
print("\n4. Data Types:")
int_tensor = Tensor([1, 2, 3])
float_tensor = Tensor([1.0, 2.0, 3.0])
print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")

print("\n✅ All examples working correctly!")

# %% [markdown]
"""
## Comprehensive Testing Suite

Final validation of all tensor functionality:
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_comprehensive_tensor_creation():
    """Test all tensor creation scenarios."""
    print("🔬 Comprehensive Creation Tests...")
    
    # Test scalar creation
    scalar_int = Tensor(42)
    assert scalar_int.shape == ()
    
    # Test vector creation
    vector_int = Tensor([1, 2, 3])
    assert vector_int.shape == (3,)

    # Test matrix creation
    matrix_2x2 = Tensor([[1, 2], [3, 4]])
    assert matrix_2x2.shape == (2, 2)
    print("✅ All creation tests passed!")

test_comprehensive_tensor_creation()

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_comprehensive_tensor_properties():
    """Test all tensor properties."""
    print("🔬 Comprehensive Properties Tests...")

    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert tensor.shape == (2, 3)
    assert tensor.size == 6
    assert np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]]))
    assert tensor.dtype in [np.int32, np.int64]
    print("✅ All properties tests passed!")

test_comprehensive_tensor_properties()

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_comprehensive_tensor_arithmetic():
    """Test all tensor arithmetic operations."""
    print("🔬 Comprehensive Arithmetic Tests...")
    
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Test all operations
    c = a + b
    assert np.array_equal(c.data, np.array([5, 7, 9]))
    
    d = a * b
    assert np.array_equal(d.data, np.array([4, 10, 18]))

    e = b - a
    assert np.array_equal(e.data, np.array([3, 3, 3]))

    f = b / a
    assert np.allclose(f.data, np.array([4.0, 2.5, 2.0]))
    print("✅ All arithmetic tests passed!")

test_comprehensive_tensor_arithmetic()

# %% nbgrader={"grade": true, "grade_id": "test_module_tensor_numpy_integration", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_tensor_numpy_integration():
    """Integration test for tensor-NumPy compatibility."""
    print("🔬 Integration Test: Tensor-NumPy Compatibility...")
    
    # Test NumPy array input
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = Tensor(numpy_array)
    assert tensor_from_numpy.shape == (2, 3)
    assert np.array_equal(tensor_from_numpy.data, numpy_array)
    
    # Test NumPy operations on tensor data
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    dot_product = np.dot(a.data, b.data)
    assert np.isclose(dot_product, 32.0)
    
    # Test broadcasting
    matrix = Tensor([[1, 2], [3, 4]])
    scalar = Tensor(10)
    result = matrix + scalar
    expected = np.array([[11, 12], [13, 14]])
    assert np.array_equal(result.data, expected)
    
    print("✅ Integration test passed!")

test_tensor_numpy_integration()

# %% [markdown]
"""
## Module 02 Summary: Tensor Foundation Complete

Congratulations! You've successfully implemented the fundamental data structure that powers all machine learning systems.

### What You've Accomplished
✅ **Clean 5 C's Structure**: Systematic approach to understanding tensors  
✅ **Professional Implementation**: Production-ready Tensor class with clean code  
✅ **Comprehensive Testing**: Separate test sections validating all functionality  
✅ **Practical Examples**: Real-world usage patterns and integration demos  
✅ **Foundation Ready**: Building block for all future TinyTorch modules  

### Key Technical Achievements
- **Multi-input Support**: Handles scalars, lists, and NumPy arrays seamlessly
- **Type Management**: Intelligent dtype handling with auto-detection
- **Broadcasting**: NumPy-compatible operations between different shapes  
- **Clean APIs**: Intuitive operator overloading (`+`, `-`, `*`, `/`)
- **Memory Efficiency**: Smart copying and data management

### The 5 C's in Practice
- **Concept**: Tensors as universal ML data containers
- **Code**: Clean, maintainable implementation patterns  
- **Connections**: Direct parallels to PyTorch and TensorFlow
- **Constraints**: Production-ready requirements and limitations
- **Context**: Real-world applications across ML domains

### Architecture Patterns Established
This module establishes the template for all future modules:
1. **5 C's Explanation** → Clear conceptual understanding
2. **Clean Implementation** → Professional, readable code  
3. **Focused Testing** → Separate validation of functionality
4. **Usage Examples** → Practical application demonstrations

### Ready for Next Level
Your tensor foundation enables:
- **Activation Functions**: Nonlinear transformations (Module 03)
- **Neural Layers**: Linear algebra operations (Module 04)  
- **Complex Networks**: Deep architectures (Module 05+)
- **Training Systems**: Optimization and learning (Module 07+)

**Next**: Let's add the mathematical functions that make neural networks capable of learning complex patterns!
""" 
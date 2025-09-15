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
# Module 02: Tensor - Core Data Structure

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

**Learning Side:** You work in `modules/source/02_tensor/tensor_dev.py`  
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
## The Mathematical Foundation: From Scalars to Tensors

Understanding tensors requires building from mathematical fundamentals:

### Scalars (Rank 0)
- **Definition**: A single number with no direction
- **Examples**: Temperature (25°C), mass (5.2 kg), probability (0.7)
- **Operations**: Addition, multiplication, comparison
- **ML Context**: Loss values, learning rates, regularization parameters

### Vectors (Rank 1)
- **Definition**: An ordered list of numbers with direction and magnitude
- **Examples**: Position [x, y, z], RGB color [255, 128, 0], word embedding [0.1, -0.5, 0.8]
- **Operations**: Dot product, cross product, norm calculation
- **ML Context**: Feature vectors, gradients, model parameters

### Matrices (Rank 2)
- **Definition**: A 2D array organizing data in rows and columns
- **Examples**: Image (height × width), weight matrix (input × output), covariance matrix
- **Operations**: Matrix multiplication, transpose, inverse, eigendecomposition
- **ML Context**: Linear layer weights, attention matrices, batch data

### Higher-Order Tensors (Rank 3+)
- **Definition**: Multi-dimensional arrays extending matrices
- **Examples**: 
  - **3D**: Video frames (time × height × width), RGB images (height × width × channels)
  - **4D**: Image batches (batch × height × width × channels)
  - **5D**: Video batches (batch × time × height × width × channels)
- **Operations**: Tensor products, contractions, decompositions
- **ML Context**: Convolutional features, RNN states, transformer attention
"""

# %% [markdown]
"""
## Why Tensors Matter in ML: The Computational Foundation

### 1. Unified Data Representation
Tensors provide a consistent way to represent all ML data:
```python
# All of these are tensors with different shapes
scalar_loss = Tensor(0.5)              # Shape: ()
feature_vector = Tensor([1, 2, 3])      # Shape: (3,)
weight_matrix = Tensor([[1, 2], [3, 4]]) # Shape: (2, 2)
image_batch = Tensor(np.random.rand(32, 224, 224, 3)) # Shape: (32, 224, 224, 3)
```

### 2. Efficient Batch Processing
ML systems process multiple samples simultaneously:
```python
# Instead of processing one image at a time:
for image in images:
    result = model(image)  # Slow: 1000 separate operations

# Process entire batch at once:
batch_result = model(image_batch)  # Fast: 1 vectorized operation
```

### 3. Hardware Acceleration
Modern hardware (GPUs, TPUs) excels at tensor operations:
- **Parallel processing**: Multiple operations simultaneously
- **Vectorization**: SIMD (Single Instruction, Multiple Data) operations
- **Memory optimization**: Contiguous memory layout for cache efficiency

### 4. Automatic Differentiation
Tensors enable gradient computation through computational graphs:
```python
# Each tensor operation creates a node in the computation graph
x = Tensor([1, 2, 3])
y = x * 2          # Node: multiplication
z = y + 1          # Node: addition
loss = z.sum()     # Node: summation
# Gradients flow backward through this graph
```
"""

# %% [markdown]
"""
## Real-World Examples: Tensors in Action

### Computer Vision
- **Grayscale image**: 2D tensor `(height, width)` - `(28, 28)` for MNIST
- **Color image**: 3D tensor `(height, width, channels)` - `(224, 224, 3)` for RGB
- **Image batch**: 4D tensor `(batch, height, width, channels)` - `(32, 224, 224, 3)`
- **Video**: 5D tensor `(batch, time, height, width, channels)`

### Natural Language Processing
- **Word embedding**: 1D tensor `(embedding_dim,)` - `(300,)` for Word2Vec
- **Sentence**: 2D tensor `(sequence_length, embedding_dim)` - `(50, 768)` for BERT
- **Batch of sentences**: 3D tensor `(batch, sequence_length, embedding_dim)`

### Audio Processing
- **Audio signal**: 1D tensor `(time_steps,)` - `(16000,)` for 1 second at 16kHz
- **Spectrogram**: 2D tensor `(time_frames, frequency_bins)`
- **Batch of audio**: 3D tensor `(batch, time_steps, features)`

### Time Series
- **Single series**: 2D tensor `(time_steps, features)`
- **Multiple series**: 3D tensor `(batch, time_steps, features)`
- **Multivariate forecasting**: 4D tensor `(batch, time_steps, features, predictions)`
"""

# %% [markdown]
"""
## Why Not Just Use NumPy?

While we use NumPy internally, our Tensor class adds ML-specific functionality:

### 1. ML-Specific Operations
- **Gradient tracking**: For automatic differentiation (coming in Module 7)
- **GPU support**: For hardware acceleration (future extension)
- **Broadcasting semantics**: ML-friendly dimension handling

### 2. Consistent API
- **Type safety**: Predictable behavior across operations
- **Error checking**: Clear error messages for debugging
- **Integration**: Seamless work with other TinyTorch components

### 3. Educational Value
- **Conceptual clarity**: Understand what tensors really are
- **Implementation insight**: See how frameworks work internally
- **Debugging skills**: Trace through tensor operations step by step

### 4. Extensibility
- **Future features**: Ready for gradients, GPU, distributed computing
- **Customization**: Add domain-specific operations
- **Optimization**: Profile and optimize specific use cases
"""

# %% [markdown]
"""
## Performance Considerations: Building Efficient Tensors

### Memory Layout
- **Contiguous arrays**: Better cache locality and performance
- **Data types**: `float32` vs `float64` trade-offs
- **Memory sharing**: Avoid unnecessary copies

### Vectorization
- **SIMD operations**: Single Instruction, Multiple Data
- **Broadcasting**: Efficient operations on different shapes
- **Batch operations**: Process multiple samples simultaneously

### Numerical Stability
- **Precision**: Balancing speed and accuracy
- **Overflow/underflow**: Handling extreme values
- **Gradient flow**: Maintaining numerical stability for training
"""

# %% [markdown]
"""
## Core Tensor Implementation

Now let's implement the Tensor class that will be the foundation of everything in TinyTorch!

### The 5 C's Framework

#### Concept
A Tensor is an N-dimensional array that carries data through neural networks. Think of it as a NumPy array with ML superpowers - same mathematical operations, but designed specifically for machine learning workflows.

#### Code Structure
```python
class Tensor:
    def __init__(self, data):     # Create from any data type
    def __add__(self, other):     # Enable tensor + tensor
    def __mul__(self, other):     # Enable tensor * tensor
    def __sub__(self, other):     # Enable tensor - tensor
    def __truediv__(self, other): # Enable tensor / tensor
    def matmul(self, other):      # Matrix multiplication
    # Properties: .shape, .size, .dtype, .data
```

#### Connections
- **PyTorch**: `torch.Tensor` - Same concept, production optimized
- **TensorFlow**: `tf.Tensor` - Distributed computing focus
- **NumPy**: `np.ndarray` - We wrap this with ML operations
- **JAX**: `jax.numpy.ndarray` - Functional programming approach

#### Constraints
- **Broadcasting**: Auto-shape matching for operations (like NumPy)
- **Type management**: Support multiple data types (float32, int32, etc.)
- **Memory efficiency**: Copy only when necessary
- **Natural notation**: `tensor + tensor` should just work

#### Context
Every ML operation flows through tensors:
- **Neural networks**: All computations operate on tensors
- **Training**: Gradients flow through tensor operations
- **Hardware**: GPUs optimized for tensor math
- **Production**: Millions of tensor ops per second in real systems

You're building the universal language of machine learning!
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
        
        APPROACH:
        1. Check if data is a scalar (int/float) - convert to numpy array
        2. Check if data is a list - convert to numpy array  
        3. Check if data is already a numpy array - use as-is
        4. Apply dtype conversion if specified
        5. Store the result in self._data
        
        EXAMPLE:
        ```python
        Tensor(5) → stores np.array(5)
        Tensor([1, 2, 3]) → stores np.array([1, 2, 3])
        Tensor(np.array([1, 2, 3])) → stores the array directly
        ```
        
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

    def mean(self) -> 'Tensor':
        """Computes the mean of the tensor's elements."""
        return Tensor(np.mean(self.data))

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Perform matrix multiplication between two tensors.
        
        TODO: Implement matrix multiplication.
        
        APPROACH:
        1. Use np.matmul() to perform matrix multiplication
        2. Return a new Tensor with the result
        3. Handle broadcasting automatically
        
        EXAMPLE:
        Tensor([[1, 2], [3, 4]]) @ Tensor([[5, 6], [7, 8]]) → Tensor([[19, 22], [43, 50]])
        
        HINTS:
        - Use np.matmul(self._data, other._data)
        - Return Tensor(result)
        - This is matrix multiplication, not element-wise multiplication
        """
        ### BEGIN SOLUTION
        result = np.matmul(self._data, other._data)
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
## Testing Your Implementation

Let's validate that your Tensor implementation works correctly! We'll run a series of comprehensive tests that verify each aspect of functionality.

### Test Structure
1. **Unit Tests**: Test individual methods in isolation
2. **Integration Tests**: Test how components work together
3. **Edge Cases**: Verify handling of special cases
4. **Performance**: Ensure efficient implementation
"""

# %% [markdown]
"""
### Unit Test: Tensor Creation

This test validates your `Tensor` class constructor, ensuring it correctly handles scalars, vectors, matrices, and higher-dimensional arrays with proper shape detection.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_tensor_creation():
    """Test tensor creation with all data types and shapes."""
    print("🔬 Testing tensor creation...")
    
    # Test scalar creation
    scalar_int = Tensor(42)
    assert scalar_int.shape == (), f"Scalar should have shape (), got {scalar_int.shape}"
    assert scalar_int.data == 42, f"Scalar data should be 42, got {scalar_int.data}"
    print("✅ Scalar creation works")
    
    # Test vector creation
    vector_int = Tensor([1, 2, 3])
    assert vector_int.shape == (3,), f"Vector should have shape (3,), got {vector_int.shape}"
    assert np.array_equal(vector_int.data, np.array([1, 2, 3])), "Vector data mismatch"
    print("✅ Vector creation works")

    # Test matrix creation
    matrix_2x2 = Tensor([[1, 2], [3, 4]])
    assert matrix_2x2.shape == (2, 2), f"Matrix should have shape (2, 2), got {matrix_2x2.shape}"
    assert np.array_equal(matrix_2x2.data, np.array([[1, 2], [3, 4]])), "Matrix data mismatch"
    print("✅ Matrix creation works")
    
    # Test NumPy array input
    numpy_input = np.array([5, 6, 7])
    tensor_from_numpy = Tensor(numpy_input)
    assert tensor_from_numpy.shape == (3,), "Should preserve NumPy array shape"
    assert np.array_equal(tensor_from_numpy.data, numpy_input), "Should preserve NumPy data"
    print("✅ NumPy array input works")
    
    print("📈 All tensor creation tests passed!")

# Run the test
test_unit_tensor_creation()

# %% [markdown]
"""
### Unit Test: Tensor Properties

This test validates your tensor property methods (shape, size, dtype, data), ensuring they correctly reflect the tensor's dimensional structure and data characteristics.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_tensor_properties():
    """Test tensor properties (shape, size, dtype, data access)."""
    print("🔬 Testing tensor properties...")

    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # Test shape property
    assert tensor.shape == (2, 3), f"Shape should be (2, 3), got {tensor.shape}"
    print("✅ Shape property works")
    
    # Test size property
    assert tensor.size == 6, f"Size should be 6, got {tensor.size}"
    print("✅ Size property works")
    
    # Test data property
    expected_data = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(tensor.data, expected_data), "Data property mismatch"
    print("✅ Data property works")
    
    # Test dtype property
    assert tensor.dtype in [np.int32, np.int64], f"Dtype should be int32 or int64, got {tensor.dtype}"
    print("✅ Dtype property works")
    
    # Test repr
    repr_str = repr(tensor)
    assert "Tensor" in repr_str, "Repr should contain 'Tensor'"
    assert "shape=(2, 3)" in repr_str, "Repr should contain shape"
    print("✅ String representation works")
    
    print("📈 All tensor property tests passed!")

# Run the test
test_unit_tensor_properties()

# %% [markdown]
"""
### Unit Test: Tensor Arithmetic Operations

This test validates your tensor arithmetic implementation (addition, multiplication, subtraction, division) and operator overloading, ensuring mathematical operations work correctly with proper broadcasting.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_tensor_arithmetic():
    """Test tensor arithmetic operations."""
    print("🔬 Testing tensor arithmetic...")
    
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Test addition
    c = a + b
    expected = np.array([5, 7, 9])
    assert np.array_equal(c.data, expected), f"Addition failed: expected {expected}, got {c.data}"
    print("✅ Addition works")
    
    # Test scalar addition
    d = a + 10
    expected = np.array([11, 12, 13])
    assert np.array_equal(d.data, expected), f"Scalar addition failed: expected {expected}, got {d.data}"
    print("✅ Scalar addition works")
    
    # Test multiplication
    e = a * b
    expected = np.array([4, 10, 18])
    assert np.array_equal(e.data, expected), f"Multiplication failed: expected {expected}, got {e.data}"
    print("✅ Multiplication works")
    
    # Test scalar multiplication
    f = a * 2
    expected = np.array([2, 4, 6])
    assert np.array_equal(f.data, expected), f"Scalar multiplication failed: expected {expected}, got {f.data}"
    print("✅ Scalar multiplication works")

    # Test subtraction
    g = b - a
    expected = np.array([3, 3, 3])
    assert np.array_equal(g.data, expected), f"Subtraction failed: expected {expected}, got {g.data}"
    print("✅ Subtraction works")

    # Test division
    h = b / Tensor([2, 5, 3])
    expected = np.array([2.0, 1.0, 2.0])
    assert np.allclose(h.data, expected), f"Division failed: expected {expected}, got {h.data}"
    print("✅ Division works")
    
    print("📈 All tensor arithmetic tests passed!")

# Run the test
test_unit_tensor_arithmetic()

# %% [markdown]
"""
### Unit Test: Matrix Multiplication

This test validates your matrix multiplication implementation, ensuring it correctly handles 2D tensors and follows proper linear algebra rules.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_matmul", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_tensor_matmul():
    """Test tensor matrix multiplication."""
    print("🔬 Testing matrix multiplication...")
    
    # Test 2x2 matrix multiplication
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    
    result = a.matmul(b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result.data, expected), f"Matmul failed: expected {expected}, got {result.data}"
    print("✅ 2x2 matrix multiplication works")
    
    # Test non-square matrices
    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3x2
    
    result = c.matmul(d)
    expected = np.array([[58, 64], [139, 154]])  # 2x2
    assert np.array_equal(result.data, expected), f"Non-square matmul failed: expected {expected}, got {result.data}"
    print("✅ Non-square matrix multiplication works")
    
    # Test vector-matrix multiplication
    vec = Tensor([1, 2])
    mat = Tensor([[3, 4], [5, 6]])
    
    result = vec.matmul(mat)
    expected = np.array([13, 16])
    assert np.array_equal(result.data, expected), f"Vector-matrix matmul failed: expected {expected}, got {result.data}"
    print("✅ Vector-matrix multiplication works")
    
    print("📈 All matrix multiplication tests passed!")

# Run the test
test_unit_tensor_matmul()

# %% [markdown]
"""
### Integration Test: Broadcasting

This test validates that your tensor operations correctly handle broadcasting - the automatic expansion of dimensions to make operations possible.
"""

# %% nbgrader={"grade": true, "grade_id": "test_integration_broadcasting", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_integration_broadcasting():
    """Test tensor broadcasting capabilities."""
    print("🔬 Testing broadcasting integration...")
    
    # Test scalar broadcasting
    matrix = Tensor([[1, 2], [3, 4]])
    scalar = Tensor(10)
    
    result = matrix + scalar
    expected = np.array([[11, 12], [13, 14]])
    assert np.array_equal(result.data, expected), f"Scalar broadcasting failed: expected {expected}, got {result.data}"
    print("✅ Scalar broadcasting works")
    
    # Test vector-matrix broadcasting
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    vector = Tensor([10, 20, 30])  # 1x3 (broadcasts to 2x3)
    
    result = matrix + vector
    expected = np.array([[11, 22, 33], [14, 25, 36]])
    assert np.array_equal(result.data, expected), f"Vector broadcasting failed: expected {expected}, got {result.data}"
    print("✅ Vector-matrix broadcasting works")
    
    # Test column vector broadcasting
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    col_vector = Tensor([[10], [20]])  # 2x1 (broadcasts to 2x3)
    
    result = matrix + col_vector
    expected = np.array([[11, 12, 13], [24, 25, 26]])
    assert np.array_equal(result.data, expected), f"Column vector broadcasting failed: expected {expected}, got {result.data}"
    print("✅ Column vector broadcasting works")
    
    print("📈 All broadcasting tests passed!")

# Run the test
test_integration_broadcasting()

# %% [markdown]
"""
### Integration Test: Type Handling

This test validates that your tensor correctly handles different data types and performs appropriate type conversions.
"""

# %% nbgrader={"grade": true, "grade_id": "test_integration_dtype", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_integration_dtype():
    """Test tensor data type handling."""
    print("🔬 Testing data type integration...")
    
    # Test integer tensor
    int_tensor = Tensor([1, 2, 3])
    assert int_tensor.dtype in [np.int32, np.int64], f"Integer tensor should have int dtype, got {int_tensor.dtype}"
    print("✅ Integer type detection works")
    
    # Test float tensor
    float_tensor = Tensor([1.0, 2.0, 3.0])
    assert float_tensor.dtype in [np.float32, np.float64], f"Float tensor should have float dtype, got {float_tensor.dtype}"
    print("✅ Float type detection works")
    
    # Test explicit dtype
    explicit_tensor = Tensor([1, 2, 3], dtype='float32')
    assert explicit_tensor.dtype == np.float32, f"Explicit dtype should be float32, got {explicit_tensor.dtype}"
    assert np.array_equal(explicit_tensor.data, np.array([1.0, 2.0, 3.0], dtype=np.float32)), "Data conversion failed"
    print("✅ Explicit dtype specification works")
    
    # Test mixed type operations
    int_t = Tensor([1, 2, 3])
    float_t = Tensor([1.5, 2.5, 3.5])
    result = int_t + float_t
    assert result.dtype in [np.float32, np.float64], "Mixed type operation should produce float"
    print("✅ Mixed type operations work")
    
    print("📈 All dtype tests passed!")

# Run the test
test_integration_dtype()

# %% [markdown]
"""
### Integration Test: Tensor-NumPy Compatibility

This test validates that tensors properly integrate with NumPy operations and maintain compatibility with the scientific Python ecosystem.
"""

# %% nbgrader={"grade": true, "grade_id": "test_integration_numpy", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_integration_numpy():
    """Test integration with NumPy ecosystem."""
    print("🔬 Testing NumPy integration...")
    
    # Test tensor from NumPy array
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = Tensor(numpy_array)
    
    assert tensor_from_numpy.shape == (2, 3), "Should preserve NumPy array shape"
    assert np.array_equal(tensor_from_numpy.data, numpy_array), "Should preserve NumPy array data"
    print("✅ Tensor from NumPy array works")
    
    # Test using tensor data with NumPy functions
    a = Tensor([1.0, 4.0, 9.0, 16.0, 25.0])
    sqrt_result = np.sqrt(a.data)
    expected_sqrt = np.array([1., 2., 3., 4., 5.])
    assert np.allclose(sqrt_result, expected_sqrt), "NumPy functions should work on tensor data"
    print("✅ NumPy functions work with tensor data")
    
    # Test dot product using tensor data
    vec1 = Tensor([1.0, 2.0, 3.0])
    vec2 = Tensor([4.0, 5.0, 6.0])
    dot_product = np.dot(vec1.data, vec2.data)
    assert np.isclose(dot_product, 32.0), f"Dot product should be 32.0, got {dot_product}"
    print("✅ NumPy dot product works with tensors")
    
    # Test creating tensor from NumPy computation
    result_array = np.sin(np.array([0, np.pi/2, np.pi]))
    tensor_from_computation = Tensor(result_array)
    expected = np.array([0., 1., 0.])
    assert np.allclose(tensor_from_computation.data, expected, atol=1e-10), "Should handle NumPy computation results"
    print("✅ Tensor from NumPy computation works")
    
    print("📈 All NumPy integration tests passed!")

# Run the test
test_integration_numpy()

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tensor Foundation Complete!

Congratulations! You've successfully implemented the fundamental data structure that powers all machine learning:

### ✅ What You've Built
- **Tensor Class**: N-dimensional array wrapper with professional interfaces
- **Core Operations**: Creation, property access, and arithmetic operations  
- **Matrix Multiplication**: Essential for neural network computations
- **Broadcasting**: Automatic shape compatibility for flexible operations
- **Type Management**: Proper handling of different data types
- **NumPy Integration**: Seamless compatibility with scientific Python

### ✅ Key Learning Outcomes
- **Understanding**: How tensors work as the foundation of machine learning
- **Implementation**: Built tensor operations from scratch with proper error handling
- **Professional Patterns**: Clean APIs, comprehensive testing, clear documentation
- **Real-World Connection**: Understanding PyTorch/TensorFlow tensor foundations
- **Systems Thinking**: Building reliable, reusable components

### ✅ Mathematical Foundations Mastered
- **N-dimensional Arrays**: Shape, size, and dimensionality concepts
- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Matrix Operations**: Matrix multiplication for linear transformations
- **Broadcasting**: Understanding how operations work with different shapes
- **Type Systems**: Managing numerical precision and memory efficiency

### ✅ Professional Skills Developed
- **API Design**: Clean, intuitive interfaces for tensor operations
- **Error Handling**: Graceful handling of invalid operations and edge cases
- **Testing Methodology**: Comprehensive validation with unit and integration tests
- **Documentation**: Clear, educational documentation with examples
- **Code Organization**: Modular, maintainable implementation

### 🔗 Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.Tensor` provides identical functionality
- **TensorFlow**: `tf.Tensor` implements similar concepts
- **NumPy**: `numpy.ndarray` serves as the foundation
- **Industry Standard**: Every major ML framework uses these exact principles

### 🚀 What's Next
Your tensor implementation is the foundation for:
- **Module 03: Activations** - Nonlinear functions that enable complex learning
- **Module 04: Layers** - Linear transformations and neural network building blocks
- **Module 05: Networks** - Composing layers into powerful architectures
- **Module 06: Training** - Optimizing networks to solve real problems

### 💡 The Power of What You've Built
You now have:
- **Universal Data Structure**: Handles all ML data types
- **Efficient Computation**: Vectorized operations for speed
- **Flexible Operations**: Broadcasting for convenient math
- **Professional Foundation**: Ready for gradients, GPU, and more

**Next Module**: Activation functions - adding the nonlinearity that makes neural networks powerful!

You've built the foundation of modern AI. Now let's add the mathematical functions that enable machines to learn complex patterns!
"""
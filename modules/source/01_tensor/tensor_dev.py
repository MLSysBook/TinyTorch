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

### The Mathematical Foundation: From Scalars to Tensors
Understanding tensors requires building from mathematical fundamentals:

#### **Scalars (Rank 0)**
- **Definition**: A single number with no direction
- **Examples**: Temperature (25°C), mass (5.2 kg), probability (0.7)
- **Operations**: Addition, multiplication, comparison
- **ML Context**: Loss values, learning rates, regularization parameters

#### **Vectors (Rank 1)**
- **Definition**: An ordered list of numbers with direction and magnitude
- **Examples**: Position [x, y, z], RGB color [255, 128, 0], word embedding [0.1, -0.5, 0.8]
- **Operations**: Dot product, cross product, norm calculation
- **ML Context**: Feature vectors, gradients, model parameters

#### **Matrices (Rank 2)**
- **Definition**: A 2D array organizing data in rows and columns
- **Examples**: Image (height × width), weight matrix (input × output), covariance matrix
- **Operations**: Matrix multiplication, transpose, inverse, eigendecomposition
- **ML Context**: Linear layer weights, attention matrices, batch data

#### **Higher-Order Tensors (Rank 3+)**
- **Definition**: Multi-dimensional arrays extending matrices
- **Examples**: 
  - **3D**: Video frames (time × height × width), RGB images (height × width × channels)
  - **4D**: Image batches (batch × height × width × channels)
  - **5D**: Video batches (batch × time × height × width × channels)
- **Operations**: Tensor products, contractions, decompositions
- **ML Context**: Convolutional features, RNN states, transformer attention

### Why Tensors Matter in ML: The Computational Foundation

#### **1. Unified Data Representation**
Tensors provide a consistent way to represent all ML data:
```python
# All of these are tensors with different shapes
scalar_loss = Tensor(0.5)              # Shape: ()
feature_vector = Tensor([1, 2, 3])      # Shape: (3,)
weight_matrix = Tensor([[1, 2], [3, 4]]) # Shape: (2, 2)
image_batch = Tensor(np.random.rand(32, 224, 224, 3)) # Shape: (32, 224, 224, 3)
```

#### **2. Efficient Batch Processing**
ML systems process multiple samples simultaneously:
```python
# Instead of processing one image at a time:
for image in images:
    result = model(image)  # Slow: 1000 separate operations

# Process entire batch at once:
batch_result = model(image_batch)  # Fast: 1 vectorized operation
```

#### **3. Hardware Acceleration**
Modern hardware (GPUs, TPUs) excels at tensor operations:
- **Parallel processing**: Multiple operations simultaneously
- **Vectorization**: SIMD (Single Instruction, Multiple Data) operations
- **Memory optimization**: Contiguous memory layout for cache efficiency

#### **4. Automatic Differentiation**
Tensors enable gradient computation through computational graphs:
```python
# Each tensor operation creates a node in the computation graph
x = Tensor([1, 2, 3])
y = x * 2          # Node: multiplication
z = y + 1          # Node: addition
loss = z.sum()     # Node: summation
# Gradients flow backward through this graph
```

### Real-World Examples: Tensors in Action

#### **Computer Vision**
- **Grayscale image**: 2D tensor `(height, width)` - `(28, 28)` for MNIST
- **Color image**: 3D tensor `(height, width, channels)` - `(224, 224, 3)` for RGB
- **Image batch**: 4D tensor `(batch, height, width, channels)` - `(32, 224, 224, 3)`
- **Video**: 5D tensor `(batch, time, height, width, channels)`

#### **Natural Language Processing**
- **Word embedding**: 1D tensor `(embedding_dim,)` - `(300,)` for Word2Vec
- **Sentence**: 2D tensor `(sequence_length, embedding_dim)` - `(50, 768)` for BERT
- **Batch of sentences**: 3D tensor `(batch, sequence_length, embedding_dim)`

#### **Audio Processing**
- **Audio signal**: 1D tensor `(time_steps,)` - `(16000,)` for 1 second at 16kHz
- **Spectrogram**: 2D tensor `(time_frames, frequency_bins)`
- **Batch of audio**: 3D tensor `(batch, time_steps, features)`

#### **Time Series**
- **Single series**: 2D tensor `(time_steps, features)`
- **Multiple series**: 3D tensor `(batch, time_steps, features)`
- **Multivariate forecasting**: 4D tensor `(batch, time_steps, features, predictions)`

### Why Not Just Use NumPy?

While we use NumPy internally, our Tensor class adds ML-specific functionality:

#### **1. ML-Specific Operations**
- **Gradient tracking**: For automatic differentiation (coming in Module 7)
- **GPU support**: For hardware acceleration (future extension)
- **Broadcasting semantics**: ML-friendly dimension handling

#### **2. Consistent API**
- **Type safety**: Predictable behavior across operations
- **Error checking**: Clear error messages for debugging
- **Integration**: Seamless work with other TinyTorch components

#### **3. Educational Value**
- **Conceptual clarity**: Understand what tensors really are
- **Implementation insight**: See how frameworks work internally
- **Debugging skills**: Trace through tensor operations step by step

#### **4. Extensibility**
- **Future features**: Ready for gradients, GPU, distributed computing
- **Customization**: Add domain-specific operations
- **Optimization**: Profile and optimize specific use cases

### Performance Considerations: Building Efficient Tensors

#### **Memory Layout**
- **Contiguous arrays**: Better cache locality and performance
- **Data types**: `float32` vs `float64` trade-offs
- **Memory sharing**: Avoid unnecessary copies

#### **Vectorization**
- **SIMD operations**: Single Instruction, Multiple Data
- **Broadcasting**: Efficient operations on different shapes
- **Batch operations**: Process multiple samples simultaneously

#### **Numerical Stability**
- **Precision**: Balancing speed and accuracy
- **Overflow/underflow**: Handling extreme values
- **Gradient flow**: Maintaining numerical stability for training

Let's start building our tensor foundation!
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

### Core Concept: Wrapping NumPy with ML Intelligence
Our Tensor class wraps NumPy arrays with ML-specific functionality. This design pattern is used by all major ML frameworks:

- **PyTorch**: `torch.Tensor` wraps ATen (C++ tensor library)
- **TensorFlow**: `tf.Tensor` wraps Eigen (C++ linear algebra library)
- **JAX**: `jax.numpy.ndarray` wraps XLA (Google's linear algebra compiler)
- **TinyTorch**: `Tensor` wraps NumPy (Python's numerical computing library)

### Design Requirements Analysis

#### **1. Input Flexibility**
Our tensor must handle diverse input types:
```python
# Scalars (Python numbers)
t1 = Tensor(5)           # int → numpy array
t2 = Tensor(3.14)        # float → numpy array

# Lists (Python sequences)
t3 = Tensor([1, 2, 3])   # list → numpy array
t4 = Tensor([[1, 2], [3, 4]])  # nested list → 2D array

# NumPy arrays (existing arrays)
t5 = Tensor(np.array([1, 2, 3]))  # array → tensor wrapper
```

#### **2. Type Management**
ML systems need consistent, predictable types:
- **Default behavior**: Auto-detect appropriate types
- **Explicit control**: Allow manual type specification
- **Performance optimization**: Prefer `float32` over `float64`
- **Memory efficiency**: Use appropriate precision

#### **3. Property Access**
Essential tensor properties for ML operations:
- **Shape**: Dimensions for compatibility checking
- **Size**: Total elements for memory estimation
- **Data type**: For numerical computation planning
- **Data access**: For integration with other libraries

#### **4. Arithmetic Operations**
Support for mathematical operations:
- **Element-wise**: Addition, multiplication, subtraction, division
- **Broadcasting**: Operations on different shapes
- **Type promotion**: Consistent result types
- **Error handling**: Clear messages for incompatible operations

### Implementation Strategy

#### **Memory Management**
- **Copy vs. Reference**: When to copy data vs. share memory
- **Type conversion**: Efficient dtype changes
- **Contiguous layout**: Ensure optimal memory access patterns

#### **Error Handling**
- **Input validation**: Check for valid input types
- **Shape compatibility**: Verify operations are mathematically valid
- **Informative messages**: Help users debug issues quickly

#### **Performance Optimization**
- **Lazy evaluation**: Defer expensive operations when possible
- **Vectorization**: Use NumPy's optimized operations
- **Memory reuse**: Minimize unnecessary allocations

### Learning Objectives for Implementation

By implementing this Tensor class, you'll learn:
1. **Wrapper pattern**: How to extend existing libraries
2. **Type system design**: Managing data types in numerical computing
3. **API design**: Creating intuitive, consistent interfaces
4. **Performance considerations**: Balancing flexibility and speed
5. **Error handling**: Providing helpful feedback to users

Let's implement our tensor foundation!
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
### 🧪 Unit Test: Tensor Creation

Let's test your tensor creation implementation right away! This gives you immediate feedback on whether your `__init__` method works correctly.

**This is a unit test** - it tests one specific function (tensor creation) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-creation-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor creation immediately after implementation
print("🔬 Unit Test: Tensor Creation...")

# Test basic tensor creation
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

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Properties

Now let's test that your tensor properties work correctly. This tests the @property methods you implemented.

**This is a unit test** - it tests specific properties (shape, size, dtype, data) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-properties-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor properties immediately after implementation
print("🔬 Unit Test: Tensor Properties...")

# Test properties with simple examples
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

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic

Let's test your tensor arithmetic operations. This tests the __add__, __mul__, __sub__, __truediv__ methods.

**This is a unit test** - it tests specific arithmetic operations in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-arithmetic-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test tensor arithmetic immediately after implementation
print("🔬 Unit Test: Tensor Arithmetic...")

# Test basic arithmetic with simple examples
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
    
    print("📈 Progress: Tensor Arithmetic ✓")
    
except Exception as e:
    print(f"❌ Tensor arithmetic test failed: {e}")
    raise

print("🎯 Tensor arithmetic behavior:")
print("   Element-wise operations on tensors")
print("   Broadcasting with scalars")
print("   Returns new Tensor objects")

# %% [markdown]
"""
### 🧪 Comprehensive Test: Tensor Creation

Let's thoroughly test your tensor creation to make sure it handles all the cases you'll encounter in ML.
This tests the foundation of everything else we'll build.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-creation-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_tensor_creation_comprehensive():
    """Comprehensive test of tensor creation with all data types and shapes."""
    print("🔬 Testing comprehensive tensor creation...")
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Scalar creation (0D tensor)
    try:
        scalar_int = Tensor(42)
        scalar_float = Tensor(3.14)
        scalar_zero = Tensor(0)
        
        assert hasattr(scalar_int, '_data'), "Tensor should have _data attribute"
        assert scalar_int._data.shape == (), f"Scalar should have shape (), got {scalar_int._data.shape}"
        assert scalar_float._data.shape == (), f"Float scalar should have shape (), got {scalar_float._data.shape}"
        assert scalar_zero._data.shape == (), f"Zero scalar should have shape (), got {scalar_zero._data.shape}"
        
        print("✅ Scalar creation: integers, floats, and zero")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Scalar creation failed: {e}")
    
    # Test 2: Vector creation (1D tensor)
    try:
        vector_int = Tensor([1, 2, 3, 4, 5])
        vector_float = Tensor([1.0, 2.5, 3.7])
        vector_single = Tensor([42])
        vector_empty = Tensor([])
        
        assert vector_int._data.shape == (5,), f"Int vector should have shape (5,), got {vector_int._data.shape}"
        assert vector_float._data.shape == (3,), f"Float vector should have shape (3,), got {vector_float._data.shape}"
        assert vector_single._data.shape == (1,), f"Single element vector should have shape (1,), got {vector_single._data.shape}"
        assert vector_empty._data.shape == (0,), f"Empty vector should have shape (0,), got {vector_empty._data.shape}"
        
        print("✅ Vector creation: integers, floats, single element, and empty")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Vector creation failed: {e}")
    
    # Test 3: Matrix creation (2D tensor)
    try:
        matrix_2x2 = Tensor([[1, 2], [3, 4]])
        matrix_3x2 = Tensor([[1, 2], [3, 4], [5, 6]])
        matrix_1x3 = Tensor([[1, 2, 3]])
        
        assert matrix_2x2._data.shape == (2, 2), f"2x2 matrix should have shape (2, 2), got {matrix_2x2._data.shape}"
        assert matrix_3x2._data.shape == (3, 2), f"3x2 matrix should have shape (3, 2), got {matrix_3x2._data.shape}"
        assert matrix_1x3._data.shape == (1, 3), f"1x3 matrix should have shape (1, 3), got {matrix_1x3._data.shape}"
        
        print("✅ Matrix creation: 2x2, 3x2, and 1x3 matrices")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Matrix creation failed: {e}")
    
    # Test 4: Data type handling
    try:
        int_tensor = Tensor([1, 2, 3])
        float_tensor = Tensor([1.0, 2.0, 3.0])
        mixed_tensor = Tensor([1, 2.5, 3])  # Should convert to float
        
        # Check that data types are reasonable
        assert int_tensor._data.dtype in [np.int32, np.int64], f"Int tensor has unexpected dtype: {int_tensor._data.dtype}"
        assert float_tensor._data.dtype in [np.float32, np.float64], f"Float tensor has unexpected dtype: {float_tensor._data.dtype}"
        assert mixed_tensor._data.dtype in [np.float32, np.float64], f"Mixed tensor should be float, got: {mixed_tensor._data.dtype}"
        
        print("✅ Data type handling: integers, floats, and mixed types")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data type handling failed: {e}")
    
    # Test 5: NumPy array input
    try:
        np_array = np.array([1, 2, 3, 4])
        tensor_from_np = Tensor(np_array)
        
        assert tensor_from_np._data.shape == (4,), f"Tensor from NumPy should have shape (4,), got {tensor_from_np._data.shape}"
        assert np.array_equal(tensor_from_np._data, np_array), "Tensor from NumPy should preserve data"
        
        print("✅ NumPy array input: conversion works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ NumPy array input failed: {e}")
    
    # Test 6: Large tensor creation
    try:
        large_tensor = Tensor(list(range(1000)))
        assert large_tensor._data.shape == (1000,), f"Large tensor should have shape (1000,), got {large_tensor._data.shape}"
        assert large_tensor._data[0] == 0, "Large tensor should start with 0"
        assert large_tensor._data[-1] == 999, "Large tensor should end with 999"
        
        print("✅ Large tensor creation: 1000 elements")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Large tensor creation failed: {e}")
    
    # Test 7: Negative numbers
    try:
        negative_tensor = Tensor([-1, -2, -3])
        mixed_signs = Tensor([-1, 0, 1])
        
        assert negative_tensor._data.shape == (3,), f"Negative tensor should have shape (3,), got {negative_tensor._data.shape}"
        assert np.array_equal(negative_tensor._data, np.array([-1, -2, -3])), "Negative numbers should be preserved"
        assert np.array_equal(mixed_signs._data, np.array([-1, 0, 1])), "Mixed signs should be preserved"
        
        print("✅ Negative numbers: handled correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Negative numbers failed: {e}")
    
    # Test 8: Edge cases
    try:
        # Very large numbers
        big_tensor = Tensor([1e6, 1e-6])
        assert big_tensor._data.shape == (2,), "Big numbers tensor should have correct shape"
        
        # Zero tensor
        zero_tensor = Tensor([0, 0, 0])
        assert np.all(zero_tensor._data == 0), "Zero tensor should contain all zeros"
        
        print("✅ Edge cases: large numbers and zeros")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Edge cases failed: {e}")
    
    # Results summary
    print(f"\n📊 Tensor Creation Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tensor creation tests passed! Your Tensor class can handle:")
        print("  • Scalars, vectors, and matrices")
        print("  • Different data types (int, float)")
        print("  • NumPy arrays")
        print("  • Large tensors and edge cases")
        print("📈 Progress: Tensor Creation ✓")
        return True
    else:
        print("⚠️  Some tensor creation tests failed. Common issues:")
        print("  • Check your __init__ method implementation")
        print("  • Make sure you're storing data in self._data")
        print("  • Verify NumPy array conversion works correctly")
        print("  • Test with different input types (int, float, list, np.array)")
        return False

# Run the comprehensive test
success = test_tensor_creation_comprehensive()

# %% [markdown]
"""
### 🧪 Comprehensive Test: Tensor Properties

Now let's test all the properties your tensor should have. These properties are essential for ML operations.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-properties-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_tensor_properties_comprehensive():
    """Comprehensive test of tensor properties (shape, size, dtype, data access)."""
    print("🔬 Testing comprehensive tensor properties...")
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Shape property
    try:
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])
        matrix = Tensor([[1, 2], [3, 4]])
        tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        assert scalar.shape == (), f"Scalar shape should be (), got {scalar.shape}"
        assert vector.shape == (3,), f"Vector shape should be (3,), got {vector.shape}"
        assert matrix.shape == (2, 2), f"Matrix shape should be (2, 2), got {matrix.shape}"
        assert tensor_3d.shape == (2, 2, 2), f"3D tensor shape should be (2, 2, 2), got {tensor_3d.shape}"
        
        print("✅ Shape property: scalar, vector, matrix, and 3D tensor")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Shape property failed: {e}")
    
    # Test 2: Size property
    try:
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])
        matrix = Tensor([[1, 2], [3, 4]])
        empty = Tensor([])
        
        assert scalar.size == 1, f"Scalar size should be 1, got {scalar.size}"
        assert vector.size == 3, f"Vector size should be 3, got {vector.size}"
        assert matrix.size == 4, f"Matrix size should be 4, got {matrix.size}"
        assert empty.size == 0, f"Empty tensor size should be 0, got {empty.size}"
        
        print("✅ Size property: scalar, vector, matrix, and empty tensor")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Size property failed: {e}")
    
    # Test 3: Data type property
    try:
        int_tensor = Tensor([1, 2, 3])
        float_tensor = Tensor([1.0, 2.0, 3.0])
        
        # Check that dtype is accessible and reasonable
        assert hasattr(int_tensor, 'dtype'), "Tensor should have dtype property"
        assert hasattr(float_tensor, 'dtype'), "Tensor should have dtype property"
        
        # Data types should be NumPy dtypes
        assert isinstance(int_tensor.dtype, np.dtype), f"dtype should be np.dtype, got {type(int_tensor.dtype)}"
        assert isinstance(float_tensor.dtype, np.dtype), f"dtype should be np.dtype, got {type(float_tensor.dtype)}"
        
        print(f"✅ Data type property: int tensor is {int_tensor.dtype}, float tensor is {float_tensor.dtype}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data type property failed: {e}")
    
    # Test 4: Data access property
    try:
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])
        matrix = Tensor([[1, 2], [3, 4]])
        
        # Test data access
        assert hasattr(scalar, 'data'), "Tensor should have data property"
        assert hasattr(vector, 'data'), "Tensor should have data property"
        assert hasattr(matrix, 'data'), "Tensor should have data property"
        
        # Test data content
        assert scalar.data.item() == 5.0, f"Scalar data should be 5.0, got {scalar.data.item()}"
        assert np.array_equal(vector.data, np.array([1, 2, 3])), "Vector data mismatch"
        assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]])), "Matrix data mismatch"
        
        print("✅ Data access: scalar, vector, and matrix data retrieval")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data access failed: {e}")
    
    # Test 5: String representation
    try:
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])
        
        # Test that __repr__ works
        scalar_str = str(scalar)
        vector_str = str(vector)
        
        assert isinstance(scalar_str, str), "Tensor string representation should be a string"
        assert isinstance(vector_str, str), "Tensor string representation should be a string"
        assert len(scalar_str) > 0, "Tensor string representation should not be empty"
        assert len(vector_str) > 0, "Tensor string representation should not be empty"
        
        print(f"✅ String representation: scalar={scalar_str[:50]}{'...' if len(scalar_str) > 50 else ''}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ String representation failed: {e}")
    
    # Test 6: Property consistency
    try:
        test_cases = [
            Tensor(42),
            Tensor([1, 2, 3, 4, 5]),
            Tensor([[1, 2, 3], [4, 5, 6]]),
            Tensor([])
        ]
        
        for i, tensor in enumerate(test_cases):
            # Size should equal product of shape
            expected_size = np.prod(tensor.shape) if tensor.shape else 1
            assert tensor.size == expected_size, f"Test case {i}: size {tensor.size} doesn't match shape {tensor.shape}"
            
            # Data shape should match tensor shape
            assert tensor.data.shape == tensor.shape, f"Test case {i}: data shape {tensor.data.shape} doesn't match tensor shape {tensor.shape}"
        
        print("✅ Property consistency: size matches shape, data shape matches tensor shape")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Property consistency failed: {e}")
    
    # Results summary
    print(f"\n📊 Tensor Properties Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tensor property tests passed! Your tensor has:")
        print("  • Correct shape property for all dimensions")
        print("  • Accurate size calculation")
        print("  • Proper data type handling")
        print("  • Working data access")
        print("  • Good string representation")
        print("📈 Progress: Tensor Creation ✓, Properties ✓")
        return True
    else:
        print("⚠️  Some property tests failed. Common issues:")
        print("  • Check your @property decorators")
        print("  • Verify shape returns self._data.shape")
        print("  • Make sure size returns self._data.size")
        print("  • Ensure dtype returns self._data.dtype")
        print("  • Test your __repr__ method")
        return False

# Run the comprehensive test
success = test_tensor_properties_comprehensive() and success

# %% [markdown]
"""
### 🧪 Comprehensive Test: Tensor Arithmetic

Let's test all arithmetic operations. These are the foundation of neural network computations!
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-arithmetic-comprehensive", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_tensor_arithmetic_comprehensive():
    """Comprehensive test of tensor arithmetic operations."""
    print("🔬 Testing comprehensive tensor arithmetic...")
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Basic addition method
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a.add(b)
        
        expected = np.array([5, 7, 9])
        assert np.array_equal(c.data, expected), f"Addition method failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "Addition should return a Tensor"
        
        print(f"✅ Addition method: {a.data} + {b.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Addition method failed: {e}")
    
    # Test 2: Basic multiplication method
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a.multiply(b)
        
        expected = np.array([4, 10, 18])
        assert np.array_equal(c.data, expected), f"Multiplication method failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "Multiplication should return a Tensor"
        
        print(f"✅ Multiplication method: {a.data} * {b.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Multiplication method failed: {e}")
    
    # Test 3: Addition operator (+)
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        
        expected = np.array([5, 7, 9])
        assert np.array_equal(c.data, expected), f"+ operator failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "+ operator should return a Tensor"
        
        print(f"✅ + operator: {a.data} + {b.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ + operator failed: {e}")
    
    # Test 4: Multiplication operator (*)
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        
        expected = np.array([4, 10, 18])
        assert np.array_equal(c.data, expected), f"* operator failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "* operator should return a Tensor"
        
        print(f"✅ * operator: {a.data} * {b.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ * operator failed: {e}")
    
    # Test 5: Subtraction operator (-)
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = b - a
        
        expected = np.array([3, 3, 3])
        assert np.array_equal(c.data, expected), f"- operator failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "- operator should return a Tensor"
        
        print(f"✅ - operator: {b.data} - {a.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ - operator failed: {e}")
    
    # Test 6: Division operator (/)
    try:
        a = Tensor([1, 2, 4])
        b = Tensor([2, 4, 8])
        c = b / a
        
        expected = np.array([2.0, 2.0, 2.0])
        assert np.allclose(c.data, expected), f"/ operator failed: expected {expected}, got {c.data}"
        assert isinstance(c, Tensor), "/ operator should return a Tensor"
        
        print(f"✅ / operator: {b.data} / {a.data} = {c.data}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ / operator failed: {e}")
    
    # Test 7: Scalar operations
    try:
        a = Tensor([1, 2, 3])
        
        # Addition with scalar
        b = a + 10
        expected_add = np.array([11, 12, 13])
        assert np.array_equal(b.data, expected_add), f"Scalar addition failed: expected {expected_add}, got {b.data}"
        
        # Multiplication with scalar
        c = a * 2
        expected_mul = np.array([2, 4, 6])
        assert np.array_equal(c.data, expected_mul), f"Scalar multiplication failed: expected {expected_mul}, got {c.data}"
        
        # Subtraction with scalar
        d = a - 1
        expected_sub = np.array([0, 1, 2])
        assert np.array_equal(d.data, expected_sub), f"Scalar subtraction failed: expected {expected_sub}, got {d.data}"
        
        # Division with scalar
        e = a / 2
        expected_div = np.array([0.5, 1.0, 1.5])
        assert np.allclose(e.data, expected_div), f"Scalar division failed: expected {expected_div}, got {e.data}"
        
        print(f"✅ Scalar operations: +10, *2, -1, /2 all work correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Scalar operations failed: {e}")
    
    # Test 8: Matrix operations
    try:
        matrix_a = Tensor([[1, 2], [3, 4]])
        matrix_b = Tensor([[5, 6], [7, 8]])
        
        # Matrix addition
        c = matrix_a + matrix_b
        expected = np.array([[6, 8], [10, 12]])
        assert np.array_equal(c.data, expected), f"Matrix addition failed: expected {expected}, got {c.data}"
        assert c.shape == (2, 2), f"Matrix addition should preserve shape, got {c.shape}"
        
        # Matrix multiplication (element-wise)
        d = matrix_a * matrix_b
        expected_mul = np.array([[5, 12], [21, 32]])
        assert np.array_equal(d.data, expected_mul), f"Matrix multiplication failed: expected {expected_mul}, got {d.data}"
        
        print(f"✅ Matrix operations: 2x2 matrix addition and multiplication")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Matrix operations failed: {e}")
    
    # Results summary
    print(f"\n📊 Tensor Arithmetic Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tensor arithmetic tests passed! Your tensor supports:")
        print("  • Basic methods: add(), multiply()")
        print("  • Python operators: +, -, *, /")
        print("  • Scalar operations: tensor + number")
        print("  • Matrix operations: element-wise operations")
        print("📈 Progress: Tensor Creation ✓, Properties ✓, Arithmetic ✓")
        return True
    else:
        print("⚠️  Some arithmetic tests failed. Common issues:")
        print("  • Check your add() and multiply() methods")
        print("  • Verify operator overloading (__add__, __mul__, __sub__, __truediv__)")
        print("  • Make sure scalar operations work (convert scalar to Tensor)")
        print("  • Test with different tensor shapes")
        return False

# Run the comprehensive test
success = test_tensor_arithmetic_comprehensive() and success

# %% [markdown]
"""
### 🧪 Comprehensive Test: Real ML Scenario

Let's test your tensor with a realistic machine learning scenario to make sure everything works together.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-comprehensive", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_tensor_comprehensive():
    """Comprehensive test with realistic ML scenario."""
    print("🔬 Testing tensor comprehensively with ML scenario...")
    
    try:
        print("🧠 Simulating a simple neural network forward pass...")
        
        # Simulate input data (batch of 2 samples, 3 features each)
        X = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print(f"📊 Input data shape: {X.shape}")
        
        # Simulate weights (3 input features, 2 output neurons)
        W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        print(f"🎯 Weights shape: {W.shape}")
        
        # Simulate bias (2 output neurons)
        b = Tensor([0.1, 0.2])
        print(f"⚖️  Bias shape: {b.shape}")
        
        # Simple linear transformation: y = X * W + b
        # Note: This is a simplified version - real matrix multiplication would be different
        # But we can test element-wise operations
        
        # Test that we can do basic operations needed for ML
        sample = Tensor([1.0, 2.0, 3.0])  # Single sample
        weight_col = Tensor([0.1, 0.3, 0.5])  # First column of weights
        
        # Compute dot product manually using element-wise operations
        products = sample * weight_col  # Element-wise multiplication
        print(f"✅ Element-wise multiplication works: {products.data}")
        
        # Test addition for bias
        result = products + Tensor([0.1, 0.1, 0.1])
        print(f"✅ Bias addition works: {result.data}")
        
        # Test with different shapes
        matrix_a = Tensor([[1, 2], [3, 4]])
        matrix_b = Tensor([[0.1, 0.2], [0.3, 0.4]])
        matrix_result = matrix_a * matrix_b
        print(f"✅ Matrix operations work: {matrix_result.data}")
        
        # Test scalar operations (common in ML)
        scaled = sample * 0.5  # Learning rate scaling
        print(f"✅ Scalar scaling works: {scaled.data}")
        
        # Test normalization-like operations
        mean_val = Tensor([2.0, 2.0, 2.0])  # Simulate mean
        normalized = sample - mean_val
        print(f"✅ Mean subtraction works: {normalized.data}")
        
        print("\n🎉 Comprehensive test passed! Your tensor class can handle:")
        print("  • Multi-dimensional data (batches, features)")
        print("  • Element-wise operations needed for ML")
        print("  • Scalar operations (learning rates, normalization)")
        print("  • Matrix operations (weights, transformations)")
        print("📈 Progress: All tensor functionality ✓")
        print("🚀 Ready for neural network layers!")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        print("\n💡 This suggests an issue with:")
        print("  • Basic tensor operations not working together")
        print("  • Shape handling problems")
        print("  • Arithmetic operation implementation")
        print("  • Check your tensor creation and arithmetic methods")
        return False

# Run the comprehensive test
success = test_tensor_comprehensive() and success

# Print final summary
print(f"\n{'='*60}")
print("🎯 TENSOR MODULE TESTING COMPLETE")
print(f"{'='*60}")

if success:
    print("🎉 CONGRATULATIONS! All tensor tests passed!")
    print("\n✅ Your Tensor class successfully implements:")
    print("  • Comprehensive tensor creation (scalars, vectors, matrices)")
    print("  • All essential properties (shape, size, dtype, data access)")
    print("  • Complete arithmetic operations (methods and operators)")
    print("  • Scalar and matrix operations")
    print("  • Real ML scenario compatibility")
    print("\n🚀 You're ready to move to the next module!")
    print("📈 Final Progress: Tensor Module ✓ COMPLETE")
else:
    print("⚠️  Some tests failed. Please review the error messages above.")
    print("\n🔧 To fix issues:")
    print("  1. Check the specific test that failed")
    print("  2. Review the error message and hints")
    print("  3. Fix your implementation")
    print("  4. Re-run the notebook cells")
    print("\n💪 Don't give up! Debugging is part of learning.")

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
print("🔬 Unit Test: Tensor Creation...")

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
print("🔬 Unit Test: Tensor Arithmetic...")

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
print("🔬 Unit Test: Tensor Broadcasting...")

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
"""

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Tensor")

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
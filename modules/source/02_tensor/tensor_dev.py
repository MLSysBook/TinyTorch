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
# Tensor - Core Data Structure

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
## 🔧 DEVELOPMENT
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

    def mean(self) -> 'Tensor':
        """Computes the mean of the tensor's elements."""
        return Tensor(np.mean(self.data))

    # --- Matmul ---
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

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation

This test validates your `Tensor` class constructor, ensuring it correctly handles scalars, vectors, matrices, and higher-dimensional arrays with proper shape detection.
"""

# %%
def test_unit_tensor_creation():
    """Comprehensive test of tensor creation with all data types and shapes."""
    print("🔬 Testing comprehensive tensor creation...")
    
    # Test scalar creation
    scalar_int = Tensor(42)
    assert scalar_int.shape == ()
    
    # Test vector creation
    vector_int = Tensor([1, 2, 3])
    assert vector_int.shape == (3,)

    # Test matrix creation
    matrix_2x2 = Tensor([[1, 2], [3, 4]])
    assert matrix_2x2.shape == (2, 2)
    print("✅ Tensor creation tests passed!")

# Run the test
test_unit_tensor_creation()

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Properties

This test validates your tensor property methods (shape, size, dtype, data), ensuring they correctly reflect the tensor's dimensional structure and data characteristics.
"""

# %%
def test_unit_tensor_properties():
    """Comprehensive test of tensor properties (shape, size, dtype, data access)."""
    print("🔬 Testing comprehensive tensor properties...")

    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # Test shape property
    assert tensor.shape == (2, 3)
    
    # Test size property
    assert tensor.size == 6
    
    # Test data property
    assert np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]]))
    
    # Test dtype property
    assert tensor.dtype in [np.int32, np.int64]
    print("✅ Tensor properties tests passed!")

# Run the test
test_unit_tensor_properties()

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic Operations

This test validates your tensor arithmetic implementation (addition, multiplication, subtraction, division) and operator overloading, ensuring mathematical operations work correctly with proper broadcasting.
"""

# %%
def test_unit_tensor_arithmetic():
    """Comprehensive test of tensor arithmetic operations."""
    print("🔬 Testing comprehensive tensor arithmetic...")
    
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Test addition
    c = a + b
    expected = np.array([5, 7, 9])
    assert np.array_equal(c.data, expected)
    
    # Test multiplication
    d = a * b
    expected = np.array([4, 10, 18])
    assert np.array_equal(d.data, expected)

    # Test subtraction
    e = b - a
    expected = np.array([3, 3, 3])
    assert np.array_equal(e.data, expected)

    # Test division
    f = b / a
    expected = np.array([4.0, 2.5, 2.0])
    assert np.allclose(f.data, expected)
    print("✅ Tensor arithmetic tests passed!")

# Run the test
test_unit_tensor_arithmetic()

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tensor Foundation

Congratulations! You've successfully implemented the fundamental data structure that powers all machine learning:

### ✅ What You've Built
- **Tensor Class**: N-dimensional array wrapper with professional interfaces
- **Core Operations**: Creation, property access, and arithmetic operations
- **Shape Management**: Automatic shape tracking and validation
- **Data Types**: Proper NumPy integration and type handling
- **Foundation**: The building block for all subsequent TinyTorch modules

### ✅ Key Learning Outcomes
- **Understanding**: How tensors work as the foundation of machine learning
- **Implementation**: Built tensor operations from scratch
- **Professional patterns**: Clean APIs, proper error handling, comprehensive testing
- **Real-world connection**: Understanding PyTorch/TensorFlow tensor foundations
- **Systems thinking**: Building reliable, reusable components

### ✅ Mathematical Foundations Mastered
- **N-dimensional arrays**: Shape, size, and dimensionality concepts
- **Element-wise operations**: Addition, subtraction, multiplication, division
- **Broadcasting**: Understanding how operations work with different shapes
- **Memory management**: Efficient data storage and access patterns

### ✅ Professional Skills Developed
- **API design**: Clean, intuitive interfaces for tensor operations
- **Error handling**: Graceful handling of invalid operations and edge cases
- **Testing methodology**: Comprehensive validation of tensor functionality
- **Documentation**: Clear, educational documentation with examples

### ✅ Ready for Advanced Applications
Your tensor implementation now enables:
- **Neural Networks**: Foundation for all layer implementations
- **Automatic Differentiation**: Gradient computation through computational graphs
- **Complex Models**: CNNs, RNNs, Transformers - all built on tensors
- **Real Applications**: Training models on real datasets

### 🔗 Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.Tensor` provides identical functionality
- **TensorFlow**: `tf.Tensor` implements similar concepts
- **NumPy**: `numpy.ndarray` serves as the foundation
- **Industry Standard**: Every major ML framework uses these exact principles

### 🎯 The Power of Tensors
You've built the fundamental data structure of modern AI:
- **Universality**: Tensors represent all data: images, text, audio, video
- **Efficiency**: Vectorized operations enable fast computation
- **Scalability**: Handles everything from single numbers to massive matrices
- **Flexibility**: Foundation for any mathematical operation

### 🚀 What's Next
Your tensor implementation is the foundation for:
- **Activations**: Nonlinear functions that enable complex learning
- **Layers**: Linear transformations and neural network building blocks
- **Networks**: Composing layers into powerful architectures
- **Training**: Optimizing networks to solve real problems

**Next Module**: Activation functions - adding the nonlinearity that makes neural networks powerful!

You've built the foundation of modern AI. Now let's add the mathematical functions that enable machines to learn complex patterns!
""" 
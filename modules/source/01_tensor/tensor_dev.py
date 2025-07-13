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

### Why Tensors Are Essential in ML
- **Efficiency**: Vectorized operations are 10-100x faster than loops
- **Flexibility**: Same operations work on scalars, vectors, matrices
- **Foundation**: Every ML framework (PyTorch, TensorFlow, JAX) uses tensors
- **Hardware**: Modern GPUs are designed for tensor operations

### Real-World Examples
```python
# Image: 3D tensor (height, width, channels)
image = Tensor(np.random.rand(224, 224, 3))

# Batch of images: 4D tensor (batch, height, width, channels)
batch = Tensor(np.random.rand(32, 224, 224, 3))

# Neural network weights: 2D tensor (input_size, output_size)
weights = Tensor(np.random.rand(784, 128))
```

### Learning Progression
We'll build tensors step by step:
1. **Creation**: Handle different input types
2. **Properties**: Access shape, size, data type
3. **Arithmetic**: Add, multiply, subtract, divide
4. **Integration**: Work with other TinyTorch components

Let's start building!
"""

# %% [markdown]
"""
## Step 1A: Tensor Creation

### The Foundation Operation
Creating tensors is the first thing you'll do in any ML system. Our Tensor class needs to:

1. **Accept various input types**: scalars, lists, numpy arrays
2. **Handle data types**: integers, floats, with automatic type management
3. **Store data efficiently**: using NumPy arrays internally
4. **Validate inputs**: ensure data is numeric and well-formed

### Design Decisions
- **Wrap NumPy**: Use NumPy arrays internally for performance
- **Consistent API**: Similar to PyTorch's `torch.tensor()` 
- **Type management**: Prefer float32 for ML (GPU efficiency)
- **Memory efficiency**: Share data when possible, copy when needed

### Real-World Context
Every ML framework does this:
- **PyTorch**: `torch.tensor([1, 2, 3])`
- **TensorFlow**: `tf.constant([1, 2, 3])`
- **JAX**: `jnp.array([1, 2, 3])`
- **TinyTorch**: `Tensor([1, 2, 3])`
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-creation", "locked": false, "schema_version": 3, "solution": true, "task": false}
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
        
        TODO: Implement tensor creation with proper type handling.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Handle different input types (int, float, list, numpy array)
        2. Convert input to numpy array using np.array()
        3. Apply dtype conversion if specified, otherwise use smart defaults
        4. Store the result in self._data
        5. Validate that the data is numeric
        
        EXAMPLE USAGE:
        ```python
        # From scalar
        t1 = Tensor(5.0)              # Creates 0D tensor
        
        # From list
        t2 = Tensor([1, 2, 3])        # Creates 1D tensor
        
        # From nested list (matrix)
        t3 = Tensor([[1, 2], [3, 4]]) # Creates 2D tensor
        
        # With specific dtype
        t4 = Tensor([1, 2, 3], dtype='float32')
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.array(data) for basic conversion
        - Handle dtype parameter: if provided, use np.array(data, dtype=dtype)
        - For smart defaults: integers → int32, floats → float32
        - Store in self._data for internal use
        - Validate: ensure data is numeric using np.issubdtype(dtype, np.number)
        
        LEARNING CONNECTIONS:
        - This is like torch.tensor() in PyTorch
        - Similar to tf.constant() in TensorFlow
        - Foundation for all tensor operations
        - Every ML computation starts with tensor creation
        """
        ### BEGIN SOLUTION
        # Convert input to numpy array
        if isinstance(data, (int, float, np.number)):
            # Handle Python and NumPy scalars
            if dtype is None:
                # Auto-detect type: int32 for integers, float32 for floats
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
            self._data = data.astype(dtype) if dtype != str(data.dtype) else data.copy()
        else:
            # Try to convert unknown types
            self._data = np.array(data, dtype=dtype)
        
        # Validate that data is numeric
        if not np.issubdtype(self._data.dtype, np.number):
            raise ValueError(f"Tensor data must be numeric, got {self._data.dtype}")
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Tensor Creation

Once you implement the `__init__` method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-creation", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_tensor_creation():
    """Test tensor creation with various input types"""
    print("Testing tensor creation...")
    
    # Test scalar creation
    t1 = Tensor(5.0)
    assert t1._data.shape == (), "Scalar tensor should have empty shape"
    assert t1._data.item() == 5.0, "Scalar value should be 5.0"
    
    # Test list creation
    t2 = Tensor([1, 2, 3])
    assert t2._data.shape == (3,), "1D tensor should have shape (3,)"
    assert np.array_equal(t2._data, [1, 2, 3]), "1D tensor values should match"
    
    # Test matrix creation
    t3 = Tensor([[1, 2], [3, 4]])
    assert t3._data.shape == (2, 2), "2D tensor should have shape (2, 2)"
    assert np.array_equal(t3._data, [[1, 2], [3, 4]]), "2D tensor values should match"
    
    # Test dtype specification
    t4 = Tensor([1, 2, 3], dtype='float32')
    assert t4._data.dtype == np.float32, "Specified dtype should be respected"
    
    # Test error handling
    try:
        Tensor(["invalid", "data"])
        assert False, "Should raise error for string data"
    except ValueError as e:
        assert "numeric" in str(e), "Error message should mention numeric requirement"
    
    print("✅ Tensor creation tests passed!")
    print(f"✅ Created tensors: scalar, vector, matrix")
    print(f"✅ Handled data types correctly")
    print(f"✅ Validated input data properly")

# Run the test
test_tensor_creation()

# %% [markdown]
"""
## Step 1B: Tensor Properties

### Essential Information Access
Every tensor needs to provide basic information about itself:

- **Shape**: Dimensions of the tensor (crucial for operations)
- **Size**: Total number of elements (important for memory)
- **Data type**: What kind of numbers we're storing
- **Data access**: Get the underlying NumPy array

### Why Properties Matter in ML
- **Debugging**: Quickly inspect tensor dimensions during development
- **Validation**: Check compatibility before operations
- **Memory management**: Understand storage requirements
- **Performance**: Optimize operations based on tensor characteristics

### Design Patterns
We use Python properties (@property) for clean access:
```python
t = Tensor([[1, 2], [3, 4]])
print(t.shape)  # (2, 2) - clean access
print(t.size)   # 4 - total elements
print(t.dtype)  # int32 - data type
```
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-properties", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
    @property
    def data(self) -> np.ndarray:
        """
        Get the underlying numpy array data.
        
        TODO: Implement data property access.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Return self._data directly
        2. This gives users access to the underlying NumPy array
        
        EXAMPLE USAGE:
        ```python
        t = Tensor([[1, 2], [3, 4]])
        print(t.data)  # [[1 2]
                          #  [3 4]]
        ```
        
        HINTS:
        - Properties provide clean access to internal data
        - Users can access the numpy array directly via this property
        - This is how PyTorch's .data property works
        """
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape (dimensions) of the tensor.
        
        TODO: Implement shape property.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Return self._data.shape
        2. This gives the dimensions as a tuple of integers
        
        EXAMPLE USAGE:
        ```python
        t = Tensor([[1, 2], [3, 4]])
        print(t.shape)  # (2, 2)
        
        # For different dimensions:
        scalar = Tensor(5)        # shape: ()
        vector = Tensor([1, 2])   # shape: (2,)
        matrix = Tensor([[1, 2], [3, 4]])  # shape: (2, 2)
        ```
        
        HINTS:
        - NumPy arrays have a .shape attribute
        - Shape is a tuple of integers
        - Essential for checking tensor compatibility
        """
        return self._data.shape

    @property
    def size(self) -> int:
        """
        Get the total number of elements in the tensor.
        
        TODO: Implement size property.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Return self._data.size
        2. This gives the total count of elements
        
        EXAMPLE USAGE:
        ```python
        t = Tensor([[1, 2], [3, 4]])
        print(t.size)  # 4 (2 * 2 elements)
        
        # For different shapes:
        scalar = Tensor(5)        # size: 1
        vector = Tensor([1, 2])   # size: 2
        matrix = Tensor([[1, 2], [3, 4]])  # size: 4
        ```
        
        HINTS:
        - NumPy arrays have a .size attribute
        - Size is the product of all dimensions
        - Important for memory calculations
        """
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the tensor elements.
        
        TODO: Implement dtype property.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Return self._data.dtype
        2. This gives the NumPy data type
        
        EXAMPLE USAGE:
        ```python
        t = Tensor([1, 2, 3])
        print(t.dtype)  # int64 (default integer type)
        
        t_float = Tensor([1.5, 2.5, 3.5])
        print(t_float.dtype)  # float64 (default float type)
        
        t_custom = Tensor([1, 2, 3], dtype='float32')
        print(t_custom.dtype)  # float32 (specified type)
        ```
        
        HINTS:
        - NumPy arrays have a .dtype attribute
        - Important for precision and memory usage
        - Affects computation speed and accuracy
        """
        return self._data.dtype

    def __repr__(self) -> str:
        """
        String representation of the tensor.
        
        TODO: Create a clear, informative string representation.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert the numpy array to a list for readable output
        2. Include shape and dtype information
        3. Format as: "Tensor(data, shape=shape, dtype=dtype)"
        
        EXAMPLE USAGE:
        ```python
        t = Tensor([1, 2, 3])
        print(t)  # Tensor([1, 2, 3], shape=(3,), dtype=int32)
        
        t2 = Tensor([[1, 2], [3, 4]])
        print(t2)  # Tensor([[1, 2], [3, 4]], shape=(2, 2), dtype=int32)
        ```
        
        IMPLEMENTATION HINTS:
        - Use .tolist() to convert numpy array to Python list
        - Include shape and dtype for debugging
        - Keep format consistent and readable
        - Use f-string formatting for clean output
        
        LEARNING CONNECTIONS:
        - Good string representation aids debugging
        - Shows essential information at a glance
        - Similar to PyTorch's tensor representation
        """
        ### BEGIN SOLUTION
        return f"Tensor({self._data.tolist()}, shape={self.shape}, dtype={self.dtype})"
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Tensor Properties

Once you implement the properties above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-properties", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_tensor_properties():
    """Test tensor properties: data, shape, size, dtype, repr"""
    print("Testing tensor properties...")
    
    # Test scalar properties
    t1 = Tensor(5.0)
    assert t1.shape == (), "Scalar shape should be empty tuple"
    assert t1.size == 1, "Scalar size should be 1"
    assert t1.data.item() == 5.0, "Scalar data should be accessible"
    assert "5.0" in str(t1), "String representation should show value"
    
    # Test vector properties
    t2 = Tensor([1, 2, 3, 4])
    assert t2.shape == (4,), "Vector shape should be (4,)"
    assert t2.size == 4, "Vector size should be 4"
    assert np.array_equal(t2.data, [1, 2, 3, 4]), "Vector data should match"
    
    # Test matrix properties
    t3 = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t3.shape == (2, 3), "Matrix shape should be (2, 3)"
    assert t3.size == 6, "Matrix size should be 6"
    assert np.array_equal(t3.data, [[1, 2, 3], [4, 5, 6]]), "Matrix data should match"
    
    # Test dtype
    t4 = Tensor([1, 2, 3], dtype='float32')
    assert t4.dtype == np.float32, "Should have float32 dtype"
    
    # Test string representation
    t5 = Tensor([1, 2])
    repr_str = str(t5)
    assert "Tensor" in repr_str, "Should contain 'Tensor'"
    assert "shape=" in repr_str, "Should show shape"
    assert "dtype=" in repr_str, "Should show dtype"
    
    print("✅ Tensor properties tests passed!")
    print(f"✅ Shape, size, and data access working correctly")
    print(f"✅ String representation is informative")
    print(f"✅ Data types are properly handled")

# Run the test
test_tensor_properties()

# %% [markdown]
"""
## Step 2: Tensor Arithmetic

### The Heart of ML: Mathematical Operations
Now we implement the core mathematical operations that make ML possible:

- **Addition**: Element-wise addition of tensors
- **Multiplication**: Element-wise multiplication  
- **Subtraction**: Element-wise subtraction
- **Division**: Element-wise division

### Why Arithmetic Matters in ML
- **Neural networks**: Every layer uses tensor arithmetic
- **Forward pass**: Data flows through arithmetic operations
- **Gradient computation**: Backpropagation relies on arithmetic
- **Optimization**: Parameter updates use arithmetic
- **Data processing**: Normalization, scaling, transformations

### Broadcasting Magic
NumPy's broadcasting allows operations between different shapes:
```python
# Same shape: element-wise
[1, 2] + [3, 4] = [4, 6]

# Broadcasting: smaller tensor expands
[[1, 2], [3, 4]] + [10, 20] = [[11, 22], [13, 24]]
```

### Real-World Context
This is used everywhere in ML:
- **Layer operations**: `output = input @ weights + bias`
- **Activation functions**: `relu(x) = max(0, x)`
- **Loss computation**: `loss = (prediction - target) ** 2`
- **Optimization**: `params = params - learning_rate * gradients`
"""

# %% [markdown]
"""
## Step 2A: Tensor Addition

### The Foundation Operation
Addition is the most basic and important tensor operation:

- **Element-wise**: Each element adds to corresponding element
- **Broadcasting**: Smaller tensors automatically expand to match larger ones
- **Commutative**: `a + b = b + a`
- **Associative**: `(a + b) + c = a + (b + c)`

### Mathematical Foundation
Addition in ML contexts:
```python
# Linear layer: output = input @ weights + bias
# The + bias is tensor addition with broadcasting

# Gradient updates: new_params = old_params + learning_rate * gradients
# This is also tensor addition

# Residual connections: output = layer1(x) + layer2(x)
# Adding outputs from different layers
```
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-addition", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
    def add(self, other: 'Tensor') -> 'Tensor':
        """
        Add two tensors element-wise.
        
        TODO: Implement tensor addition with broadcasting support.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get the numpy data from both tensors
        2. Use numpy's + operator for element-wise addition
        3. Create a new Tensor with the result
        4. Return the new tensor
        
        EXAMPLE USAGE:
        ```python
        # Same shape addition
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])
        result = t1.add(t2)
        print(result.data)  # [[6, 8], [10, 12]]
        
        # Broadcasting addition
        t3 = Tensor([[1, 2], [3, 4]])
        t4 = Tensor([10, 20])
        result = t3.add(t4)
        print(result.data)  # [[11, 22], [13, 24]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use self._data + other._data for numpy addition
        - Wrap result in new Tensor: return Tensor(result)
        - NumPy handles broadcasting automatically
        - Don't modify original tensors (create new one)
        
        LEARNING CONNECTIONS:
        - This is used in every neural network layer
        - Gradient updates use addition: params = params + learning_rate * gradients
        - Bias terms are added to layer outputs
        - Residual connections add tensors from different layers
        """
        ### BEGIN SOLUTION
        result = self._data + other._data
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Tensor Addition

Once you implement the `add` method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-addition", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_tensor_addition():
    """Test tensor addition with various shapes"""
    print("Testing tensor addition...")
    
    # Test same-shape addition
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    result = t1.add(t2)
    expected = np.array([[6, 8], [10, 12]])
    assert np.array_equal(result.data, expected), "Same-shape addition failed"
    
    # Test scalar addition (broadcasting)
    t3 = Tensor([[1, 2], [3, 4]])
    t4 = Tensor(10)
    result = t3.add(t4)
    expected = np.array([[11, 12], [13, 14]])
    assert np.array_equal(result.data, expected), "Scalar addition failed"
    
    # Test vector addition (broadcasting)
    t5 = Tensor([[1, 2], [3, 4]])
    t6 = Tensor([10, 20])
    result = t5.add(t6)
    expected = np.array([[11, 22], [13, 24]])
    assert np.array_equal(result.data, expected), "Vector addition failed"
    
    # Test that original tensors are unchanged
    original_t1 = np.array([[1, 2], [3, 4]])
    assert np.array_equal(t1.data, original_t1), "Original tensor should be unchanged"
    
    print("✅ Tensor addition tests passed!")
    print(f"✅ Same-shape addition working correctly")
    print(f"✅ Broadcasting with scalars and vectors working")
    print(f"✅ Original tensors preserved (immutable operations)")

# Run the test
test_tensor_addition()

# %% [markdown]
"""
## Step 2B: Tensor Multiplication

### Scaling and Element-wise Products
Multiplication is crucial for scaling values and computing element-wise products:

- **Element-wise**: Each element multiplies with corresponding element
- **Broadcasting**: Works with different shapes automatically
- **Commutative**: `a * b = b * a`
- **Distributive**: `a * (b + c) = a * b + a * c`

### ML Applications
```python
# Scaling by learning rate: gradients * learning_rate
# Attention mechanisms: attention_weights * values
# Gating mechanisms: gate_signal * input_signal
# Dropout: input * dropout_mask
```
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-multiplication", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiply two tensors element-wise.
        
        TODO: Implement tensor multiplication with broadcasting support.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get the numpy data from both tensors
        2. Use numpy's * operator for element-wise multiplication
        3. Create a new Tensor with the result
        4. Return the new tensor
        
        EXAMPLE USAGE:
        ```python
        # Same shape multiplication
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[2, 3], [4, 5]])
        result = t1.multiply(t2)
        print(result.data)  # [[2, 6], [12, 20]]
        
        # Broadcasting multiplication
        t3 = Tensor([[1, 2], [3, 4]])
        t4 = Tensor(2)
        result = t3.multiply(t4)
        print(result.data)  # [[2, 4], [6, 8]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use self._data * other._data for numpy multiplication
        - Wrap result in new Tensor: return Tensor(result)
        - NumPy handles broadcasting automatically
        - Don't modify original tensors (create new one)
        
        LEARNING CONNECTIONS:
        - Used in activation functions: ReLU uses multiplication with masks
        - Attention mechanisms: attention weights * values
        - Scaling operations: learning_rate * gradients
        - Gating in LSTM/GRU: gate_values * hidden_state
        """
        ### BEGIN SOLUTION
        result = self._data * other._data
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Tensor Multiplication

Once you implement the `multiply` method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-multiplication", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_tensor_multiplication():
    """Test tensor multiplication with various shapes"""
    print("Testing tensor multiplication...")
    
    # Test same-shape multiplication
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[2, 3], [4, 5]])
    result = t1.multiply(t2)
    expected = np.array([[2, 6], [12, 20]])
    assert np.array_equal(result.data, expected), "Same-shape multiplication failed"
    
    # Test scalar multiplication (broadcasting)
    t3 = Tensor([[1, 2], [3, 4]])
    t4 = Tensor(2)
    result = t3.multiply(t4)
    expected = np.array([[2, 4], [6, 8]])
    assert np.array_equal(result.data, expected), "Scalar multiplication failed"
    
    # Test vector multiplication (broadcasting)
    t5 = Tensor([[1, 2], [3, 4]])
    t6 = Tensor([2, 3])
    result = t5.multiply(t6)
    expected = np.array([[2, 6], [6, 12]])
    assert np.array_equal(result.data, expected), "Vector multiplication failed"
    
    # Test that original tensors are unchanged
    original_t1 = np.array([[1, 2], [3, 4]])
    assert np.array_equal(t1.data, original_t1), "Original tensor should be unchanged"
    
    print("✅ Tensor multiplication tests passed!")
    print(f"✅ Same-shape multiplication working correctly")
    print(f"✅ Broadcasting with scalars and vectors working")
    print(f"✅ Original tensors preserved (immutable operations)")

# Run the test
test_tensor_multiplication()

# %% [markdown]
"""
## Step 2C: Python Operators (Syntactic Sugar)

### Making Tensors Feel Natural
Python allows us to overload operators to make tensor operations feel natural:

- `t1 + t2` instead of `t1.add(t2)`
- `t1 * t2` instead of `t1.multiply(t2)`
- `t1 - t2` for subtraction
- `t1 / t2` for division

### Why This Matters
- **Readability**: Code looks like mathematical expressions
- **Familiarity**: Works like NumPy, PyTorch, TensorFlow
- **Convenience**: Shorter, more expressive code
- **Chaining**: `(t1 + t2) * t3` reads naturally
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-operators", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Addition operator: t1 + t2
        
        TODO: Implement the + operator for tensors.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor or scalar (int/float)
        2. If scalar, convert to Tensor first
        3. Use the add method we already implemented
        4. Return the result
        
        EXAMPLE USAGE:
        ```python
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2  # Same as t1.add(t2)
        
        # Also works with scalars
        result = t1 + 5  # Same as t1.add(Tensor(5))
        ```
        
        IMPLEMENTATION HINTS:
        - Use isinstance(other, (int, float)) to check for scalars
        - Convert scalars: other = Tensor(other)
        - Then use: return self.add(other)
        - This delegates to our existing add method
        """
        ### BEGIN SOLUTION
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self.add(other)
        ### END SOLUTION

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Multiplication operator: t1 * t2
        
        TODO: Implement the * operator for tensors.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor or scalar (int/float)
        2. If scalar, convert to Tensor first
        3. Use the multiply method we already implemented
        4. Return the result
        
        EXAMPLE USAGE:
        ```python
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 * t2  # Same as t1.multiply(t2)
        
        # Also works with scalars
        result = t1 * 2  # Same as t1.multiply(Tensor(2))
        ```
        
        IMPLEMENTATION HINTS:
        - Use isinstance(other, (int, float)) to check for scalars
        - Convert scalars: other = Tensor(other)
        - Then use: return self.multiply(other)
        - This delegates to our existing multiply method
        """
        ### BEGIN SOLUTION
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self.multiply(other)
        ### END SOLUTION

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Subtraction operator: t1 - t2
        
        TODO: Implement the - operator for tensors.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor or scalar (int/float)
        2. If scalar, convert to Tensor first
        3. Use numpy subtraction: self._data - other._data
        4. Return new Tensor with result
        
        EXAMPLE USAGE:
        ```python
        t1 = Tensor([5, 6, 7])
        t2 = Tensor([1, 2, 3])
        result = t1 - t2  # [4, 4, 4]
        
        # Also works with scalars
        result = t1 - 1  # [4, 5, 6]
        ```
        
        IMPLEMENTATION HINTS:
        - Similar pattern to __add__ and __mul__
        - Convert scalars to Tensor if needed
        - Use self._data - other._data for subtraction
        - Return Tensor(result)
        """
        ### BEGIN SOLUTION
        if isinstance(other, (int, float)):
            other = Tensor(other)
        result = self._data - other._data
        return Tensor(result)
        ### END SOLUTION

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Division operator: t1 / t2
        
        TODO: Implement the / operator for tensors.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor or scalar (int/float)
        2. If scalar, convert to Tensor first
        3. Use numpy division: self._data / other._data
        4. Return new Tensor with result
        
        EXAMPLE USAGE:
        ```python
        t1 = Tensor([6, 8, 10])
        t2 = Tensor([2, 4, 5])
        result = t1 / t2  # [3.0, 2.0, 2.0]
        
        # Also works with scalars
        result = t1 / 2  # [3.0, 4.0, 5.0]
        ```
        
        IMPLEMENTATION HINTS:
        - Similar pattern to other operators
        - Convert scalars to Tensor if needed
        - Use self._data / other._data for division
        - Return Tensor(result)
        - Division always produces floats
        """
        ### BEGIN SOLUTION
        if isinstance(other, (int, float)):
            other = Tensor(other)
        result = self._data / other._data
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Python Operators

Once you implement the operators above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-operators", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_tensor_operators():
    """Test Python operators (+, -, *, /) for tensors"""
    print("Testing tensor operators...")
    
    # Test addition operator
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 + t2
    expected = np.array([5, 7, 9])
    assert np.array_equal(result.data, expected), "Addition operator failed"
    
    # Test scalar addition
    result = t1 + 5
    expected = np.array([6, 7, 8])
    assert np.array_equal(result.data, expected), "Scalar addition failed"
    
    # Test multiplication operator
    result = t1 * t2
    expected = np.array([4, 10, 18])
    assert np.array_equal(result.data, expected), "Multiplication operator failed"
    
    # Test scalar multiplication
    result = t1 * 2
    expected = np.array([2, 4, 6])
    assert np.array_equal(result.data, expected), "Scalar multiplication failed"
    
    # Test subtraction operator
    result = t2 - t1
    expected = np.array([3, 3, 3])
    assert np.array_equal(result.data, expected), "Subtraction operator failed"
    
    # Test division operator
    t3 = Tensor([6, 8, 10])
    t4 = Tensor([2, 4, 5])
    result = t3 / t4
    expected = np.array([3.0, 2.0, 2.0])
    assert np.array_equal(result.data, expected), "Division operator failed"
    
    # Test chained operations
    result = (t1 + t2) * 2
    expected = np.array([10, 14, 18])
    assert np.array_equal(result.data, expected), "Chained operations failed"
    
    print("✅ Tensor operators tests passed!")
    print(f"✅ Addition, multiplication, subtraction, division working")
    print(f"✅ Scalar operations working correctly")
    print(f"✅ Chained operations working correctly")

# Run the test
test_tensor_operators()

# %% [markdown]
"""
## 🎯 Step 3: Integration Test

### Putting It All Together
Now let's test that all our tensor operations work together in realistic ML scenarios:

- **Chained operations**: Multiple operations in sequence
- **Broadcasting**: Different shapes working together
- **Mixed operations**: Addition, multiplication, etc. combined
- **Real ML patterns**: Mimicking actual neural network computations
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-integration", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_tensor_integration():
    """Test complete tensor functionality in realistic ML scenarios"""
    print("Testing tensor integration...")
    
    # Simulate a simple linear transformation: y = x @ W + b
    # where x is input, W is weights, b is bias
    x = Tensor([[1, 2], [3, 4]])  # Input batch
    W = Tensor([[0.5, 0.3], [0.2, 0.7]])  # Weight matrix
    b = Tensor([0.1, 0.2])  # Bias vector
    
    # Manual matrix multiplication (we'll implement real matmul later)
    # For now, test element-wise operations
    scaled_x = x * 0.5  # Scale input
    shifted_x = scaled_x + 0.1  # Add bias
    
    # Test properties after operations
    assert shifted_x.shape == x.shape, "Shape should be preserved"
    assert isinstance(shifted_x, Tensor), "Result should be a Tensor"
    
    # Test chained operations with broadcasting
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([10, 20])
    scalar = Tensor(0.5)
    
    # Complex expression: (t1 + t2) * scalar - 1
    result = (t1 + t2) * scalar - 1
    expected = np.array([[4.5, 10.0], [5.5, 11.0]])
    assert np.array_equal(result.data, expected), "Complex chained operations failed"
    
    # Test that operations are immutable
    original_t1 = np.array([[1, 2], [3, 4]])
    t1 + t2  # This should not modify t1
    assert np.array_equal(t1.data, original_t1), "Operations should be immutable"
    
    # Test broadcasting with different shapes
    t3 = Tensor([1, 2, 3])
    t4 = Tensor([[1], [2], [3]])
    result = t3 + t4
    assert result.shape == (3, 3), "Broadcasting result should be (3, 3)"
    
    # Test mixed data types
    int_tensor = Tensor([1, 2, 3])
    float_tensor = Tensor([1.0, 2.0, 3.0])
    result = int_tensor + float_tensor
    assert result.dtype in [np.float32, np.float64], "Mixed types should produce float"
    
    print("✅ Tensor integration tests passed!")
    print(f"✅ All tensor operations work together correctly")
    print(f"✅ Complex chained operations working")
    print(f"✅ Broadcasting working across different shapes")
    print(f"✅ Immutable operations preserve original tensors")
    print(f"✅ Ready to build neural networks!")

# Run the integration test
test_tensor_integration()

# %% [markdown]
"""
## 🎯 Module Summary: Tensor Mastery Achieved!

Congratulations! You've successfully implemented the core Tensor class with comprehensive functionality:

### ✅ What You've Built
- **Tensor Creation**: Handle scalars, lists, arrays with smart dtype management
- **Properties**: Access shape, size, dtype, and data efficiently
- **Core Arithmetic**: Add, multiply, subtract, divide with broadcasting support
- **Python Operators**: Natural syntax with +, -, *, / operators
- **Integration**: Operations work together seamlessly in complex expressions

### ✅ Key Learning Outcomes
- **Understanding**: Tensors as the foundation of all ML systems
- **Implementation**: Built tensor operations from scratch using NumPy
- **Testing**: Comprehensive validation at each step with immediate feedback
- **Broadcasting**: Automatic shape compatibility for flexible operations
- **Immutability**: Operations create new tensors without modifying originals

### ✅ Real-World Skills Developed
- **Systems thinking**: Understanding how components fit together
- **Progressive development**: Building complexity step by step
- **Testing discipline**: Validating each component before integration
- **API design**: Creating user-friendly interfaces

### ✅ Ready for Next Steps
Your tensor implementation is now the foundation for:
- **Activations**: ReLU, Sigmoid, Tanh will operate on your tensors
- **Layers**: Dense layers will use tensor arithmetic for transformations
- **Networks**: Complete neural networks built on your tensor foundation
- **Autograd**: Automatic differentiation will track tensor operations

### 🔗 Connection to Real ML Systems
Your implementation mirrors the core concepts in:
- **PyTorch**: `torch.Tensor` with similar operations and broadcasting
- **TensorFlow**: `tf.Tensor` with comparable functionality
- **NumPy**: Direct integration with the scientific Python ecosystem
- **JAX**: `jnp.array` with similar mathematical operations

### 🎯 Professional Development
You've demonstrated:
- **Systems thinking**: Understanding how components fit together
- **Progressive development**: Building complexity step by step
- **Testing discipline**: Validating each component before integration
- **API design**: Creating user-friendly interfaces

**Next Module**: Activations - Adding nonlinearity to enable complex learning!

Your tensor foundation is solid. Now let's build the functions that make neural networks powerful!
""" 
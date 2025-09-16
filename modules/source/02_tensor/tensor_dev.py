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
from typing import Union, Tuple, Optional, Any

# %% nbgrader={"grade": false, "grade_id": "tensor-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Tensor Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build tensors!")

# %% [markdown]
"""
## Where This Code Lives in the Final Package

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
## Mathematical Foundation: From Scalars to Tensors

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

### Unified Data Representation
Tensors provide a consistent way to represent all ML data:
```python
# All of these are tensors with different shapes
scalar_loss = Tensor(0.5)              # Shape: ()
feature_vector = Tensor([1, 2, 3])      # Shape: (3,)
weight_matrix = Tensor([[1, 2], [3, 4]]) # Shape: (2, 2)
image_batch = Tensor(np.random.rand(32, 224, 224, 3)) # Shape: (32, 224, 224, 3)
```

### Efficient Batch Processing
ML systems process multiple samples simultaneously:
```python
# Instead of processing one image at a time:
for image in images:
    result = model(image)  # Slow: 1000 separate operations

# Process entire batch at once:
batch_result = model(image_batch)  # Fast: 1 vectorized operation
```

### Hardware Acceleration
Modern hardware (GPUs, TPUs) excels at tensor operations:
- **Parallel processing**: Multiple operations simultaneously
- **Vectorization**: SIMD (Single Instruction, Multiple Data) operations
- **Memory optimization**: Contiguous memory layout for cache efficiency

### Automatic Differentiation
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

### ML-Specific Operations
- **Gradient tracking**: For automatic differentiation (coming in Module 7)
- **GPU support**: For hardware acceleration (future extension)
- **Broadcasting semantics**: ML-friendly dimension handling

### Consistent API
- **Type safety**: Predictable behavior across operations
- **Error checking**: Clear error messages for debugging
- **Integration**: Seamless work with other TinyTorch components

### Educational Value
- **Conceptual clarity**: Understand what tensors really are
- **Implementation insight**: See how frameworks work internally
- **Debugging skills**: Trace through tensor operations step by step

### Extensibility
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
# CONCEPT
Tensors are N-dimensional arrays that carry data through neural networks.
Think NumPy arrays with ML superpowers - same math, more capabilities.

"""
# %% [markdown]
"""
# CODE STRUCTURE
```python
class Tensor:
    def __init__(self, data):     # Create from any data type
    def __add__(self, other):     # Enable tensor + tensor
    def __mul__(self, other):     # Enable tensor * tensor
    # Properties: .shape, .size, .dtype, .data
```

"""
# %% [markdown]
"""
# CONNECTIONS
- torch.Tensor (PyTorch) - same concept, production optimized
- tf.Tensor (TensorFlow) - distributed computing focus
- np.ndarray (NumPy) - we wrap this with ML operations

"""
# %% [markdown]
"""
# CONSTRAINTS
- Handle broadcasting (auto-shape matching for operations)
- Support multiple data types (float32, int32, etc.)
- Efficient memory usage (copy only when necessary)
- Natural math notation (tensor + tensor should just work)

"""
# %% [markdown]
"""
# CONTEXT
Every ML operation flows through tensors:
- Neural networks: All computations operate on tensors
- Training: Gradients flow through tensor operations  
- Hardware: GPUs optimized for tensor math
- Production: Millions of tensor ops per second in real systems

**You're building the universal language of machine learning.**

"""
# %% nbgrader={"grade": false, "grade_id": "tensor-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Access the internal _data attribute
        2. Return the numpy array directly
        3. This provides access to underlying data for NumPy operations
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - PyTorch: tensor.numpy() converts to NumPy for visualization/analysis
        - TensorFlow: tensor.numpy() enables integration with scientific Python
        - Production: Data scientists need to access raw arrays for debugging
        - Performance: Direct access avoids copying for read-only operations
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Access the _data attribute (the NumPy array)
        2. Get the shape property from the NumPy array
        3. Return the shape tuple directly
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Neural networks: Layer compatibility requires matching shapes
        - Computer vision: Image shape (height, width, channels) determines architecture
        - NLP: Sequence length and vocabulary size affect model design
        - Debugging: Shape mismatches are the #1 cause of ML errors
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Access the _data attribute (the NumPy array)
        2. Get the size property from the NumPy array
        3. Return the total element count as an integer
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Memory planning: Calculate RAM requirements for large tensors
        - Model architecture: Determine parameter counts for layers
        - Performance optimization: Size affects computation time
        - Batch processing: Total elements determines vectorization efficiency
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Access the _data attribute (the NumPy array)
        2. Get the dtype property from the NumPy array
        3. Return the NumPy dtype object directly
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Precision vs speed: float32 is faster, float64 more accurate
        - Memory optimization: int8 uses 1/4 memory of int32
        - GPU compatibility: Some operations only work with specific types
        - Model deployment: Mobile/edge devices prefer smaller data types
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert the numpy array to a list using .tolist()
        2. Get shape and dtype information from properties
        3. Format as "Tensor([data], shape=shape, dtype=dtype)"
        4. Return the formatted string
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Debugging: Clear tensor representation speeds debugging
        - Jupyter notebooks: Good __repr__ improves data exploration
        - Logging: Production systems log tensor info for monitoring
        - Education: Students understand tensors better with clear output
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Extract numpy arrays from both tensors
        2. Use NumPy's + operator for element-wise addition
        3. Create a new Tensor object with the result
        4. Return the new tensor
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Neural networks: Adding bias terms to linear layer outputs
        - Residual connections: skip connections in ResNet architectures
        - Gradient updates: Adding computed gradients to parameters
        - Ensemble methods: Combining predictions from multiple models
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Extract numpy arrays from both tensors
        2. Use NumPy's * operator for element-wise multiplication
        3. Create a new Tensor object with the result
        4. Return the new tensor
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Activation functions: Element-wise operations like ReLU masking
        - Attention mechanisms: Element-wise scaling in transformer models
        - Feature scaling: Multiplying features by learned scaling factors
        - Gating: Element-wise gating in LSTM and GRU cells
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor object
        2. If Tensor, call the add() method directly
        3. If scalar, convert to Tensor then call add()
        4. Return the result from add() method
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Natural syntax: tensor + scalar enables intuitive code
        - Broadcasting: Adding scalars to tensors is common in ML
        - Operator overloading: Python's magic methods enable math-like syntax
        - API design: Clean interfaces reduce cognitive load for researchers
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor object
        2. If Tensor, call the multiply() method directly
        3. If scalar, convert to Tensor then call multiply()
        4. Return the result from multiply() method
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Scaling features: tensor * learning_rate for gradient updates
        - Masking: tensor * mask for attention mechanisms
        - Regularization: tensor * dropout_mask during training
        - Normalization: tensor * scale_factor in batch normalization
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor object
        2. If Tensor, subtract other._data from self._data
        3. If scalar, subtract scalar directly from self._data
        4. Create new Tensor with result and return
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Gradient computation: parameter - learning_rate * gradient
        - Residual connections: output - skip_connection in some architectures
        - Error calculation: predicted - actual for loss computation
        - Centering data: tensor - mean for zero-centered inputs
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check if other is a Tensor object
        2. If Tensor, divide self._data by other._data
        3. If scalar, divide self._data by scalar directly
        4. Create new Tensor with result and return
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Normalization: tensor / std_deviation for standard scaling
        - Learning rate decay: parameter / decay_factor over time
        - Probability computation: counts / total_counts for frequencies
        - Temperature scaling: logits / temperature in softmax functions
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Extract numpy arrays from both tensors
        2. Use np.matmul() for proper matrix multiplication
        3. Create new Tensor object with the result
        4. Return the new tensor
        
        LEARNING CONNECTIONS:
        Real-world relevance:
        - Linear layers: input @ weight matrices in neural networks
        - Transformer attention: Q @ K^T for attention scores
        - CNN convolutions: Implemented as matrix multiplications
        - Batch processing: Matrix ops enable parallel computation
        
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
# Testing Your Implementation

Now let's test our tensor implementation with comprehensive tests that validate all functionality.

"""
# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation

Let's test your tensor creation implementation right away! This gives you immediate feedback on whether your `__init__` method works correctly.

**This is a unit test** - it tests one specific function (tensor creation) in isolation.
"""
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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
### 🔬 Comprehensive Tests

Now let's run comprehensive tests that validate all tensor functionality together. These tests ensure your implementation is production-ready.

**These are comprehensive tests** - they test multiple features and edge cases to ensure robustness.
"""
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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

# Test function defined (called in main block)

# %% [markdown]
"""
### Unit Test: Tensor Properties

This test validates your tensor property methods (shape, size, dtype, data), ensuring they correctly reflect the tensor's dimensional structure and data characteristics.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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

# Test function defined (called in main block)

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic Operations

Now let's test all your arithmetic operations working together! This comprehensive test validates that addition, subtraction, multiplication, and division all work correctly with your tensor implementation.

**What This Tests:**
- Element-wise addition, subtraction, multiplication, division
- Proper NumPy array handling in arithmetic
- Result correctness across different operations

**Why This Matters:**
- Arithmetic operations are the foundation of all neural network computations
- These operations must be fast and mathematically correct
- Your implementation should match NumPy's behavior exactly
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
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

# Test function defined (called in main block)

# %% [markdown]
"""
### 🧪 Integration Test: Tensor-NumPy Integration

This integration test validates that your tensor system works seamlessly with NumPy, the foundation of the scientific Python ecosystem.

**What This Tests:**
- Creating tensors from NumPy arrays
- Converting tensors back to NumPy arrays  
- Mixed operations between tensors and NumPy
- Data type preservation and consistency

**Why This Matters:**
- Real ML systems must integrate with NumPy seamlessly
- Data scientists expect tensors to work with existing NumPy code
- Performance optimizations often involve NumPy operations
- This compatibility is what makes PyTorch and TensorFlow so powerful

**Real-World Connection:**
- PyTorch tensors have `.numpy()` and `torch.from_numpy()` methods
- TensorFlow has similar NumPy integration
- This test ensures your tensors work in real data science workflows
"""

# %% nbgrader={"grade": true, "grade_id": "test_module_tensor_numpy_integration", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_module_tensor_numpy_integration():
    """
    Integration test for tensor operations with NumPy arrays.
    
    Tests that tensors properly integrate with NumPy operations and maintain
    compatibility with the scientific Python ecosystem.
    """
    print("🔬 Running Integration Test: Tensor-NumPy Integration...")
    
    # Test 1: Tensor from NumPy array
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = Tensor(numpy_array)
    
    assert tensor_from_numpy.shape == (2, 3), "Tensor should preserve NumPy array shape"
    assert np.array_equal(tensor_from_numpy.data, numpy_array), "Tensor should preserve NumPy array data"
    
    # Test 2: Tensor arithmetic with NumPy-compatible operations
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    
    # Test operations that would be used in neural networks
    dot_product_result = np.dot(a.data, b.data)  # Common in layers
    assert np.isclose(dot_product_result, 32.0), "Dot product should work with tensor data"
    
    # Test 3: Broadcasting compatibility
    matrix = Tensor([[1, 2], [3, 4]])
    scalar = Tensor(10)
    
    result = matrix + scalar
    expected = np.array([[11, 12], [13, 14]])
    assert np.array_equal(result.data, expected), "Broadcasting should work like NumPy"
    
    # Test 4: Integration with scientific computing patterns
    data = Tensor([1, 4, 9, 16, 25])
    sqrt_result = Tensor(np.sqrt(data.data))  # Using NumPy functions on tensor data
    expected_sqrt = np.array([1., 2., 3., 4., 5.])
    assert np.allclose(sqrt_result.data, expected_sqrt), "Should integrate with NumPy functions"
    
    print("✅ Integration Test Passed: Tensor-NumPy integration works correctly.")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Reflection Questions

Now that you've built a working tensor system, take a moment to reflect on how this simple foundation connects to the massive ML systems powering today's AI revolution:

### System Design - How does this fit into larger systems?
1. **The Universal Language**: You just built the fundamental data structure of modern AI. How does having one unified way to represent data (as tensors) enable frameworks like PyTorch to handle everything from image recognition to language models using the same operations?

2. **Memory and Scale**: Your tensor creates a new result for every operation. When OpenAI trains GPT models with billions of parameters, why might memory management become a critical system design challenge? What trade-offs do you think they make?

3. **Hardware Abstraction**: Your tensor works on CPU through NumPy. How does abstracting tensor operations allow the same neural network code to run on CPUs, GPUs, and specialized AI chips like TPUs without changing the model logic?

### Production ML - How is this used in real ML workflows?
4. **Batch Processing Power**: Real ML systems process thousands of images or sentences simultaneously in batches. How does your element-wise tensor math suddenly become a superpower when applied to batches? Why is this crucial for real-time applications?

5. **Data Pipeline Integration**: Your tensor wraps NumPy arrays, making it compatible with the entire scientific Python ecosystem. How does this design choice affect a data scientist's ability to move from data exploration to model deployment?

6. **Production Reliability**: In a live recommendation system serving millions of users, what happens when tensor operations encounter unexpected data shapes or types? How might this influence framework design priorities?

### Framework Design - Why do frameworks make certain choices?
7. **API Philosophy**: You chose property-style access (`tensor.shape`) like PyTorch, rather than function calls. How does this seemingly small design decision affect the developer experience when building complex neural architectures?

8. **Broadcasting Magic**: PyTorch tensors can add a scalar to a matrix automatically, while your current implementation requires matching shapes. Why do you think frameworks invest heavily in broadcasting logic? What problems does it solve for researchers?

9. **Type System Design**: Your tensor supports different data types (int32, float32). How do these choices ripple through an entire ML pipeline, from training speed to model accuracy to deployment costs?

### Performance & Scale - What happens when systems get large?
10. **Memory Explosion**: Every operation in your tensor creates new memory. Imagine a neural network with millions of operations during training. How might this connect to why modern frameworks offer both copying and in-place operations?

11. **Computational Graphs**: Your operations happen immediately. Large language models track millions of tensor operations to compute gradients. How does the shift from immediate computation to deferred execution change system architecture?

12. **Hardware Optimization**: Your simple `+` operation translates to highly optimized kernel code on GPUs. How does having a clean tensor abstraction enable frameworks to automatically optimize operations for different hardware without changing user code?

**💡 Systems Insight**: The tensor abstraction you just built is one of computing's most elegant solutions—it provides a simple, mathematical interface that hides incredible complexity in optimization, hardware acceleration, and distributed computing. Every major AI breakthrough, from computer vision to large language models, flows through tensor operations just like the ones you implemented.
"""

# %% [markdown]
"""
# MODULE SUMMARY: Tensor Foundation

Congratulations! You've successfully implemented the fundamental data structure that powers all machine learning:

## What You've Built
- **Tensor Class**: N-dimensional array wrapper with professional interfaces
- **Core Operations**: Creation, property access, and arithmetic operations
- **Shape Management**: Automatic shape tracking and validation
- **Data Types**: Proper NumPy integration and type handling
- **Foundation**: The building block for all subsequent TinyTorch modules

## Key Learning Outcomes
- **Understanding**: How tensors work as the foundation of machine learning
- **Implementation**: Built tensor operations from scratch
- **Professional patterns**: Clean APIs, proper error handling, comprehensive testing
- **Real-world connection**: Understanding PyTorch/TensorFlow tensor foundations
- **Systems thinking**: Building reliable, reusable components

## Mathematical Foundations Mastered
- **N-dimensional arrays**: Shape, size, and dimensionality concepts
- **Element-wise operations**: Addition, subtraction, multiplication, division
- **Broadcasting**: Understanding how operations work with different shapes
- **Memory management**: Efficient data storage and access patterns

## Professional Skills Developed
- **API design**: Clean, intuitive interfaces for tensor operations
- **Error handling**: Graceful handling of invalid operations and edge cases
- **Testing methodology**: Comprehensive validation of tensor functionality
- **Documentation**: Clear, educational documentation with examples

## Ready for Advanced Applications
Your tensor implementation now enables:
- **Neural Networks**: Foundation for all layer implementations
- **Automatic Differentiation**: Gradient computation through computational graphs
- **Complex Models**: CNNs, RNNs, Transformers - all built on tensors
- **Real Applications**: Training models on real datasets

## Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.Tensor` provides identical functionality
- **TensorFlow**: `tf.Tensor` implements similar concepts
- **NumPy**: `numpy.ndarray` serves as the foundation
- **Industry Standard**: Every major ML framework uses these exact principles

## The Power of Tensors
You've built the fundamental data structure of modern AI:
- **Universality**: Tensors represent all data: images, text, audio, video
- **Efficiency**: Vectorized operations enable fast computation
- **Scalability**: Handles everything from single numbers to massive matrices
- **Flexibility**: Foundation for any mathematical operation

## What's Next
Your tensor implementation is the foundation for:
- **Activations**: Nonlinear functions that enable complex learning
- **Layers**: Linear transformations and neural network building blocks
- **Networks**: Composing layers into powerful architectures
- **Training**: Optimizing networks to solve real problems

**Next Module**: Activation functions - adding the nonlinearity that makes neural networks powerful!

You've built the foundation of modern AI. Now let's add the mathematical functions that enable machines to learn complex patterns!
"""

if __name__ == "__main__":
    # Run all tensor tests
    test_unit_tensor_creation()
    test_unit_tensor_properties()
    test_unit_tensor_arithmetic()
    test_module_tensor_numpy_integration()
    
    print("All tests passed!")
    print("Tensor module complete!")
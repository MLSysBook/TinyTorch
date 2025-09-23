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
# Tensor - Core Data Structure and Memory Management

Welcome to the Tensor module! You'll implement the fundamental data structure that powers all neural networks and understand why memory layout determines performance.

## Learning Goals
- Systems understanding: How tensor memory layout affects cache performance and computational efficiency
- Core implementation skill: Build a complete Tensor class with shape management and arithmetic operations
- Pattern recognition: Understand how tensors abstract N-dimensional data for ML algorithms
- Framework connection: See how your implementation mirrors PyTorch's tensor design and memory model
- Performance insight: Learn why contiguous memory layout and vectorized operations are critical for ML performance

## Build → Use → Reflect
1. **Build**: Complete Tensor class with shape management, broadcasting, and vectorized operations
2. **Use**: Perform tensor arithmetic and transformations on real multi-dimensional data
3. **Reflect**: Why does tensor memory layout become the performance bottleneck in large neural networks?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how N-dimensional arrays are stored and manipulated in memory
- Practical capability to build efficient tensor operations that form the foundation of neural networks
- Systems insight into why memory access patterns determine whether ML operations run fast or slow
- Performance consideration of when tensor operations trigger expensive memory copies vs efficient in-place updates
- Connection to production ML systems and how PyTorch optimizes tensor storage for GPU acceleration

## Systems Reality Check
💡 **Production Context**: PyTorch tensors automatically choose optimal memory layouts and can seamlessly move between CPU and GPU - your implementation reveals these design decisions
⚡ **Performance Note**: Non-contiguous tensors can be 10-100x slower than contiguous ones - memory layout is often more important than algorithm choice in ML systems
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
    
    def __init__(self, data: Any, dtype: Optional[str] = None, requires_grad: bool = False):
        """
        Create a new tensor from data.
        
        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
            requires_grad: Whether this tensor needs gradients for training. Defaults to False.
            
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
        elif isinstance(data, Tensor):
            # Input is another Tensor - extract its data
            if dtype is None:
                # Keep existing dtype, but prefer float32 for float64
                if data.data.dtype == np.float64:
                    dtype = 'float32'
                else:
                    dtype = str(data.data.dtype)
            self._data = data.data.astype(dtype) if dtype != str(data.data.dtype) else data.data.copy()
        else:
            # Try to convert unknown types
            self._data = np.array(data, dtype=dtype)
        
        # Initialize gradient tracking attributes
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self._grad_fn = None
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

    def sum(self) -> 'Tensor':
        """
        Sum all elements in the tensor.
        
        Returns a new tensor containing the sum of all elements.
        This is commonly used in loss functions and gradient computation.
        
        Returns:
            Tensor: A scalar tensor containing the sum of all elements
            
        Example:
            Tensor([1, 2, 3]).sum() → Tensor(6)
            Tensor([[1, 2], [3, 4]]).sum() → Tensor(10)
        """
        return Tensor(np.sum(self.data))
    
    @property
    def T(self) -> 'Tensor':
        """
        Transpose of the tensor.
        
        Returns a new tensor with transposed data. For 1D tensors,
        returns the tensor unchanged. For 2D+ tensors, swaps the dimensions.
        
        Returns:
            Tensor: Transposed tensor
            
        Example:
            Tensor([[1, 2], [3, 4]]).T → Tensor([[1, 3], [2, 4]])
        """
        return Tensor(self.data.T)

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

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication operator: tensor @ other
        
        Enables the @ operator for matrix multiplication, providing
        clean syntax for neural network operations.
        """
        return self.matmul(other)
    
    def backward(self, gradient=None):
        """
        Compute gradients for this tensor and propagate backward.
        
        This is a stub for now - full implementation in Module 09 (Autograd).
        For now, just accumulates gradients if requires_grad=True.
        
        Args:
            gradient: Gradient from upstream. If None, assumes scalar with grad=1
        """
        if not self.requires_grad:
            return
            
        if gradient is None:
            # Scalar case - gradient is 1
            gradient = Tensor(np.ones_like(self._data))
        
        # Accumulate gradients
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

    def reshape(self, *shape: int) -> 'Tensor':
        """
        Return a new tensor with the same data but different shape.
        
        Args:
            *shape: New shape dimensions. Use -1 for automatic sizing.
            
        Returns:
            New Tensor with reshaped data
            
        Example:
            tensor.reshape(2, -1)  # Reshape to 2 rows, auto columns
            tensor.reshape(4, 3)   # Reshape to 4x3 matrix
        """
        reshaped_data = self._data.reshape(*shape)
        return Tensor(reshaped_data)

# %% [markdown]
"""
# Testing Your Implementation

Now let's test our tensor implementation with comprehensive tests that validate all functionality.

**Testing Standards**: All tests follow the immediate testing pattern where each test is:
1. **Wrapped in a test_ function** for clear organization
2. **Called immediately after definition** for instant feedback  
3. **Educational and explanatory** to help you understand what's being verified

This approach ensures you get immediate verification that your implementation works correctly.

"""
# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation

Let's test your tensor creation implementation right away! This gives you immediate feedback on whether your `__init__` method works correctly.

**This is a unit test** - it tests one specific function (tensor creation) in isolation.
"""
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_creation_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_tensor_creation_immediate():
    """Test tensor creation immediately after implementation."""
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

# Test immediately after definition
test_tensor_creation_immediate()

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Properties

Now let's test that your tensor properties work correctly. This tests the @property methods you implemented.

**This is a unit test** - it tests specific properties (shape, size, dtype, data) in isolation.
"""
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_properties_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_tensor_properties_immediate():
    """Test tensor properties immediately after implementation."""
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

# Test immediately after definition
test_tensor_properties_immediate()

# %% [markdown]
"""
### 🧪 Educational Deep Dive: Matrix Multiplication Understanding

Before we test matrix multiplication, let's understand HOW it works by implementing it with loops. This educational section helps you understand what `np.matmul()` does internally.

**Educational Goal**: Understand the mathematical operations behind matrix multiplication.
"""

# %% nbgrader={"grade": false, "grade_id": "matrix-multiplication-education", "locked": false, "schema_version": 3, "solution": false, "task": false}
def educational_matmul_with_loops(a_data, b_data):
    """
    Educational implementation of matrix multiplication using loops.
    
    This shows exactly how matrix multiplication works mathematically.
    DO NOT use this in production - it's for understanding only!
    
    Args:
        a_data: 2D numpy array (rows, cols)
        b_data: 2D numpy array (cols, new_cols)
        
    Returns:
        2D numpy array: Result of matrix multiplication
    """
    # Get dimensions
    a_rows, a_cols = a_data.shape
    b_rows, b_cols = b_data.shape
    
    # Check compatibility
    if a_cols != b_rows:
        raise ValueError(f"Cannot multiply {a_data.shape} @ {b_data.shape}: inner dimensions don't match")
    
    # Initialize result matrix
    result = np.zeros((a_rows, b_cols))
    
    # Triple nested loop - this is why we use optimized libraries!
    for i in range(a_rows):          # For each row in A
        for j in range(b_cols):      # For each column in B
            for k in range(a_cols):  # For each element in the dot product
                result[i, j] += a_data[i, k] * b_data[k, j]
    
    return result

print("🎓 Educational Matrix Multiplication with Loops")
print("This shows HOW matrix multiplication works mathematically.")
print("In production, we use optimized np.matmul() for speed.")
print()

# Example: 2x2 @ 2x2 matrix multiplication
a_example = np.array([[1, 2], [3, 4]])
b_example = np.array([[5, 6], [7, 8]])

# Educational implementation (slow but clear)
result_loops = educational_matmul_with_loops(a_example, b_example)
print(f"Educational result (loops): \n{result_loops}")

# Production implementation (fast and optimized)
result_numpy = np.matmul(a_example, b_example)
print(f"Production result (numpy): \n{result_numpy}")

# Verify they match
print(f"Results match: {np.array_equal(result_loops, result_numpy)}")
print()
print("💡 Key insight: The math is identical, but numpy is ~100x faster!")
print("   That's why we use np.matmul() in our Tensor implementation.")

# %% [markdown]
"""
### 🧪 Unit Test: New Tensor Operations

Let's test the new `sum()` and `transpose()` operations we just added.

**This is a unit test** - it tests specific new operations in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test_unit_new_tensor_operations", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_tensor_sum():
    """Test sum operation on tensors."""
    print("🔬 Testing tensor sum operation...")
    
    # Test vector sum
    vector = Tensor([1, 2, 3, 4])
    sum_result = vector.sum()
    assert sum_result.data == 10, f"Vector sum should be 10, got {sum_result.data}"
    print("✅ Vector sum works")
    
    # Test matrix sum
    matrix = Tensor([[1, 2], [3, 4]])
    matrix_sum = matrix.sum()
    assert matrix_sum.data == 10, f"Matrix sum should be 10, got {matrix_sum.data}"
    print("✅ Matrix sum works")

def test_tensor_transpose():
    """Test transpose operation on tensors."""
    print("🔬 Testing tensor transpose operation...")
    
    # Test matrix transpose
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    transposed = matrix.T
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(transposed.data, expected), f"Transpose failed: expected \n{expected}, got \n{transposed.data}"
    print("✅ Matrix transpose works")
    
    # Test vector transpose (should be unchanged for 1D)
    vector = Tensor([1, 2, 3])
    vector_t = vector.T
    assert np.array_equal(vector_t.data, vector.data), "Vector transpose should preserve 1D arrays"
    print("✅ Vector transpose works")

def test_new_tensor_operations_immediate():
    """Test new tensor operations immediately after definition."""
    print("🔬 Unit Test: New Tensor Operations (sum, transpose)...")
    test_tensor_sum()
    test_tensor_transpose()
    print("📈 Progress: New Tensor Operations (sum, transpose) ✓")
    print("🎯 New operations behavior:")
    print("   sum(): Returns scalar tensor with sum of all elements")
    print("   .T: Returns new tensor with transposed dimensions")

# Test immediately after definition
test_new_tensor_operations_immediate()

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Arithmetic

Let's test your tensor arithmetic operations. This tests the __add__, __mul__, __sub__, __truediv__ methods.

**This is a unit test** - it tests specific arithmetic operations in isolation.
"""
# %% nbgrader={"grade": true, "grade_id": "test_unit_tensor_arithmetic_immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_tensor_arithmetic_immediate():
    """Test tensor arithmetic immediately after implementation."""
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

# Test immediately after definition
test_tensor_arithmetic_immediate()

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
    
    # Test new operations: sum and transpose
    sum_a = a.sum()
    assert sum_a.data == 6, f"Sum should be 6, got {sum_a.data}"
    
    # Test matrix operations
    matrix = Tensor([[1, 2], [3, 4]])
    matrix_t = matrix.T
    expected_t = np.array([[1, 3], [2, 4]])
    assert np.array_equal(matrix_t.data, expected_t), "Transpose should swap dimensions"
    
    # Test matrix multiplication
    mat_a = Tensor([[1, 2], [3, 4]])
    mat_b = Tensor([[5, 6], [7, 8]])
    result = mat_a @ mat_b
    expected_matmul = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result.data, expected_matmul), "Matrix multiplication should work correctly"
    
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

if __name__ == "__main__":
    # Run all tensor tests
    test_tensor_creation_immediate()
    test_tensor_properties_immediate()
    test_new_tensor_operations_immediate()
    test_tensor_arithmetic_immediate()
    test_unit_tensor_creation()
    test_unit_tensor_properties()
    test_unit_tensor_arithmetic()
    test_module_tensor_numpy_integration()
    
    print("\n🎯 All tensor functionality verified:")
    print("   ✅ Tensor creation from various data types")
    print("   ✅ Property access (shape, size, dtype, data)")
    print("   ✅ Arithmetic operations (+, -, *, /)")
    print("   ✅ Matrix operations (matmul @, sum, transpose .T)")
    print("   ✅ NumPy integration and compatibility")
    print("\n🚀 Tensor module complete!")
    print("Ready to build neural networks with your tensor foundation!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built a working tensor system, let's connect this foundational work to broader ML systems challenges. These questions help you think critically about how tensor operations scale to production ML environments.

Take time to reflect thoughtfully on each question - your insights will help you understand how the tensor concepts you've implemented connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Memory Layout and Cache Efficiency

**Context**: Your tensor implementation wraps NumPy arrays and creates new tensors for each operation. In production ML systems, tensor operations happen millions of times per second, making memory layout and cache efficiency critical for performance.

**Reflection Question**: Design a memory-efficient tensor system for training large neural networks (billions of parameters). How would you balance memory layout optimization with cache efficiency? Consider scenarios where you need to process massive image batches (1000+ images) while maintaining memory locality for CPU cache optimization. What trade-offs would you make between memory copying and in-place operations?

Think about: contiguous memory layout, cache line utilization, memory fragmentation, and the difference between row-major vs column-major storage in different computational contexts.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-memory-layout", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON MEMORY LAYOUT AND CACHE EFFICIENCY:

TODO: Replace this text with your thoughtful response about memory-efficient tensor system design.

Consider addressing:
- How would you optimize memory layout for large batch processing?
- What strategies would you use to minimize cache misses during tensor operations?
- How would you handle the trade-off between memory copying and in-place operations?
- What role does contiguous memory layout play in computational efficiency?
- How would different storage patterns (row-major vs column-major) affect performance?

Write a practical design connecting your tensor implementation to real memory optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of memory layout impact on performance (3 points)
- Addresses cache efficiency and locality concerns appropriately (3 points)
- Shows practical knowledge of memory optimization strategies (2 points)
- Demonstrates systems thinking about large-scale tensor operations (2 points)
- Clear technical reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of memory optimization
# Students should demonstrate understanding of cache efficiency and memory layout optimization
### END SOLUTION

# %% [markdown]
"""
### Question 2: Hardware Abstraction and Multi-Platform Deployment

**Context**: Your tensor class currently operates on CPU through NumPy. Production ML systems must run efficiently across diverse hardware: development laptops (CPU), training clusters (GPU), mobile devices (ARM processors), and edge devices (specialized AI chips).

**Reflection Question**: Architect a hardware-abstraction layer for your tensor system that enables the same tensor operations to run optimally across CPU, GPU, and specialized AI accelerators. How would you handle the complexity of different memory models, precision requirements, and computational paradigms while maintaining a simple user interface? Consider the challenges of automatic device placement and memory management across heterogeneous hardware.

Think about: device-specific optimizations, memory transfer costs, precision trade-offs, and automatic kernel selection for different hardware architectures.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-hardware-abstraction", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON HARDWARE ABSTRACTION AND MULTI-PLATFORM DEPLOYMENT:

TODO: Replace this text with your thoughtful response about hardware abstraction design.

Consider addressing:
- How would you design an abstraction layer that works across CPU, GPU, and AI accelerators?
- What strategies would you use for automatic device placement and memory management?
- How would you handle different precision requirements across hardware platforms?
- What role would kernel selection and optimization play in your design?
- How would you minimize memory transfer costs between different compute devices?

Write an architectural analysis connecting your tensor foundation to real hardware deployment challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of multi-platform hardware challenges (3 points)
- Designs practical abstraction layer for device management (3 points)
- Addresses precision and optimization considerations (2 points)
- Demonstrates systems thinking about hardware-software interfaces (2 points)
- Clear architectural reasoning with practical insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of hardware abstraction challenges
# Students should demonstrate knowledge of multi-platform deployment and device optimization
### END SOLUTION

# %% [markdown]
"""
### Question 3: Computational Graph Integration and Automatic Differentiation

**Context**: Your tensor performs operations immediately (eager execution). Modern deep learning frameworks build computational graphs to track operations for automatic differentiation, enabling gradient-based optimization that powers neural network training.

**Reflection Question**: Extend your tensor design to support computational graph construction for automatic differentiation. How would you modify your tensor operations to build a graph of dependencies while maintaining performance for both training (graph construction) and inference (optimized execution)? Consider the challenge of supporting both eager execution for debugging and graph mode for production deployment.

Think about: operation tracking, gradient flow, memory management for large graphs, and the trade-offs between flexibility and performance in different execution modes.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-computational-graphs", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON COMPUTATIONAL GRAPH INTEGRATION:

TODO: Replace this text with your thoughtful response about computational graph design.

Consider addressing:
- How would you modify your tensor class to support computational graph construction?
- What strategies would you use to balance eager execution with graph-based optimization?
- How would you handle gradient flow and automatic differentiation in your design?
- What memory management challenges arise with large computational graphs?
- How would you support both debugging-friendly and production-optimized execution modes?

Write a design analysis connecting your tensor operations to automatic differentiation and training systems.

GRADING RUBRIC (Instructor Use):
- Understands computational graph concepts and gradient tracking (3 points)
- Designs practical approach to eager vs graph execution modes (3 points)
- Addresses memory management and performance considerations (2 points)
- Shows systems thinking about training vs inference requirements (2 points)
- Clear design reasoning with automatic differentiation insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of computational graphs and automatic differentiation
# Students should demonstrate knowledge of how tensor operations enable gradient computation
### END SOLUTION

# %% [markdown]
"""
## Parameter Helper Function

Now that we have Tensor with gradient support, let's add a convenient helper function for creating trainable parameters:
"""

# %% nbgrader={"grade": false, "grade_id": "parameter-helper", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| export
def Parameter(data, dtype=None):
    """
    Convenience function for creating trainable tensors.
    
    This is equivalent to Tensor(data, requires_grad=True) but provides
    cleaner syntax for neural network parameters.
    
    Args:
        data: Input data (scalar, list, or numpy array)
        dtype: Data type ('float32', 'int32', etc.). Defaults to auto-detect.
        
    Returns:
        Tensor with requires_grad=True
        
    Examples:
        weight = Parameter(np.random.randn(784, 128))  # Neural network weight
        bias = Parameter(np.zeros(128))                # Neural network bias
    """
    return Tensor(data, dtype=dtype, requires_grad=True)

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
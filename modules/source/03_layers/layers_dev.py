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
# Module 3: Layers - Building Blocks of Neural Networks

Welcome to the Layers module! This is where we build the fundamental components that stack together to form neural networks.

## Learning Goals
- Understand how matrix multiplication powers neural networks
- Implement naive matrix multiplication from scratch for deep understanding
- Build the Dense (Linear) layer - the foundation of all neural networks
- Learn weight initialization strategies and their importance
- See how layers compose with activations to create powerful networks

## Build → Use → Understand
1. **Build**: Matrix multiplication and Dense layers from scratch
2. **Use**: Create and test layers with real data
3. **Understand**: How linear transformations enable feature learning
"""

# %% nbgrader={"grade": false, "grade_id": "layers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.layers

#| export
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Union, List, Tuple, Optional

# Import our dependencies - try from package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_activations'))
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax

# %% nbgrader={"grade": false, "grade_id": "layers-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% nbgrader={"grade": false, "grade_id": "layers-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Layers Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network layers!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/03_layers/layers_dev.py`  
**Building Side:** Code exports to `tinytorch.core.layers`

```python
# Final package structure:
from tinytorch.core.layers import Dense, Conv2D  # All layer types together!
from tinytorch.core.tensor import Tensor  # The foundation
from tinytorch.core.activations import ReLU, Sigmoid  # Nonlinearity
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.Linear`
- **Consistency:** All layer types live together in `core.layers`
- **Integration:** Works seamlessly with tensors and activations
"""

# %% [markdown]
"""
## 🧠 The Mathematical Foundation of Neural Layers

### Linear Algebra at the Heart of ML
Neural networks are fundamentally about **linear transformations** followed by **nonlinear activations**:

```
Layer: y = Wx + b (linear transformation)
Activation: z = σ(y) (nonlinear transformation)
```

### Matrix Multiplication: The Engine of Deep Learning
Every forward pass in a neural network involves matrix multiplication:
- **Dense layers**: Matrix multiplication between inputs and weights
- **Convolutional layers**: Convolution as matrix multiplication
- **Attention**: Query-key-value matrix operations
- **Transformers**: Self-attention through matrix operations

### Why Matrix Multiplication Matters
- **Parallel computation**: GPUs excel at matrix operations
- **Batch processing**: Handle multiple samples simultaneously
- **Feature learning**: Each row/column learns different patterns
- **Composability**: Layers stack naturally through matrix chains

### Connection to Real ML Systems
Every framework optimizes matrix multiplication:
- **PyTorch**: `torch.nn.Linear` uses optimized BLAS
- **TensorFlow**: `tf.keras.layers.Dense` uses cuDNN
- **JAX**: `jax.numpy.dot` uses XLA compilation
- **TinyTorch**: `tinytorch.core.layers.Dense` (what we're building!)

### Performance Considerations
- **Memory layout**: Contiguous arrays for cache efficiency
- **Vectorization**: SIMD operations for speed
- **Parallelization**: Multi-threading and GPU acceleration
- **Numerical stability**: Proper initialization and normalization
"""

# %% [markdown]
"""
## Step 1: Understanding Matrix Multiplication

### What is Matrix Multiplication?
Matrix multiplication is the **fundamental operation** that powers neural networks. When we multiply matrices A and B:

```
C = A @ B
```

Each element C[i,j] is the **dot product** of row i from A and column j from B.

### Why Matrix Multiplication in Neural Networks?
- **Dense layers**: Transform inputs through learned weights
- **Batch processing**: Handle multiple samples at once
- **Feature learning**: Each neuron learns different patterns
- **Efficiency**: GPUs are optimized for matrix operations

### Visual Example
```
A = [[1, 2],     B = [[5, 6],     C = [[19, 22],
     [3, 4]]          [7, 8]]          [43, 50]]

C[0,0] = 1*5 + 2*7 = 19
C[0,1] = 1*6 + 2*8 = 22
C[1,0] = 3*5 + 4*7 = 43
C[1,1] = 3*6 + 4*8 = 50
```

### The Algorithm
For matrices A(m×n) and B(n×p) → C(m×p):
```
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i,j] += A[i,k] * B[k,j]
```

Let's implement this to truly understand it!
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-naive", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication using explicit for-loops.
    
    This helps you understand what matrix multiplication really does!
    
    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (n, p)
        
    Returns:
        Matrix of shape (m, p) where C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
        
    TODO: Implement matrix multiplication using three nested for-loops.
    
    APPROACH:
    1. Get the dimensions: m, n from A and n2, p from B
    2. Check that n == n2 (matrices must be compatible)
    3. Create output matrix C of shape (m, p) filled with zeros
    4. Use three nested loops:
       - i loop: rows of A (0 to m-1)
       - j loop: columns of B (0 to p-1) 
       - k loop: shared dimension (0 to n-1)
    5. For each (i,j), compute: C[i,j] += A[i,k] * B[k,j]
    
    EXAMPLE:
    A = [[1, 2],     B = [[5, 6],
         [3, 4]]          [7, 8]]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
    
    HINTS:
    - Start with C = np.zeros((m, p))
    - Use three nested for loops: for i in range(m): for j in range(p): for k in range(n):
    - Accumulate the sum: C[i,j] += A[i,k] * B[k,j]
    """
    ### BEGIN SOLUTION
    # Get matrix dimensions
    m, n = A.shape
    n2, p = B.shape
    
    # Check compatibility
    if n != n2:
        raise ValueError(f"Incompatible matrix dimensions: A is {m}x{n}, B is {n2}x{p}")
    
    # Initialize result matrix
    C = np.zeros((m, p))
    
    # Triple nested loop for matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Quick Test: Matrix Multiplication

Let's test your matrix multiplication implementation right away! This is the foundation of neural networks.
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test matrix multiplication immediately after implementation
print("🔬 Testing matrix multiplication...")

# Test simple 2x2 case
try:
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_naive(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Matrix multiplication failed: expected {expected}, got {result}"
    print(f"✅ Simple 2x2 test: {A.tolist()} @ {B.tolist()} = {result.tolist()}")
    
    # Compare with NumPy
    numpy_result = A @ B
    assert np.allclose(result, numpy_result), f"Doesn't match NumPy: got {result}, expected {numpy_result}"
    print("✅ Matches NumPy's result")
    
except Exception as e:
    print(f"❌ Matrix multiplication test failed: {e}")
    raise

# Test different shapes
try:
    A2 = np.array([[1, 2, 3]], dtype=np.float32)  # 1x3
    B2 = np.array([[4], [5], [6]], dtype=np.float32)  # 3x1
    result2 = matmul_naive(A2, B2)
    expected2 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
    
    assert np.allclose(result2, expected2), f"Different shapes failed: got {result2}, expected {expected2}"
    print(f"✅ Different shapes test: {A2.tolist()} @ {B2.tolist()} = {result2.tolist()}")
    
except Exception as e:
    print(f"❌ Different shapes test failed: {e}")
    raise

# Show the algorithm in action
print("🎯 Matrix multiplication algorithm:")
print("   C[i,j] = Σ(A[i,k] * B[k,j]) for all k")
print("   Triple nested loops compute each element")
print("📈 Progress: Matrix multiplication ✓")

# %% [markdown]
"""
## Step 2: Building the Dense Layer

Now let's build the **Dense layer**, the most fundamental building block of neural networks. A Dense layer performs a linear transformation: `y = Wx + b`

### What is a Dense Layer?
- **Linear transformation**: `y = Wx + b`
- **W**: Weight matrix (learnable parameters)
- **x**: Input tensor
- **b**: Bias vector (learnable parameters)
- **y**: Output tensor

### Why Dense Layers Matter
- **Universal approximation**: Can approximate any function with enough neurons
- **Feature learning**: Each neuron learns a different feature
- **Nonlinearity**: When combined with activation functions, becomes very powerful
- **Foundation**: All other layers build on this concept

### The Math
For input x of shape (batch_size, input_size):
- **W**: Weight matrix of shape (input_size, output_size)
- **b**: Bias vector of shape (output_size)
- **y**: Output of shape (batch_size, output_size)

### Visual Example
```
Input: x = [1, 2, 3] (3 features)
Weights: W = [[0.1, 0.2],    Bias: b = [0.1, 0.2]
              [0.3, 0.4],
              [0.5, 0.6]]

Step 1: Wx = [0.1*1 + 0.3*2 + 0.5*3,  0.2*1 + 0.4*2 + 0.6*3]
            = [2.2, 3.2]

Step 2: y = Wx + b = [2.2 + 0.1, 3.2 + 0.2] = [2.3, 3.4]
```

Let's implement this!
"""

# %% nbgrader={"grade": false, "grade_id": "dense-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Dense:
    """
    Dense (Linear) Layer: y = Wx + b
    
    The fundamental building block of neural networks.
    Performs linear transformation: matrix multiplication + bias addition.
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, 
                 use_naive_matmul: bool = False):
        """
        Initialize Dense layer with random weights.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            use_bias: Whether to include bias term (default: True)
            use_naive_matmul: Whether to use naive matrix multiplication (for learning)
            
        TODO: Implement Dense layer initialization with proper weight initialization.
        
        APPROACH:
        1. Store layer parameters (input_size, output_size, use_bias, use_naive_matmul)
        2. Initialize weights with Xavier/Glorot initialization
        3. Initialize bias to zeros (if use_bias=True)
        4. Convert to float32 for consistency
        
        EXAMPLE:
        Dense(3, 2) creates:
        - weights: shape (3, 2) with small random values
        - bias: shape (2,) with zeros
        
        HINTS:
        - Use np.random.randn() for random initialization
        - Scale weights by sqrt(2/(input_size + output_size)) for Xavier init
        - Use np.zeros() for bias initialization
        - Convert to float32 with .astype(np.float32)
        """
        ### BEGIN SOLUTION
        # Store parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.use_naive_matmul = use_naive_matmul
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * scale
        
        # Initialize bias
        if use_bias:
            self.bias = np.zeros(output_size, dtype=np.float32)
        else:
            self.bias = None
        ### END SOLUTION
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = Wx + b
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
            
        TODO: Implement matrix multiplication and bias addition.
        
        APPROACH:
        1. Choose matrix multiplication method based on use_naive_matmul flag
        2. Perform matrix multiplication: Wx
        3. Add bias if use_bias=True
        4. Return result wrapped in Tensor
        
        EXAMPLE:
        Input x: Tensor([[1, 2, 3]])  # shape (1, 3)
        Weights: shape (3, 2)
        Output: Tensor([[val1, val2]])  # shape (1, 2)
        
        HINTS:
        - Use self.use_naive_matmul to choose between matmul_naive and @
        - x.data gives you the numpy array
        - Use broadcasting for bias addition: result + self.bias
        - Return Tensor(result) to wrap the result
        """
        ### BEGIN SOLUTION
        # Matrix multiplication
        if self.use_naive_matmul:
            result = matmul_naive(x.data, self.weights)
        else:
            result = x.data @ self.weights
        
        # Add bias
        if self.use_bias:
            result += self.bias
        
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Quick Test: Dense Layer

Let's test your Dense layer implementation! This is the fundamental building block of neural networks.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test Dense layer immediately after implementation
print("🔬 Testing Dense layer...")

# Test basic Dense layer
try:
    layer = Dense(input_size=3, output_size=2, use_bias=True)
    x = Tensor([[1, 2, 3]])  # batch_size=1, input_size=3
    
    print(f"Input shape: {x.shape}")
    print(f"Layer weights shape: {layer.weights.shape}")
    if layer.bias is not None:
        print(f"Layer bias shape: {layer.bias.shape}")
    
    y = layer(x)
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
    
    # Test shape compatibility
    assert y.shape == (1, 2), f"Output shape should be (1, 2), got {y.shape}"
    print("✅ Dense layer produces correct output shape")
    
    # Test weights initialization
    assert layer.weights.shape == (3, 2), f"Weights shape should be (3, 2), got {layer.weights.shape}"
    if layer.bias is not None:
        assert layer.bias.shape == (2,), f"Bias shape should be (2,), got {layer.bias.shape}"
    print("✅ Dense layer has correct weight and bias shapes")
    
    # Test that weights are not all zeros (proper initialization)
    assert not np.allclose(layer.weights, 0), "Weights should not be all zeros"
    if layer.bias is not None:
        assert np.allclose(layer.bias, 0), "Bias should be initialized to zeros"
    print("✅ Dense layer has proper weight initialization")
    
except Exception as e:
    print(f"❌ Dense layer test failed: {e}")
    raise

# Test without bias
try:
    layer_no_bias = Dense(input_size=2, output_size=1, use_bias=False)
    x2 = Tensor([[1, 2]])
    y2 = layer_no_bias(x2)
    
    assert y2.shape == (1, 1), f"No bias output shape should be (1, 1), got {y2.shape}"
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("✅ Dense layer works without bias")
    
except Exception as e:
    print(f"❌ Dense layer no-bias test failed: {e}")
    raise

# Test naive matrix multiplication
try:
    layer_naive = Dense(input_size=2, output_size=2, use_naive_matmul=True)
    x3 = Tensor([[1, 2]])
    y3 = layer_naive(x3)
    
    assert y3.shape == (1, 2), f"Naive matmul output shape should be (1, 2), got {y3.shape}"
    print("✅ Dense layer works with naive matrix multiplication")
    
except Exception as e:
    print(f"❌ Dense layer naive matmul test failed: {e}")
    raise

# Show the linear transformation in action
print("🎯 Dense layer behavior:")
print("   y = Wx + b (linear transformation)")
print("   W: learnable weight matrix")
print("   b: learnable bias vector")
print("📈 Progress: Matrix multiplication ✓, Dense layer ✓")

# %% [markdown]
"""
### 🧪 Test Your Implementations

Once you implement the functions above, run these cells to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul-naive", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test matrix multiplication
print("Testing matrix multiplication...")

# Test case 1: Simple 2x2 matrices
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)

result = matmul_naive(A, B)
expected = np.array([[19, 22], [43, 50]], dtype=np.float32)

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"Your result:\n{result}")
print(f"Expected:\n{expected}")

assert np.allclose(result, expected), f"Result doesn't match expected: got {result}, expected {expected}"

# Test case 2: Compare with NumPy
numpy_result = A @ B
assert np.allclose(result, numpy_result), f"Doesn't match NumPy result: got {result}, expected {numpy_result}"

# Test case 3: Different shapes
A2 = np.array([[1, 2, 3]], dtype=np.float32)  # 1x3
B2 = np.array([[4], [5], [6]], dtype=np.float32)  # 3x1
result2 = matmul_naive(A2, B2)
expected2 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
assert np.allclose(result2, expected2), f"Different shapes failed: got {result2}, expected {expected2}"

print("✅ Matrix multiplication tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-dense-layer", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test Dense layer
print("Testing Dense layer...")

# Test basic Dense layer
layer = Dense(input_size=3, output_size=2, use_bias=True)
x = Tensor([[1, 2, 3]])  # batch_size=1, input_size=3

print(f"Input shape: {x.shape}")
print(f"Layer weights shape: {layer.weights.shape}")
if layer.bias is not None:
    print(f"Layer bias shape: {layer.bias.shape}")
else:
    print("Layer bias: None")

y = layer(x)
print(f"Output shape: {y.shape}")
print(f"Output: {y}")

# Test shape compatibility
assert y.shape == (1, 2), f"Output shape should be (1, 2), got {y.shape}"

# Test without bias
layer_no_bias = Dense(input_size=2, output_size=1, use_bias=False)
x2 = Tensor([[1, 2]])
y2 = layer_no_bias(x2)
assert y2.shape == (1, 1), f"No bias output shape should be (1, 1), got {y2.shape}"
assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"

# Test naive matrix multiplication
layer_naive = Dense(input_size=2, output_size=2, use_naive_matmul=True)
x3 = Tensor([[1, 2]])
y3 = layer_naive(x3)
assert y3.shape == (1, 2), f"Naive matmul output shape should be (1, 2), got {y3.shape}"

print("✅ Dense layer tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-layer-composition", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test layer composition
print("Testing layer composition...")

# Create a simple network: Dense → ReLU → Dense
dense1 = Dense(input_size=3, output_size=2)
relu = ReLU()
dense2 = Dense(input_size=2, output_size=1)

# Test input
x = Tensor([[1, 2, 3]])
print(f"Input: {x}")

# Forward pass through the network
h1 = dense1(x)
print(f"After Dense1: {h1}")

h2 = relu(h1)
print(f"After ReLU: {h2}")

h3 = dense2(h2)
print(f"After Dense2: {h3}")

# Test shapes
assert h1.shape == (1, 2), f"Dense1 output should be (1, 2), got {h1.shape}"
assert h2.shape == (1, 2), f"ReLU output should be (1, 2), got {h2.shape}"
assert h3.shape == (1, 1), f"Dense2 output should be (1, 1), got {h3.shape}"

# Test that ReLU actually applied (non-negative values)
assert np.all(h2.data >= 0), "ReLU should produce non-negative values"

print("✅ Layer composition tests passed!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the core building blocks of neural networks:

### What You've Accomplished
✅ **Matrix Multiplication**: Implemented from scratch with triple nested loops  
✅ **Dense Layer**: The fundamental linear transformation y = Wx + b  
✅ **Weight Initialization**: Xavier/Glorot initialization for stable training  
✅ **Layer Composition**: Combining layers with activations  
✅ **Flexible Implementation**: Support for both naive and optimized matrix multiplication  

### Key Concepts You've Learned
- **Matrix multiplication** is the engine of neural networks
- **Dense layers** perform linear transformations that learn features
- **Weight initialization** is crucial for stable training
- **Layer composition** creates powerful nonlinear functions
- **Batch processing** enables efficient computation

### Mathematical Foundations
- **Linear algebra**: Matrix operations power all neural computations
- **Universal approximation**: Dense layers can approximate any function
- **Feature learning**: Each neuron learns different patterns
- **Composability**: Simple operations combine to create complex behaviors

### Next Steps
1. **Export your code**: `tito package nbdev --export 03_layers`
2. **Test your implementation**: `tito module test 03_layers`
3. **Use your layers**: 
   ```python
   from tinytorch.core.layers import Dense
   from tinytorch.core.activations import ReLU
   layer = Dense(10, 5)
   activation = ReLU()
   ```
4. **Move to Module 4**: Start building complete neural networks!

**Ready for the next challenge?** Let's compose these layers into complete neural network architectures!
""" 
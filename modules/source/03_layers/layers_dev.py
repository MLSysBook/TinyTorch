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
### 🧪 Unit Test: Matrix Multiplication

Let's test your matrix multiplication implementation right away! This is the foundation of neural networks.

**This is a unit test** - it tests one specific function (matmul_naive) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test matrix multiplication immediately after implementation
print("🔬 Unit Test: Matrix Multiplication...")

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
### 🧪 Unit Test: Dense Layer

Let's test your Dense layer implementation! This is the fundamental building block of neural networks.

**This is a unit test** - it tests one specific class (Dense layer) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test Dense layer immediately after implementation
print("🔬 Unit Test: Dense Layer...")

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
## 🧪 Comprehensive Testing: Matrix Multiplication and Dense Layers

Let's thoroughly test your implementations to make sure they work correctly in all scenarios.
This comprehensive testing ensures your layers are robust and ready for real neural networks.
"""

# %% nbgrader={"grade": true, "grade_id": "test-layers-comprehensive", "locked": true, "points": 30, "schema_version": 3, "solution": false, "task": false}
def test_layers_comprehensive():
    """Comprehensive test of matrix multiplication and Dense layers."""
    print("🔬 Testing matrix multiplication and Dense layers comprehensively...")
    
    tests_passed = 0
    total_tests = 10
    
    # Test 1: Matrix Multiplication Basic Cases
    try:
        # Test 2x2 matrices
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul_naive(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        assert np.allclose(result, expected), f"2x2 multiplication failed: expected {expected}, got {result}"
        
        # Compare with NumPy
        numpy_result = A @ B
        assert np.allclose(result, numpy_result), f"Doesn't match NumPy: expected {numpy_result}, got {result}"
        
        print(f"✅ Matrix multiplication 2x2: {A.shape} × {B.shape} = {result.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Matrix multiplication basic failed: {e}")
    
    # Test 2: Matrix Multiplication Different Shapes
    try:
        # Test 1x3 × 3x1 = 1x1
        A1 = np.array([[1, 2, 3]], dtype=np.float32)
        B1 = np.array([[4], [5], [6]], dtype=np.float32)
        result1 = matmul_naive(A1, B1)
        expected1 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
        assert np.allclose(result1, expected1), f"1x3 × 3x1 failed: expected {expected1}, got {result1}"
        
        # Test 3x2 × 2x4 = 3x4
        A2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        B2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        result2 = matmul_naive(A2, B2)
        expected2 = A2 @ B2
        assert np.allclose(result2, expected2), f"3x2 × 2x4 failed: expected {expected2}, got {result2}"
        
        print(f"✅ Matrix multiplication shapes: (1,3)×(3,1), (3,2)×(2,4)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Matrix multiplication shapes failed: {e}")
    
    # Test 3: Matrix Multiplication Edge Cases
    try:
        # Test with zeros
        A_zero = np.zeros((2, 3), dtype=np.float32)
        B_zero = np.zeros((3, 2), dtype=np.float32)
        result_zero = matmul_naive(A_zero, B_zero)
        expected_zero = np.zeros((2, 2), dtype=np.float32)
        assert np.allclose(result_zero, expected_zero), "Zero matrix multiplication failed"
        
        # Test with identity
        A_id = np.array([[1, 2]], dtype=np.float32)
        B_id = np.array([[1, 0], [0, 1]], dtype=np.float32)
        result_id = matmul_naive(A_id, B_id)
        expected_id = np.array([[1, 2]], dtype=np.float32)
        assert np.allclose(result_id, expected_id), "Identity matrix multiplication failed"
        
        # Test with negative values
        A_neg = np.array([[-1, 2]], dtype=np.float32)
        B_neg = np.array([[3], [-4]], dtype=np.float32)
        result_neg = matmul_naive(A_neg, B_neg)
        expected_neg = np.array([[-11]], dtype=np.float32)  # -1*3 + 2*(-4) = -11
        assert np.allclose(result_neg, expected_neg), "Negative matrix multiplication failed"
        
        print("✅ Matrix multiplication edge cases: zeros, identity, negatives")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Matrix multiplication edge cases failed: {e}")
    
    # Test 4: Dense Layer Initialization
    try:
        # Test with bias
        layer_bias = Dense(input_size=3, output_size=2, use_bias=True)
        assert layer_bias.weights.shape == (3, 2), f"Weights shape should be (3, 2), got {layer_bias.weights.shape}"
        assert layer_bias.bias is not None, "Bias should not be None when use_bias=True"
        assert layer_bias.bias.shape == (2,), f"Bias shape should be (2,), got {layer_bias.bias.shape}"
        
        # Check weight initialization (should not be all zeros)
        assert not np.allclose(layer_bias.weights, 0), "Weights should not be all zeros"
        assert np.allclose(layer_bias.bias, 0), "Bias should be initialized to zeros"
        
        # Test without bias
        layer_no_bias = Dense(input_size=4, output_size=3, use_bias=False)
        assert layer_no_bias.weights.shape == (4, 3), f"No-bias weights shape should be (4, 3), got {layer_no_bias.weights.shape}"
        assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
        
        print("✅ Dense layer initialization: weights, bias, shapes")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dense layer initialization failed: {e}")
    
    # Test 5: Dense Layer Forward Pass
    try:
        layer = Dense(input_size=3, output_size=2, use_bias=True)
        
        # Test single sample
        x_single = Tensor([[1, 2, 3]])  # shape: (1, 3)
        y_single = layer(x_single)
        assert y_single.shape == (1, 2), f"Single sample output should be (1, 2), got {y_single.shape}"
        
        # Test batch of samples
        x_batch = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape: (3, 3)
        y_batch = layer(x_batch)
        assert y_batch.shape == (3, 2), f"Batch output should be (3, 2), got {y_batch.shape}"
        
        # Verify computation manually for single sample
        expected_single = np.dot(x_single.data, layer.weights) + layer.bias
        assert np.allclose(y_single.data, expected_single), "Single sample computation incorrect"
        
        print("✅ Dense layer forward pass: single sample, batch processing")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dense layer forward pass failed: {e}")
    
    # Test 6: Dense Layer Without Bias
    try:
        layer_no_bias = Dense(input_size=2, output_size=3, use_bias=False)
        x = Tensor([[1, 2]])
        y = layer_no_bias(x)
        
        assert y.shape == (1, 3), f"No-bias output should be (1, 3), got {y.shape}"
        
        # Verify computation (should be just matrix multiplication)
        expected = np.dot(x.data, layer_no_bias.weights)
        assert np.allclose(y.data, expected), "No-bias computation incorrect"
        
        print("✅ Dense layer without bias: correct computation")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dense layer without bias failed: {e}")
    
    # Test 7: Dense Layer with Naive Matrix Multiplication
    try:
        layer_naive = Dense(input_size=2, output_size=2, use_naive_matmul=True)
        layer_optimized = Dense(input_size=2, output_size=2, use_naive_matmul=False)
        
        # Set same weights for comparison
        layer_optimized.weights = layer_naive.weights.copy()
        layer_optimized.bias = layer_naive.bias.copy() if layer_naive.bias is not None else None
        
        x = Tensor([[1, 2]])
        y_naive = layer_naive(x)
        y_optimized = layer_optimized(x)
        
        # Both should give same results
        assert np.allclose(y_naive.data, y_optimized.data), "Naive and optimized should give same results"
        
        print("✅ Dense layer naive vs optimized: consistent results")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dense layer naive matmul failed: {e}")
    
    # Test 8: Layer Composition
    try:
        # Create a simple network: Dense → ReLU → Dense
        dense1 = Dense(input_size=3, output_size=4)
        relu = ReLU()
        dense2 = Dense(input_size=4, output_size=2)
        
        x = Tensor([[1, -2, 3]])
        
        # Forward pass
        h1 = dense1(x)
        h2 = relu(h1)
        h3 = dense2(h2)
        
        # Check shapes
        assert h1.shape == (1, 4), f"Dense1 output should be (1, 4), got {h1.shape}"
        assert h2.shape == (1, 4), f"ReLU output should be (1, 4), got {h2.shape}"
        assert h3.shape == (1, 2), f"Dense2 output should be (1, 2), got {h3.shape}"
        
        # Check ReLU effect
        assert np.all(h2.data >= 0), "ReLU should produce non-negative values"
        
        print("✅ Layer composition: Dense → ReLU → Dense pipeline")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Layer composition failed: {e}")
    
    # Test 9: Different Layer Sizes
    try:
        # Test various layer sizes
        test_configs = [
            (1, 1),    # Minimal
            (10, 5),   # Medium
            (100, 50), # Large
            (784, 128) # MNIST-like
        ]
        
        for input_size, output_size in test_configs:
            layer = Dense(input_size=input_size, output_size=output_size)
            
            # Test with single sample
            x = Tensor(np.random.randn(1, input_size))
            y = layer(x)
            
            assert y.shape == (1, output_size), f"Size ({input_size}, {output_size}) failed: got {y.shape}"
            assert layer.weights.shape == (input_size, output_size), f"Weights shape wrong for ({input_size}, {output_size})"
        
        print("✅ Different layer sizes: (1,1), (10,5), (100,50), (784,128)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Different layer sizes failed: {e}")
    
    # Test 10: Real Neural Network Scenario
    try:
        # Simulate MNIST-like scenario: 784 → 128 → 64 → 10
        input_layer = Dense(input_size=784, output_size=128)
        hidden_layer = Dense(input_size=128, output_size=64)
        output_layer = Dense(input_size=64, output_size=10)
        
        relu1 = ReLU()
        relu2 = ReLU()
        softmax = Softmax()
        
        # Simulate flattened MNIST image
        x = Tensor(np.random.randn(32, 784))  # Batch of 32 images
        
        # Forward pass through network
        h1 = input_layer(x)
        h1_activated = relu1(h1)
        h2 = hidden_layer(h1_activated)
        h2_activated = relu2(h2)
        logits = output_layer(h2_activated)
        probabilities = softmax(logits)
        
        # Check final output
        assert probabilities.shape == (32, 10), f"Final output should be (32, 10), got {probabilities.shape}"
        
        # Check that probabilities sum to 1 for each sample
        row_sums = np.sum(probabilities.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each sample should have probabilities summing to 1"
        
        # Check that all intermediate shapes are correct
        assert h1.shape == (32, 128), f"Hidden 1 shape should be (32, 128), got {h1.shape}"
        assert h2.shape == (32, 64), f"Hidden 2 shape should be (32, 64), got {h2.shape}"
        assert logits.shape == (32, 10), f"Logits shape should be (32, 10), got {logits.shape}"
        
        print("✅ Real neural network scenario: MNIST-like 784→128→64→10 classification")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Real neural network scenario failed: {e}")
    
    # Results summary
    print(f"\n📊 Layers Module Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All layers tests passed! Your implementations support:")
        print("  • Matrix multiplication: naive implementation from scratch")
        print("  • Dense layers: linear transformations with learnable parameters")
        print("  • Weight initialization: proper random initialization")
        print("  • Bias handling: optional bias terms")
        print("  • Batch processing: multiple samples at once")
        print("  • Layer composition: building complete neural networks")
        print("  • Real ML scenarios: MNIST-like classification networks")
        print("📈 Progress: All Layer Functionality ✓")
        return True
    else:
        print("⚠️  Some layers tests failed. Common issues:")
        print("  • Check matrix multiplication implementation (triple nested loops)")
        print("  • Verify Dense layer forward pass (y = Wx + b)")
        print("  • Ensure proper weight initialization (not all zeros)")
        print("  • Check shape handling for different input/output sizes")
        print("  • Verify bias handling when use_bias=False")
        return False

# Run the comprehensive test
success = test_layers_comprehensive()

# %% [markdown]
"""
### 🧪 Integration Test: Layers in Complete Neural Networks

Let's test how your layers work in realistic neural network architectures.
"""

# %% nbgrader={"grade": true, "grade_id": "test-layers-integration", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_layers_integration():
    """Integration test with complete neural network architectures."""
    print("🔬 Testing layers in complete neural network architectures...")
    
    try:
        print("🧠 Building and testing different network architectures...")
        
        # Architecture 1: Simple Binary Classifier
        print("\n📊 Architecture 1: Binary Classification Network")
        binary_net = [
            Dense(input_size=4, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=1),
            Sigmoid()
        ]
        
        # Test with batch of samples
        x_binary = Tensor(np.random.randn(10, 4))  # 10 samples, 4 features
        
        # Forward pass through network
        current = x_binary
        for i, layer in enumerate(binary_net):
            current = layer(current)
            print(f"  Layer {i}: {current.shape}")
        
        # Verify final output is valid probabilities
        assert current.shape == (10, 1), f"Binary classifier output should be (10, 1), got {current.shape}"
        assert np.all((current.data >= 0) & (current.data <= 1)), "Binary probabilities should be in [0,1]"
        
        print("✅ Binary classification network: 4→8→4→1 with ReLU/Sigmoid")
        
        # Architecture 2: Multi-class Classifier
        print("\n📊 Architecture 2: Multi-class Classification Network")
        multiclass_net = [
            Dense(input_size=784, output_size=256),
            ReLU(),
            Dense(input_size=256, output_size=128),
            ReLU(),
            Dense(input_size=128, output_size=10),
            Softmax()
        ]
        
        # Simulate MNIST-like input
        x_mnist = Tensor(np.random.randn(5, 784))  # 5 images, 784 pixels
        
        current = x_mnist
        for i, layer in enumerate(multiclass_net):
            current = layer(current)
            print(f"  Layer {i}: {current.shape}")
        
        # Verify final output is valid probability distribution
        assert current.shape == (5, 10), f"Multi-class output should be (5, 10), got {current.shape}"
        row_sums = np.sum(current.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each sample should have probabilities summing to 1"
        
        print("✅ Multi-class classification network: 784→256→128→10 with Softmax")
        
        # Architecture 3: Deep Network
        print("\n📊 Architecture 3: Deep Network (5 layers)")
        deep_net = [
            Dense(input_size=100, output_size=80),
            ReLU(),
            Dense(input_size=80, output_size=60),
            ReLU(),
            Dense(input_size=60, output_size=40),
            ReLU(),
            Dense(input_size=40, output_size=20),
            ReLU(),
            Dense(input_size=20, output_size=3),
            Softmax()
        ]
        
        x_deep = Tensor(np.random.randn(8, 100))  # 8 samples, 100 features
        
        current = x_deep
        for i, layer in enumerate(deep_net):
            current = layer(current)
            if i % 2 == 0:  # Print every other layer to save space
                print(f"  Layer {i}: {current.shape}")
        
        assert current.shape == (8, 3), f"Deep network output should be (8, 3), got {current.shape}"
        
        print("✅ Deep network: 100→80→60→40→20→3 with multiple ReLU layers")
        
        # Test 4: Network with Different Activation Functions
        print("\n📊 Architecture 4: Mixed Activation Functions")
        mixed_net = [
            Dense(input_size=6, output_size=4),
            Tanh(),  # Zero-centered activation
            Dense(input_size=4, output_size=3),
            ReLU(),  # Sparse activation
            Dense(input_size=3, output_size=2),
            Sigmoid()  # Bounded activation
        ]
        
        x_mixed = Tensor(np.random.randn(3, 6))
        
        current = x_mixed
        for i, layer in enumerate(mixed_net):
            current = layer(current)
            print(f"  Layer {i}: {current.shape}, range: [{np.min(current.data):.3f}, {np.max(current.data):.3f}]")
        
        assert current.shape == (3, 2), f"Mixed network output should be (3, 2), got {current.shape}"
        
        print("✅ Mixed activations network: Tanh→ReLU→Sigmoid combinations")
        
        # Test 5: Parameter Counting
        print("\n📊 Parameter Analysis")
        
        def count_parameters(layer):
            """Count trainable parameters in a Dense layer."""
            if isinstance(layer, Dense):
                weight_params = layer.weights.size
                bias_params = layer.bias.size if layer.bias is not None else 0
                return weight_params + bias_params
            return 0
        
        # Count parameters in binary classifier
        total_params = sum(count_parameters(layer) for layer in binary_net)
        print(f"Binary classifier parameters: {total_params}")
        
        # Manual verification for first layer: 4*8 + 8 = 40
        first_dense = binary_net[0]
        expected_first = 4 * 8 + 8  # weights + bias
        actual_first = count_parameters(first_dense)
        assert actual_first == expected_first, f"First layer params: expected {expected_first}, got {actual_first}"
        
        print("✅ Parameter counting: weight and bias parameters calculated correctly")
        
        # Test 6: Gradient Flow Preparation
        print("\n📊 Gradient Flow Preparation")
        
        # Test that network can handle different input types
        test_inputs = [
            Tensor(np.zeros((1, 4))),      # All zeros
            Tensor(np.ones((1, 4))),       # All ones
            Tensor(np.random.randn(1, 4)), # Random
            Tensor(np.random.randn(1, 4) * 10)  # Large values
        ]
        
        for i, test_input in enumerate(test_inputs):
            current = test_input
            for layer in binary_net:
                current = layer(current)
            
            # Check for numerical stability
            assert not np.any(np.isnan(current.data)), f"Input {i} produced NaN"
            assert not np.any(np.isinf(current.data)), f"Input {i} produced Inf"
        
        print("✅ Numerical stability: networks handle various input ranges")
        
        print("\n🎉 Integration test passed! Your layers work correctly in:")
        print("  • Binary classification networks")
        print("  • Multi-class classification networks") 
        print("  • Deep networks with multiple hidden layers")
        print("  • Networks with mixed activation functions")
        print("  • Parameter counting and analysis")
        print("  • Numerical stability across input ranges")
        print("📈 Progress: Layers ready for complete neural networks!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("\n💡 This suggests an issue with:")
        print("  • Layer composition and chaining")
        print("  • Shape compatibility between layers")
        print("  • Activation function integration")
        print("  • Numerical stability in deep networks")
        print("  • Check your Dense layer and matrix multiplication")
        return False

# Run the integration test
success = test_layers_integration() and success

# Print final summary
print(f"\n{'='*60}")
print("🎯 LAYERS MODULE TESTING COMPLETE")
print(f"{'='*60}")

if success:
    print("🎉 CONGRATULATIONS! All layers tests passed!")
    print("\n✅ Your layers module successfully implements:")
    print("  • Matrix multiplication: naive implementation from scratch")
    print("  • Dense layers: y = Wx + b linear transformations")
    print("  • Weight initialization: proper random weight setup")
    print("  • Bias handling: optional bias terms")
    print("  • Batch processing: efficient multi-sample computation")
    print("  • Layer composition: building complete neural networks")
    print("  • Integration: works with all activation functions")
    print("  • Real ML scenarios: MNIST-like classification networks")
    print("\n🚀 You're ready to build complete neural network architectures!")
    print("📈 Final Progress: Layers Module ✓ COMPLETE")
else:
    print("⚠️  Some tests failed. Please review the error messages above.")
    print("\n🔧 To fix issues:")
    print("  1. Check your matrix multiplication implementation")
    print("  2. Verify Dense layer forward pass computation")
    print("  3. Ensure proper weight and bias initialization")
    print("  4. Test shape compatibility between layers")
    print("  5. Verify integration with activation functions")
    print("\n💪 Keep building! These layers are the foundation of all neural networks.")

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
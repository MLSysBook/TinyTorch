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
    try:
        from tensor_dev import Tensor
        from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    except ImportError:
        # If the local modules are not available, use relative imports
        from ..tensor.tensor_dev import Tensor
        from ..activations.activations_dev import ReLU, Sigmoid, Tanh, Softmax

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
## What Are Neural Network Layers?

### The Building Block Pattern
Neural networks are built by stacking **layers** - each layer is a function that:
1. **Takes input**: Tensor data from previous layer
2. **Transforms**: Applies mathematical operations (linear transformation + activation)
3. **Produces output**: New tensor data for next layer

### The Universal Pattern
Every layer follows this pattern:
```python
def layer(x):
    # 1. Linear transformation
    linear_output = x @ weights + bias
    
    # 2. Nonlinear activation
    output = activation(linear_output)
    
    return output
```

### Why This Works
- **Linear part**: Learns feature combinations
- **Nonlinear part**: Enables complex patterns
- **Stacking**: Multiple layers = more complex functions

### Mathematical Foundation
A neural network is function composition:
```
f(x) = layer_n(layer_{n-1}(...layer_2(layer_1(x))))
```

Each layer transforms the representation to be more useful for the final task.

### What We'll Build
1. **Matrix Multiplication**: The core operation powering all layers
2. **Dense Layer**: The fundamental building block of neural networks
3. **Integration**: How layers work with activations and tensors
"""

# %% [markdown]
"""
## Step 1: Matrix Multiplication - The Engine of Neural Networks

### What is Matrix Multiplication?
Matrix multiplication is the core operation that powers all neural network layers:

```
C = A @ B
```

Where:
- **A**: Input data (batch_size × input_features)
- **B**: Weight matrix (input_features × output_features)  
- **C**: Output data (batch_size × output_features)

### Why It's Essential
- **Feature combination**: Each output combines all input features
- **Learned weights**: B contains the learned parameters
- **Efficient computation**: Vectorized operations are much faster
- **Parallel processing**: GPUs are designed for matrix operations

### The Mathematical Definition
For matrices A (m×n) and B (n×p), the result C (m×p) is:
```
C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
```

### Visual Understanding
```
[1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
[3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
```

### Real-World Context
Every major operation in deep learning uses matrix multiplication:
- **Dense layers**: Linear transformations
- **Convolutional layers**: Convolution as matrix multiplication
- **Attention mechanisms**: Query-Key-Value computations
- **Embeddings**: Lookup tables as matrix multiplication
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-naive", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication using explicit for-loops.
    
    This helps you understand what matrix multiplication really does!
    
    TODO: Implement matrix multiplication using three nested for-loops.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Get the dimensions: m, n from A.shape and n2, p from B.shape
    2. Check compatibility: n must equal n2
    3. Create output matrix C of shape (m, p) filled with zeros
    4. Use three nested loops:
       - i loop: iterate through rows of A (0 to m-1)
       - j loop: iterate through columns of B (0 to p-1)
       - k loop: iterate through shared dimension (0 to n-1)
    5. For each (i,j), accumulate: C[i,j] += A[i,k] * B[k,j]
    
    EXAMPLE WALKTHROUGH:
    ```python
    A = [[1, 2],     B = [[5, 6],
         [3, 4]]          [7, 8]]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
    
    Result: [[19, 22], [43, 50]]
    ```
    
    IMPLEMENTATION HINTS:
    - Get dimensions: m, n = A.shape; n2, p = B.shape
    - Check compatibility: if n != n2: raise ValueError
    - Initialize result: C = np.zeros((m, p))
    - Triple nested loop: for i in range(m): for j in range(p): for k in range(n):
    - Accumulate sum: C[i,j] += A[i,k] * B[k,j]
    
    LEARNING CONNECTIONS:
    - This is what every neural network layer does internally
    - Understanding this helps debug shape mismatches
    - Forms the basis for efficient GPU computations
    - Essential for implementing custom layers
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
### 🧪 Test Your Matrix Multiplication

Once you implement the `matmul_naive` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_matrix_multiplication():
    """Test matrix multiplication implementation"""
    print("Testing matrix multiplication...")
    
    # Test simple 2x2 case
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_naive(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Matrix multiplication failed: expected {expected}, got {result}"
    
    # Compare with NumPy
    numpy_result = A @ B
    assert np.allclose(result, numpy_result), f"Doesn't match NumPy: got {result}, expected {numpy_result}"
    
    # Test different shapes
    A2 = np.array([[1, 2, 3]], dtype=np.float32)  # 1x3
    B2 = np.array([[4], [5], [6]], dtype=np.float32)  # 3x1
    result2 = matmul_naive(A2, B2)
    expected2 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
    
    assert np.allclose(result2, expected2), f"1x3 @ 3x1 failed: expected {expected2}, got {result2}"
    
    # Test 3x3 case
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    B3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # Identity
    result3 = matmul_naive(A3, B3)
    
    assert np.allclose(result3, A3), "Multiplication by identity should preserve matrix"
    
    # Test incompatible shapes
    A4 = np.array([[1, 2]], dtype=np.float32)  # 1x2
    B4 = np.array([[3], [4], [5]], dtype=np.float32)  # 3x1
    
    try:
        matmul_naive(A4, B4)
        assert False, "Should raise error for incompatible shapes"
    except ValueError as e:
        assert "Incompatible matrix dimensions" in str(e)
    
    print("✅ Matrix multiplication tests passed!")
    print(f"✅ 2x2 multiplication working correctly")
    print(f"✅ Matches NumPy's implementation")
    print(f"✅ Handles different shapes correctly")
    print(f"✅ Proper error handling for incompatible shapes")

# Run the test
test_matrix_multiplication()

# %% [markdown]
"""
## Step 2: Dense Layer - The Foundation of Neural Networks

### What is a Dense Layer?
A **Dense layer** (also called Linear or Fully Connected layer) is the fundamental building block of neural networks:

```python
output = input @ weights + bias
```

Where:
- **input**: Input data (batch_size × input_features)
- **weights**: Learned parameters (input_features × output_features)
- **bias**: Learned bias terms (output_features,)
- **output**: Transformed data (batch_size × output_features)

### Why Dense Layers Are Essential
1. **Feature transformation**: Learn meaningful combinations of input features
2. **Universal approximation**: Stack enough layers to approximate any function
3. **Learnable parameters**: Weights and biases are optimized during training
4. **Composability**: Can be stacked to create complex architectures

### The Mathematical Foundation
For input x, weight matrix W, and bias b:
```
y = xW + b
```

This is a linear transformation that:
- **Combines features**: Each output is a weighted sum of all inputs
- **Learns relationships**: Weights encode feature interactions
- **Adds flexibility**: Bias allows shifting the output

### Real-World Applications
- **Classification**: Transform features to class logits
- **Regression**: Transform features to continuous outputs
- **Representation learning**: Learn useful intermediate representations
- **Attention mechanisms**: Compute queries, keys, and values

### Design Decisions
- **Weight initialization**: Random initialization to break symmetry
- **Bias usage**: Usually included for flexibility
- **Activation**: Often followed by nonlinear activation
"""

# %% nbgrader={"grade": false, "grade_id": "dense-layer", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Dense:
    """
    Dense (Linear/Fully Connected) Layer
    
    Applies a linear transformation: y = xW + b
    
    This is the fundamental building block of neural networks.
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, use_naive_matmul: bool = False):
        """
        Initialize Dense layer with random weights and optional bias.
        
        TODO: Implement Dense layer initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store the layer parameters (input_size, output_size, use_bias, use_naive_matmul)
        2. Initialize weights with random values using proper scaling
        3. Initialize bias (if use_bias=True) with zeros
        4. Convert weights and bias to Tensor objects
        
        WEIGHT INITIALIZATION STRATEGY:
        - Use Xavier/Glorot initialization for better gradient flow
        - Scale: sqrt(2 / (input_size + output_size))
        - Random values: np.random.randn() * scale
        
        EXAMPLE USAGE:
        ```python
        layer = Dense(input_size=3, output_size=2)
        # Creates weight matrix of shape (3, 2) and bias of shape (2,)
        ```
        
        IMPLEMENTATION HINTS:
        - Store parameters: self.input_size, self.output_size, self.use_bias, self.use_naive_matmul
        - Weight shape: (input_size, output_size)
        - Bias shape: (output_size,) if use_bias else None
        - Use Xavier initialization: scale = np.sqrt(2.0 / (input_size + output_size))
        - Initialize weights: np.random.randn(input_size, output_size) * scale
        - Initialize bias: np.zeros(output_size) if use_bias else None
        - Convert to Tensors: self.weights = Tensor(weight_data), self.bias = Tensor(bias_data)
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.Linear() in PyTorch
        - Proper initialization prevents vanishing/exploding gradients
        - Bias adds flexibility to the linear transformation
        - Weight sharing across the layer enables parameter efficiency
        """
        ### BEGIN SOLUTION
        # Store layer parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.use_naive_matmul = use_naive_matmul
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        
        # Initialize weights with random values
        weight_data = np.random.randn(input_size, output_size) * scale
        self.weights = Tensor(weight_data)
        
        # Initialize bias
        if use_bias:
            bias_data = np.zeros(output_size)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Dense layer.
        
        TODO: Implement the forward pass: y = xW + b
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Perform matrix multiplication: x @ self.weights
        2. Add bias if present: result + self.bias
        3. Return the result as a Tensor
        
        MATRIX MULTIPLICATION OPTIONS:
        - If use_naive_matmul=True: Use our custom matmul_naive function
        - If use_naive_matmul=False: Use NumPy's built-in @ operator
        
        EXAMPLE USAGE:
        ```python
        layer = Dense(input_size=3, output_size=2)
        input_data = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        output = layer(input_data)        # Shape: (1, 2)
        ```
        
        IMPLEMENTATION HINTS:
        - Matrix multiplication: x.data @ self.weights.data (or use matmul_naive)
        - For naive implementation: matmul_naive(x.data, self.weights.data)
        - Add bias: result + self.bias.data (broadcasting handles shape)
        - Return as Tensor: return Tensor(final_result)
        - Handle both cases: with and without bias
        
        LEARNING CONNECTIONS:
        - This is the core operation in every neural network layer
        - Matrix multiplication combines all input features
        - Bias addition allows shifting the output distribution
        - The result feeds into activation functions
        """
        ### BEGIN SOLUTION
        # Perform matrix multiplication
        if self.use_naive_matmul:
            linear_output = matmul_naive(x.data, self.weights.data)
        else:
            linear_output = x.data @ self.weights.data
        
        # Add bias if present
        if self.use_bias and self.bias is not None:
            linear_output = linear_output + self.bias.data
        
        return Tensor(linear_output)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Make the layer callable: layer(x) instead of layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Dense Layer

Once you implement the Dense layer above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense-layer", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_dense_layer():
    """Test Dense layer implementation"""
    print("Testing Dense layer...")
    
    # Test layer creation
    layer = Dense(input_size=3, output_size=2)
    
    # Check weight and bias shapes
    assert layer.weights.shape == (3, 2), f"Weight shape should be (3, 2), got {layer.weights.shape}"
    assert layer.bias is not None, "Bias should not be None when use_bias=True"
    assert layer.bias.shape == (2,), f"Bias shape should be (2,), got {layer.bias.shape}"
    
    # Test forward pass
    input_data = Tensor([[1, 2, 3]])  # Shape: (1, 3)
    output = layer(input_data)
    
    # Check output shape
    assert output.shape == (1, 2), f"Output shape should be (1, 2), got {output.shape}"
    
    # Test batch processing
    batch_input = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
    batch_output = layer(batch_input)
    
    assert batch_output.shape == (2, 2), f"Batch output shape should be (2, 2), got {batch_output.shape}"
    
    # Test without bias
    no_bias_layer = Dense(input_size=3, output_size=2, use_bias=False)
    assert no_bias_layer.bias is None, "Layer without bias should have None bias"
    
    no_bias_output = no_bias_layer(input_data)
    assert no_bias_output.shape == (1, 2), "No-bias layer should still produce correct shape"
    
    # Test with naive matrix multiplication
    naive_layer = Dense(input_size=3, output_size=2, use_naive_matmul=True)
    naive_output = naive_layer(input_data)
    assert naive_output.shape == (1, 2), "Naive matmul should produce correct shape"
    
    # Test that different inputs produce different outputs
    input1 = Tensor([[1, 0, 0]])
    input2 = Tensor([[0, 1, 0]])
    
    output1 = layer(input1)
    output2 = layer(input2)
    
    # Should not be equal (with high probability due to random initialization)
    assert not np.allclose(output1.data, output2.data), "Different inputs should produce different outputs"
    
    # Test linearity property: layer(a*x) = a*layer(x)
    scale = 2.0
    scaled_input = Tensor([[2, 4, 6]])  # 2 * [1, 2, 3]
    scaled_output = layer(scaled_input)
    
    # Due to bias, this won't be exactly 2*output, but the linear part should scale
    print("✅ Dense layer tests passed!")
    print(f"✅ Correct weight and bias initialization")
    print(f"✅ Forward pass produces correct shapes")
    print(f"✅ Batch processing works correctly")
    print(f"✅ Bias and no-bias variants work")
    print(f"✅ Naive matrix multiplication option works")

# Run the test
test_dense_layer()

# %% [markdown]
"""
## Step 3: Layer Integration with Activations

### Building Complete Neural Network Components
Now let's see how Dense layers work with activation functions to create complete neural network components:

```python
# Complete neural network layer
x = input_data
linear_output = dense_layer(x)
final_output = activation_function(linear_output)
```

### Why This Combination Works
1. **Linear transformation**: Dense layer learns feature combinations
2. **Nonlinear activation**: Enables complex pattern recognition
3. **Stacking**: Multiple layer+activation pairs create deep networks
4. **Universal approximation**: Can approximate any continuous function

### Real-World Layer Patterns
- **Hidden layers**: Dense + ReLU (most common)
- **Output layers**: Dense + Softmax (classification) or Dense + Sigmoid (binary)
- **Gated layers**: Dense + Sigmoid (for gates in LSTM/GRU)
- **Attention layers**: Dense + Softmax (for attention weights)
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-activation-integration", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_layer_activation_integration():
    """Test Dense layer integration with activation functions"""
    print("Testing layer-activation integration...")
    
    # Create layer and activation functions
    layer = Dense(input_size=4, output_size=3)
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    softmax = Softmax()
    
    # Test input
    input_data = Tensor([[1, -2, 3, -4], [2, 1, -1, 3]])  # Shape: (2, 4)
    
    # Test Dense + ReLU (common hidden layer pattern)
    linear_output = layer(input_data)
    relu_output = relu(linear_output)
    
    assert relu_output.shape == (2, 3), "ReLU output should preserve shape"
    assert np.all(relu_output.data >= 0), "ReLU output should be non-negative"
    
    # Test Dense + Softmax (classification output pattern)
    softmax_output = softmax(linear_output)
    
    assert softmax_output.shape == (2, 3), "Softmax output should preserve shape"
    
    # Each row should sum to 1 (probability distribution)
    for i in range(2):
        row_sum = np.sum(softmax_output.data[i])
        assert abs(row_sum - 1.0) < 1e-6, f"Row {i} should sum to 1, got {row_sum}"
    
    # Test Dense + Sigmoid (binary classification pattern)
    sigmoid_output = sigmoid(linear_output)
    
    assert sigmoid_output.shape == (2, 3), "Sigmoid output should preserve shape"
    assert np.all(sigmoid_output.data > 0), "Sigmoid output should be positive"
    assert np.all(sigmoid_output.data < 1), "Sigmoid output should be less than 1"
    
    # Test Dense + Tanh (hidden layer with centered outputs)
    tanh_output = tanh(linear_output)
    
    assert tanh_output.shape == (2, 3), "Tanh output should preserve shape"
    assert np.all(tanh_output.data > -1), "Tanh output should be > -1"
    assert np.all(tanh_output.data < 1), "Tanh output should be < 1"
    
    # Test chained layers (simple 2-layer network)
    layer1 = Dense(input_size=4, output_size=5)
    layer2 = Dense(input_size=5, output_size=3)
    
    # Forward pass through 2-layer network
    hidden = relu(layer1(input_data))
    output = softmax(layer2(hidden))
    
    assert output.shape == (2, 3), "2-layer network should produce correct output shape"
    
    # Each output should be a valid probability distribution
    for i in range(2):
        row_sum = np.sum(output.data[i])
        assert abs(row_sum - 1.0) < 1e-6, f"Network output row {i} should sum to 1"
    
    # Test that layers are learning-ready (have parameters)
    assert hasattr(layer1, 'weights'), "Layer should have weights"
    assert hasattr(layer1, 'bias'), "Layer should have bias"
    assert isinstance(layer1.weights, Tensor), "Weights should be Tensor"
    assert isinstance(layer1.bias, Tensor), "Bias should be Tensor"
    
    print("✅ Layer-activation integration tests passed!")
    print(f"✅ Dense + ReLU working correctly")
    print(f"✅ Dense + Softmax producing valid probabilities")
    print(f"✅ Dense + Sigmoid bounded correctly")
    print(f"✅ Dense + Tanh centered correctly")
    print(f"✅ Multi-layer networks working")
    print(f"✅ All components ready for training!")

# Run the test
test_layer_activation_integration()

# %% [markdown]
"""
## 🎯 Module Summary: Neural Network Layers Mastery!

Congratulations! You've successfully implemented the fundamental building blocks of neural networks:

### ✅ What You've Built
- **Matrix Multiplication**: The core operation powering all neural network computations
- **Dense Layer**: The fundamental building block with proper weight initialization
- **Integration**: How layers work with activation functions to create complete neural components
- **Flexibility**: Support for bias/no-bias and naive/optimized matrix multiplication

### ✅ Key Learning Outcomes
- **Understanding**: How linear transformations enable feature learning
- **Implementation**: Built layers from scratch with proper initialization
- **Testing**: Progressive validation with immediate feedback
- **Integration**: Saw how layers compose with activations for complete functionality
- **Real-world skills**: Understanding the mathematics behind neural networks

### ✅ Mathematical Mastery
- **Matrix Multiplication**: C[i,j] = Σ(A[i,k] * B[k,j]) - implemented with loops
- **Linear Transformation**: y = xW + b - the heart of neural networks
- **Xavier Initialization**: Proper weight scaling for stable gradients
- **Composition**: How multiple layers create complex functions

### ✅ Professional Skills Developed
- **Algorithm implementation**: From mathematical definition to working code
- **Performance considerations**: Naive vs optimized implementations
- **API design**: Clean, consistent interfaces for layer creation and usage
- **Testing methodology**: Unit tests, integration tests, and edge case handling

### ✅ Ready for Next Steps
Your layers are now ready to power:
- **Complete Networks**: Stack multiple layers with activations
- **Training**: Gradient computation and parameter updates
- **Specialized Architectures**: CNNs, RNNs, Transformers all use these foundations
- **Real Applications**: Image classification, NLP, game playing, etc.

### 🔗 Connection to Real ML Systems
Your implementations mirror production frameworks:
- **PyTorch**: `torch.nn.Linear()` - same mathematical operations
- **TensorFlow**: `tf.keras.layers.Dense()` - identical functionality
- **Industry**: Every major neural network uses these exact computations

### 🎯 The Power of Linear Algebra
You've unlocked the mathematical foundation of AI:
- **Feature combination**: Each layer learns how to combine input features
- **Representation learning**: Layers automatically discover useful representations
- **Universal approximation**: Stack enough layers to approximate any function
- **Scalability**: Same operations work from small networks to massive language models

### 🧠 Deep Learning Insights
- **Why deep networks work**: Multiple layers = multiple levels of abstraction
- **Parameter efficiency**: Shared weights enable learning with limited data
- **Gradient flow**: Proper initialization enables training deep networks
- **Composability**: Simple components combine to create complex intelligence

**Next Module**: Networks - Composing your layers into complete neural network architectures!

Your layers are the building blocks. Now let's assemble them into powerful neural networks that can learn to solve complex problems!
""" 
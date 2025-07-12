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
# Module 2: Layers - Neural Network Building Blocks

Welcome to the Layers module! This is where neural networks begin. You'll implement the fundamental building blocks that transform tensors.

## Learning Goals
- Understand layers as functions that transform tensors: `y = f(x)`
- Implement Dense layers with linear transformations: `y = Wx + b`
- Use activation functions from the activations module for nonlinearity
- See how neural networks are just function composition
- Build intuition before diving into training

## Build → Use → Understand
1. **Build**: Dense layers using activation functions as building blocks
2. **Use**: Transform tensors and see immediate results
3. **Understand**: How neural networks transform information

## Module Dependencies
This module builds on the **activations** module:
- **activations** → **layers** → **networks**
- Clean separation of concerns: math functions → layer building blocks → full networks
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/layers/layers_dev.py`  
**Building Side:** Code exports to `tinytorch.core.layers`

```python
# Final package structure:
from tinytorch.core.layers import Dense, Conv2D  # All layers together!
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.tensor import Tensor
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn`
- **Consistency:** All layers (Dense, Conv2D) live together in `core.layers`
"""

# %%
#| default_exp core.layers

# Setup and imports
import numpy as np
import sys
from typing import Union, Optional, Callable
import math

# %%
#| export
import numpy as np
import math
import sys
from typing import Union, Optional, Callable

# Import from the main package (rock solid foundation)
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Tanh

# print("🔥 TinyTorch Layers Module")
# print(f"NumPy version: {np.__version__}")
# print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
# print("Ready to build neural network layers!")

# %% [markdown]
"""
## Step 1: What is a Layer?

### Definition
A **layer** is a function that transforms tensors. Think of it as a mathematical operation that takes input data and produces output data:

```
Input Tensor → Layer → Output Tensor
```

### Why Layers Matter in Neural Networks
Layers are the fundamental building blocks of all neural networks because:
- **Modularity**: Each layer has a specific job (linear transformation, nonlinearity, etc.)
- **Composability**: Layers can be combined to create complex functions
- **Learnability**: Each layer has parameters that can be learned from data
- **Interpretability**: Different layers learn different features

### The Fundamental Insight
**Neural networks are just function composition!**
```
x → Layer1 → Layer2 → Layer3 → y
```

Each layer transforms the data, and the final output is the composition of all these transformations.

### Real-World Examples
- **Dense Layer**: Learns linear relationships between features
- **Convolutional Layer**: Learns spatial patterns in images
- **Recurrent Layer**: Learns temporal patterns in sequences
- **Activation Layer**: Adds nonlinearity to make networks powerful

### Visual Intuition
```
Input: [1, 2, 3] (3 features)
Dense Layer: y = Wx + b
Weights W: [[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]] (2×3 matrix)
Bias b: [0.1, 0.2] (2 values)
Output: [0.1*1 + 0.2*2 + 0.3*3 + 0.1,
         0.4*1 + 0.5*2 + 0.6*3 + 0.2] = [1.4, 3.2]
```

Let's start with the most important layer: **Dense** (also called Linear or Fully Connected).
"""

# %% [markdown]
"""
## Step 2: Understanding Matrix Multiplication

Before we build layers, let's understand the core operation: **matrix multiplication**. This is what powers all neural network computations.

### Why Matrix Multiplication Matters
- **Efficiency**: Process multiple inputs at once
- **Parallelization**: GPU acceleration works great with matrix operations
- **Batch processing**: Handle multiple samples simultaneously
- **Mathematical foundation**: Linear algebra is the language of neural networks

### The Math Behind It
For matrices A (m×n) and B (n×p), the result C (m×p) is:
```
C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
```

### Visual Example
```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

C = A @ B = [[1*5 + 2*7,  1*6 + 2*8],
              [3*5 + 4*7,  3*6 + 4*8]]
  = [[19, 22],
     [43, 50]]
```

Let's implement this step by step!
"""

# %%
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
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication using explicit for-loops.
    
    This helps you understand what matrix multiplication really does!
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"Matrix shapes don't match: A({m},{n}) @ B({n2},{p})"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# %% [markdown]
"""
### 🧪 Test Your Matrix Multiplication
"""

# %%
# Test matrix multiplication
print("Testing matrix multiplication...")

try:
    # Test case 1: Simple 2x2 matrices
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_naive(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    print(f"✅ Matrix A:\n{A}")
    print(f"✅ Matrix B:\n{B}")
    print(f"✅ Your result:\n{result}")
    print(f"✅ Expected:\n{expected}")
    
    assert np.allclose(result, expected), "❌ Result doesn't match expected!"
    print("🎉 Matrix multiplication works!")
    
    # Test case 2: Compare with NumPy
    numpy_result = A @ B
    assert np.allclose(result, numpy_result), "❌ Doesn't match NumPy result!"
    print("✅ Matches NumPy implementation!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement matmul_naive above!")

# %% [markdown]
"""
## Step 3: Building the Dense Layer

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

# %%
#| export
class Dense:
    """
    Dense (Linear) Layer: y = Wx + b
    
    The fundamental building block of neural networks.
    Performs linear transformation: matrix multiplication + bias addition.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        use_bias: Whether to include bias term (default: True)
        use_naive_matmul: Whether to use naive matrix multiplication (for learning)
        
    TODO: Implement the Dense layer with weight initialization and forward pass.
    
    APPROACH:
    1. Store layer parameters (input_size, output_size, use_bias, use_naive_matmul)
    2. Initialize weights with small random values (Xavier/Glorot initialization)
    3. Initialize bias to zeros (if use_bias=True)
    4. Implement forward pass using matrix multiplication and bias addition
    
    EXAMPLE:
    layer = Dense(input_size=3, output_size=2)
    x = Tensor([[1, 2, 3]])  # batch_size=1, input_size=3
    y = layer(x)  # shape: (1, 2)
    
    HINTS:
    - Use np.random.randn() for random initialization
    - Scale weights by sqrt(2/(input_size + output_size)) for Xavier init
    - Store weights and bias as numpy arrays
    - Use matmul_naive or @ operator based on use_naive_matmul flag
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, 
                 use_naive_matmul: bool = False):
        """
        Initialize Dense layer with random weights.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            use_bias: Whether to include bias term
            use_naive_matmul: Use naive matrix multiplication (for learning)
            
        TODO: 
        1. Store layer parameters (input_size, output_size, use_bias, use_naive_matmul)
        2. Initialize weights with small random values
        3. Initialize bias to zeros (if use_bias=True)
        
        STEP-BY-STEP:
        1. Store the parameters as instance variables
        2. Calculate scale factor for Xavier initialization: sqrt(2/(input_size + output_size))
        3. Initialize weights: np.random.randn(input_size, output_size) * scale
        4. If use_bias=True, initialize bias: np.zeros(output_size)
        5. If use_bias=False, set bias to None
        
        EXAMPLE:
        Dense(3, 2) creates:
        - weights: shape (3, 2) with small random values
        - bias: shape (2,) with zeros
        """
        raise NotImplementedError("Student implementation required")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = Wx + b
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
            
        TODO: Implement matrix multiplication and bias addition
        - Use self.use_naive_matmul to choose between NumPy and naive implementation
        - If use_naive_matmul=True, use matmul_naive(x.data, self.weights)
        - If use_naive_matmul=False, use x.data @ self.weights
        - Add bias if self.use_bias=True
        
        STEP-BY-STEP:
        1. Perform matrix multiplication: Wx
           - If use_naive_matmul: result = matmul_naive(x.data, self.weights)
           - Else: result = x.data @ self.weights
        2. Add bias if use_bias: result += self.bias
        3. Return Tensor(result)
        
        EXAMPLE:
        Input x: Tensor([[1, 2, 3]])  # shape (1, 3)
        Weights: shape (3, 2)
        Output: Tensor([[val1, val2]])  # shape (1, 2)
        
        HINTS:
        - x.data gives you the numpy array
        - self.weights is your weight matrix
        - Use broadcasting for bias addition: result + self.bias
        - Return Tensor(result) to wrap the result
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %%
#| hide
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
            use_bias: Whether to include bias term
            use_naive_matmul: Use naive matrix multiplication (for learning)
        """
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
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = Wx + b
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Matrix multiplication
        if self.use_naive_matmul:
            result = matmul_naive(x.data, self.weights)
        else:
            result = x.data @ self.weights
        
        # Add bias
        if self.use_bias:
            result += self.bias
        
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Dense Layer
"""

# %%
# Test Dense layer
print("Testing Dense layer...")

try:
    # Test basic Dense layer
    layer = Dense(input_size=3, output_size=2, use_bias=True)
    x = Tensor([[1, 2, 3]])  # batch_size=1, input_size=3
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Layer weights shape: {layer.weights.shape}")
    print(f"✅ Layer bias shape: {layer.bias.shape}")
    
    y = layer(x)
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output: {y}")
    
    # Test without bias
    layer_no_bias = Dense(input_size=2, output_size=1, use_bias=False)
    x2 = Tensor([[1, 2]])
    y2 = layer_no_bias(x2)
    print(f"✅ No bias output: {y2}")
    
    # Test naive matrix multiplication
    layer_naive = Dense(input_size=2, output_size=2, use_naive_matmul=True)
    x3 = Tensor([[1, 2]])
    y3 = layer_naive(x3)
    print(f"✅ Naive matmul output: {y3}")
    
    print("\n🎉 All Dense layer tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Dense layer above!")

# %% [markdown]
"""
## Step 4: Composing Layers with Activations

Now let's see how layers work together! A neural network is just layers composed with activation functions.

### Why Layer Composition Matters
- **Nonlinearity**: Activation functions make networks powerful
- **Feature learning**: Each layer learns different levels of features
- **Universal approximation**: Can approximate any function
- **Modularity**: Easy to experiment with different architectures

### The Pattern
```
Input → Dense → Activation → Dense → Activation → Output
```

### Real-World Example
```
Input: [1, 2, 3] (3 features)
Dense(3→2): [1.4, 2.8] (linear transformation)
ReLU: [1.4, 2.8] (nonlinearity)
Dense(2→1): [3.2] (final prediction)
```

Let's build a simple network!
"""

# %%
# Test layer composition
print("Testing layer composition...")

try:
    # Create a simple network: Dense → ReLU → Dense
    dense1 = Dense(input_size=3, output_size=2)
    relu = ReLU()
    dense2 = Dense(input_size=2, output_size=1)
    
    # Test input
    x = Tensor([[1, 2, 3]])
    print(f"✅ Input: {x}")
    
    # Forward pass through the network
    h1 = dense1(x)
    print(f"✅ After Dense1: {h1}")
    
    h2 = relu(h1)
    print(f"✅ After ReLU: {h2}")
    
    y = dense2(h2)
    print(f"✅ Final output: {y}")
    
    print("\n🎉 Layer composition works!")
    print("This is how neural networks work: layers + activations!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure all your layers and activations are working!")

# %% [markdown]
"""
## Step 5: Performance Comparison

Let's compare our naive matrix multiplication with NumPy's optimized version to understand why optimization matters in ML.

### Why Performance Matters
- **Training time**: Neural networks train for hours/days
- **Inference speed**: Real-time applications need fast predictions
- **GPU utilization**: Optimized operations use hardware efficiently
- **Scalability**: Large models need efficient implementations
"""

# %%
# Performance comparison
print("Comparing naive vs NumPy matrix multiplication...")

try:
    import time
    
    # Create test matrices
    A = np.random.randn(100, 100).astype(np.float32)
    B = np.random.randn(100, 100).astype(np.float32)
    
    # Time naive implementation
    start_time = time.time()
    result_naive = matmul_naive(A, B)
    naive_time = time.time() - start_time
    
    # Time NumPy implementation
    start_time = time.time()
    result_numpy = A @ B
    numpy_time = time.time() - start_time
    
    print(f"✅ Naive time: {naive_time:.4f} seconds")
    print(f"✅ NumPy time: {numpy_time:.4f} seconds")
    print(f"✅ Speedup: {naive_time/numpy_time:.1f}x faster")
    
    # Verify correctness
    assert np.allclose(result_naive, result_numpy), "Results don't match!"
    print("✅ Results are identical!")
    
    print("\n💡 This is why we use optimized libraries in production!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've built the foundation of neural network layers:

### What You've Accomplished
✅ **Matrix Multiplication**: Understanding the core operation  
✅ **Dense Layer**: Linear transformation with weights and bias  
✅ **Layer Composition**: Combining layers with activations  
✅ **Performance Awareness**: Understanding optimization importance  
✅ **Testing**: Immediate feedback on your implementations  

### Key Concepts You've Learned
- **Layers** are functions that transform tensors
- **Matrix multiplication** powers all neural network computations
- **Dense layers** perform linear transformations: `y = Wx + b`
- **Layer composition** creates complex functions from simple building blocks
- **Performance** matters for real-world ML applications

### What's Next
In the next modules, you'll build on this foundation:
- **Networks**: Compose layers into complete models
- **Training**: Learn parameters with gradients and optimization
- **Convolutional layers**: Process spatial data like images
- **Recurrent layers**: Process sequential data like text

### Real-World Connection
Your Dense layer is now ready to:
- Learn patterns in data through weight updates
- Transform features for classification and regression
- Serve as building blocks for complex architectures
- Integrate with the rest of the TinyTorch ecosystem

**Ready for the next challenge?** Let's move on to building complete neural networks!
"""

# %%
# Final verification
print("\n" + "="*50)
print("🎉 LAYERS MODULE COMPLETE!")
print("="*50)
print("✅ Matrix multiplication understanding")
print("✅ Dense layer implementation")
print("✅ Layer composition with activations")
print("✅ Performance awareness")
print("✅ Comprehensive testing")
print("\n🚀 Ready to build networks in the next module!") 
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

## Module → Package Structure
**🎓 Teaching vs. 🔧 Building**: 
- **Learning side**: Work in `modules/layers/layers_dev.py`  
- **Building side**: Exports to `tinytorch/core/layers.py`

This module builds the fundamental transformations that compose into neural networks.
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
from tinytorch.core.tensor import Tensor

# Import activation functions from the activations module
from tinytorch.core.activations import ReLU, Sigmoid, Tanh

# Import our Tensor class
# sys.path.append('../../')
# from modules.tensor.tensor_dev import Tensor

# print("🔥 TinyTorch Layers Module")
# print(f"NumPy version: {np.__version__}")
# print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
# print("Ready to build neural network layers!")

# %% [markdown]
"""
## Step 1: What is a Layer?

A **layer** is a function that transforms tensors. Think of it as:
- **Input**: Tensor with some shape
- **Transformation**: Mathematical operation (linear, nonlinear, etc.)
- **Output**: Tensor with possibly different shape

**The fundamental insight**: Neural networks are just function composition!
```
x → Layer1 → Layer2 → Layer3 → y
```

**Why layers matter**:
- They're the building blocks of all neural networks
- Each layer learns a different transformation
- Composing layers creates complex functions
- Understanding layers = understanding neural networks

Let's start with the most important layer: **Dense** (also called Linear or Fully Connected).
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
        - If use_naive_matmul=True, use matmul_naive(x.data, self.weights.data)
        - If use_naive_matmul=False, use x.data @ self.weights.data
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
        """Initialize Dense layer with random weights."""
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.use_naive_matmul = use_naive_matmul
        
        # Initialize weights with Xavier/Glorot initialization
        # This helps with gradient flow during training
        limit = math.sqrt(6.0 / (input_size + output_size))
        self.weights = Tensor(
            np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        )
        
        # Initialize bias to zeros
        if use_bias:
            self.bias = Tensor(np.zeros(output_size, dtype=np.float32))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = Wx + b"""
        # Choose matrix multiplication implementation
        if self.use_naive_matmul:
            # Use naive implementation (for learning)
            output = Tensor(matmul_naive(x.data, self.weights.data))
        else:
            # Use NumPy's optimized implementation (for speed)
            output = Tensor(x.data @ self.weights.data)
        
        # Add bias if present
        if self.bias is not None:
            output = Tensor(output.data + self.bias.data)
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Dense Layer

Once you implement the Dense layer above, run this cell to test it:
"""

# %%
# Test the Dense layer
try:
    print("=== Testing Dense Layer ===")
    
    # Create a simple Dense layer: 3 inputs → 2 outputs
    layer = Dense(input_size=3, output_size=2)
    print(f"Created Dense layer: {layer.input_size} → {layer.output_size}")
    print(f"Weights shape: {layer.weights.shape}")
    print(f"Bias shape: {layer.bias.shape if layer.bias else 'No bias'}")
    
    # Test with a single example
    x = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input: {x.data}")
    print(f"Output: {y.data}")
    
    # Test with batch
    x_batch = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    y_batch = layer(x_batch)
    print(f"\nBatch input shape: {x_batch.shape}")
    print(f"Batch output shape: {y_batch.shape}")
    
    print("✅ Dense layer working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Dense layer above!")

# %% [markdown]
"""
## Step 1.5: Understanding Matrix Multiplication

Let's compare the naive matrix multiplication with NumPy's optimized version!
"""

# %%
# Test matrix multiplication implementations
try:
    print("=== Testing Matrix Multiplication Implementations ===")
    
    # Create small test matrices
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2x2
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)  # 2x2
    
    print(f"Matrix A (2x2):\n{A}")
    print(f"Matrix B (2x2):\n{B}")
    
    # Test NumPy's implementation
    C_numpy = A @ B
    print(f"\nNumPy result (A @ B):\n{C_numpy}")
    
    # Test naive implementation
    C_naive = matmul_naive(A, B)
    print(f"Naive result:\n{C_naive}")
    
    # Compare results
    if np.allclose(C_numpy, C_naive):
        print("✅ Both implementations give the same result!")
    else:
        print("❌ Results differ! Check your naive implementation.")
    
    # Show the computation step by step
    print(f"\n📊 Step-by-step computation for C[0,0]:")
    print(f"C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]")
    print(f"C[0,0] = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]}")
    print(f"C[0,0] = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]}")
    print(f"C[0,0] = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
    print(f"Expected: {C_numpy[0,0]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement matmul_naive above!")

# %%
# Performance comparison
try:
    print("=== Performance Comparison ===")
    
    # Create larger matrices for timing
    size = 50
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    import time
    
    # Time NumPy implementation
    start_time = time.time()
    C_numpy = A @ B
    numpy_time = time.time() - start_time
    
    # Time naive implementation
    start_time = time.time()
    C_naive = matmul_naive(A, B)
    naive_time = time.time() - start_time
    
    print(f"Matrix size: {size}x{size}")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"Naive time: {naive_time:.6f} seconds")
    print(f"Speedup: {naive_time/numpy_time:.1f}x slower")
    
    # Verify results are the same
    if np.allclose(C_numpy, C_naive):
        print("✅ Results are identical!")
    else:
        print("❌ Results differ!")
    
    print(f"\n💡 Why is NumPy so much faster?")
    print(f"   • Vectorized operations (no Python loops)")
    print(f"   • Optimized C/Fortran backend")
    print(f"   • Cache-friendly memory access")
    print(f"   • Parallel processing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement matmul_naive above!")

# %%
# Test Dense layer with both implementations
try:
    print("=== Testing Dense Layer with Both Implementations ===")
    
    # Create test data
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    
    # Test with NumPy implementation
    layer_numpy = Dense(input_size=3, output_size=2, use_naive_matmul=False)
    y_numpy = layer_numpy(x)
    
    # Test with naive implementation
    layer_naive = Dense(input_size=3, output_size=2, use_naive_matmul=True)
    y_naive = layer_naive(x)
    
    print(f"Input shape: {x.shape}")
    print(f"NumPy output: {y_numpy.data}")
    print(f"Naive output: {y_naive.data}")
    
    # Compare results
    if np.allclose(y_numpy.data, y_naive.data):
        print("✅ Both Dense implementations give the same result!")
    else:
        print("❌ Results differ! Check your implementations.")
    
    print(f"\n🎯 Key Insight:")
    print(f"   • Both implementations compute the same mathematical operation")
    print(f"   • NumPy is much faster but hides the computation")
    print(f"   • Naive implementation shows you exactly what's happening")
    print(f"   • Understanding the naive version helps you understand neural networks!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement both matmul_naive and Dense layer!")

# %% [markdown]
"""
## Step 2: Activation Functions - Adding Nonlinearity

Now we'll use the activation functions from the **activations** module! 

**Clean Architecture**: We import the activation functions rather than redefining them:
```python
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
```

**Why this matters**:
- **Separation of concerns**: Math functions vs. layer building blocks
- **Reusability**: Activations can be used anywhere in the system
- **Maintainability**: One place to update activation implementations
- **Composability**: Clean imports make neural networks easier to build

**Why nonlinearity matters**: Without it, stacking layers is pointless!
```
Linear → Linear → Linear = Just one big Linear transformation
Linear → NonLinear → Linear = Can learn complex patterns
```
"""

# %% [markdown]
"""
### 🧪 Test Activation Functions from Activations Module

Let's test that we can use the activation functions from the activations module:
"""

# %%
# Test activation functions from activations module
try:
    print("=== Testing Activation Functions from Activations Module ===")
    
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    print(f"Input: {x.data}")
    
    # Test ReLU from activations module
    relu = ReLU()
    y_relu = relu(x)
    print(f"ReLU output: {y_relu.data}")
    
    # Test Sigmoid from activations module
    sigmoid = Sigmoid()
    y_sigmoid = sigmoid(x)
    print(f"Sigmoid output: {y_sigmoid.data}")
    
    # Test Tanh from activations module
    tanh = Tanh()
    y_tanh = tanh(x)
    print(f"Tanh output: {y_tanh.data}")
    
    print("✅ Activation functions from activations module working!")
    print("🎉 Clean architecture: layers module uses activations module!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the activations module is properly exported!")

# %% [markdown]
"""
## Step 3: Layer Composition - Building Neural Networks

Now comes the magic! We can **compose** layers to build neural networks:

```
Input → Dense → ReLU → Dense → Sigmoid → Output
```

This is a 2-layer neural network that can learn complex nonlinear patterns!

**Notice the clean architecture**:
- Dense layers handle linear transformations
- Activation functions (from activations module) handle nonlinearity
- Composition creates complex behaviors from simple building blocks
"""

# %%
# Build a simple 2-layer neural network
try:
    print("=== Building a 2-Layer Neural Network ===")
    
    # Network architecture: 3 → 4 → 2
    # Input: 3 features
    # Hidden: 4 neurons with ReLU
    # Output: 2 neurons with Sigmoid
    
    layer1 = Dense(input_size=3, output_size=4)
    activation1 = ReLU()  # From activations module
    layer2 = Dense(input_size=4, output_size=2)
    activation2 = Sigmoid()  # From activations module
    
    print("Network architecture:")
    print(f"  Input: 3 features")
    print(f"  Hidden: {layer1.input_size} → {layer1.output_size} (Dense + ReLU)")
    print(f"  Output: {layer2.input_size} → {layer2.output_size} (Dense + Sigmoid)")
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 examples, 3 features each
    print(f"\nInput shape: {x.shape}")
    print(f"Input data: {x.data}")
    
    # Forward pass through the network
    h1 = layer1(x)           # Dense layer 1
    h1_activated = activation1(h1)  # ReLU activation
    h2 = layer2(h1_activated)       # Dense layer 2  
    output = activation2(h2)        # Sigmoid activation
    
    print(f"\nAfter layer 1: {h1.shape}")
    print(f"After ReLU: {h1_activated.shape}")
    print(f"After layer 2: {h2.shape}")
    print(f"Final output: {output.shape}")
    print(f"Output values: {output.data}")
    
    print("\n🎉 Neural network working! You just built your first neural network!")
    print("🏗️  Clean architecture: Dense layers + Activations module = Neural Network")
    print("Notice how the network transforms 3D input into 2D output through learned transformations.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the layers and check activations module!")

# %% [markdown]
"""
## Step 4: Understanding What We Built

Congratulations! You just implemented a clean, modular neural network architecture:

### 🧱 **What You Built**
1. **Dense Layer**: Linear transformation `y = Wx + b`
2. **Activation Functions**: Imported from activations module (ReLU, Sigmoid, Tanh)
3. **Layer Composition**: Chaining layers to build networks

### 🏗️ **Clean Architecture Benefits**
- **Separation of concerns**: Math functions vs. layer building blocks
- **Reusability**: Activations can be used across different modules
- **Maintainability**: One place to update activation implementations
- **Composability**: Clean imports make complex networks easier to build

### 🎯 **Key Insights**
- **Layers are functions**: They transform tensors from one space to another
- **Composition creates complexity**: Simple layers → complex networks
- **Nonlinearity is crucial**: Without it, deep networks are just linear transformations
- **Neural networks are function approximators**: They learn to map inputs to outputs
- **Modular design**: Building blocks can be combined in many ways

### 🚀 **What's Next**
In the next modules, you'll learn:
- **Training**: How networks learn from data (backpropagation, optimizers)
- **Architectures**: Specialized layers for different problems (CNNs, RNNs)
- **Applications**: Using networks for real problems

### 🔧 **Export to Package**
Run this to export your layers to the TinyTorch package:
```bash
python bin/tito.py sync
```

Then test your implementation:
```bash
python bin/tito.py test --module layers
```

**Great job! You've built a clean, modular foundation for neural networks!** 🎉
"""

# %%
# Final demonstration: A more complex example
try:
    print("=== Final Demo: Image Classification Network ===")
    
    # Simulate a small image: 28x28 pixels flattened to 784 features
    # This is like a tiny MNIST digit
    image_size = 28 * 28  # 784 pixels
    num_classes = 10      # 10 digits (0-9)
    
    # Build a 3-layer network for digit classification
    # 784 → 128 → 64 → 10
    layer1 = Dense(input_size=image_size, output_size=128)
    relu1 = ReLU()  # From activations module
    layer2 = Dense(input_size=128, output_size=64)
    relu2 = ReLU()  # From activations module
    layer3 = Dense(input_size=64, output_size=num_classes)
    softmax = Sigmoid()  # Using Sigmoid as a simple "probability-like" output
    
    print(f"Image classification network:")
    print(f"  Input: {image_size} pixels (28x28 image)")
    print(f"  Hidden 1: {layer1.input_size} → {layer1.output_size} (Dense + ReLU)")
    print(f"  Hidden 2: {layer2.input_size} → {layer2.output_size} (Dense + ReLU)")
    print(f"  Output: {layer3.input_size} → {layer3.output_size} (Dense + Sigmoid)")
    
    # Simulate a batch of 5 images
    batch_size = 5
    fake_images = Tensor(np.random.randn(batch_size, image_size).astype(np.float32))
    
    # Forward pass
    h1 = relu1(layer1(fake_images))
    h2 = relu2(layer2(h1))
    predictions = softmax(layer3(h2))
    
    print(f"\nBatch processing:")
    print(f"  Input batch shape: {fake_images.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions.data[0]}")  # First image predictions
    
    print("\n🎉 You built a neural network that could classify images!")
    print("🏗️  Clean architecture: Dense layers + Activations module = Image Classifier")
    print("With training, this network could learn to recognize handwritten digits!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your layer implementations and activations module!")

# %% [markdown]
"""
## 🎓 Module Summary

### What You Learned
1. **Layer Architecture**: Dense layers as linear transformations
2. **Clean Dependencies**: Layers module uses activations module
3. **Function Composition**: Simple building blocks → complex networks
4. **Modular Design**: Separation of concerns for maintainable code

### Key Architectural Insight
```
activations (math functions) → layers (building blocks) → networks (applications)
```

This clean dependency graph makes the system:
- **Understandable**: Each module has a clear purpose
- **Testable**: Each module can be tested independently
- **Reusable**: Components can be used across different contexts
- **Maintainable**: Changes are localized to appropriate modules

### Next Steps
- **Training**: Learn how networks learn from data
- **Advanced Architectures**: CNNs, RNNs, Transformers
- **Applications**: Real-world machine learning problems

**Congratulations on building a clean, modular neural network foundation!** 🚀
""" 
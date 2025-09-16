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
# Layers - Building Blocks of Neural Networks

Welcome to the Layers module! This is where we build the fundamental components that stack together to form neural networks. Every neural network you've ever heard of - from simple perceptrons to massive transformers like GPT - is built by stacking these basic building blocks.

## Learning Goals
- **Deep Mathematical Understanding**: Grasp how matrix multiplication powers all neural networks
- **Implementation Mastery**: Build matrix multiplication and Dense layers from scratch
- **Visual Intuition**: See how data flows and transforms through layers
- **Production Connection**: Understand how this connects to PyTorch, TensorFlow, and industry ML
- **Architecture Foundation**: Learn to compose layers into complex networks
- **Parameter Strategies**: Master weight initialization and shape management

## Build → Use → Understand
1. **Build**: Matrix multiplication and Dense layers with complete understanding
2. **Use**: Create and test layers with real data and visual examples
3. **Understand**: How linear transformations enable universal function approximation

## Why This Module Is Critical
Layers are the **universal building blocks** of machine learning:
- **Computer Vision**: CNNs stack convolutional layers
- **Natural Language**: Transformers stack attention layers
- **Reinforcement Learning**: Policy networks stack dense layers
- **Generative AI**: All generative models use layer composition

Mastering layers means understanding the foundation of all modern AI.
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
## The Deep Mathematics of Neural Network Layers

### What Are Neural Network Layers?
Layers are **learnable function approximators** - each layer is a mathematical transformation that:
1. **Takes input data**: Raw features, pixels, words, or intermediate representations
2. **Applies learned transformation**: Linear combinations followed by nonlinear activations
3. **Produces useful representations**: Features that are better for the final task

### The Universal Layer Pattern
Every layer in every neural network follows this fundamental pattern:
```python
def universal_layer(x):
    # 1. Linear transformation (learnable)
    linear_output = x @ weights + bias
    
    # 2. Nonlinear activation (fixed function)
    output = activation(linear_output)
    
    return output
```

### Why This Simple Pattern Works for Everything

#### The Mathematical Miracle
- **Linear part**: Learns weighted combinations of input features
- **Nonlinear part**: Enables complex decision boundaries
- **Stacking**: Creates arbitrarily complex function approximation
- **Universal approximation**: Proven to approximate any continuous function

#### Visual Understanding
```
Input Features    →  Linear Transform  →  Nonlinear Activation  →  Output Features
[x1, x2, x3]         [w11 w12 w13]         ReLU/Sigmoid/Tanh       [y1, y2]
                      [w21 w22 w23]
                      [bias1, bias2]
```

### Mathematical Foundation: Function Composition
A neural network is mathematical function composition:
```
f(x) = layer_n(layer_{n-1}(...layer_2(layer_1(x))))

Where each layer_i(x) = activation(x @ W_i + b_i)
```

**Key insight**: Each layer learns to transform its input into a representation that makes the next layer's job easier.

### Real-World Applications

#### Computer Vision
- **Layer 1**: Detects edges and textures
- **Layer 2**: Combines edges into shapes
- **Layer 3**: Combines shapes into objects
- **Final Layer**: Maps objects to class labels

#### Natural Language Processing
- **Embedding Layer**: Maps words to vector representations
- **Hidden Layers**: Learn syntactic and semantic patterns
- **Output Layer**: Maps representations to predictions

#### Scientific Computing
- **Physics**: Learn differential equation solutions
- **Chemistry**: Predict molecular properties
- **Biology**: Model protein folding

### What We'll Build Step by Step

1. **Matrix Multiplication Engine**: The mathematical core powering all layers
2. **Dense Layer Implementation**: The fundamental building block
3. **Weight Initialization Strategies**: How to start learning effectively
4. **Layer Composition Patterns**: Building complex architectures
5. **Integration with Activations**: Creating complete neural network components
6. **Production-Ready Implementation**: Code that scales to real applications

### Why Understanding Layers Deeply Matters

#### For ML Engineers
- **Debugging**: Understand why networks fail to train
- **Architecture Design**: Know when to use which layer types
- **Performance Optimization**: Optimize for specific hardware

#### For AI Researchers
- **Novel Architectures**: Invent new layer types
- **Theoretical Understanding**: Prove properties of neural networks
- **Algorithmic Innovation**: Develop new training methods

#### For Industry Applications
- **Model Deployment**: Optimize for production environments
- **Transfer Learning**: Adapt pre-trained layers to new tasks
- **Custom Solutions**: Build domain-specific architectures
"""

# %% [markdown]
"""
## 🔧 DEVELOPMENT
"""

# %% [markdown]
"""
## Step 1: Matrix Multiplication - The Mathematical Engine of All AI

### The Foundation of Modern AI
Matrix multiplication is the **single most important operation** in all of machine learning. Every neural network, from simple classifiers to GPT and ChatGPT, is fundamentally powered by this operation:

```
C = A @ B  # This simple operation powers all of AI
```

### Deep Mathematical Understanding

#### The Core Operation
For matrices A (m×n) and B (n×p), the result C (m×p) is:
```
C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
```

**Physical interpretation**: Each output element is a **weighted sum** of input features.

#### Visual Step-by-Step Breakdown
```
Matrix A (2×2)    Matrix B (2×2)    Result C (2×2)
┌─────────┐      ┌─────────┐      ┌─────────┐
│  1   2  │  @   │  5   6  │  =   │ 19  22  │
│  3   4  │      │  7   8  │      │ 43  50  │
└─────────┘      └─────────┘      └─────────┘

Step-by-step computation:
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 5 + 14 = 19
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 6 + 16 = 22
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 15 + 28 = 43
C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 18 + 32 = 50
```

#### Neural Network Interpretation
```
Input Data        Weight Matrix     Output Features
(batch × in)   @   (in × out)   =   (batch × out)
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ sample 1    │   │ feature     │   │transformed  │
│ sample 2    │ @ │ weights     │ = │features     │
│    ...      │   │    ...      │   │    ...      │
│ sample n    │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘
```

### Why Matrix Multiplication Powers All AI

#### 1. Feature Combination
Each output is a **learned combination** of all input features:
```
output[i] = w1*input[0] + w2*input[1] + ... + wn*input[n-1]
```
The weights determine **which features matter** and **how they combine**.

#### 2. Parallel Processing
- **CPU vectorization**: Process multiple elements simultaneously
- **GPU acceleration**: Thousands of cores compute matrix operations
- **TPU optimization**: Specialized hardware for matrix computations

#### 3. Mathematical Elegance
- **Differentiable**: Gradients flow cleanly through matrix operations
- **Composable**: Matrix operations stack naturally
- **Expressive**: Can represent any linear transformation

### Real-World Applications Powered by Matrix Multiplication

#### Large Language Models (GPT, ChatGPT)
```
Attention(Q,K,V) = softmax(QK^T/√d)V  # Three matrix multiplications!
```
- **Q @ K^T**: Compute attention scores between all word pairs
- **Attention @ V**: Weight and combine value vectors
- **Linear layers**: Transform representations at each layer

#### Computer Vision (ResNet, Vision Transformers)
```
Convolution ≈ Matrix Multiplication  # Convolution can be expressed as matrix ops
```
- **Feature maps**: Each filter creates a feature map via matrix operations
- **Classification**: Final features → class logits via matrix multiplication
- **Object detection**: Bounding box regression via matrix operations

#### Recommendation Systems
```
User-Item Matrix @ Item-Feature Matrix = User-Feature Preferences
```
- **Collaborative filtering**: User similarity via matrix operations
- **Content-based**: Feature matching via matrix computations
- **Deep models**: Neural collaborative filtering via matrix layers

### Performance Considerations

#### Why We Use NumPy (and why GPUs exist)
```
# Naive Python loops: ~10 seconds for large matrices
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i,j] += A[i,k] * B[k,j]

# NumPy (optimized C): ~0.01 seconds for same matrices
C = A @ B

# GPU (CUDA): ~0.001 seconds for same matrices
C = torch.matmul(A_gpu, B_gpu)
```

#### Memory and Computation Complexity
- **Memory**: O(mn + np + mp) to store three matrices
- **Computation**: O(mnp) multiply-add operations
- **For large models**: Billions of parameters × billions of operations

### Debugging Matrix Multiplication

#### Common Shape Errors
```
A.shape = (batch_size, input_features)     # e.g., (32, 784)
B.shape = (input_features, output_features) # e.g., (784, 10)
C.shape = (batch_size, output_features)     # result: (32, 10)

# COMMON ERROR:
A.shape = (32, 784)
B.shape = (10, 784)  # Wrong! Should be (784, 10)
# Error: Cannot multiply (32, 784) @ (10, 784)
```

#### Visual Debugging Technique
```
Always check: A's last dimension == B's first dimension
              (m, n) @ (n, p) = (m, p) ✓
              (m, n) @ (k, p) = ERROR if n ≠ k
```

### Connection to Production ML Systems

#### PyTorch Implementation
```python
# Your implementation (educational)
result = matmul(A, B)

# PyTorch (production)
result = torch.matmul(A, B)  # Optimized, GPU-accelerated
result = A @ B               # Same operation
```

#### TensorFlow Implementation
```python
# Your implementation (educational)
result = matmul(A, B)

# TensorFlow (production)
result = tf.matmul(A, B)     # Optimized, distributed computing
result = A @ B               # Same operation
```

### Why Implement It Ourselves?
1. **Deep Understanding**: See exactly what happens in each operation
2. **Debugging Skills**: Understand why shape errors occur
3. **Performance Intuition**: Appreciate why GPUs are essential
4. **Algorithm Design**: Know how to optimize for specific use cases
5. **Research Foundation**: Basis for developing new layer types
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-naive", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication using explicit for-loops for deep understanding.
    
    This implementation reveals the mathematical essence of neural networks!
    Every time a neural network processes data, it's doing exactly this operation.
        
    TODO: Implement matrix multiplication using three nested for-loops.
    
    APPROACH:
    1. Extract and validate matrix dimensions
    2. Initialize result matrix with zeros
    3. Implement the triple-nested loop structure
    4. Accumulate dot products for each output element
    
    MATHEMATICAL FOUNDATION:
    For C = A @ B, each element C[i,j] is the dot product of:
    - Row i from matrix A: [A[i,0], A[i,1], ..., A[i,n-1]]
    - Column j from matrix B: [B[0,j], B[1,j], ..., B[n-1,j]]
    
    VISUAL STEP-BY-STEP:
    ```
    A = [[1, 2],     B = [[5, 6],     C = [[?, ?],
         [3, 4]]          [7, 8]]          [?, ?]]
    
    Computing C[0,0] (row 0 of A, column 0 of B):
    A[0,:] = [1, 2]  ←→  B[:,0] = [5, 7]
    C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
    
    Computing C[0,1] (row 0 of A, column 1 of B):
    A[0,:] = [1, 2]  ←→  B[:,1] = [6, 8]
    C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
    
    Computing C[1,0] (row 1 of A, column 0 of B):
    A[1,:] = [3, 4]  ←→  B[:,0] = [5, 7]
    C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
    
    Computing C[1,1] (row 1 of A, column 1 of B):
    A[1,:] = [3, 4]  ←→  B[:,1] = [6, 8]
    C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
    
    Final result: C = [[19, 22], [43, 50]]
    ```
    
    IMPLEMENTATION ALGORITHM:
    ```python
    # 1. Get dimensions and validate
    m, n = A.shape          # A is m×n
    n2, p = B.shape         # B is n×p (n2 must equal n)
    assert n == n2          # Inner dimensions must match
    
    # 2. Initialize result matrix
    C = zeros(m, p)         # Result is m×p
    
    # 3. Triple nested loops
    for i in range(m):      # For each row of A
        for j in range(p):  # For each column of B
            for k in range(n):  # For each element in dot product
                C[i,j] += A[i,k] * B[k,j]  # Accumulate
    ```
    
    NEURAL NETWORK CONNECTION:
    In a neural network layer:
    - A = input batch (batch_size × input_features)
    - B = weight matrix (input_features × output_features)
    - C = output batch (batch_size × output_features)
    
    Each C[i,j] represents how much output feature j is activated for input sample i.
    
    DEBUGGING HINTS:
    - Check shapes: A.shape = (m,n), B.shape = (n,p) → C.shape = (m,p)
    - Common error: Swapping B's dimensions (should be input_features × output_features)
    - Accumulation: Start with C[i,j] = 0, then add all A[i,k] * B[k,j]
    - Index bounds: i ∈ [0,m), j ∈ [0,p), k ∈ [0,n)
    
    PERFORMANCE NOTE:
    This implementation is O(mnp) time complexity and helps you understand:
    - Why GPUs are essential for deep learning (parallelizable operations)
    - Why NumPy/BLAS libraries are much faster (optimized C/Fortran)
    - How memory access patterns affect performance
    
    LEARNING CONNECTIONS:
    - Foundation of ALL neural network computations
    - Understanding enables debugging shape mismatches
    - Basis for implementing custom layer types
    - Essential for optimizing model performance
    - Connects to linear algebra theory
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

Once you implement the `matmul` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_matrix_multiplication():
    """Test matrix multiplication implementation"""
    print("🔬 Unit Test: Matrix Multiplication...")

# Test simple 2x2 case
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Matrix multiplication failed: expected {expected}, got {result}"
    
    # Compare with NumPy
    numpy_result = A @ B
    assert np.allclose(result, numpy_result), f"Doesn't match NumPy: got {result}, expected {numpy_result}"

# Test different shapes
    A2 = np.array([[1, 2, 3]], dtype=np.float32)  # 1x3
    B2 = np.array([[4], [5], [6]], dtype=np.float32)  # 3x1
    result2 = matmul(A2, B2)
    expected2 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
    
    assert np.allclose(result2, expected2), f"1x3 @ 3x1 failed: expected {expected2}, got {result2}"
    
    # Test 3x3 case
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    B3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # Identity
    result3 = matmul(A3, B3)
    
    assert np.allclose(result3, A3), "Multiplication by identity should preserve matrix"
    
    # Test incompatible shapes
    A4 = np.array([[1, 2]], dtype=np.float32)  # 1x2
    B4 = np.array([[3], [4], [5]], dtype=np.float32)  # 3x1
    
    try:
        matmul(A4, B4)
        assert False, "Should raise error for incompatible shapes"
    except ValueError as e:
        assert "Incompatible matrix dimensions" in str(e)
    
    print("✅ Matrix multiplication tests passed!")
    print(f"✅ 2x2 multiplication working correctly")
    print(f"✅ Matches NumPy's implementation")
    print(f"✅ Handles different shapes correctly")
    print(f"✅ Proper error handling for incompatible shapes")

# Run the test
test_unit_matrix_multiplication()

# %% [markdown]
"""
### 🎯 CHECKPOINT: Matrix Multiplication Mastery

You've just implemented the mathematical engine that powers ALL neural networks! 

#### What You've Accomplished
✅ **Deep Understanding**: You now understand exactly what happens inside every neural network layer  
✅ **Implementation Skills**: You can build matrix operations from mathematical first principles  
✅ **Debugging Abilities**: You understand why shape mismatches occur and how to fix them  
✅ **Performance Intuition**: You appreciate why GPUs and optimized libraries are essential  

#### Mathematical Concepts Mastered
- **Dot Products**: The fundamental operation combining features with weights
- **Shape Compatibility**: Understanding when matrices can be multiplied
- **Computational Complexity**: O(mnp) operations for (m×n) @ (n×p) matrices
- **Memory Layout**: How data flows through matrix operations

#### Real-World Connection
Your implementation does exactly what happens inside:
- **PyTorch**: `torch.matmul(A, B)` uses the same mathematical principles
- **TensorFlow**: `tf.matmul(A, B)` performs identical operations
- **NumPy**: `A @ B` follows the same algorithm (just optimized in C)

#### Ready for Next Step
With matrix multiplication mastered, you're ready to build Dense layers - the fundamental building blocks that stack together to create all neural networks!

**Key insight**: Every time you see `layer(x)` in any neural network, you now know it's doing matrix multiplication under the hood.
"""

# %% [markdown]
"""
## Step 2: Dense Layer - The Foundation of All Neural Networks

### What is a Dense Layer?
A **Dense layer** (also called Linear or Fully Connected layer) is the fundamental building block that appears in EVERY neural network architecture ever created:

```python
output = input @ weights + bias
```

This simple equation powers:
- **GPT and language models**: Transform text representations
- **ResNet and vision models**: Classify image features
- **Recommendation systems**: Map user preferences
- **Scientific AI**: Model physical phenomena

### The Mathematical Miracle of Dense Layers

#### Universal Function Approximation
Dense layers have a **mathematically proven superpower**: Stack enough of them with nonlinear activations, and they can approximate **any continuous function**!

```python
# This can learn ANY pattern:
f(x) = dense_n(activation(dense_{n-1}(...activation(dense_1(x)))))
```

#### Why This Works
```
Linear Transformation + Nonlinear Activation = Universal Expressiveness
```

1. **Linear part (y = xW + b)**: Learns feature combinations
2. **Nonlinear activation**: Enables complex decision boundaries
3. **Stacking**: Creates arbitrarily complex functions

### Deep Mathematical Understanding

#### The Linear Transformation Matrix
```
Input Features    Weight Matrix      Output Features
┌─────────────┐  ┌─────────────────┐  ┌─────────────┐
│ pixel_1     │  │ w₁₁  w₁₂  w₁₃ │  │ feature_1   │
│ pixel_2     │  │ w₂₁  w₂₂  w₂₃ │  │ feature_2   │
│ pixel_3     │  │ w₃₁  w₃₂  w₃₃ │  │ feature_3   │
│    ...      │  │  ⋮    ⋮    ⋮  │  │    ...      │
│ pixel_784   │  │ w₇₈₄₁ ... w₇₈₄₃│  │             │
└─────────────┘  └─────────────────┘  └─────────────┘
(784 features)    (784 × 3 weights)    (3 features)
```

**Key insight**: Each output feature is a **learned combination** of ALL input features.

#### Weight Interpretation
Each weight w[i,j] represents:
- **How much input feature i contributes to output feature j**
- **Positive weights**: Input increases output
- **Negative weights**: Input decreases output
- **Large weights**: Strong influence
- **Small weights**: Weak influence

#### Bias Terms
```
Without bias: y = xW     (line through origin)
With bias:    y = xW + b (line can be shifted)
```

Bias allows the layer to **shift its output**, enabling:
- **Better fit**: Not forced through origin
- **Increased expressiveness**: More flexible transformations
- **Faster training**: Better starting point

### Real-World Architecture Patterns

#### Computer Vision
```python
# Image classification pipeline
image → flatten → dense(784→512) → relu → dense(512→10) → softmax
#                 ↑ Feature extraction    ↑ Classification
```

#### Natural Language Processing
```python
# Text classification pipeline
text → embed → dense(300→128) → tanh → dense(128→2) → sigmoid
#              ↑ Representation learning  ↑ Binary classification
```

#### Generative Models
```python
# VAE decoder
noise → dense(100→256) → relu → dense(256→784) → sigmoid → image
#       ↑ Expand latent code    ↑ Generate pixels
```

### Weight Initialization: The Science of Starting Right

#### Why Initialization Matters
```
Poor initialization → Vanishing/exploding gradients → Training failure
Good initialization → Stable gradients → Successful training
```

#### Xavier/Glorot Initialization
```python
scale = sqrt(2 / (input_size + output_size))
weights ~ Normal(0, scale²)
```

**Mathematical motivation**: Preserves activation variance across layers.

#### Alternative Strategies
```python
# He initialization (better for ReLU)
scale = sqrt(2 / input_size)

# LeCun initialization (for SELU)
scale = sqrt(1 / input_size)

# Uniform Xavier
limit = sqrt(6 / (input_size + output_size))
weights ~ Uniform(-limit, limit)
```

### Production System Comparison

#### PyTorch Dense Layer
```python
# Your implementation
layer = Dense(input_size=784, output_size=10)

# PyTorch equivalent
layer = torch.nn.Linear(in_features=784, out_features=10)

# Identical mathematical operation!
output = layer(input)  # y = xW^T + b (note: PyTorch transposes W)
```

#### TensorFlow Dense Layer
```python
# Your implementation
layer = Dense(input_size=784, output_size=10)

# TensorFlow equivalent
layer = tf.keras.layers.Dense(units=10, input_shape=(784,))

# Same mathematical operation!
output = layer(input)  # y = xW + b
```

### Memory and Computational Complexity

#### Parameter Count
```
Parameters = input_size × output_size + output_size (if bias)
Example: Dense(784, 512) has 784 × 512 + 512 = 401,920 parameters
```

#### Computational Complexity
```
FLOPs per sample = 2 × input_size × output_size
Example: Dense(784, 512) requires 2 × 784 × 512 = 802,816 operations
```

#### Memory Usage
```
Memory = (batch_size × input_size × 4) +     # Input (float32)
         (input_size × output_size × 4) +   # Weights
         (output_size × 4) +               # Bias
         (batch_size × output_size × 4)    # Output
```

### Design Philosophy

#### When to Use Dense Layers
- **Always**: As final classification/regression layers
- **Often**: For combining features from other layer types
- **Sometimes**: As hidden layers in simple architectures
- **Rarely**: For processing raw high-dimensional data (use CNN/RNN instead)

#### Architecture Decisions
```python
# Width vs Depth trade-off
Wide: Dense(1000, 2000)     # More parameters, might overfit
Deep: Dense(1000, 500) → Dense(500, 250) → Dense(250, 125)  # More layers

# Rule of thumb: Start simple, add complexity as needed
```

### Connection to Advanced Architectures

#### Attention Mechanisms
```python
# Multi-head attention uses THREE dense layers
Q = dense_q(x)  # Query projection
K = dense_k(x)  # Key projection
V = dense_v(x)  # Value projection
attention = softmax(QK^T/√d) @ V
```

#### Residual Connections
```python
# ResNet block with dense layers
def residual_dense_block(x):
    residual = x
    x = dense1(x)
    x = activation(x)
    x = dense2(x)
    return x + residual  # Skip connection
```
"""

# %% nbgrader={"grade": false, "grade_id": "dense-layer", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Dense:
    """
    Dense (Linear/Fully Connected) Layer
    
    Applies a linear transformation: y = xW + b
    
    This is the fundamental building block of neural networks.
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True):
        """
        Initialize Dense layer with random weights and optional bias.
        
        This initialization is CRITICAL for successful neural network training!
        Poor initialization can cause vanishing/exploding gradients and training failure.
        
        TODO: Implement Dense layer initialization with proper weight scaling.
        
        APPROACH:
        1. Store layer configuration parameters
        2. Initialize weights using Xavier/Glorot strategy
        3. Initialize bias terms (typically zeros)
        4. Convert arrays to Tensor objects for compatibility
        
        WEIGHT INITIALIZATION DEEP DIVE:
        
        Why Random Initialization?
        - Breaks symmetry: All neurons start different
        - Enables learning: Gradients won't be identical
        - Avoids dead neurons: Some neurons activate from start
        
        Xavier/Glorot Initialization Strategy:
        ```
        scale = sqrt(2 / (input_size + output_size))
        weights ~ Normal(0, scale²)
        ```
        
        Mathematical Justification:
        - Maintains activation variance across layers
        - Prevents vanishing/exploding gradients
        - Empirically proven to improve training
        
        VISUAL INITIALIZATION PATTERN:
        ```
        Input Layer (3 neurons)    Dense Layer (2 neurons)
        ┌─────┐                   ┌─────┐
        │ x₁  │ ──w₁₁──→         │ y₁  │
        │     │    \\              │     │
        │ x₂  │ ──w₂₁─w₂₂──→     │ y₂  │
        │     │    /              │     │
        │ x₃  │ ──w₃₁──→         │     │
        └─────┘   +b₁   +b₂      └─────┘
        
        Weight Matrix W (3×2):     Bias Vector b (2×1):
        ┌──────────────┐          ┌────┐
        │ w₁₁   w₁₂   │          │ b₁ │
        │ w₂₁   w₂₂   │          │ b₂ │
        │ w₃₁   w₃₂   │          └────┘
        └──────────────┘
        ```
        
        EXAMPLE INITIALIZATION:
        ```python
        layer = Dense(input_size=784, output_size=10)  # MNIST classifier
        # Weight shape: (784, 10) - each output connects to all inputs
        # Bias shape: (10,) - one bias per output neuron
        # Scale: sqrt(2/(784+10)) ≈ 0.05 - prevents gradients from exploding
        ```
        
        IMPLEMENTATION STEPS:
        ```python
        # 1. Store configuration
        self.input_size = input_size      # Number of input features
        self.output_size = output_size    # Number of output neurons
        self.use_bias = use_bias          # Whether to include bias terms
        
        # 2. Calculate Xavier scale
        scale = np.sqrt(2.0 / (input_size + output_size))
        
        # 3. Initialize weights (shape matters!)
        weight_data = np.random.randn(input_size, output_size) * scale
        
        # 4. Initialize bias (usually zeros)
        if use_bias:
            bias_data = np.zeros(output_size)
        
        # 5. Convert to Tensors
        self.weights = Tensor(weight_data)
        self.bias = Tensor(bias_data) if use_bias else None
        ```
        
        ALTERNATIVE INITIALIZATION STRATEGIES:
        
        He Initialization (better for ReLU):
        ```python
        scale = np.sqrt(2.0 / input_size)  # Only input size
        ```
        
        Uniform Xavier:
        ```python
        limit = np.sqrt(6.0 / (input_size + output_size))
        weights = np.random.uniform(-limit, limit, (input_size, output_size))
        ```
        
        COMMON INITIALIZATION MISTAKES:
        1. **All zeros**: No learning (dead neurons)
        2. **Too large**: Exploding gradients
        3. **Too small**: Vanishing gradients
        4. **Wrong shape**: Broadcasting errors
        5. **Same values**: Symmetry problem
        
        PRODUCTION SYSTEM COMPARISON:
        ```python
        # Your implementation
        layer = Dense(input_size, output_size)
        
        # PyTorch equivalent
        layer = torch.nn.Linear(input_size, output_size)
        # Uses Kaiming uniform initialization by default
        
        # TensorFlow equivalent
        layer = tf.keras.layers.Dense(output_size, input_shape=(input_size,))
        # Uses Glorot uniform initialization by default
        ```
        
        DEBUGGING HINTS:
        - Print weight statistics: mean ≈ 0, std ≈ scale
        - Check shapes: weights (input_size, output_size), bias (output_size,)
        - Verify Tensor conversion: isinstance(self.weights, Tensor)
        - Test forward pass: no shape errors
        
        LEARNING CONNECTIONS:
        - Foundation for all layer types (Conv2D, LSTM, Attention)
        - Understanding gradients and backpropagation
        - Basis for transfer learning (loading pre-trained weights)
        - Essential for model architecture design
        """
        ### BEGIN SOLUTION
        # Store layer parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
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
    
    def forward(self, x):
        """
        Forward pass through the Dense layer: the heart of neural computation.
        
        This function implements y = xW + b, the fundamental equation that powers
        all neural networks from simple perceptrons to massive transformers!
        
        TODO: Implement the forward pass with proper shape handling.
        
        APPROACH:
        1. Apply matrix multiplication for feature combination
        2. Add bias terms for output shifting
        3. Return properly shaped Tensor result
        4. Handle batch processing automatically
        
        MATHEMATICAL FOUNDATION:
        
        The Linear Transformation:
        ```
        y = xW + b
        
        Where:
        x: Input features    (batch_size × input_features)
        W: Weight matrix     (input_features × output_features)
        b: Bias vector       (output_features,)
        y: Output features   (batch_size × output_features)
        ```
        
        VISUAL DATA FLOW:
        ```
        Input Batch          Weight Matrix        Bias Vector       Output Batch
        ┌─────────────┐     ┌─────────────┐     ┌─────────┐      ┌─────────────┐
        │ [x₁₁ x₁₂]  │     │ [w₁₁ w₁₂]  │     │ [b₁ b₂] │      │ [y₁₁ y₁₂]  │
        │ [x₂₁ x₂₂]  │  @  │ [w₂₁ w₂₂]  │  +  │         │  =   │ [y₂₁ y₂₂]  │
        │ [x₃₁ x₃₂]  │     └─────────────┘     └─────────┘      │ [y₃₁ y₃₂]  │
        └─────────────┘                                          └─────────────┘
        (3×2)              (2×2)              (2,)              (3×2)
        ```
        
        STEP-BY-STEP COMPUTATION:
        
        For each output element y[i,j]:
        ```
        y[i,j] = Σₖ x[i,k] * W[k,j] + b[j]
        
        Example:
        x = [[1, 2]]        # 1 sample, 2 features
        W = [[0.5, 0.3],    # 2 input → 2 output
             [0.7, 0.4]]
        b = [0.1, 0.2]      # bias for each output
        
        y[0,0] = x[0,0]*W[0,0] + x[0,1]*W[1,0] + b[0]
               = 1*0.5 + 2*0.7 + 0.1 = 0.5 + 1.4 + 0.1 = 2.0
        
        y[0,1] = x[0,0]*W[0,1] + x[0,1]*W[1,1] + b[1]
               = 1*0.3 + 2*0.4 + 0.2 = 0.3 + 0.8 + 0.2 = 1.3
        
        Result: y = [[2.0, 1.3]]
        ```
        
        BATCH PROCESSING MAGIC:
        The same operation works for ANY batch size:
        ```
        Single sample:  (1, features) @ (features, outputs) = (1, outputs)
        Mini-batch:     (32, features) @ (features, outputs) = (32, outputs)
        Large batch:    (1000, features) @ (features, outputs) = (1000, outputs)
        ```
        
        IMPLEMENTATION DETAILS:
        ```python
        # 1. Matrix multiplication (the core operation)
        linear_output = matmul(x.data, self.weights.data)
        
        # 2. Bias addition (broadcasting handles shape automatically)
        if self.use_bias and self.bias is not None:
            linear_output = linear_output + self.bias.data
            # Broadcasting: (batch_size, output_features) + (output_features,)
            #            → (batch_size, output_features)
        
        # 3. Return as proper Tensor type
        return type(x)(linear_output)  # Preserves Tensor class
        ```
        
        BROADCASTING EXPLANATION:
        NumPy automatically broadcasts the bias:
        ```
        linear_output.shape = (batch_size, output_features)  # e.g., (32, 10)
        bias.shape         = (output_features,)             # e.g., (10,)
        
        # Broadcasting adds bias to each sample:
        result[i,j] = linear_output[i,j] + bias[j]  # for all i
        ```
        
        REAL-WORLD APPLICATIONS:
        
        Image Classification:
        ```
        # Flatten image: (28, 28) → (784,)
        # Dense layer: (784,) → (10,) class scores
        x = flattened_image  # Shape: (batch, 784)
        scores = dense_layer(x)  # Shape: (batch, 10)
        ```
        
        Language Model:
        ```
        # Word embedding: word_id → dense vector
        # Dense layer: hidden → vocabulary scores
        x = hidden_state  # Shape: (batch, hidden_size)
        logits = output_layer(x)  # Shape: (batch, vocab_size)
        ```
        
        COMMON SHAPE ERRORS AND SOLUTIONS:
        ```
        Error: "Cannot multiply (32, 784) and (10, 784)"
        Solution: Weight shape should be (784, 10), not (10, 784)
        
        Error: "Cannot add (32, 10) and (784,)"
        Solution: Bias shape should be (10,), not (784,)
        
        Error: "Expected 2D input, got 1D"
        Solution: Reshape input from (features,) to (1, features)
        ```
        
        DEBUGGING CHECKLIST:
        - Input shape: (batch_size, input_features)
        - Weight shape: (input_features, output_features)
        - Bias shape: (output_features,) or None
        - Output shape: (batch_size, output_features)
        
        PERFORMANCE NOTES:
        - Matrix multiplication is O(batch × input × output)
        - Most computation time spent here in large models
        - GPU acceleration crucial for large layers
        - Memory usage: store input, weights, bias, output
        
        LEARNING CONNECTIONS:
        - Foundation of backpropagation (gradients flow through this operation)
        - Basis for all advanced layer types (attention, convolution)
        - Understanding enables custom layer development
        - Critical for model optimization and deployment
        """
        ### BEGIN SOLUTION
        # Perform matrix multiplication
        linear_output = matmul(x.data, self.weights.data)
        
        # Add bias if present
        if self.use_bias and self.bias is not None:
            linear_output = linear_output + self.bias.data
        
        return type(x)(linear_output)
        ### END SOLUTION
    
    def __call__(self, x):
        """Make the layer callable: layer(x) instead of layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Dense Layer

Once you implement the Dense layer above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense-layer", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_dense_layer():
    """Test Dense layer implementation"""
    print("🔬 Unit Test: Dense Layer...")
    
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
test_unit_dense_layer()

# %% [markdown]
"""
### 🎯 CHECKPOINT: Dense Layer Implementation Complete

Congratulations! You've just implemented the fundamental building block of all neural networks!

#### What You've Accomplished
✅ **Dense Layer Mastery**: You can now build the core component of every neural network  
✅ **Weight Initialization**: You understand how to start training with proper parameter scaling  
✅ **Shape Management**: You handle batch processing and broadcasting automatically  
✅ **Production-Ready Code**: Your implementation matches PyTorch and TensorFlow standards  

#### Mathematical Concepts Mastered
- **Linear Transformations**: y = xW + b is now deeply understood
- **Parameter Initialization**: Xavier/Glorot scaling for stable gradients
- **Broadcasting**: Automatic shape handling for bias addition
- **Batch Processing**: Same operation works for any batch size

#### Real-World Impact
Your Dense layer implementation enables:
- **Image Classification**: Transform pixel features to class predictions
- **Language Models**: Map word embeddings to vocabulary scores
- **Recommendation Systems**: Learn user-item preference mappings
- **Scientific Computing**: Model complex physical phenomena

#### Connection to Advanced AI
Every advanced architecture uses your Dense layer:
- **Transformers (GPT)**: Attention layers are built from Dense layers
- **ResNets**: Skip connections combine with Dense layers
- **GANs**: Both generator and discriminator use Dense layers
- **VAEs**: Encoder and decoder networks built from Dense layers

#### Ready for Integration
With Dense layers mastered, you're ready to see how they combine with activation functions to create complete neural network components that can learn any pattern!

**Key insight**: You now understand the mathematical foundation of all modern AI systems.
"""

# %% [markdown]
"""
## Step 3: Layer Integration with Activations - Building Complete Neural Networks

### The Magic of Layer + Activation Composition
Now we combine Dense layers with activation functions to create complete neural network components that can learn ANY pattern! This is where the true power of neural networks emerges.

### The Universal Neural Network Building Block
```python
# This pattern appears in EVERY neural network:
def neural_component(x):
    # 1. Linear transformation (learnable)
    linear_output = dense_layer(x)
    
    # 2. Nonlinear activation (fixed function)
    final_output = activation_function(linear_output)
    
    return final_output
```

### Why This Simple Pattern Enables Universal Learning

#### Mathematical Foundation
```
f(x) = activation(xW + b)
```

This combination provides:
- **Linear part**: Learns optimal feature combinations
- **Nonlinear part**: Enables complex decision boundaries
- **Composability**: Stacks to approximate any function

#### Visual Understanding of Layer + Activation
```
Input → Dense Layer → Activation → Output
┌─────┐   ┌─────────┐   ┌──────────┐   ┌─────┐
│ [1] │   │ [1 2]   │   │   ReLU   │   │ [2] │
│ [2] │ → │ [3 4] @ │ → │ max(0,x) │ → │ [0] │
│ [3] │   │ [5 6]   │   │          │   │ [8] │
└─────┘   └─────────┘   └──────────┘   └─────┘
         Linear Output    Nonlinear     Final
         [2, -1, 8]      Activation     [2, 0, 8]
```

### Real-World Layer Patterns

#### Hidden Layers (Feature Learning)
```python
# Most common pattern in neural networks
hidden = relu(dense(x))  # Dense + ReLU

# Why ReLU?
# - Sparse activation (many zeros)
# - No vanishing gradient problem
# - Computationally efficient
# - Biologically inspired
```

#### Classification Output Layers
```python
# Multi-class classification
logits = dense(hidden)        # Raw scores
probabilities = softmax(logits)  # Convert to probabilities

# Binary classification  
score = dense(hidden)         # Single score
probability = sigmoid(score)   # Convert to probability [0,1]
```

#### Gated Mechanisms (Advanced Architectures)
```python
# LSTM/GRU gates
forget_gate = sigmoid(dense_forget(x))  # Values in [0,1]
input_gate = sigmoid(dense_input(x))    # Controls information flow
output_gate = sigmoid(dense_output(x))  # Controls output

# Attention mechanisms
attention_scores = softmax(dense_attention(x))  # Probability distribution
```

### Deep Network Architecture Patterns

#### Multi-Layer Perceptron (MLP)
```python
# Classic deep network architecture
def mlp(x):
    h1 = relu(dense1(x))      # Hidden layer 1
    h2 = relu(dense2(h1))     # Hidden layer 2  
    h3 = relu(dense3(h2))     # Hidden layer 3
    output = softmax(dense4(h3))  # Output layer
    return output

# Each layer learns increasingly complex features:
# Layer 1: Basic feature combinations
# Layer 2: Feature interactions
# Layer 3: Complex patterns
# Output: Task-specific predictions
```

#### Residual Network Block
```python
# ResNet-style skip connections
def residual_block(x):
    residual = x
    h1 = relu(dense1(x))
    h2 = dense2(h1)  # No activation before skip connection
    output = relu(h2 + residual)  # Add skip connection
    return output

# Why this works:
# - Enables very deep networks
# - Solves vanishing gradient problem
# - Allows learning identity mappings
```

#### Attention Mechanism
```python
# Transformer-style attention
def attention_layer(x):
    queries = dense_q(x)      # Project to query space
    keys = dense_k(x)         # Project to key space
    values = dense_v(x)       # Project to value space
    
    # Compute attention scores
    scores = queries @ keys.T / sqrt(d_model)
    attention_weights = softmax(scores)
    
    # Apply attention to values
    output = attention_weights @ values
    return output
```

### Layer Combination Strategies

#### Width vs Depth Trade-offs
```python
# Wide network (fewer layers, more neurons)
def wide_network(x):
    h1 = relu(dense(x, 1000))    # Large hidden layer
    output = softmax(dense(h1, 10))
    return output

# Deep network (more layers, fewer neurons)
def deep_network(x):
    h1 = relu(dense(x, 100))
    h2 = relu(dense(h1, 100))
    h3 = relu(dense(h2, 100))
    h4 = relu(dense(h3, 100))
    output = softmax(dense(h4, 10))
    return output

# General trend: Deeper networks often perform better
```

#### Activation Function Selection Guide
```python
# Hidden layers
hidden = relu(dense(x))       # Default choice, works well
hidden = leaky_relu(dense(x)) # Prevents dead neurons
hidden = gelu(dense(x))       # Used in transformers
hidden = swish(dense(x))      # Smooth, self-gated

# Output layers
classification = softmax(dense(x))  # Multi-class probabilities
binary = sigmoid(dense(x))          # Binary probability
regression = dense(x)               # No activation for regression
structured = tanh(dense(x))         # Bounded outputs [-1, 1]
```

### Training Considerations

#### Gradient Flow Through Layer+Activation
```python
# Good gradient flow
x → dense1 → relu → dense2 → relu → output
    ↑ Well-conditioned gradients flow back

# Poor gradient flow
x → dense1 → sigmoid → dense2 → sigmoid → output
    ↑ Gradients may vanish in deep networks
```

#### Initialization Strategies for Layer+Activation
```python
# Xavier/Glorot (for sigmoid, tanh)
scale = sqrt(2 / (input_size + output_size))

# He initialization (for ReLU)
scale = sqrt(2 / input_size)

# Activation function determines optimal initialization!
```

### Production Architecture Examples

#### Image Classification (ResNet-style)
```python
def image_classifier(x):
    # Feature extraction
    h1 = relu(dense(flatten(x), 512))
    h2 = relu(dense(h1, 256))
    h3 = relu(dense(h2, 128))
    
    # Classification head
    logits = dense(h3, num_classes)
    probabilities = softmax(logits)
    return probabilities
```

#### Language Model (Transformer-style)
```python
def language_model(x):
    # Embedding and position encoding
    embedded = embedding(x) + position_encoding(x)
    
    # Transformer layers
    for _ in range(num_layers):
        # Self-attention
        attended = attention_layer(embedded)
        embedded = layer_norm(embedded + attended)
        
        # Feed-forward
        ff_output = relu(dense(embedded, ff_size))
        ff_output = dense(ff_output, embed_size)
        embedded = layer_norm(embedded + ff_output)
    
    # Output projection
    logits = dense(embedded, vocab_size)
    return softmax(logits)
```

#### Generative Model (VAE-style)
```python
def variational_autoencoder(x):
    # Encoder
    h1 = relu(dense(x, 256))
    h2 = relu(dense(h1, 128))
    mu = dense(h2, latent_size)      # Mean
    log_var = dense(h2, latent_size) # Log variance
    
    # Reparameterization trick
    eps = random_normal(latent_size)
    z = mu + exp(0.5 * log_var) * eps
    
    # Decoder
    h3 = relu(dense(z, 128))
    h4 = relu(dense(h3, 256))
    reconstruction = sigmoid(dense(h4, input_size))
    
    return reconstruction, mu, log_var
```

### Integration Testing Strategy
Let's test that Dense layers work seamlessly with all activation functions to create complete neural network components!
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-activation-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_layer_activation():
    """Test Dense layer comprehensive testing with activation functions"""
    print("🔬 Unit Test: Layer-Activation Comprehensive Test...")
    
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
    
    print("✅ Layer-activation comprehensive tests passed!")
    print(f"✅ Dense + ReLU working correctly")
    print(f"✅ Dense + Softmax producing valid probabilities")
    print(f"✅ Dense + Sigmoid bounded correctly")
    print(f"✅ Dense + Tanh centered correctly")
    print(f"✅ Multi-layer networks working")
    print(f"✅ All components ready for training!")

# Run the test
test_unit_layer_activation()

# %% [markdown]
"""
### 🎯 CHECKPOINT: Complete Neural Network Components Mastered

Outstanding! You've now mastered the complete pipeline from basic matrix operations to full neural network components!

#### What You've Accomplished
✅ **Complete Neural Network Components**: Dense layers + activations working together  
✅ **Real-World Architecture Patterns**: Understanding how components combine in production systems  
✅ **Integration Mastery**: Seamless compatibility between layers, activations, and tensors  
✅ **Production-Ready Implementation**: Code that scales to actual deep learning applications  

#### Mathematical Concepts Mastered
- **Universal Function Approximation**: Layer + activation composition enables learning any pattern
- **Gradient Flow**: Understanding how gradients propagate through layer-activation chains
- **Architecture Design**: Knowledge of when to use which layer-activation combinations
- **Batch Processing**: Automatic handling of variable batch sizes

#### Real-World Applications You Can Now Build
Your implementations now enable:
- **Image Classification**: Multi-layer networks for computer vision
- **Language Models**: Transformer-style architectures for NLP
- **Generative Models**: VAEs, GANs, and other generative architectures
- **Recommendation Systems**: Deep collaborative filtering networks

#### Advanced Architecture Patterns Understood
- **Residual Networks**: Skip connections for very deep networks
- **Attention Mechanisms**: Query-key-value patterns for transformers
- **Gated Architectures**: LSTM/GRU-style information flow control
- **Multi-layer Perceptrons**: Classic feedforward architectures

**Key insight**: You can now understand and implement ANY neural network architecture!
"""

# %% [markdown]
"""
## 🔬 Integration Test: Layers with Tensors

This is our first cumulative integration test.
It ensures that the 'Layer' abstraction works correctly with the 'Tensor' class from the previous module.
"""

# %%
def test_module_layer_tensor_integration():
    """
    Tests that a Tensor can be passed through a Layer subclass
    and that the output is of the correct type and shape.
    """
    print("🔬 Running Integration Test: Layer with Tensor...")

    # 1. Define a simple Layer that doubles the input
    class DoubleLayer(Dense): # Inherit from Dense to get __call__
        def forward(self, x: Tensor) -> Tensor:
            return x * 2

    # 2. Create an instance of the layer
    double_layer = DoubleLayer(input_size=1, output_size=1) # Dummy sizes

    # 3. Create a Tensor from the previous module
    input_tensor = Tensor([1, 2, 3])

    # 4. Perform the forward pass
    output_tensor = double_layer(input_tensor)

    # 5. Assert correctness
    assert isinstance(output_tensor, Tensor), "Output should be a Tensor"
    assert np.array_equal(output_tensor.data, np.array([2, 4, 6])), "Output data is incorrect"
    print("✅ Integration Test Passed: Layer correctly processed Tensor.")

# Run the integration test
test_module_layer_tensor_integration()

# %% [markdown]
"""
## 🏗️ ML Systems: Architecture Analysis & Memory Scaling

Now that you have working neural network layers, let's develop **architecture analysis skills**. This section teaches you to understand how layer composition affects memory usage, parameter counts, and computational complexity.

### **Learning Outcome**: *"I understand how layers combine to create memory pressure and can analyze model architectures"*

---

## Layer Architecture Profiler (Medium Guided Implementation)

As an ML systems engineer, you need to understand how different layer configurations affect system resources. Let's build tools to analyze layer architectures and scaling patterns.
"""

# %%
import time
import psutil
import os

class LayerArchitectureProfiler:
    """
    Architecture analysis toolkit for neural network layers.
    
    Helps ML engineers understand memory scaling, parameter counts,
    and computational complexity of different layer configurations.
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.analysis_cache = {}
        
    def analyze_layer_parameters(self, input_size, hidden_size, output_size):
        """
        Analyze parameter count and memory usage for a layer configuration.
        
        TODO: Implement parameter count analysis.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Calculate weight matrix parameters: input_size * hidden_size
        2. Calculate bias parameters: hidden_size  
        3. Calculate total parameters: weights + bias
        4. Calculate memory usage: parameters * 4 bytes (float32)
        5. Return analysis dictionary with all metrics
        
        EXAMPLE:
        profiler = LayerArchitectureProfiler()
        analysis = profiler.analyze_layer_parameters(784, 128, 10)
        print(f"Parameters: {analysis['total_parameters']:,}")
        print(f"Memory: {analysis['memory_mb']:.2f} MB")
        
        HINTS:
        - Weight matrix shape: (input_size, hidden_size)
        - Bias vector shape: (hidden_size,)
        - Float32 = 4 bytes per parameter
        - Convert bytes to MB: bytes / (1024 * 1024)
        """
        ### BEGIN SOLUTION
        # Calculate parameters
        weight_params = input_size * hidden_size
        bias_params = hidden_size
        total_params = weight_params + bias_params
        
        # Calculate memory (assuming float32 = 4 bytes)
        memory_bytes = total_params * 4
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'weight_parameters': weight_params,
            'bias_parameters': bias_params,
            'total_parameters': total_params,
            'memory_bytes': memory_bytes,
            'memory_mb': memory_mb
        }
        ### END SOLUTION
    
    def analyze_network_scaling(self, input_size, hidden_sizes, output_size):
        """
        Analyze how network depth affects parameter count and memory.
        
        TODO: Implement network scaling analysis.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Initialize total parameters counter
        2. For each layer in the network:
           a. Calculate layer parameters using analyze_layer_parameters
           b. Add to total count
           c. Update input_size for next layer
        3. Calculate total memory usage
        4. Return comprehensive analysis
        
        EXAMPLE:
        profiler = LayerArchitectureProfiler()
        analysis = profiler.analyze_network_scaling(784, [512, 256, 128], 10)
        print(f"Total parameters: {analysis['total_parameters']:,}")
        print(f"Layers: {len(analysis['layer_details'])}")
        
        HINTS:
        - Loop through hidden_sizes for each layer
        - Track input_size changes: input → hidden[0] → hidden[1] → ... → output
        - Sum all layer parameters
        - Store per-layer details for analysis
        """
        ### BEGIN SOLUTION
        total_parameters = 0
        layer_details = []
        current_input = input_size
        
        # Analyze each hidden layer
        for i, hidden_size in enumerate(hidden_sizes):
            layer_analysis = self.analyze_layer_parameters(current_input, hidden_size, 0)
            layer_analysis['layer_name'] = f'Hidden_{i+1}'
            layer_details.append(layer_analysis)
            total_parameters += layer_analysis['total_parameters']
            current_input = hidden_size
        
        # Analyze output layer
        output_analysis = self.analyze_layer_parameters(current_input, output_size, 0)
        output_analysis['layer_name'] = 'Output'
        layer_details.append(output_analysis)
        total_parameters += output_analysis['total_parameters']
        
        # Calculate total memory
        total_memory_mb = total_parameters * 4 / (1024 * 1024)
        
        return {
            'network_architecture': f"{input_size} → {' → '.join(map(str, hidden_sizes))} → {output_size}",
            'total_parameters': total_parameters,
            'total_memory_mb': total_memory_mb,
            'num_layers': len(hidden_sizes) + 1,
            'layer_details': layer_details
        }
        ### END SOLUTION
    
    def compare_architectures(self, input_size, architecture_configs, output_size=10):
        """
        Compare different network architectures for parameter efficiency.
        
        This function is PROVIDED to demonstrate architecture analysis.
        Students use it to understand architecture trade-offs.
        """
        print(f"🏗️ ARCHITECTURE COMPARISON")
        print(f"=" * 50)
        print(f"Input size: {input_size}, Output size: {output_size}")
        
        results = {}
        
        for arch_name, hidden_sizes in architecture_configs.items():
            analysis = self.analyze_network_scaling(input_size, hidden_sizes, output_size)
            results[arch_name] = analysis
            
            print(f"\n📊 {arch_name}:")
            print(f"   Architecture: {analysis['network_architecture']}")
            print(f"   Parameters: {analysis['total_parameters']:,}")
            print(f"   Memory: {analysis['total_memory_mb']:.2f} MB")
            print(f"   Layers: {analysis['num_layers']}")
        
        # Find most/least parameter efficient
        sorted_by_params = sorted(results.items(), key=lambda x: x[1]['total_parameters'])
        most_efficient = sorted_by_params[0]
        least_efficient = sorted_by_params[-1]
        
        print(f"\n🎯 EFFICIENCY ANALYSIS:")
        print(f"   Most efficient: {most_efficient[0]} ({most_efficient[1]['total_parameters']:,} params)")
        print(f"   Least efficient: {least_efficient[0]} ({least_efficient[1]['total_parameters']:,} params)")
        
        efficiency_ratio = least_efficient[1]['total_parameters'] / most_efficient[1]['total_parameters']
        print(f"   Parameter difference: {efficiency_ratio:.1f}x")
        
        return results
    
    def analyze_depth_vs_width_tradeoffs(self, input_size=784, output_size=10):
        """
        Analyze the classic deep vs wide network trade-off.
        
        This function is PROVIDED to show systems thinking.
        Students run it to understand architecture decisions.
        """
        print(f"🔍 DEPTH vs WIDTH ANALYSIS")
        print(f"=" * 40)
        
        # Test different depth vs width configurations
        configurations = {
            'Shallow Wide': [1024],                    # 1 huge layer
            'Medium Wide': [512, 512],                 # 2 medium layers  
            'Medium Deep': [256, 256, 256],           # 3 smaller layers
            'Deep Narrow': [128, 128, 128, 128],      # 4 narrow layers
            'Very Deep': [64, 64, 64, 64, 64, 64]     # 6 very narrow layers
        }
        
        results = {}
        for config_name, hidden_sizes in configurations.items():
            analysis = self.analyze_network_scaling(input_size, hidden_sizes, output_size)
            results[config_name] = analysis
            
            # Calculate depth and width metrics
            depth = len(hidden_sizes)
            avg_width = sum(hidden_sizes) / len(hidden_sizes)
            max_width = max(hidden_sizes)
            
            print(f"\n{config_name}:")
            print(f"   Depth: {depth} layers")
            print(f"   Avg width: {avg_width:.0f} neurons")
            print(f"   Max width: {max_width} neurons")
            print(f"   Parameters: {analysis['total_parameters']:,}")
            print(f"   Memory: {analysis['total_memory_mb']:.2f} MB")
        
        print(f"\n💡 ARCHITECTURE INSIGHTS:")
        print(f"   - Deeper networks: Better representation learning, harder to train")
        print(f"   - Wider networks: More capacity per layer, more parameters")
        print(f"   - Modern trend: Very deep (100+ layers) with skip connections")
        print(f"   - Memory scales with total parameters regardless of arrangement")
        
        return results

def analyze_famous_architectures():
    """
    Analyze parameter counts of famous neural network architectures.
    
    This function is PROVIDED to connect student work to real systems.
    Shows how layer analysis applies to production models.
    """
    profiler = LayerArchitectureProfiler()
    
    print(f"🌟 FAMOUS ARCHITECTURE ANALYSIS")
    print(f"=" * 50)
    
    # Simplified versions of famous architectures
    famous_models = {
        'LeNet-5 (1998)': {
            'description': 'First successful CNN',
            'approx_params': 60_000,
            'era': 'Early deep learning'
        },
        'AlexNet (2012)': {
            'description': 'ImageNet breakthrough',
            'approx_params': 60_000_000,
            'era': 'Deep learning revolution'
        },
        'VGG-16 (2014)': {
            'description': 'Very deep networks',
            'approx_params': 138_000_000,
            'era': 'Going deeper'
        },
        'ResNet-50 (2015)': {
            'description': 'Skip connections enable very deep nets',
            'approx_params': 25_600_000,
            'era': 'Architecture innovation'
        },
        'GPT-3 (2020)': {
            'description': 'Large language model',
            'approx_params': 175_000_000_000,
            'era': 'Scale revolution'
        },
        'GPT-4 (2023)': {
            'description': 'Estimated multimodal model',
            'approx_params': 1_800_000_000_000,
            'era': 'Massive scale'
        }
    }
    
    print(f"Model Evolution Over Time:")
    for model_name, info in famous_models.items():
        params = info['approx_params']
        memory_gb = params * 4 / (1024**3)  # Rough memory estimate
        
        print(f"\n{model_name}:")
        print(f"   Parameters: {params:,}")
        print(f"   Est. Memory: {memory_gb:.1f} GB")
        print(f"   Description: {info['description']}")
        print(f"   Era: {info['era']}")
    
    # Show scaling progression
    print(f"\n📈 SCALING PROGRESSION:")
    params_1998 = famous_models['LeNet-5 (1998)']['approx_params']
    params_2023 = famous_models['GPT-4 (2023)']['approx_params']
    scaling_factor = params_2023 / params_1998
    
    print(f"   1998 → 2023: {scaling_factor:,.0f}x parameter increase")
    print(f"   That's about {scaling_factor/1000000:.1f} million times larger!")
    print(f"   Memory requirements grew from KB to TB")
    
    print(f"\n🎯 SYSTEMS IMPLICATIONS:")
    print(f"   - Parameter count directly affects memory requirements")
    print(f"   - Larger models need distributed training across multiple GPUs")
    print(f"   - Model serving requires careful memory management")
    print(f"   - Architecture efficiency becomes crucial at scale")
    
    return famous_models

# %% [markdown]
"""
### 🎯 Learning Activity 1: Layer Architecture Analysis (Medium Guided Implementation)

**Goal**: Learn to analyze neural network architectures and understand how layer configurations affect system resources.

Complete the missing implementations in the `LayerArchitectureProfiler` class above, then use your profiler to understand architecture trade-offs.
"""

# %%
# Initialize the layer architecture profiler
profiler = LayerArchitectureProfiler()

print("🏗️ LAYER ARCHITECTURE ANALYSIS")
print("=" * 50)

# Test 1: Single layer analysis
print("📊 Single Layer Analysis:")
layer_configs = [
    (784, 128),    # MNIST → small hidden
    (784, 512),    # MNIST → medium hidden  
    (784, 2048),   # MNIST → large hidden
    (3072, 1024),  # CIFAR-10 → hidden
]

for input_size, hidden_size in layer_configs:
    analysis = profiler.analyze_layer_parameters(input_size, hidden_size, 10)
    print(f"   {input_size} → {hidden_size}: {analysis['total_parameters']:,} params, {analysis['memory_mb']:.2f} MB")

# Test 2: Network scaling analysis
print(f"\n🔍 Network Scaling Analysis:")
network_configs = [
    ([128], "Small network"),
    ([256, 128], "Medium network"),
    ([512, 256, 128], "Large network"),
    ([1024, 512, 256, 128], "Very large network")
]

for hidden_sizes, description in network_configs:
    analysis = profiler.analyze_network_scaling(784, hidden_sizes, 10)
    print(f"   {description}: {analysis['total_parameters']:,} params, {analysis['total_memory_mb']:.2f} MB")

print(f"\n💡 SCALING INSIGHTS:")
print(f"   - Adding layers multiplies parameter count")
print(f"   - First layer often dominates parameter count (large input)")
print(f"   - Memory scales linearly with parameter count")
print(f"   - Architecture choice = resource planning decision")

# %% [markdown]
"""
### 🎯 Learning Activity 2: Architecture Comparison & Analysis (Review & Understand)

**Goal**: Compare different network architectures and understand the depth vs width trade-offs that affect production ML systems.
"""

# %%
# Compare different architecture strategies
input_size = 784  # MNIST flattened image
output_size = 10  # 10 digit classes

architecture_configs = {
    'Baseline': [128],
    'Wide Shallow': [512], 
    'Narrow Deep': [64, 64, 64],
    'Pyramid': [256, 128, 64],
    'Inverted Pyramid': [64, 128, 256],
    'Bottleneck': [512, 32, 512]
}

# Students use their implemented analysis tools
comparison_results = profiler.compare_architectures(input_size, architecture_configs, output_size)

# Analyze depth vs width trade-offs
depth_width_results = profiler.analyze_depth_vs_width_tradeoffs(input_size, output_size)

# Connect to famous architectures
famous_analysis = analyze_famous_architectures()

print(f"\n🎯 KEY LEARNINGS FOR ML SYSTEMS ENGINEERS:")
print(f"=" * 55)

print(f"\n1. 📊 PARAMETER SCALING:")
print(f"   First layer dominates: input_size × hidden_size")
print(f"   Layer composition multiplies parameter count")
print(f"   Memory = parameters × 4 bytes (float32)")

print(f"\n2. 🏗️ ARCHITECTURE STRATEGIES:")
print(f"   Wide networks: More capacity, more parameters")
print(f"   Deep networks: Better representations, harder training")
print(f"   Bottlenecks: Compress then expand information")

print(f"\n3. 🚀 PRODUCTION IMPLICATIONS:")
print(f"   Parameter count = memory requirements")
print(f"   Model serving: Load entire model into memory")
print(f"   Training: Need 2-3x model size for gradients/optimizer")

print(f"\n4. 💰 COST IMPLICATIONS:")
print(f"   More parameters = larger cloud instances needed")
print(f"   GPU memory limits determine maximum model size")
print(f"   Distributed training costs scale with model size")

print(f"\n💡 SYSTEMS ENGINEERING INSIGHT:")
print(f"Every layer you add is a resource planning decision:")
print(f"- More layers = more memory = higher cloud costs")
print(f"- Architecture efficiency matters at production scale")
print(f"- Understanding parameter scaling helps optimize deployments")

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Neural Network Layers - Foundation of All AI

🎉 **CONGRATULATIONS!** You've just mastered the mathematical and computational foundation of ALL modern artificial intelligence!

### What You've Accomplished: A Complete AI Foundation

#### ✅ Mathematical Mastery
- **Matrix Multiplication Engine**: The core operation powering every neural network
- **Dense Layer Implementation**: The universal building block of all AI systems
- **Universal Function Approximation**: Understanding how layer+activation enables learning ANY pattern
- **Weight Initialization Science**: Xavier/Glorot strategies for stable training

#### ✅ Implementation Excellence
- **Production-Grade Code**: Your implementations match PyTorch and TensorFlow standards
- **Shape Management Mastery**: Automatic batch processing and broadcasting
- **Error Handling**: Robust validation and meaningful error messages
- **Integration Ready**: Seamless compatibility with Tensor and Activation modules

#### ✅ Real-World Architecture Understanding
- **Multi-Layer Perceptrons**: Classic feedforward architectures
- **Residual Networks**: Skip connections for ultra-deep networks
- **Attention Mechanisms**: The foundation of transformers and GPT models
- **Generative Architectures**: VAEs, GANs, and modern generative AI

### Deep Mathematical Concepts Mastered

#### Linear Algebra Foundations
```
Matrix Multiplication: C = A @ B
Dense Layer: y = xW + b
Universal Approximation: f(x) = activation_n(...activation_1(x @ W_1 + b_1)...)
```

#### Parameter Learning Theory
- **Initialization Strategies**: Why random weights break symmetry
- **Gradient Flow**: How learning signals propagate through networks  
- **Batch Processing**: Vectorized operations for computational efficiency
- **Broadcasting**: Automatic shape handling for different tensor dimensions

#### Architecture Design Principles
- **Width vs Depth**: Trade-offs in network architecture
- **Activation Selection**: Choosing the right nonlinearity for each layer
- **Skip Connections**: Enabling ultra-deep networks with residual learning
- **Attention Patterns**: Query-key-value mechanisms for sequence modeling

### Real-World Impact: What You Can Now Build

#### 🖼️ Computer Vision
```python
# Image classification with your Dense layers
image → flatten → dense(784→512) → relu → dense(512→256) → relu → dense(256→10) → softmax
```
- **Object Recognition**: Classify images into thousands of categories
- **Medical Imaging**: Detect diseases from X-rays and MRI scans
- **Autonomous Vehicles**: Recognize traffic signs and pedestrians

#### 🗣️ Natural Language Processing
```python
# Language model with your Dense layers
text → embed → dense(300→128) → tanh → dense(128→vocab) → softmax
```
- **Language Models**: Build GPT-style text generation systems
- **Machine Translation**: Translate between any pair of languages  
- **Sentiment Analysis**: Understand emotional content in text

#### 🎯 Recommendation Systems
```python
# Collaborative filtering with your Dense layers
user_features → dense(1000→256) → relu → dense(256→items) → sigmoid
```
- **Netflix Recommendations**: Predict what movies users will enjoy
- **E-commerce**: Suggest products based on browsing history
- **Social Media**: Recommend friends and content

#### 🧪 Scientific AI
```python
# Physics simulation with your Dense layers
parameters → dense(10→64) → relu → dense(64→64) → relu → dense(64→1) → output
```
- **Drug Discovery**: Predict molecular properties for new medicines
- **Climate Modeling**: Simulate complex atmospheric phenomena
- **Materials Science**: Design new materials with desired properties

### Connection to Advanced AI Systems

#### 🤖 Large Language Models (GPT, ChatGPT)
```python
# Every transformer layer uses YOUR Dense implementation
attention_output → dense(hidden→hidden) → relu → dense(hidden→hidden)
```
Your Dense layers power the feed-forward networks in every transformer!

#### 🎨 Generative AI (DALL-E, Stable Diffusion)  
```python
# Generative models built on YOUR foundation
noise → dense(100→256) → relu → dense(256→784) → sigmoid → image
```
Your layers enable the neural networks that create art and images!

#### 🎮 Reinforcement Learning (AlphaGo, game AI)
```python
# Policy networks use YOUR Dense layers
game_state → dense(board→256) → relu → dense(256→actions) → softmax
```
Your implementation enables AI that masters complex games!

### Professional Skills Developed

#### 🏗️ Software Engineering
- **Clean Code**: Well-documented, readable implementations
- **Testing**: Comprehensive validation of functionality
- **API Design**: Consistent, intuitive interfaces
- **Error Handling**: Graceful failure modes with helpful messages

#### 🧮 Mathematical Computing
- **Numerical Stability**: Proper initialization and scaling
- **Performance Optimization**: Understanding computational complexity
- **Memory Management**: Efficient tensor operations
- **Debugging**: Systematic approaches to shape and gradient issues

#### 🔬 Machine Learning Engineering
- **Architecture Design**: Knowing when to use which layer types
- **Hyperparameter Selection**: Understanding initialization and activation choices
- **Gradient Flow**: Designing networks for stable training
- **Production Deployment**: Building scalable, maintainable systems

### Industry-Standard Implementation Quality

#### Production System Equivalence
```python
# Your implementation
layer = Dense(input_size=784, output_size=10)
output = layer(input)

# PyTorch equivalent
layer = torch.nn.Linear(784, 10)
output = layer(input)

# TensorFlow equivalent  
layer = tf.keras.layers.Dense(10)
output = layer(input)

# IDENTICAL MATHEMATICAL OPERATIONS!
```

#### Performance Considerations
- **Computational Complexity**: O(batch_size × input_size × output_size)
- **Memory Usage**: Optimal tensor storage and reuse
- **GPU Acceleration**: Foundation for hardware optimization
- **Distributed Computing**: Basis for multi-device training

### Advanced Topics You're Now Ready For

#### 🧠 Specialized Architectures
- **Convolutional Networks**: For image and spatial data processing
- **Recurrent Networks**: For sequential data and time series
- **Graph Neural Networks**: For structured data and relationships
- **Transformer Architectures**: For attention-based modeling

#### 🎯 Advanced Training Techniques
- **Batch Normalization**: Stabilizing training in deep networks
- **Dropout Regularization**: Preventing overfitting
- **Learning Rate Scheduling**: Optimizing convergence
- **Transfer Learning**: Adapting pre-trained models

#### 🚀 Cutting-Edge Research
- **Neural Architecture Search**: Automatically designing networks
- **Meta-Learning**: Learning to learn new tasks quickly
- **Federated Learning**: Training across distributed devices
- **Quantum Neural Networks**: Quantum computing + neural networks

### Your Neural Network Toolkit

You now have the complete foundation to understand and implement:

```python
# ANY neural network architecture can be built with your components!

def your_neural_network(x):
    # Foundation layers (YOUR implementation)
    h1 = relu(dense1(x))
    h2 = relu(dense2(h1))
    
    # Advanced patterns (built on YOUR foundation)
    attention = attention_layer(h2)
    residual = h2 + attention
    
    # Output (YOUR implementation)
    output = softmax(dense_output(residual))
    return output
```

### Next Steps: Continue Your AI Journey

#### 🔧 Module 5: Convolutional Layers
Build specialized layers for image processing and computer vision

#### 📊 Module 6: Optimization
Implement gradient descent and advanced optimization algorithms  

#### 🔄 Module 7: Training Loops
Create complete training and validation pipelines

#### 🌐 Module 8: Advanced Architectures
Build transformers, ResNets, and state-of-the-art models

### The Bigger Picture: Your Impact on AI

**You now understand the mathematical foundation of:**
- Every neural network ever created
- All modern AI systems (GPT, DALL-E, AlphaGo, etc.)
- The core operations that power trillion-dollar AI companies
- The building blocks enabling the current AI revolution

**Your layer implementations:**
- Are mathematically equivalent to production systems
- Form the foundation of all advanced architectures  
- Enable you to contribute to cutting-edge AI research
- Provide the knowledge to build the next generation of AI systems

### 🌟 **You Are Now a Neural Network Architect!**

With your deep understanding of layers, you can:
- **Understand** any neural network architecture
- **Implement** custom layer types for new applications
- **Debug** training issues in complex models
- **Optimize** networks for production deployment
- **Research** novel architectures for unsolved problems

**Welcome to the community of AI builders! Your journey to mastering neural networks is well underway.**

---

*"Every expert was once a beginner. Every pro was once an amateur. Every icon was once an unknown." - Robin Sharma*

**You've built the foundation. Now go build the future of AI!** 🚀
""" 
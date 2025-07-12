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
# Module 5: CNN - Convolutional Neural Networks

Welcome to the CNN module! Here you'll implement the core building block of modern computer vision: the convolutional layer.

## Learning Goals
- Understand the convolution operation and its importance in computer vision
- Implement Conv2D with explicit for-loops to understand the sliding window mechanism
- Build convolutional layers that can detect spatial patterns in images
- Compose Conv2D with other layers to build complete convolutional networks
- See how convolution enables parameter sharing and translation invariance

## Build → Use → Understand
1. **Build**: Conv2D layer using sliding window convolution from scratch
2. **Use**: Transform images and see feature maps emerge
3. **Understand**: How CNNs learn hierarchical spatial patterns
"""

# %% nbgrader={"grade": false, "grade_id": "cnn-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.cnn

#| export
import numpy as np
import os
import sys
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Import from the main package - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_activations'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_layers'))
    from tensor_dev import Tensor
    from activations_dev import ReLU
    from layers_dev import Dense

# %% nbgrader={"grade": false, "grade_id": "cnn-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
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

# %% nbgrader={"grade": false, "grade_id": "cnn-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch CNN Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build convolutional neural networks!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/05_cnn/cnn_dev.py`  
**Building Side:** Code exports to `tinytorch.core.cnn`

```python
# Final package structure:
from tinytorch.core.cnn import Conv2D, conv2d_naive, flatten  # CNN operations!
from tinytorch.core.layers import Dense  # Fully connected layers
from tinytorch.core.activations import ReLU  # Nonlinearity
from tinytorch.core.tensor import Tensor  # Foundation
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding of convolution
- **Production:** Proper organization like PyTorch's `torch.nn.Conv2d`
- **Consistency:** All CNN operations live together in `core.cnn`
- **Integration:** Works seamlessly with other TinyTorch components
"""

# %% [markdown]
"""
## 🧠 The Mathematical Foundation of Convolution

### The Convolution Operation
Convolution is a mathematical operation that combines two functions to produce a third function:

```
(f * g)(t) = ∫ f(τ)g(t - τ)dτ
```

In discrete 2D computer vision, this becomes:
```
(I * K)[i,j] = ΣΣ I[i+m, j+n] × K[m,n]
```

### Why Convolution is Perfect for Images
- **Local connectivity**: Each output depends only on a small region of input
- **Weight sharing**: Same filter applied everywhere (translation invariance)
- **Spatial hierarchy**: Multiple layers build increasingly complex features
- **Parameter efficiency**: Much fewer parameters than fully connected layers

### The Three Core Principles
1. **Sparse connectivity**: Each neuron connects to only a small region
2. **Parameter sharing**: Same weights used across all spatial locations
3. **Equivariant representation**: If input shifts, output shifts correspondingly

### Connection to Real ML Systems
Every vision framework uses convolution:
- **PyTorch**: `torch.nn.Conv2d` with optimized CUDA kernels
- **TensorFlow**: `tf.keras.layers.Conv2D` with cuDNN acceleration
- **JAX**: `jax.lax.conv_general_dilated` with XLA compilation
- **TinyTorch**: `tinytorch.core.cnn.Conv2D` (what we're building!)

### Performance Considerations
- **Memory layout**: Efficient data access patterns
- **Vectorization**: SIMD operations for parallel computation
- **Cache efficiency**: Spatial locality in memory access
- **Optimization**: im2col, FFT-based convolution, Winograd algorithm
"""

# %% [markdown]
"""
## Step 1: Understanding Convolution

### What is Convolution?
A **convolutional layer** applies a small filter (kernel) across the input, producing a feature map. This operation captures local patterns and is the foundation of modern vision models.

### Why Convolution Matters in Computer Vision
- **Local connectivity**: Each output value depends only on a small region of the input
- **Weight sharing**: The same filter is applied everywhere (translation invariance)
- **Spatial hierarchy**: Multiple layers build increasingly complex features
- **Parameter efficiency**: Much fewer parameters than fully connected layers

### The Fundamental Insight
**Convolution is pattern matching!** The kernel learns to detect specific patterns:
- **Edge detectors**: Find boundaries between objects
- **Texture detectors**: Recognize surface patterns
- **Shape detectors**: Identify geometric forms
- **Feature detectors**: Combine simple patterns into complex features

### Real-World Examples
- **Image processing**: Detect edges, blur, sharpen
- **Computer vision**: Recognize objects, faces, text
- **Medical imaging**: Detect tumors, analyze scans
- **Autonomous driving**: Identify traffic signs, pedestrians

### Visual Intuition
```
Input Image:     Kernel:        Output Feature Map:
[1, 2, 3]       [1,  0]       [1*1+2*0+4*0+5*(-1), 2*1+3*0+5*0+6*(-1)]
[4, 5, 6]       [0, -1]       [4*1+5*0+7*0+8*(-1), 5*1+6*0+8*0+9*(-1)]
[7, 8, 9]
```

The kernel slides across the input, computing dot products at each position.

Let's implement this step by step!
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-naive", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def conv2d_naive(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Naive 2D convolution (single channel, no stride, no padding).
    
    Args:
        input: 2D input array (H, W)
        kernel: 2D filter (kH, kW)
    Returns:
        2D output array (H-kH+1, W-kW+1)
        
    TODO: Implement the sliding window convolution using for-loops.
    
    APPROACH:
    1. Get input dimensions: H, W = input.shape
    2. Get kernel dimensions: kH, kW = kernel.shape
    3. Calculate output dimensions: out_H = H - kH + 1, out_W = W - kW + 1
    4. Create output array: np.zeros((out_H, out_W))
    5. Use nested loops to slide the kernel:
       - i loop: output rows (0 to out_H-1)
       - j loop: output columns (0 to out_W-1)
       - di loop: kernel rows (0 to kH-1)
       - dj loop: kernel columns (0 to kW-1)
    6. For each (i,j), compute: output[i,j] += input[i+di, j+dj] * kernel[di, dj]
    
    EXAMPLE:
    Input: [[1, 2, 3],     Kernel: [[1, 0],
            [4, 5, 6],               [0, -1]]
            [7, 8, 9]]
    
    Output[0,0] = 1*1 + 2*0 + 4*0 + 5*(-1) = 1 - 5 = -4
    Output[0,1] = 2*1 + 3*0 + 5*0 + 6*(-1) = 2 - 6 = -4
    Output[1,0] = 4*1 + 5*0 + 7*0 + 8*(-1) = 4 - 8 = -4
    Output[1,1] = 5*1 + 6*0 + 8*0 + 9*(-1) = 5 - 9 = -4
    
    HINTS:
    - Start with output = np.zeros((out_H, out_W))
    - Use four nested loops: for i in range(out_H): for j in range(out_W): for di in range(kH): for dj in range(kW):
    - Accumulate the sum: output[i,j] += input[i+di, j+dj] * kernel[di, dj]
    """
    ### BEGIN SOLUTION
    # Get input and kernel dimensions
    H, W = input.shape
    kH, kW = kernel.shape
    
    # Calculate output dimensions
    out_H, out_W = H - kH + 1, W - kW + 1
    
    # Initialize output array
    output = np.zeros((out_H, out_W), dtype=input.dtype)
    
    # Sliding window convolution with four nested loops
    for i in range(out_H):
        for j in range(out_W):
            for di in range(kH):
                for dj in range(kW):
                    output[i, j] += input[i + di, j + dj] * kernel[di, dj]
    
    return output
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Quick Test: Convolution Operation

Let's test your convolution implementation right away! This is the core operation that powers computer vision.
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-naive-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test conv2d_naive function immediately after implementation
print("🔬 Testing convolution operation...")

# Test simple 3x3 input with 2x2 kernel
try:
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    kernel_array = np.array([[1, 0], [0, 1]], dtype=np.float32)  # Identity-like kernel
    
    result = conv2d_naive(input_array, kernel_array)
    expected = np.array([[6, 8], [12, 14]], dtype=np.float32)  # 1+5, 2+6, 4+8, 5+9
    
    print(f"Input:\n{input_array}")
    print(f"Kernel:\n{kernel_array}")
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    
    assert np.allclose(result, expected), f"Convolution failed: expected {expected}, got {result}"
    print("✅ Simple convolution test passed")
    
except Exception as e:
    print(f"❌ Simple convolution test failed: {e}")
    raise

# Test edge detection kernel
try:
    input_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    edge_kernel = np.array([[-1, -1], [-1, 3]], dtype=np.float32)  # Edge detection
    
    result = conv2d_naive(input_array, edge_kernel)
    expected = np.array([[0, 0], [0, 0]], dtype=np.float32)  # Uniform region = no edges
    
    assert np.allclose(result, expected), f"Edge detection failed: expected {expected}, got {result}"
    print("✅ Edge detection test passed")
    
except Exception as e:
    print(f"❌ Edge detection test failed: {e}")
    raise

# Test output shape
try:
    input_5x5 = np.random.randn(5, 5).astype(np.float32)
    kernel_3x3 = np.random.randn(3, 3).astype(np.float32)
    
    result = conv2d_naive(input_5x5, kernel_3x3)
    expected_shape = (3, 3)  # 5-3+1 = 3
    
    assert result.shape == expected_shape, f"Output shape wrong: expected {expected_shape}, got {result.shape}"
    print("✅ Output shape test passed")
    
except Exception as e:
    print(f"❌ Output shape test failed: {e}")
    raise

# Show the convolution process
print("🎯 Convolution behavior:")
print("   Slides kernel across input")
print("   Computes dot product at each position")
print("   Output size = Input size - Kernel size + 1")
print("📈 Progress: Convolution operation ✓")

# %% [markdown]
"""
## Step 2: Building the Conv2D Layer

### What is a Conv2D Layer?
A **Conv2D layer** is a learnable convolutional layer that:
- Has learnable kernel weights (initialized randomly)
- Applies convolution to input tensors
- Integrates with the rest of the neural network

### Why Conv2D Layers Matter
- **Feature learning**: Kernels learn to detect useful patterns
- **Composability**: Can be stacked with other layers
- **Efficiency**: Shared weights reduce parameters dramatically
- **Translation invariance**: Same patterns detected anywhere in the image

### Real-World Applications
- **Image classification**: Recognize objects in photos
- **Object detection**: Find and locate objects
- **Medical imaging**: Detect anomalies in scans
- **Autonomous driving**: Identify road features

### Design Decisions
- **Kernel size**: Typically 3×3 or 5×5 for balance of locality and capacity
- **Initialization**: Small random values to break symmetry
- **Integration**: Works with Tensor class and other layers
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Conv2D:
    """
    2D Convolutional Layer (single channel, single filter, no stride/pad).
    
    A learnable convolutional layer that applies a kernel to detect spatial patterns.
    Perfect for building the foundation of convolutional neural networks.
    """
    
    def __init__(self, kernel_size: Tuple[int, int]):
        """
        Initialize Conv2D layer with random kernel.
        
        Args:
            kernel_size: (kH, kW) - size of the convolution kernel
            
        TODO: Initialize a random kernel with small values.
        
        APPROACH:
        1. Store kernel_size as instance variable
        2. Initialize random kernel with small values
        3. Use proper initialization for stable training
        
        EXAMPLE:
        Conv2D((2, 2)) creates:
        - kernel: shape (2, 2) with small random values
        
        HINTS:
        - Store kernel_size as self.kernel_size
        - Initialize kernel: np.random.randn(kH, kW) * 0.1 (small values)
        - Convert to float32 for consistency
        """
        ### BEGIN SOLUTION
        # Store kernel size
        self.kernel_size = kernel_size
        kH, kW = kernel_size
        
        # Initialize random kernel with small values
        self.kernel = np.random.randn(kH, kW).astype(np.float32) * 0.1
        ### END SOLUTION
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: apply convolution to input tensor.
        
        Args:
            x: Input tensor (2D for simplicity)
            
        Returns:
            Output tensor after convolution
            
        TODO: Implement forward pass using conv2d_naive function.
        
        APPROACH:
        1. Extract numpy array from input tensor
        2. Apply conv2d_naive with stored kernel
        3. Return result wrapped in Tensor
        
        EXAMPLE:
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3)
        layer = Conv2D((2, 2))
        y = layer(x)  # shape (2, 2)
        
        HINTS:
        - Use x.data to get numpy array
        - Use conv2d_naive(x.data, self.kernel)
        - Return Tensor(result) to wrap the result
        """
        ### BEGIN SOLUTION
        # Apply convolution using naive implementation
        result = conv2d_naive(x.data, self.kernel)
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Quick Test: Conv2D Layer

Let's test your Conv2D layer implementation! This is a learnable convolutional layer that can be trained.
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-layer-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test Conv2D layer immediately after implementation
print("🔬 Testing Conv2D layer...")

# Create a Conv2D layer
try:
    layer = Conv2D(kernel_size=(2, 2))
    print(f"Conv2D layer created with kernel size: {layer.kernel_size}")
    print(f"Kernel shape: {layer.kernel.shape}")
    
    # Test that kernel is initialized properly
    assert layer.kernel.shape == (2, 2), f"Kernel shape should be (2, 2), got {layer.kernel.shape}"
    assert not np.allclose(layer.kernel, 0), "Kernel should not be all zeros"
    print("✅ Conv2D layer initialization successful")
    
    # Test with sample input
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Input shape: {x.shape}")
    
    y = layer(x)
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
    
    # Verify shapes
    assert y.shape == (2, 2), f"Output shape should be (2, 2), got {y.shape}"
    assert isinstance(y, Tensor), "Output should be a Tensor"
    print("✅ Conv2D layer forward pass successful")
    
except Exception as e:
    print(f"❌ Conv2D layer test failed: {e}")
    raise

# Test different kernel sizes
try:
    layer_3x3 = Conv2D(kernel_size=(3, 3))
    x_5x5 = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    y_3x3 = layer_3x3(x_5x5)
    
    assert y_3x3.shape == (3, 3), f"3x3 kernel output should be (3, 3), got {y_3x3.shape}"
    print("✅ Different kernel sizes work correctly")
    
except Exception as e:
    print(f"❌ Different kernel sizes test failed: {e}")
    raise

# Show the layer behavior
print("🎯 Conv2D layer behavior:")
print("   Learnable kernel weights")
print("   Applies convolution to detect patterns")
print("   Can be trained end-to-end")
print("📈 Progress: Convolution operation ✓, Conv2D layer ✓")

# %% [markdown]
"""
## Step 3: Flattening for Dense Layers

### What is Flattening?
**Flattening** converts multi-dimensional tensors to 1D vectors, enabling connection between convolutional and dense layers.

### Why Flattening is Needed
- **Interface compatibility**: Conv2D outputs 2D, Dense expects 1D
- **Network composition**: Connect spatial features to classification
- **Standard practice**: Almost all CNNs use this pattern
- **Dimension management**: Preserve information while changing shape

### The Pattern
```
Conv2D → ReLU → Conv2D → ReLU → Flatten → Dense → Output
```

### Real-World Usage
- **Classification**: Final layers need 1D input for class probabilities
- **Feature extraction**: Convert spatial features to vector representations
- **Transfer learning**: Extract features from pre-trained CNNs
"""

# %% nbgrader={"grade": false, "grade_id": "flatten-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def flatten(x: Tensor) -> Tensor:
    """
    Flatten a 2D tensor to 1D (for connecting to Dense layers).
    
    Args:
        x: Input tensor to flatten
        
    Returns:
        Flattened tensor with batch dimension preserved
        
    TODO: Implement flattening operation.
    
    APPROACH:
    1. Get the numpy array from the tensor
    2. Use .flatten() to convert to 1D
    3. Add batch dimension with [None, :]
    4. Return Tensor wrapped around the result
    
    EXAMPLE:
    Input: Tensor([[1, 2], [3, 4]])  # shape (2, 2)
    Output: Tensor([[1, 2, 3, 4]])  # shape (1, 4)
    
    HINTS:
    - Use x.data.flatten() to get 1D array
    - Add batch dimension: result[None, :]
    - Return Tensor(result)
    """
    ### BEGIN SOLUTION
    # Flatten the tensor and add batch dimension
    flattened = x.data.flatten()
    result = flattened[None, :]  # Add batch dimension
    return Tensor(result)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Quick Test: Flatten Function

Let's test your flatten function! This connects convolutional layers to dense layers.
"""

# %% nbgrader={"grade": true, "grade_id": "test-flatten-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test flatten function immediately after implementation
print("🔬 Testing flatten function...")

# Test case 1: 2x2 tensor
try:
    x = Tensor([[1, 2], [3, 4]])
    flattened = flatten(x)
    
    print(f"Input: {x}")
    print(f"Flattened: {flattened}")
    print(f"Flattened shape: {flattened.shape}")
    
    # Verify shape and content
    assert flattened.shape == (1, 4), f"Flattened shape should be (1, 4), got {flattened.shape}"
    expected_data = np.array([[1, 2, 3, 4]])
    assert np.array_equal(flattened.data, expected_data), f"Flattened data should be {expected_data}, got {flattened.data}"
    print("✅ 2x2 flatten test passed")
    
except Exception as e:
    print(f"❌ 2x2 flatten test failed: {e}")
    raise

# Test case 2: 3x3 tensor
try:
    x2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    flattened2 = flatten(x2)
    
    assert flattened2.shape == (1, 9), f"Flattened shape should be (1, 9), got {flattened2.shape}"
    expected_data2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert np.array_equal(flattened2.data, expected_data2), f"Flattened data should be {expected_data2}, got {flattened2.data}"
    print("✅ 3x3 flatten test passed")
    
except Exception as e:
    print(f"❌ 3x3 flatten test failed: {e}")
    raise

# Test case 3: Different shapes
try:
    x3 = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4
    flattened3 = flatten(x3)
    
    assert flattened3.shape == (1, 8), f"Flattened shape should be (1, 8), got {flattened3.shape}"
    expected_data3 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    assert np.array_equal(flattened3.data, expected_data3), f"Flattened data should be {expected_data3}, got {flattened3.data}"
    print("✅ Different shapes flatten test passed")
    
except Exception as e:
    print(f"❌ Different shapes flatten test failed: {e}")
    raise

# Show the flattening behavior
print("🎯 Flatten behavior:")
print("   Converts 2D tensor to 1D")
print("   Preserves batch dimension")
print("   Enables connection to Dense layers")
print("📈 Progress: Convolution operation ✓, Conv2D layer ✓, Flatten ✓")
print("🚀 CNN pipeline ready!")

# %% [markdown]
"""
### 🧪 Test Your CNN Implementations

Once you implement the functions above, run these cells to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-naive", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test conv2d_naive function
print("Testing conv2d_naive function...")

# Test case 1: Simple 3x3 input with 2x2 kernel
input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
kernel_array = np.array([[1, 0], [0, -1]], dtype=np.float32)

result = conv2d_naive(input_array, kernel_array)
expected = np.array([[-4, -4], [-4, -4]], dtype=np.float32)

print(f"Input:\n{input_array}")
print(f"Kernel:\n{kernel_array}")
print(f"Result:\n{result}")
print(f"Expected:\n{expected}")

assert np.allclose(result, expected), f"conv2d_naive failed: expected {expected}, got {result}"

# Test case 2: Different kernel
kernel2 = np.array([[1, 1], [1, 1]], dtype=np.float32)
result2 = conv2d_naive(input_array, kernel2)
expected2 = np.array([[12, 16], [24, 28]], dtype=np.float32)

assert np.allclose(result2, expected2), f"conv2d_naive failed: expected {expected2}, got {result2}"

print("✅ conv2d_naive tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-layer", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test Conv2D layer
print("Testing Conv2D layer...")

# Create a Conv2D layer
layer = Conv2D(kernel_size=(2, 2))
print(f"Kernel size: {layer.kernel_size}")
print(f"Kernel shape: {layer.kernel.shape}")

# Test with sample input
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Input shape: {x.shape}")

y = layer(x)
print(f"Output shape: {y.shape}")
print(f"Output: {y}")

# Verify shapes
assert y.shape == (2, 2), f"Output shape should be (2, 2), got {y.shape}"
assert isinstance(y, Tensor), "Output should be a Tensor"

print("✅ Conv2D layer tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-flatten", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test flatten function
print("Testing flatten function...")

# Test case 1: 2x2 tensor
x = Tensor([[1, 2], [3, 4]])
flattened = flatten(x)

print(f"Input: {x}")
print(f"Flattened: {flattened}")
print(f"Flattened shape: {flattened.shape}")

# Verify shape and content
assert flattened.shape == (1, 4), f"Flattened shape should be (1, 4), got {flattened.shape}"
expected_data = np.array([[1, 2, 3, 4]])
assert np.array_equal(flattened.data, expected_data), f"Flattened data should be {expected_data}, got {flattened.data}"

# Test case 2: 3x3 tensor
x2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
flattened2 = flatten(x2)

assert flattened2.shape == (1, 9), f"Flattened shape should be (1, 9), got {flattened2.shape}"
expected_data2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
assert np.array_equal(flattened2.data, expected_data2), f"Flattened data should be {expected_data2}, got {flattened2.data}"

print("✅ Flatten tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-cnn-pipeline", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test complete CNN pipeline
print("Testing complete CNN pipeline...")

# Create a simple CNN pipeline: Conv2D → ReLU → Flatten → Dense
conv_layer = Conv2D(kernel_size=(2, 2))
relu = ReLU()
dense_layer = Dense(input_size=4, output_size=2)

# Test input (3x3 image)
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Input shape: {x.shape}")

# Forward pass through pipeline
h1 = conv_layer(x)
print(f"After Conv2D: {h1.shape}")

h2 = relu(h1)
print(f"After ReLU: {h2.shape}")

h3 = flatten(h2)
print(f"After Flatten: {h3.shape}")

h4 = dense_layer(h3)
print(f"After Dense: {h4.shape}")

# Verify pipeline works
assert h1.shape == (2, 2), f"Conv2D output should be (2, 2), got {h1.shape}"
assert h2.shape == (2, 2), f"ReLU output should be (2, 2), got {h2.shape}"
assert h3.shape == (1, 4), f"Flatten output should be (1, 4), got {h3.shape}"
assert h4.shape == (1, 2), f"Dense output should be (1, 2), got {h4.shape}"

print("✅ CNN pipeline tests passed!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the core components of convolutional neural networks:

### What You've Accomplished
✅ **Convolution Operation**: Implemented conv2d_naive with sliding window from scratch  
✅ **Conv2D Layer**: Built a learnable convolutional layer with random kernel initialization  
✅ **Flattening**: Created the bridge between convolutional and dense layers  
✅ **CNN Pipeline**: Composed Conv2D → ReLU → Flatten → Dense for complete networks  
✅ **Spatial Pattern Detection**: Understanding how convolution detects local features  

### Key Concepts You've Learned
- **Convolution is pattern matching**: Kernels detect specific spatial patterns
- **Parameter sharing**: Same kernel applied everywhere for translation invariance
- **Local connectivity**: Each output depends only on a small input region
- **Spatial hierarchy**: Multiple layers build increasingly complex features
- **Dimension management**: Flattening connects spatial and vector representations

### Mathematical Foundations
- **Convolution operation**: (I * K)[i,j] = ΣΣ I[i+m, j+n] × K[m,n]
- **Sliding window**: Kernel moves across input computing dot products
- **Feature maps**: Convolution outputs that highlight detected patterns
- **Translation invariance**: Same pattern detected regardless of position

### Real-World Applications
- **Computer vision**: Object recognition, face detection, medical imaging
- **Image processing**: Edge detection, noise reduction, enhancement
- **Autonomous systems**: Traffic sign recognition, obstacle detection
- **Scientific imaging**: Satellite imagery, microscopy, astronomy

### Next Steps
1. **Export your code**: `tito package nbdev --export 05_cnn`
2. **Test your implementation**: `tito module test 05_cnn`
3. **Use your CNN components**: 
   ```python
   from tinytorch.core.cnn import Conv2D, conv2d_naive, flatten
   from tinytorch.core.layers import Dense
   from tinytorch.core.activations import ReLU
   
   # Create CNN pipeline
   conv = Conv2D((3, 3))
   relu = ReLU()
   dense = Dense(16, 10)
   
   # Process image
   features = conv(image)
   activated = relu(features)
   flattened = flatten(activated)
   output = dense(flattened)
   ```
4. **Move to Module 6**: Start building data loading and preprocessing pipelines!

**Ready for the next challenge?** Let's build efficient data loading systems to feed our networks!
""" 
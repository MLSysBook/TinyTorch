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
# CNN - Convolutional Neural Networks

Welcome to the CNN module! Here you'll implement the core building block of modern computer vision: the convolutional layer.

## Learning Goals
- Understand the convolution operation and its importance in computer vision
- Implement Conv2D with explicit for-loops to understand the sliding window mechanism
- Build convolutional layers that can detect spatial patterns in images
- Compose Conv2D with other layers to build complete convolutional networks
- See how convolution enables parameter sharing and translation invariance

## Build → Use → Reflect
1. **Build**: Conv2D layer using sliding window convolution from scratch
2. **Use**: Transform images and see feature maps emerge
3. **Reflect**: How CNNs learn hierarchical spatial patterns

## What You'll Learn
By the end of this module, you'll understand:
- How convolution works as a sliding window operation
- Why convolution is perfect for spatial data like images
- How to build learnable convolutional layers
- The CNN pipeline: Conv2D → Activation → Flatten → Dense
- How parameter sharing makes CNNs efficient
"""

# %% nbgrader={"grade": false, "grade_id": "cnn-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.spatial

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
## Step 1: Understanding Convolution

### What is Convolution?
**Convolution** is a mathematical operation that slides a small filter (kernel) across an input, computing dot products at each position.

### Why Convolution is Perfect for Images
- **Local patterns**: Images have local structure (edges, textures)
- **Translation invariance**: Same pattern can appear anywhere
- **Parameter sharing**: One filter detects the pattern everywhere
- **Spatial hierarchy**: Multiple layers build increasingly complex features

### The Fundamental Insight
**Convolution is pattern matching!** The kernel learns to detect specific patterns:
- **Edge detectors**: Find boundaries between objects
- **Texture detectors**: Recognize surface patterns
- **Shape detectors**: Identify geometric forms
- **Feature detectors**: Combine simple patterns into complex features

### Real-World Applications
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
            [4, 5, 6],              [0, -1]]
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
### 🧪 Unit Test: Convolution Operation

Let's test your convolution implementation right away! This is the core operation that powers computer vision.

**This is a unit test** - it tests one specific function (conv2d_naive) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-naive-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test conv2d_naive function immediately after implementation
print("🔬 Unit Test: Convolution Operation...")

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
    
    def forward(self, x):
        """
        Forward pass through the Conv2D layer.
        
        Args:
            x: Input tensor (batch_size, H, W)
        Returns:
            Output tensor after convolution
        """
        # Handle batches by iterating through each item
        if len(x.shape) == 3:
            batch_size, H, W = x.shape
            # Calculate output shape once
            kH, kW = self.kernel.shape
            out_H, out_W = H - kH + 1, W - kW + 1
            
            # Create an empty list to store results
            results = []
            # Iterate over each image in the batch
            for i in range(batch_size):
                # Apply naive convolution to each image
                convolved = conv2d_naive(x.data[i], self.kernel)
                results.append(convolved)
            # Stack results into a single NumPy array
            output_data = np.stack(results)

        else: # Handle single image case
            output_data = conv2d_naive(x.data, self.kernel)

        return Tensor(output_data)
    
    def __call__(self, x):
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Conv2D Layer

Let's test your Conv2D layer implementation! This is a learnable convolutional layer that can be trained.

**This is a unit test** - it tests one specific class (Conv2D) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d-layer-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test Conv2D layer immediately after implementation
print("🔬 Unit Test: Conv2D Layer...")

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
def flatten(x):
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
    return type(x)(result)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Flatten Function

Let's test your flatten function! This connects convolutional layers to dense layers.

**This is a unit test** - it tests one specific function (flatten) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-flatten-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test flatten function immediately after implementation
print("🔬 Unit Test: Flatten Function...")

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

# %% [markdown]
"""
## Step 4: Comprehensive Test - Complete CNN Pipeline

### Real-World CNN Applications
Let's test our CNN components in realistic scenarios:

#### **Image Classification Pipeline**
```python
# The standard CNN pattern
Conv2D → ReLU → Flatten → Dense → Output
```

#### **Multi-layer CNN**
```python
# Deeper pattern for complex features
Conv2D → ReLU → Conv2D → ReLU → Flatten → Dense → Output
```

#### **Feature Extraction**
```python
# Extract spatial features then classify
image → CNN features → dense classifier → predictions
```

This comprehensive test ensures our CNN components work together for real computer vision applications!
"""

# %% nbgrader={"grade": true, "grade_id": "test-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
# Comprehensive test - complete CNN applications
print("🔬 Comprehensive Test: Complete CNN Applications...")

try:
    # Test 1: Simple CNN Pipeline
    print("\n1. Simple CNN Pipeline Test:")
    
    # Create pipeline: Conv2D → ReLU → Flatten → Dense
    conv = Conv2D(kernel_size=(2, 2))
    relu = ReLU()
    dense = Dense(input_size=4, output_size=3)
    
    # Input image
    image = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Forward pass
    features = conv(image)          # (3,3) → (2,2)
    activated = relu(features)      # (2,2) → (2,2)
    flattened = flatten(activated)  # (2,2) → (1,4)
    output = dense(flattened)       # (1,4) → (1,3)
    
    assert features.shape == (2, 2), f"Conv output shape wrong: {features.shape}"
    assert activated.shape == (2, 2), f"ReLU output shape wrong: {activated.shape}"
    assert flattened.shape == (1, 4), f"Flatten output shape wrong: {flattened.shape}"
    assert output.shape == (1, 3), f"Dense output shape wrong: {output.shape}"
    
    print("✅ Simple CNN pipeline works correctly")
    
    # Test 2: Multi-layer CNN
    print("\n2. Multi-layer CNN Test:")
    
    # Create deeper pipeline: Conv2D → ReLU → Conv2D → ReLU → Flatten → Dense
    conv1 = Conv2D(kernel_size=(2, 2))
    relu1 = ReLU()
    conv2 = Conv2D(kernel_size=(2, 2))
    relu2 = ReLU()
    dense_multi = Dense(input_size=9, output_size=2)
    
    # Larger input for multi-layer processing
    large_image = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    
    # Forward pass
    h1 = conv1(large_image)  # (5,5) → (4,4)
    h2 = relu1(h1)           # (4,4) → (4,4)
    h3 = conv2(h2)           # (4,4) → (3,3)
    h4 = relu2(h3)           # (3,3) → (3,3)
    h5 = flatten(h4)         # (3,3) → (1,9)
    output_multi = dense_multi(h5)  # (1,9) → (1,2)
    
    assert h1.shape == (4, 4), f"Conv1 output wrong: {h1.shape}"
    assert h3.shape == (3, 3), f"Conv2 output wrong: {h3.shape}"
    assert h5.shape == (1, 9), f"Flatten output wrong: {h5.shape}"
    assert output_multi.shape == (1, 2), f"Final output wrong: {output_multi.shape}"
    
    print("✅ Multi-layer CNN works correctly")
    
    # Test 3: Image Classification Scenario
    print("\n3. Image Classification Test:")
    
    # Simulate digit classification with 8x8 image
    digit_image = Tensor([[1, 0, 0, 1, 1, 0, 0, 1],
                     [0, 1, 0, 1, 1, 0, 1, 0],
                     [0, 0, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0, 1, 1, 1],
                     [1, 0, 0, 1, 1, 0, 0, 1],
                     [0, 1, 1, 0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0, 0, 1, 1]])
    
    # CNN for digit classification
    feature_extractor = Conv2D(kernel_size=(3, 3))  # (8,8) → (6,6)
    activation = ReLU()
    classifier = Dense(input_size=36, output_size=10)  # 10 digit classes
    
    # Forward pass
    features = feature_extractor(digit_image)
    activated_features = activation(features)
    feature_vector = flatten(activated_features)
    digit_scores = classifier(feature_vector)
    
    assert features.shape == (6, 6), f"Feature extraction shape wrong: {features.shape}"
    assert feature_vector.shape == (1, 36), f"Feature vector shape wrong: {feature_vector.shape}"
    assert digit_scores.shape == (1, 10), f"Digit scores shape wrong: {digit_scores.shape}"
    
    print("✅ Image classification scenario works correctly")
    
    # Test 4: Feature Extraction and Composition
    print("\n4. Feature Extraction Test:")
    
    # Create modular feature extractor
    feature_conv = Conv2D(kernel_size=(2, 2))
    feature_activation = ReLU()
    
    # Create classifier head
    classifier_head = Dense(input_size=4, output_size=3)
    
    # Test composition
    test_image = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Extract features
    extracted_features = feature_conv(test_image)
    activated_features = feature_activation(extracted_features)
    feature_representation = flatten(activated_features)
    
    # Classify
    predictions = classifier_head(feature_representation)
    
    assert extracted_features.shape == (2, 2), f"Feature extraction wrong: {extracted_features.shape}"
    assert feature_representation.shape == (1, 4), f"Feature representation wrong: {feature_representation.shape}"
    assert predictions.shape == (1, 3), f"Predictions wrong: {predictions.shape}"
    
    print("✅ Feature extraction and composition works correctly")
    
    print("\n🎉 Comprehensive test passed! Your CNN components work correctly for:")
    print("  • Image classification pipelines")
    print("  • Multi-layer feature extraction")
    print("  • Spatial pattern recognition")
    print("  • End-to-end CNN workflows")
    print("📈 Progress: Complete CNN architecture ready for computer vision!")
    
except Exception as e:
    print(f"❌ Comprehensive test failed: {e}")
    raise

print("📈 Final Progress: Complete CNN system ready for computer vision!")

def test_unit_convolution_operation():
    """Unit test for the convolution operation implementation."""
    print("🔬 Unit Test: Convolution Operation...")
    
    # Test basic convolution
    input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 0], [0, 1]])
    result = conv2d_naive(input_data, kernel)
    
    assert result.shape == (2, 2), "Convolution should produce correct output shape"
    expected = np.array([[6, 8], [12, 14]])
    assert np.array_equal(result, expected), "Convolution should produce correct values"
    
    print("✅ Convolution operation works correctly")

def test_unit_conv2d_layer():
    """Unit test for the Conv2D layer implementation."""
    print("🔬 Unit Test: Conv2D Layer...")
    
    # Test Conv2D layer
    conv = Conv2D(kernel_size=(3, 3))
    input_tensor = Tensor(np.random.randn(6, 6))
    output = conv(input_tensor)
    
    assert output.shape == (4, 4), "Conv2D should produce correct output shape"
    assert hasattr(conv, 'kernel'), "Conv2D should have kernel attribute"
    assert conv.kernel.shape == (3, 3), "Kernel should have correct shape"
    
    print("✅ Conv2D layer works correctly")

def test_unit_flatten_function():
    """Unit test for the flatten function implementation."""
    print("🔬 Unit Test: Flatten Function...")
    
    # Test flatten function
    input_2d = Tensor([[1, 2], [3, 4]])
    flattened = flatten(input_2d)
    
    assert flattened.shape == (1, 4), "Flatten should produce output with batch dimension"
    expected = np.array([[1, 2, 3, 4]])
    assert np.array_equal(flattened.data, expected), "Flatten should preserve values"
    
    print("✅ Flatten function works correctly")

# CNN pipeline integration test moved to tests/integration/test_cnn_pipeline.py

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

# %% [markdown]
"""
## 🔬 Integration Test: Conv2D Layer with Tensors
"""

# %%
def test_module_conv2d_tensor_compatibility():
    """
    Integration test for the Conv2D layer and the Tensor class.
    
    Tests that the Conv2D layer correctly processes a batch of image-like Tensors.
    """
    print("🔬 Running Integration Test: Conv2D with Tensors...")

    # 1. Define a Conv2D layer
    # Kernel of size 3x3
    conv_layer = Conv2D((3, 3))

    # 2. Create a batch of 5 grayscale images (10x10)
    # Shape: (batch_size, height, width)
    input_images = np.random.randn(5, 10, 10)
    input_tensor = Tensor(input_images)

    # 3. Perform a forward pass
    output_tensor = conv_layer(input_tensor)

    # 4. Assert the output shape is correct
    # Output height = 10 - 3 + 1 = 8
    # Output width = 10 - 3 + 1 = 8
    expected_shape = (5, 8, 8)
    assert isinstance(output_tensor, Tensor), "Conv2D output must be a Tensor"
    assert output_tensor.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_tensor.shape}"
    print("✅ Integration Test Passed: Conv2D layer correctly transformed image tensor.")

if __name__ == "__main__":
    test_module_conv2d_tensor_compatibility()
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("CNN")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the core components of convolutional neural networks:

### What You've Accomplished
✅ **Convolution Operation**: Implemented the sliding window mechanism from scratch  
✅ **Conv2D Layer**: Built learnable convolutional layers with random initialization  
✅ **Flatten Function**: Created the bridge between convolutional and dense layers  
✅ **CNN Pipelines**: Composed complete systems for image processing  
✅ **Real Applications**: Tested on image classification and feature extraction

### Key Concepts You've Learned
- **Convolution as pattern matching**: Kernels detect specific features
- **Sliding window mechanism**: How convolution processes spatial data
- **Parameter sharing**: Same kernel applied across the entire image
- **Spatial hierarchy**: Multiple layers build complex features
- **CNN architecture**: Conv2D → Activation → Flatten → Dense pattern

### Mathematical Foundations
- **Convolution operation**: dot product of kernel and image patches
- **Output size calculation**: (input_size - kernel_size + 1)
- **Translation invariance**: Same pattern detected anywhere in input
- **Feature maps**: Spatial representations of detected patterns

### Real-World Applications
- **Image classification**: Object recognition, medical imaging
- **Computer vision**: Face detection, autonomous driving
- **Pattern recognition**: Texture analysis, edge detection
- **Feature extraction**: Transfer learning, representation learning

### CNN Architecture Insights
- **Kernel size**: 3×3 most common, balances locality and capacity
- **Stacking layers**: Builds hierarchical feature representations
- **Spatial reduction**: Each layer reduces spatial dimensions
- **Channel progression**: Typically increase channels while reducing spatial size

### Performance Characteristics
- **Parameter efficiency**: Dramatic reduction vs. fully connected
- **Translation invariance**: Robust to object location changes
- **Computational efficiency**: Parallel processing of spatial regions
- **Memory considerations**: Feature maps require storage during forward pass

### Next Steps
1. **Export your code**: Use NBDev to export to the `tinytorch` package
2. **Test your implementation**: Run the complete test suite
3. **Build CNN architectures**: 
   ```python
   from tinytorch.core.cnn import Conv2D, flatten
   from tinytorch.core.layers import Dense
   from tinytorch.core.activations import ReLU
   
   # Create CNN
   conv = Conv2D(kernel_size=(3, 3))
   relu = ReLU()
   dense = Dense(input_size=36, output_size=10)
   
   # Process image
   features = relu(conv(image))
   predictions = dense(flatten(features))
   ```
4. **Explore advanced CNNs**: Pooling, multiple channels, modern architectures!

**Ready for the next challenge?** Let's build data loaders to handle real datasets efficiently!
""" 
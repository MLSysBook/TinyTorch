# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Spatial - Convolutional Neural Networks

Welcome to Spatial! You'll implement the fundamental spatial operations that make CNNs work for image processing and pattern recognition.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 03 (Layers): Neural network building blocks
- Module 04 (Networks): Multi-layer architectures

**What's Working**: You can build fully connected networks that process flattened data.

**The Gap**: Your networks can't recognize spatial patterns in images - they lose all spatial structure when flattening.

**This Module's Solution**: Implement convolution and pooling operations that preserve and process spatial relationships.

**Connection Map**:
```
Networks → Spatial → Autograd
(1D data)  (2D images) (gradient computation)
```

## Learning Objectives
1. **Core Implementation**: Build Conv2D and MaxPool2D layers for spatial pattern recognition
2. **Systems Understanding**: Analyze memory usage and computational complexity of spatial operations
3. **Integration Knowledge**: Connect convolutional layers with existing neural network components
4. **Testing Skills**: Validate spatial operations with immediate unit testing

## Build → Test → Use
1. **Build**: Implement convolution and pooling from scratch
2. **Test**: Validate each operation immediately after implementation
3. **Use**: Combine operations into CNN architectures for image processing
"""

# %% nbgrader={"grade": false, "grade_id": "spatial-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp spatial

# Core imports for spatial operations
import numpy as np
from typing import Tuple, Union, Optional

# Import previous modules
import sys
sys.path.append('../../')
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Module, Linear
except ImportError:
    # Fallback for development
    sys.path.extend([
        '../01_tensor',
        '../03_layers'
    ])
    from tensor_dev import Tensor
    from layers_dev import Module, Linear

print("✅ Spatial module imports successful!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/08_spatial/spatial_dev.py
**Building Side:** Code exports to tinytorch.core.spatial

```python
# Final package structure:
from tinytorch.core.spatial import Conv2D, MaxPool2D, flatten  # This module
from tinytorch.core.tensor import Tensor  # Foundation (always needed)
from tinytorch.core.layers import Module  # Base class for layers
```

**Why this matters:**
- **Learning:** Complete spatial processing system in one focused module
- **Production:** Organized like PyTorch's torch.nn with spatial operations
- **Consistency:** All spatial operations and utilities in core.spatial
- **Integration:** Works seamlessly with layers for complete CNN architectures
"""

# %% [markdown]
"""
## 🏗️ Understanding Spatial Operations

### What is Convolution?

Convolution is a mathematical operation that slides a small filter (kernel) across an image to detect patterns:

```
Input Image (5×5)      Filter (3×3)       Output (3×3)
┌─────────────────┐    ┌───────┐         ┌─────────┐
│ 1 2 3 4 5      │    │ 1 0-1 │         │ ? ? ? │
│ 6 7 8 9 0      │  × │ 2 1 0 │    =    │ ? ? ? │
│ 1 2 3 4 5      │    │-1 0 1 │         │ ? ? ? │
│ 6 7 8 9 0      │    └───────┘         └─────────┘
│ 1 2 3 4 5      │
└─────────────────┘
```

**Why Spatial Operations Matter:**
- **Pattern Recognition**: Detect edges, textures, and complex features
- **Translation Invariance**: Same pattern detected regardless of position
- **Parameter Sharing**: One filter detects patterns across entire image
- **Spatial Hierarchy**: Simple patterns → complex patterns → objects

### Memory Efficiency vs Fully Connected

**Fully Connected Approach** (wasteful):
- 28×28 image = 784 inputs
- Hidden layer: 784 × 128 = 100,352 parameters per neuron!
- No spatial understanding

**Convolutional Approach** (efficient):
- 3×3 filter = 9 parameters
- Applied everywhere via sliding
- Preserves spatial relationships
"""
# %% [markdown]
"""
## Implementation: Core Spatial Operations

Let's build the essential spatial operations: convolution, pooling, and flattening.
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-naive", "locked": false, "schema_version": 3, "solution": true, "task": false}
def conv2d_naive(input_array, kernel, bias=None):
    """
    Naive 2D convolution implementation for educational understanding.

    Args:
        input_array: np.ndarray of shape (height, width) or (channels, height, width)
        kernel: np.ndarray of shape (kernel_height, kernel_width)
        bias: Optional bias value to add to each output

    Returns:
        np.ndarray: Convolved output

    TODO: Implement 2D convolution by sliding kernel across input

    APPROACH:
    1. Handle input dimensions (add channel dimension if needed)
    2. Calculate output dimensions based on input and kernel sizes
    3. Slide kernel across input and compute dot products
    4. Add bias if provided

    EXAMPLE:
    >>> input_img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> edge_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    >>> result = conv2d_naive(input_img, edge_kernel)
    >>> print(result.shape)
    (1, 1)

    HINTS:
    - Use nested loops to slide kernel across input
    - Multiply overlapping regions element-wise and sum
    - Handle single-channel inputs by adding channel dimension
    """
    ### BEGIN SOLUTION
    # Ensure input has channel dimension
    if input_array.ndim == 2:
        input_array = input_array[np.newaxis, :, :]  # Add channel dimension

    channels, height, width = input_array.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions (no padding, stride=1)
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1

    # Initialize output
    output = np.zeros((channels, out_height, out_width))

    # Slide kernel across input
    for c in range(channels):
        for i in range(out_height):
            for j in range(out_width):
                # Extract region and compute convolution
                region = input_array[c, i:i+kernel_height, j:j+kernel_width]
                output[c, i, j] = np.sum(region * kernel)

                # Add bias if provided
                if bias is not None:
                    output[c, i, j] += bias

    return output
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Convolution Operation

This test validates our basic convolution implementation works correctly.
"""

# %%
def test_unit_conv2d_naive():
    """Test convolution operation with educational feedback"""
    print("🔬 Unit Test: Convolution Operation...")

    # Test 1: Simple edge detection
    input_img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    edge_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Vertical edge detector

    result = conv2d_naive(input_img, edge_kernel)

    # Verify output shape (3x3 input, 3x3 kernel -> 1x1 output)
    assert result.shape == (1, 1, 1), f"Expected shape (1, 1, 1), got {result.shape}"

    # Test 2: Multi-channel input
    multi_channel = np.random.randn(3, 5, 5)  # 3 channels, 5x5 each
    kernel = np.array([[1, 0], [0, 1]])  # 2x2 kernel

    result = conv2d_naive(multi_channel, kernel)
    assert result.shape == (3, 4, 4), f"Expected shape (3, 4, 4), got {result.shape}"

    # Test 3: Bias addition
    simple_input = np.array([[1, 1], [1, 1]])
    simple_kernel = np.array([[1]])
    bias_value = 5

    result_with_bias = conv2d_naive(simple_input, simple_kernel, bias=bias_value)
    result_without_bias = conv2d_naive(simple_input, simple_kernel)

    bias_diff = result_with_bias - result_without_bias
    assert np.allclose(bias_diff, bias_value), "Bias not added correctly"

    print("✅ Convolution operation works correctly!")

test_unit_conv2d_naive()

# %% [markdown]
"""
## Implementation: Conv2D Layer

Now let's build a proper convolutional layer class that can be used in neural networks.
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
class Conv2D(Module):
    """
    2D Convolutional Layer for spatial pattern recognition.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kernel_size: Size of convolution kernel (int or tuple)
        bias: Whether to use bias term

    TODO: Implement a convolutional layer that can process multi-channel inputs

    APPROACH:
    1. Initialize weights and bias with proper shapes
    2. Handle kernel_size as int or tuple
    3. Implement forward pass with multi-channel convolution
    4. Use conv2d_naive for each input-output channel combination

    EXAMPLE:
    >>> conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    >>> x = Tensor(np.random.randn(3, 28, 28))  # RGB image
    >>> output = conv(x)
    >>> print(output.shape)
    (16, 26, 26)

    HINTS:
    - Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
    - For each output channel, convolve with all input channels and sum
    - Use He initialization for weights: scale by sqrt(2 / fan_in)
    """
    ### BEGIN SOLUTION
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        # Initialize weights with He initialization
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        weight_scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, *self.kernel_size) * weight_scale
        )

        # Initialize bias
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass of 2D convolution.

        Args:
            x: Input tensor of shape (in_channels, height, width)

        Returns:
            Output tensor of shape (out_channels, out_height, out_width)
        """
        if x.data.ndim != 3:
            raise ValueError(f"Expected 3D input (channels, height, width), got {x.data.ndim}D")

        in_channels, height, width = x.data.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}")

        # Calculate output dimensions
        out_height = height - self.kernel_size[0] + 1
        out_width = width - self.kernel_size[1] + 1

        # Initialize output
        output = np.zeros((self.out_channels, out_height, out_width))

        # Convolve each output channel
        for out_ch in range(self.out_channels):
            channel_sum = np.zeros((out_height, out_width))

            # Sum convolutions across all input channels
            for in_ch in range(self.in_channels):
                kernel = self.weight.data[out_ch, in_ch]
                conv_result = conv2d_naive(x.data[in_ch], kernel)
                channel_sum += conv_result.squeeze()  # Remove extra dimensions

            output[out_ch] = channel_sum

            # Add bias if enabled
            if self.use_bias:
                output[out_ch] += self.bias.data[out_ch]

        return Tensor(output)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Conv2D Layer

This test validates our Conv2D layer implementation.
"""

# %%
def test_unit_conv2d():
    """Test Conv2D layer with educational feedback"""
    print("🔬 Unit Test: Conv2D Layer...")

    # Test 1: Single channel to multiple channels
    conv = Conv2D(in_channels=1, out_channels=3, kernel_size=3)
    x = Tensor(np.random.randn(1, 5, 5))

    output = conv(x)
    expected_shape = (3, 3, 3)  # 3 output channels, 3x3 spatial
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # Test 2: RGB to feature maps (realistic scenario)
    rgb_conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    rgb_input = Tensor(np.random.randn(3, 28, 28))  # RGB image

    features = rgb_conv(rgb_input)
    expected_shape = (16, 26, 26)  # 16 feature maps, 26x26 spatial
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"

    # Test 3: Different kernel sizes
    large_kernel_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=5)
    test_input = Tensor(np.random.randn(1, 10, 10))

    large_output = large_kernel_conv(test_input)
    expected_shape = (1, 6, 6)  # 10-5+1 = 6
    assert large_output.shape == expected_shape, f"Expected {expected_shape}, got {large_output.shape}"

    # Test 4: Parameter counting
    conv_params = Conv2D(in_channels=3, out_channels=64, kernel_size=3)
    # Weights: 64 * 3 * 3 * 3 = 1728, Bias: 64, Total: 1792
    weight_params = 64 * 3 * 3 * 3
    bias_params = 64
    total_expected = weight_params + bias_params

    weight_actual = conv_params.weight.data.size
    bias_actual = conv_params.bias.data.size if conv_params.bias else 0
    total_actual = weight_actual + bias_actual

    assert total_actual == total_expected, f"Expected {total_expected} parameters, got {total_actual}"

    print("✅ Conv2D layer works correctly!")

test_unit_conv2d()

# %% [markdown]
"""
## Implementation: MaxPool2D Layer

Pooling layers reduce spatial dimensions while preserving important features.
"""

# %% nbgrader={"grade": false, "grade_id": "maxpool2d-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
class MaxPool2D(Module):
    """
    2D Max Pooling Layer for spatial downsampling.

    Args:
        pool_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (defaults to pool_size)

    TODO: Implement max pooling that reduces spatial dimensions

    APPROACH:
    1. Handle pool_size and stride as int or tuple
    2. Calculate output dimensions based on input size and pooling parameters
    3. Slide pooling window and take maximum in each region
    4. Handle multi-channel inputs by pooling each channel independently

    EXAMPLE:
    >>> pool = MaxPool2D(pool_size=2)
    >>> x = Tensor(np.random.randn(16, 26, 26))  # Feature maps from Conv2D
    >>> output = pool(x)
    >>> print(output.shape)
    (16, 13, 13)

    HINTS:
    - Default stride equals pool_size for non-overlapping pooling
    - Output size = (input_size - pool_size) // stride + 1
    - Use np.max on each pooling region
    """
    ### BEGIN SOLUTION
    def __init__(self, pool_size, stride=None):
        super().__init__()

        # Handle pool_size as int or tuple
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        # Default stride equals pool_size (non-overlapping)
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def forward(self, x):
        """
        Forward pass of 2D max pooling.

        Args:
            x: Input tensor of shape (channels, height, width)

        Returns:
            Output tensor with reduced spatial dimensions
        """
        if x.data.ndim != 3:
            raise ValueError(f"Expected 3D input (channels, height, width), got {x.data.ndim}D")

        channels, height, width = x.data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride

        # Calculate output dimensions
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1

        # Initialize output
        output = np.zeros((channels, out_height, out_width))

        # Apply max pooling to each channel
        for c in range(channels):
            for i in range(out_height):
                for j in range(out_width):
                    # Calculate pooling region bounds
                    h_start = i * stride_h
                    h_end = h_start + pool_h
                    w_start = j * stride_w
                    w_end = w_start + pool_w

                    # Extract region and take maximum
                    region = x.data[c, h_start:h_end, w_start:w_end]
                    output[c, i, j] = np.max(region)

        return Tensor(output)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: MaxPool2D Layer

This test validates our MaxPool2D layer implementation.
"""

# %%
def test_unit_maxpool2d():
    """Test MaxPool2D layer with educational feedback"""
    print("🔬 Unit Test: MaxPool2D Layer...")

    # Test 1: Basic 2x2 pooling
    pool = MaxPool2D(pool_size=2)
    x = Tensor(np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]]]))  # 1x4x4 input

    output = pool(x)
    expected_shape = (1, 2, 2)  # 4x4 -> 2x2 with pool_size=2
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # Verify max values are correct
    expected_values = np.array([[[6, 8], [14, 16]]])  # Max in each 2x2 region
    assert np.allclose(output.data, expected_values), "MaxPool values incorrect"

    # Test 2: Multi-channel pooling
    multi_input = Tensor(np.random.randn(3, 8, 8))
    multi_output = pool(multi_input)

    expected_shape = (3, 4, 4)  # Each channel pooled independently
    assert multi_output.shape == expected_shape, f"Expected {expected_shape}, got {multi_output.shape}"

    # Test 3: Different pool sizes
    pool_3x3 = MaxPool2D(pool_size=3)
    large_input = Tensor(np.random.randn(1, 9, 9))

    pool_output = pool_3x3(large_input)
    expected_shape = (1, 3, 3)  # 9x9 with 3x3 pooling and stride=3
    assert pool_output.shape == expected_shape, f"Expected {expected_shape}, got {pool_output.shape}"

    # Test 4: Integration with Conv2D
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3)
    pooling = MaxPool2D(pool_size=2)

    test_image = Tensor(np.random.randn(1, 10, 10))
    conv_features = conv(test_image)  # Should be (4, 8, 8)
    pooled_features = pooling(conv_features)  # Should be (4, 4, 4)

    expected_shape = (4, 4, 4)
    assert pooled_features.shape == expected_shape, f"Expected {expected_shape}, got {pooled_features.shape}"

    print("✅ MaxPool2D layer works correctly!")

test_unit_maxpool2d()

# %% [markdown]
"""
## Implementation: Flatten Function

Convert spatial feature maps to 1D for fully connected layers.
"""

# %% nbgrader={"grade": false, "grade_id": "flatten-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
def flatten(x):
    """
    Flatten multi-dimensional tensor to 1D for fully connected layers.

    Args:
        x: Input tensor of any shape

    Returns:
        Tensor: Flattened tensor with shape (total_elements,)

    TODO: Flatten tensor while preserving all data

    APPROACH:
    1. Calculate total number of elements
    2. Reshape to 1D preserving data order
    3. Return as new Tensor

    EXAMPLE:
    >>> x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))  # (2, 2, 2)
    >>> flat = flatten(x)
    >>> print(flat.shape)
    (8,)

    HINTS:
    - Use numpy.reshape with -1 to flatten
    - Ensure data order is preserved (row-major/C-style)
    """
    ### BEGIN SOLUTION
    # Calculate total elements and reshape to 1D
    flattened_data = x.data.reshape(-1)
    return Tensor(flattened_data)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Flatten Function

This test validates our flatten function implementation.
"""

# %%
def test_unit_flatten():
    """Test flatten function with educational feedback"""
    print("🔬 Unit Test: Flatten Function...")

    # Test 1: 2D tensor
    x_2d = Tensor(np.array([[1, 2], [3, 4]]))
    flat_2d = flatten(x_2d)

    expected_shape = (4,)
    assert flat_2d.shape == expected_shape, f"Expected {expected_shape}, got {flat_2d.shape}"
    assert np.array_equal(flat_2d.data, [1, 2, 3, 4]), "Flatten values incorrect"

    # Test 2: 3D tensor (typical CNN output)
    x_3d = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))  # (2, 2, 2)
    flat_3d = flatten(x_3d)

    expected_shape = (8,)
    assert flat_3d.shape == expected_shape, f"Expected {expected_shape}, got {flat_3d.shape}"
    assert np.array_equal(flat_3d.data, [1, 2, 3, 4, 5, 6, 7, 8]), "3D flatten values incorrect"

    # Test 3: Real CNN scenario - feature maps to classifier
    # Simulate: Conv2D(64 filters, 5x5 output) -> Flatten -> Linear
    feature_maps = Tensor(np.random.randn(64, 5, 5))  # 64 feature maps of 5x5
    flattened_features = flatten(feature_maps)

    expected_shape = (64 * 5 * 5,)  # 1600 features
    assert flattened_features.shape == expected_shape, f"Expected {expected_shape}, got {flattened_features.shape}"

    # Test 4: Preserve data integrity
    original = Tensor(np.arange(24).reshape(2, 3, 4))
    flattened = flatten(original)

    # Check that all values are preserved
    assert np.array_equal(flattened.data, np.arange(24)), "Data not preserved during flattening"

    print("✅ Flatten function works correctly!")

test_unit_flatten()

# %% [markdown]
"""
## 🔍 Systems Analysis

Now that your implementation is complete and tested, let's analyze its behavior:
"""

# %%
def analyze_spatial_complexity():
    """
    📊 SYSTEMS MEASUREMENT: Spatial Operations Complexity

    Measure how spatial operations scale with input size and parameters.
    """
    print("📊 SPATIAL COMPLEXITY ANALYSIS")
    print("Testing how spatial operations scale with different inputs...")

    import time

    # Test convolution scaling
    input_sizes = [16, 32, 64, 128]
    conv_times = []

    print("\n🔍 Convolution Scaling Analysis:")
    for size in input_sizes:
        # Create test input and kernel
        test_input = np.random.randn(3, size, size)  # 3-channel image
        test_kernel = np.random.randn(3, 3)  # 3x3 kernel

        # Time the convolution
        start = time.perf_counter()
        result = conv2d_naive(test_input, test_kernel)
        elapsed = time.perf_counter() - start

        conv_times.append(elapsed)
        flops = 3 * (size-2) * (size-2) * 9  # channels * output_pixels * kernel_size

        print(f"  Size {size}×{size}: {elapsed*1000:.2f}ms, {flops:,} FLOPs")

        if elapsed > 1.0:  # Stop if too slow
            break

    # Analyze scaling pattern
    if len(conv_times) >= 3:
        size_ratio = input_sizes[2] / input_sizes[0]  # 4x increase
        time_ratio = conv_times[2] / conv_times[0]
        print(f"💡 COMPLEXITY INSIGHT: {size_ratio:.0f}x size increase → {time_ratio:.1f}x time increase")
        print(f"   This suggests ~O(N²) scaling as expected for spatial convolution")

    # Test memory usage
    print("\n💾 Memory Usage Analysis:")
    channel_configs = [(1, 16), (3, 32), (16, 64), (32, 128)]

    for in_ch, out_ch in channel_configs:
        conv = Conv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=3)

        # Calculate parameter memory
        weight_params = out_ch * in_ch * 3 * 3
        bias_params = out_ch
        total_params = weight_params + bias_params
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32

        print(f"  Conv2D({in_ch}→{out_ch}): {total_params:,} params, {memory_mb:.2f}MB")

        if total_params > 1_000_000:
            print(f"    💥 Parameter explosion! {total_params/1e6:.1f}M parameters")
            print(f"    This shows why depthwise separable convolutions were invented")
            break

    print(f"\n💡 SYSTEMS INSIGHT: Spatial operations have quadratic scaling")
    print(f"   Input size matters more than you might expect!")
    print(f"   Modern optimizations: im2col, FFT convolution, optimized BLAS")

# Run the analysis
analyze_spatial_complexity()

# %% [markdown]
"""
## 🧪 Complete Module Testing

Test all spatial components together.
"""

# %%
def test_module():
    """Run comprehensive test of spatial module"""
    print("🧪 Testing Complete Spatial Module...")

    print("\n1. Testing individual components...")
    test_unit_conv2d_naive()
    test_unit_conv2d()
    test_unit_maxpool2d()
    test_unit_flatten()

    print("\n2. Testing CNN pipeline integration...")

    # Build a simple CNN pipeline
    print("   Building CNN: Conv2D → MaxPool2D → Flatten → Linear")

    # Create layers
    conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3)  # RGB → 16 features
    pool1 = MaxPool2D(pool_size=2)  # Spatial downsampling
    conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3)  # 16 → 32 features
    pool2 = MaxPool2D(pool_size=2)  # More downsampling
    classifier = Linear(input_size=32*5*5, output_size=10)  # To 10 classes

    # Test forward pass with realistic input
    test_image = Tensor(np.random.randn(3, 28, 28))  # RGB image like CIFAR-10
    print(f"   Input shape: {test_image.shape}")

    # Forward pass through CNN
    x = conv1(test_image)
    print(f"   After Conv1: {x.shape}")

    x = pool1(x)
    print(f"   After Pool1: {x.shape}")

    x = conv2(x)
    print(f"   After Conv2: {x.shape}")

    x = pool2(x)
    print(f"   After Pool2: {x.shape}")

    x = flatten(x)
    print(f"   After Flatten: {x.shape}")

    x = classifier(x)
    print(f"   Final output: {x.shape}")

    # Verify final shape
    assert x.shape == (10,), f"Expected (10,) output for classification, got {x.shape}"

    print("\n✅ All spatial module tests passed!")
    print("🎯 CNN pipeline working correctly - ready for image classification!")

# %% [markdown]
"""
## Main Execution Block

All tests run when module is executed directly.
"""

# %%
if __name__ == "__main__":
    print("🚀 SPATIAL MODULE - CONVOLUTIONAL NEURAL NETWORKS")
    print("=" * 60)

    # Run complete module test
    test_module()

    # Run systems analysis
    print("\n" + "=" * 60)
    analyze_spatial_complexity()

    print("\n" + "=" * 60)
    print("🎯 SPATIAL MODULE COMPLETE!")
    print("📈 Progress: Spatial Operations ✓")
    print("🔥 Next: Autograd - Automatic Differentiation!")
    print("💪 You can now build CNNs for image recognition!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Analyze your spatial implementations and their systems implications:

### Question 1: Convolution Memory Access Patterns

In your `conv2d_naive` implementation, you used nested loops to slide the kernel across the input. Analyze the memory access patterns in your nested loop structure:

```python
for c in range(channels):
    for i in range(out_height):
        for j in range(out_width):
            region = input_array[c, i:i+kernel_height, j:j+kernel_width]
```

**Analysis Question**: How could you reorder these loops or modify the memory access pattern to improve cache locality? Consider that modern CPUs have L1 cache sizes of ~32KB and cache lines of 64 bytes. Design specific modifications to your current implementation that would minimize cache misses.

Think about:
- Which loop order accesses memory most sequentially?
- How does kernel size affect cache efficiency?
- What happens with large input images that don't fit in cache?
- How would you implement cache-blocking for very large convolutions?

### Question 2: Multi-Channel Convolution Scaling

Your `Conv2D` class processes multiple input and output channels. Looking at your implementation:

```python
for out_ch in range(self.out_channels):
    for in_ch in range(self.in_channels):
        # Convolution operation
```

**Analysis Question**: Design a parallelization strategy for your multi-channel convolution that could efficiently utilize 8 GPU cores. How would you distribute the work across channels and spatial dimensions? What are the memory bandwidth requirements, and how would you handle synchronization?

Think about:
- Which loops can be parallelized independently?
- How do you minimize memory transfers between GPU cores?
- What's the optimal work distribution for different input sizes?
- How does memory coalescing affect your parallel algorithm?

### Question 3: CNN Architecture Memory Management

You built a complete CNN pipeline: Conv2D → MaxPool2D → Conv2D → MaxPool2D → Flatten → Linear. Analyze the memory footprint of your pipeline:

**Analysis Question**: For a batch of 32 CIFAR-10 images (32×32×3), calculate the peak memory usage during forward pass through your CNN architecture. Include intermediate activations, parameters, and gradients. At what point does memory become the limiting factor for larger models?

Think about:
- Memory usage of each intermediate activation
- Parameter storage for each layer
- Gradient storage during backpropagation
- When would you need gradient checkpointing?
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Spatial Operations Complete!

Congratulations! You've successfully implemented the core spatial operations that make CNNs work:

### What You've Accomplished
✅ **Convolution Implementation**: Built conv2d_naive() and Conv2D class with multi-channel support
✅ **Pooling Operations**: Implemented MaxPool2D for spatial downsampling and translation invariance
✅ **Pipeline Integration**: Created complete CNN pipeline from images to classification
✅ **Systems Analysis**: Analyzed computational complexity and memory scaling of spatial operations
✅ **Testing Framework**: Validated each component with immediate unit testing

### Key Learning Outcomes
- **Spatial Pattern Recognition**: Understanding how convolution detects local patterns
- **Parameter Efficiency**: How weight sharing makes CNNs practical for image processing
- **Computational Complexity**: Why spatial operations scale as O(N²) with input size
- **Memory Management**: How multi-channel operations affect parameter and activation memory

### Mathematical Foundations Mastered
- **Convolution Operation**: Discrete convolution as correlation with flipped kernels
- **Spatial Dimensions**: How kernel size, stride, and padding affect output dimensions
- **Multi-Channel Processing**: Combining features across input channels to create output channels

### Professional Skills Developed
- **CNN Architecture Design**: Building complete pipelines for image classification
- **Performance Analysis**: Understanding scaling bottlenecks in spatial operations
- **Memory Optimization**: Recognizing when spatial operations become memory-bound

### Ready for Advanced Applications
Your spatial implementation now enables:
- **Image Classification**: CNNs for CIFAR-10, ImageNet-style datasets
- **Feature Extraction**: Hierarchical feature learning in deep networks
- **Computer Vision**: Foundation for object detection, segmentation, and more

### Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.nn.Conv2d` and `torch.nn.MaxPool2d` with similar APIs
- **TensorFlow**: `tf.keras.layers.Conv2D` for production computer vision
- **Industry Standard**: Weight sharing and spatial convolution are universal in CV

### Next Steps
1. **Export your module**: `tito module complete 08_spatial`
2. **Validate integration**: `tito test --module spatial`
3. **Explore optimizations**: Consider im2col convolution algorithms
4. **Ready for Module 09**: Autograd will add automatic differentiation to your spatial operations

**🚀 Achievement Unlocked**: Your spatial operations form the foundation for any computer vision application! CNNs + backpropagation = modern AI vision systems.
"""
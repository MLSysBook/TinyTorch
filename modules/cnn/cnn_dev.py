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
# Module X: CNN - Convolutional Neural Networks

Welcome to the CNN module! Here you'll implement the core building block of modern computer vision: the convolutional layer.

## Learning Goals
- Understand the convolution operation (sliding window, local connectivity, weight sharing)
- Implement Conv2D with explicit for-loops
- Visualize how convolution builds feature maps
- Compose Conv2D with other layers to build a simple ConvNet
- (Stretch) Explore stride, padding, pooling, and multi-channel input

## Build → Use → Understand
1. **Build**: Conv2D layer using sliding window convolution
2. **Use**: Transform images and see feature maps
3. **Understand**: How CNNs learn spatial patterns
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/cnn/cnn_dev.py`  
**Building Side:** Code exports to `tinytorch.core.layers`

```python
# Final package structure:
from tinytorch.core.layers import Dense, Conv2D  # Both layers together!
from tinytorch.core.activations import ReLU
from tinytorch.core.tensor import Tensor
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn`
- **Consistency:** All layers (Dense, Conv2D) live together in `core.layers`
"""

# %%
#| default_exp core.cnn

# Setup and imports
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU

# %% [markdown]
"""
## Step 1: What is Convolution?

### Definition
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

### The Math Behind It
For input I (H×W) and kernel K (kH×kW), the output O (out_H×out_W) is:
```
O[i,j] = sum(I[i+di, j+dj] * K[di, dj] for di in range(kH), dj in range(kW))
```

Let's implement this step by step!
"""

# %%
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
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def conv2d_naive(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = input.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    output = np.zeros((out_H, out_W), dtype=input.dtype)
    for i in range(out_H):
        for j in range(out_W):
            for di in range(kH):
                for dj in range(kW):
                    output[i, j] += input[i + di, j + dj] * kernel[di, dj]
    return output

# %% [markdown]
"""
### 🧪 Test Your Conv2D Implementation

Try your function on this simple example:
"""

# %%
# Test case for conv2d_naive
input = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32)
kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

expected = np.array([
    [1*1+2*0+4*0+5*(-1), 2*1+3*0+5*0+6*(-1)],
    [4*1+5*0+7*0+8*(-1), 5*1+6*0+8*0+9*(-1)]
], dtype=np.float32)

try:
    output = conv2d_naive(input, kernel)
    print("✅ Input:\n", input)
    print("✅ Kernel:\n", kernel)
    print("✅ Your output:\n", output)
    print("✅ Expected:\n", expected)
    assert np.allclose(output, expected), "❌ Output does not match expected!"
    print("🎉 conv2d_naive works!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement conv2d_naive above!")

# %% [markdown]
"""
## Step 2: Understanding What Convolution Does

Let's visualize how different kernels detect different patterns:
"""

# %%
# Visualize different convolution kernels
print("Visualizing different convolution kernels...")

try:
    # Test different kernels
    test_input = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    # Edge detection kernel (horizontal)
    edge_kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Sharpening kernel
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Test edge detection
    edge_output = conv2d_naive(test_input, edge_kernel)
    print("✅ Edge detection kernel:")
    print("   Detects horizontal edges (boundaries between light and dark)")
    print("   Output:\n", edge_output)
    
    # Test sharpening
    sharpen_output = conv2d_naive(test_input, sharpen_kernel)
    print("✅ Sharpening kernel:")
    print("   Enhances edges and details")
    print("   Output:\n", sharpen_output)
    
    print("\n💡 Different kernels detect different patterns!")
    print("   Neural networks learn these kernels automatically!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
"""
## Step 3: Conv2D Layer Class

Now let's wrap your convolution function in a layer class for use in networks. This makes it consistent with other layers like Dense.

### Why Layer Classes Matter
- **Consistent API**: Same interface as Dense layers
- **Learnable parameters**: Kernels can be learned from data
- **Composability**: Can be combined with other layers
- **Integration**: Works seamlessly with the rest of TinyTorch

### The Pattern
```
Input Tensor → Conv2D → Output Tensor
```

Just like Dense layers, but with spatial operations instead of linear transformations.
"""

# %%
#| export
class Conv2D:
    """
    2D Convolutional Layer (single channel, single filter, no stride/pad).
    
    Args:
        kernel_size: (kH, kW) - size of the convolution kernel
        
    TODO: Initialize a random kernel and implement the forward pass using conv2d_naive.
    
    APPROACH:
    1. Store kernel_size as instance variable
    2. Initialize random kernel with small values
    3. Implement forward pass using conv2d_naive function
    4. Return Tensor wrapped around the result
    
    EXAMPLE:
    layer = Conv2D(kernel_size=(2, 2))
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3)
    y = layer(x)  # shape (2, 2)
    
    HINTS:
    - Store kernel_size as (kH, kW)
    - Initialize kernel with np.random.randn(kH, kW) * 0.1 (small values)
    - Use conv2d_naive(x.data, self.kernel) in forward pass
    - Return Tensor(result) to wrap the result
    """
    def __init__(self, kernel_size: Tuple[int, int]):
        """
        Initialize Conv2D layer with random kernel.
        
        Args:
            kernel_size: (kH, kW) - size of the convolution kernel
            
        TODO: 
        1. Store kernel_size as instance variable
        2. Initialize random kernel with small values
        3. Scale kernel values to prevent large outputs
        
        STEP-BY-STEP:
        1. Store kernel_size as self.kernel_size
        2. Unpack kernel_size into kH, kW
        3. Initialize kernel: np.random.randn(kH, kW) * 0.1
        4. Convert to float32 for consistency
        
        EXAMPLE:
        Conv2D((2, 2)) creates:
        - kernel: shape (2, 2) with small random values
        """
        raise NotImplementedError("Student implementation required")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: apply convolution to input.
        
        Args:
            x: Input tensor of shape (H, W)
            
        Returns:
            Output tensor of shape (H-kH+1, W-kW+1)
            
        TODO: Implement convolution using conv2d_naive function.
        
        STEP-BY-STEP:
        1. Use conv2d_naive(x.data, self.kernel)
        2. Return Tensor(result)
        
        EXAMPLE:
        Input x: Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3)
        Kernel: shape (2, 2)
        Output: Tensor([[val1, val2], [val3, val4]])  # shape (2, 2)
        
        HINTS:
        - x.data gives you the numpy array
        - self.kernel is your learned kernel
        - Use conv2d_naive(x.data, self.kernel)
        - Return Tensor(result) to wrap the result
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Conv2D:
    def __init__(self, kernel_size: Tuple[int, int]):
        self.kernel_size = kernel_size
        kH, kW = kernel_size
        # Initialize with small random values
        self.kernel = np.random.randn(kH, kW).astype(np.float32) * 0.1
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(conv2d_naive(x.data, self.kernel))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Conv2D Layer
"""

# %%
# Test Conv2D layer
print("Testing Conv2D layer...")

try:
    # Test basic Conv2D layer
    conv = Conv2D(kernel_size=(2, 2))
    x = Tensor(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32))
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Kernel shape: {conv.kernel.shape}")
    print(f"✅ Kernel values:\n{conv.kernel}")
    
    y = conv(x)
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output: {y}")
    
    # Test with different kernel size
    conv2 = Conv2D(kernel_size=(3, 3))
    y2 = conv2(x)
    print(f"✅ 3x3 kernel output shape: {y2.shape}")
    
    print("\n🎉 Conv2D layer works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Conv2D layer above!")

# %% [markdown]
"""
## Step 4: Building a Simple ConvNet

Now let's compose Conv2D layers with other layers to build a complete convolutional neural network!

### Why ConvNets Matter
- **Spatial hierarchy**: Each layer learns increasingly complex features
- **Parameter sharing**: Same kernel applied everywhere (efficiency)
- **Translation invariance**: Can recognize objects regardless of position
- **Real-world success**: Power most modern computer vision systems

### The Architecture
```
Input Image → Conv2D → ReLU → Flatten → Dense → Output
```

This simple architecture can learn to recognize patterns in images!
"""

# %%
#| export
def flatten(x: Tensor) -> Tensor:
    """
    Flatten a 2D tensor to 1D (for connecting to Dense).
    
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
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def flatten(x: Tensor) -> Tensor:
    """Flatten a 2D tensor to 1D (for connecting to Dense)."""
    return Tensor(x.data.flatten()[None, :])

# %% [markdown]
"""
### 🧪 Test Your Flatten Function
"""

# %%
# Test flatten function
print("Testing flatten function...")

try:
    # Test flattening
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
    flattened = flatten(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Flattened shape: {flattened.shape}")
    print(f"✅ Flattened values: {flattened}")
    
    # Verify the flattening worked correctly
    expected = np.array([[1, 2, 3, 4, 5, 6]])
    assert np.allclose(flattened.data, expected), "❌ Flattening incorrect!"
    print("✅ Flattening works correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the flatten function above!")

# %% [markdown]
"""
## Step 5: Composing a Complete ConvNet

Now let's build a simple convolutional neural network that can process images!
"""

# %%
# Compose a simple ConvNet
print("Building a simple ConvNet...")

try:
    # Create network components
    conv = Conv2D((2, 2))
    relu = ReLU()
    dense = Dense(input_size=4, output_size=1)  # 4 features from 2x2 output
    
    # Test input (small 3x3 "image")
    x = Tensor(np.random.randn(3, 3).astype(np.float32))
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Input: {x}")
    
    # Forward pass through the network
    conv_out = conv(x)
    print(f"✅ After Conv2D: {conv_out}")
    
    relu_out = relu(conv_out)
    print(f"✅ After ReLU: {relu_out}")
    
    flattened = flatten(relu_out)
    print(f"✅ After flatten: {flattened}")
    
    final_out = dense(flattened)
    print(f"✅ Final output: {final_out}")
    
    print("\n🎉 Simple ConvNet works!")
    print("This network can learn to recognize patterns in images!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your Conv2D, flatten, and Dense implementations!")

# %% [markdown]
"""
## Step 6: Understanding the Power of Convolution

Let's see how convolution captures different types of patterns:
"""

# %%
# Demonstrate pattern detection
print("Demonstrating pattern detection...")

try:
    # Create a simple "image" with a pattern
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    # Different kernels detect different patterns
    edge_kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    
    blur_kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32)
    
    # Test edge detection
    edge_result = conv2d_naive(image, edge_kernel)
    print("✅ Edge detection:")
    print("   Detects boundaries around the white square")
    print("   Result:\n", edge_result)
    
    # Test blurring
    blur_result = conv2d_naive(image, blur_kernel)
    print("✅ Blurring:")
    print("   Smooths the image")
    print("   Result:\n", blur_result)
    
    print("\n💡 Different kernels = different feature detectors!")
    print("   Neural networks learn these automatically from data!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've built the foundation of convolutional neural networks:

### What You've Accomplished
✅ **Convolution Operation**: Understanding the sliding window mechanism  
✅ **Conv2D Layer**: Learnable convolutional layer implementation  
✅ **Pattern Detection**: Visualizing how kernels detect different features  
✅ **ConvNet Architecture**: Composing Conv2D with other layers  
✅ **Real-world Applications**: Understanding computer vision applications  

### Key Concepts You've Learned
- **Convolution** is pattern matching with sliding windows
- **Local connectivity** means each output depends on a small input region
- **Weight sharing** makes CNNs parameter-efficient
- **Spatial hierarchy** builds complex features from simple patterns
- **Translation invariance** allows recognition regardless of position

### What's Next
In the next modules, you'll build on this foundation:
- **Advanced CNN features**: Stride, padding, pooling
- **Multi-channel convolution**: RGB images, multiple filters
- **Training**: Learning kernels from data
- **Real applications**: Image classification, object detection

### Real-World Connection
Your Conv2D layer is now ready to:
- Learn edge detectors, texture recognizers, and shape detectors
- Process real images for computer vision tasks
- Integrate with the rest of the TinyTorch ecosystem
- Scale to complex architectures like ResNet, VGG, etc.

**Ready for the next challenge?** Let's move on to training these networks!
"""

# %%
# Final verification
print("\n" + "="*50)
print("🎉 CNN MODULE COMPLETE!")
print("="*50)
print("✅ Convolution operation understanding")
print("✅ Conv2D layer implementation")
print("✅ Pattern detection visualization")
print("✅ ConvNet architecture composition")
print("✅ Real-world computer vision context")
print("\n🚀 Ready to train networks in the next module!") 
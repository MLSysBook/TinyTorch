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

A convolutional layer applies a small filter (kernel) across the input, producing a feature map. This operation captures local patterns and is the foundation of modern vision models.

- **Local connectivity:** Each output value depends only on a small region of the input.
- **Weight sharing:** The same filter is applied everywhere.
- **Sliding window:** The filter moves across the input spatially.

We'll start with a single-channel (grayscale) 2D convolution, no stride or padding.
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
            output[i, j] = np.sum(input[i:i+kH, j:j+kW] * kernel)
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
    print("Output:\n", output)
    print("Expected:\n", expected)
    assert np.allclose(output, expected), "❌ Output does not match expected!"
    print("✅ conv2d_naive works!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement conv2d_naive above!")

# %% [markdown]
"""
## Step 2: Conv2D Layer Class

Now let's wrap your function in a layer class for use in networks.
"""

# %%
#| export
class Conv2D:
    """
    2D Convolutional Layer (single channel, single filter, no stride/pad).
    Args:
        kernel_size: (kH, kW)
    TODO: Initialize a random kernel and implement the forward pass using conv2d_naive.
    """
    def __init__(self, kernel_size: Tuple[int, int]):
        raise NotImplementedError("Student implementation required")
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Student implementation required")
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Conv2D:
    def __init__(self, kernel_size: Tuple[int, int]):
        self.kernel = np.random.randn(*kernel_size).astype(np.float32)
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(conv2d_naive(x.data, self.kernel))
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
## Step 3: Provided Utilities (Stretch Goals)
- Stride and padding utilities (provided or as stretch)
- Multi-channel and multi-filter support (provided or as stretch)
- Pooling (optional)
"""

# %%
#| export
def flatten(x: Tensor) -> Tensor:
    """Flatten a 2D tensor to 1D (for connecting to Dense)."""
    return Tensor(x.data.flatten()[None, :])

# %% [markdown]
"""
## Step 4: Compose a Simple ConvNet

Now you can build a simple ConvNet:
- Conv2D → ReLU → Flatten → Dense
"""

# %%
# Compose a simple ConvNet
try:
    conv = Conv2D((2, 2))
    relu = ReLU()
    dense = Dense(4, 1)
    x = Tensor(np.random.randn(3, 3).astype(np.float32))
    out = dense(flatten(relu(conv(x))))
    print("ConvNet output:", out.data)
    print("✅ Simple ConvNet works!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your Conv2D and flatten implementations!")

# %% [markdown]
"""
## Step 5: Visualization (Provided)

Visualize the sliding window and feature map construction.
"""

# %%
# Provided visualization for convolution
import matplotlib.patches as patches
def visualize_conv2d(input, kernel, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(input, cmap='Blues')
    axes[0].set_title('Input')
    axes[1].imshow(kernel, cmap='Reds')
    axes[1].set_title('Kernel')
    axes[2].imshow(output, cmap='Greens')
    axes[2].set_title('Output Feature Map')
    plt.show()

# Example visualization
try:
    output = conv2d_naive(input, kernel)
    visualize_conv2d(input, kernel, output)
except Exception as e:
    print("Visualization skipped (conv2d_naive not implemented yet)")

# %% [markdown]
"""
## Step 6: Tests (Provided)

Test your Conv2D layer on more examples and edge cases.
"""

# %%
# More tests for Conv2D
try:
    x = Tensor(np.ones((5, 5), dtype=np.float32))
    conv = Conv2D((3, 3))
    out = conv(x)
    print("Conv2D output shape:", out.shape)
    assert out.shape == (3, 3), "❌ Output shape incorrect!"
    print("✅ Conv2D layer passes shape test!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your Conv2D implementation!") 
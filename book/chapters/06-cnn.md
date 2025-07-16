---
title: "CNN"
description: "Convolutional Neural Network layers and operations"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# Module: CNN

```{div} badges
⭐⭐⭐⭐ | ⏱️ 6-8 hours
```


## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 6-8 hours
- **Prerequisites**: Tensor, Activations, Layers, Networks modules
- **Next Steps**: Training, Computer Vision modules

Implement the core building block of modern computer vision: the convolutional layer. This module teaches you how convolution transforms computer vision from hand-crafted features to learned hierarchical representations that power everything from image recognition to autonomous vehicles.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Understand convolution fundamentals**: Master the sliding window operation, local connectivity, and weight sharing principles
- **Implement Conv2D from scratch**: Build convolutional layers using explicit loops to understand the core operation
- **Visualize feature learning**: See how convolution builds feature maps and hierarchical representations
- **Design CNN architectures**: Compose convolutional layers with pooling and dense layers into complete networks
- **Apply computer vision principles**: Understand how CNNs revolutionized image processing and pattern recognition

## 🧠 Build → Use → Analyze

This module follows TinyTorch's **Build → Use → Analyze** framework:

1. **Build**: Implement Conv2D from scratch using explicit for-loops to understand the core convolution operation
2. **Use**: Compose Conv2D with activation functions and other layers to build complete convolutional networks
3. **Analyze**: Visualize learned features, understand architectural choices, and compare CNN performance characteristics

## 📚 What You'll Build

### Core Convolution Implementation
```python
# Conv2D layer: the heart of computer vision
conv_layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
input_image = Tensor([[[[...]]]])  # (batch, channels, height, width)
feature_maps = conv_layer(input_image)  # Learned features

# Understanding the operation
print(f"Input shape: {input_image.shape}")     # (1, 3, 32, 32)
print(f"Output shape: {feature_maps.shape}")   # (1, 16, 30, 30)
print(f"Learned {feature_maps.shape[1]} different feature detectors")
```

### Complete CNN Architecture
```python
# Simple CNN for image classification
cnn = Sequential([
    Conv2D(3, 16, kernel_size=3),    # Feature extraction
    ReLU(),                          # Nonlinearity
    MaxPool2D(kernel_size=2),        # Dimensionality reduction
    Conv2D(16, 32, kernel_size=3),   # Higher-level features
    ReLU(),                          # More nonlinearity
    Flatten(),                       # Prepare for dense layers
    Dense(32 * 13 * 13, 128),        # Feature integration
    ReLU(),
    Dense(128, 10),                  # Classification head
    Sigmoid()                        # Probability outputs
])

# End-to-end image classification
image_batch = Tensor([[[[...]]]])  # Batch of images
predictions = cnn(image_batch)     # Class probabilities
```

### Convolution Operation Details
- **Sliding Window**: Filter moves across input to detect local patterns
- **Weight Sharing**: Same filter applied everywhere for translation invariance
- **Local Connectivity**: Each output depends only on local input region
- **Feature Maps**: Multiple filters learn different feature detectors

### CNN Building Blocks
- **Conv2D Layer**: Core convolution operation with learnable filters
- **Pooling Layers**: MaxPool and AvgPool for spatial downsampling
- **Flatten Layer**: Converts 2D feature maps to 1D for dense layers
- **Complete Networks**: Integration with existing Dense and activation layers

## 🚀 Getting Started

### Prerequisites
Ensure you have mastered the foundational network building blocks:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify all prerequisite modules
tito test --module tensor
tito test --module activations
tito test --module layers
tito test --module networks
```

### Development Workflow
1. **Open the development file**: `modules/source/06_cnn/cnn_dev.py`
2. **Implement convolution operation**: Start with explicit for-loop implementation for understanding
3. **Build Conv2D layer class**: Wrap convolution in reusable layer interface
4. **Add pooling operations**: Implement MaxPool and AvgPool for spatial reduction
5. **Create complete CNNs**: Compose layers into full computer vision architectures
6. **Export and verify**: `tito export --module cnn && tito test --module cnn`

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify computer vision functionality:

```bash
# TinyTorch CLI (recommended)
tito test --module cnn

# Direct pytest execution
python -m pytest tests/ -k cnn -v
```

### Test Coverage Areas
- ✅ **Convolution Operation**: Verify sliding window operation and local connectivity
- ✅ **Filter Learning**: Test weight initialization and parameter management
- ✅ **Shape Transformations**: Ensure proper input/output shape handling
- ✅ **Pooling Operations**: Verify spatial downsampling and feature preservation
- ✅ **CNN Integration**: Test complete networks with real image-like data

### Inline Testing & Visualization
The module includes comprehensive educational feedback and visual analysis:
```python
# Example inline test output
🔬 Unit Test: Conv2D implementation...
✅ Convolution sliding window works correctly
✅ Weight sharing applied consistently
✅ Output shapes match expected dimensions
📈 Progress: Conv2D ✓

# Visualization feedback
📊 Visualizing convolution operation...
📈 Showing filter sliding across input
📊 Feature map generation: 3→16 channels
```

### Manual Testing Examples
```python
from tinytorch.core.tensor import Tensor
from cnn_dev import Conv2D, MaxPool2D, Flatten
from activations_dev import ReLU

# Test basic convolution
conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3)
input_img = Tensor([[[[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25]]]])
feature_maps = conv(input_img)
print(f"Input: {input_img.shape}, Features: {feature_maps.shape}")

# Test complete CNN pipeline
relu = ReLU()
pool = MaxPool2D(kernel_size=2)
flatten = Flatten()

# Forward pass through CNN layers
activated = relu(feature_maps)
pooled = pool(activated)
flattened = flatten(pooled)
print(f"Final shape: {flattened.shape}")
```

## 🎯 Key Concepts

### Real-World Applications
- **Image Classification**: CNNs power systems like ImageNet winners (AlexNet, ResNet, EfficientNet)
- **Object Detection**: YOLO and R-CNN families use CNN backbones for feature extraction
- **Medical Imaging**: CNNs analyze X-rays, MRIs, and CT scans for diagnostic assistance
- **Autonomous Vehicles**: CNN-based perception systems process camera feeds for navigation

### Computer Vision Fundamentals
- **Translation Invariance**: Convolution detects patterns regardless of position in image
- **Hierarchical Features**: Early layers detect edges, later layers detect objects and concepts
- **Parameter Efficiency**: Weight sharing dramatically reduces parameters compared to dense layers
- **Spatial Structure**: CNNs preserve and leverage 2D spatial relationships in images

### Convolution Mathematics
- **Sliding Window Operation**: Filter moves across input with stride and padding parameters
- **Cross-Correlation vs Convolution**: Deep learning typically uses cross-correlation operation
- **Feature Map Computation**: Output[i,j] = sum(input[i:i+k, j:j+k] * filter)
- **Receptive Field**: Region of input that influences each output activation

### CNN Architecture Patterns
- **Feature Extraction**: Convolution + ReLU + Pooling blocks extract hierarchical features
- **Classification Head**: Flatten + Dense layers perform final classification
- **Progressive Filtering**: Increasing filter count with decreasing spatial dimensions
- **Skip Connections**: Advanced architectures add residual connections for deeper networks

## 🎉 Ready to Build?

You're about to implement the technology that revolutionized computer vision! CNNs transformed image processing from hand-crafted features to learned representations, enabling everything from photo tagging to medical diagnosis to autonomous driving.

Understanding convolution from the ground up—implementing the sliding window operation yourself—will give you deep insight into why CNNs work so well for visual tasks. Take your time with the core operation, visualize what's happening, and enjoy building the foundation of modern computer vision!




Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/06_cnn/cnn_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/06_cnn/cnn_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/06_cnn/cnn_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/05_networks.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/07_dataloader.html" title="next page">Next Module →</a>
</div>

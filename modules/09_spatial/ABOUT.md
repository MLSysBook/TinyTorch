---
title: "Convolutional Networks"
description: "Build CNNs from scratch for computer vision and spatial pattern recognition"
difficulty: 3
time_estimate: "6-8 hours"
prerequisites: ["Tensor", "Activations", "Layers", "DataLoader"]
next_steps: ["Tokenization"]
learning_objectives:
  - "Implement convolution as sliding window operations with weight sharing"
  - "Design CNN architectures with feature extraction and classification components"
  - "Understand translation invariance and hierarchical feature learning"
  - "Build pooling operations for spatial downsampling and invariance"
  - "Apply computer vision principles to image classification tasks"
---

# 09. Convolutional Networks

**🏛️ ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 6-8 hours

## Overview

Implement convolutional neural networks (CNNs) from scratch. This module teaches you how convolution transforms computer vision from hand-crafted features to learned hierarchical representations that power everything from image classification to autonomous driving.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement convolution** as sliding window operations with explicit loops, understanding weight sharing and local connectivity
2. **Design CNN architectures** by composing convolutional, pooling, and dense layers for image classification
3. **Understand translation invariance** and why CNNs are superior to dense networks for spatial data
4. **Build pooling operations** (MaxPool, AvgPool) for spatial downsampling and feature invariance
5. **Apply computer vision principles** to achieve >75% accuracy on CIFAR-10 image classification

## Why This Matters

### Production Context

CNNs are the backbone of modern computer vision systems:

- **Meta's Vision AI** uses CNN architectures to tag 2 billion photos daily across Facebook and Instagram
- **Tesla Autopilot** processes camera feeds through CNN backbones for object detection and lane recognition
- **Google Photos** built a CNN-based system that automatically organizes billions of images
- **Medical Imaging** systems use CNNs to detect cancer in X-rays and MRIs with superhuman accuracy

### Historical Context

The convolution revolution transformed computer vision:

- **LeNet (1998)**: Yann LeCun's CNN read zip codes on mail; convolution proved viable but limited by compute
- **AlexNet (2012)**: Won ImageNet with 16% error rate (vs 26% previous); GPUs + convolution = computer vision revolution
- **ResNet (2015)**: 152-layer CNN achieved 3.6% error (better than human 5%); proved depth matters
- **Modern Era (2020+)**: CNNs power production vision systems processing trillions of images daily

The patterns you're implementing revolutionized how machines see.

## Pedagogical Pattern: Build → Use → Analyze

### 1. Build

Implement from first principles:
- Convolution as explicit sliding window operation
- Conv2D layer with learnable filters and weight sharing
- MaxPool2D and AvgPool2D for spatial downsampling
- Flatten layer to connect spatial and dense layers
- Complete CNN architecture with feature extraction and classification

### 2. Use

Apply to real problems:
- Build CNN for CIFAR-10 image classification
- Extract and visualize learned feature maps
- Compare CNN vs MLP performance on spatial data
- Achieve >75% accuracy with proper architecture
- Understand impact of kernel size, stride, and padding

### 3. Analyze

Deep-dive into architectural choices:
- Why does weight sharing reduce parameters dramatically?
- How do early vs late layers learn different features?
- What's the trade-off between depth and width in CNNs?
- Why are pooling operations crucial for translation invariance?
- How does spatial structure preservation improve learning?

## Implementation Guide

### Core Components

**Conv2D Layer - The Heart of Computer Vision**
```python
class Conv2D:
    """2D Convolutional layer with learnable filters.
    
    Implements sliding window convolution:
    - Applies same filter across all spatial positions (weight sharing)
    - Each filter learns to detect different features (edges, textures, objects)
    - Output is feature map showing where filter activates strongly
    
    Args:
        in_channels: Number of input channels (3 for RGB, 16 for feature maps)
        out_channels: Number of learned filters (feature detectors)
        kernel_size: Size of sliding window (typically 3 or 5)
        stride: Step size when sliding (1 = no downsampling)
        padding: Border padding to preserve spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # Initialize learnable filters
        self.weight = Tensor(shape=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Tensor(shape=(out_channels,))
        
    def forward(self, x):
        # x shape: (batch, in_channels, height, width)
        batch, _, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        # Calculate output dimensions
        out_h = (H + 2 * self.padding - kh) // self.stride + 1
        out_w = (W + 2 * self.padding - kw) // self.stride + 1
        
        # Sliding window convolution
        output = Tensor(shape=(batch, self.out_channels, out_h, out_w))
        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract local patch
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x[b, :, i_start:i_start+kh, j_start:j_start+kw]
                        
                        # Convolution: element-wise multiply and sum
                        output[b, oc, i, j] = (patch * self.weight[oc]).sum() + self.bias[oc]
        
        return output
```

**Pooling Layers - Spatial Downsampling**
```python
class MaxPool2D:
    """Max pooling for spatial downsampling and translation invariance.
    
    Takes maximum value in each local region:
    - Reduces spatial dimensions while preserving important features
    - Provides invariance to small translations
    - Reduces computation in later layers
    """
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    
    def forward(self, x):
        batch, channels, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        out_h = (H - kh) // self.stride + 1
        out_w = (W - kw) // self.stride + 1
        
        output = Tensor(shape=(batch, channels, out_h, out_w))
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x[b, c, i_start:i_start+kh, j_start:j_start+kw]
                        output[b, c, i, j] = patch.max()
        
        return output
```

**Complete CNN Architecture**
```python
class SimpleCNN:
    """CNN for CIFAR-10 classification.
    
    Architecture:
        Conv(3→32, 3x3) → ReLU → MaxPool(2x2)    # 32x32 → 16x16
        Conv(32→64, 3x3) → ReLU → MaxPool(2x2)   # 16x16 → 8x8
        Flatten → Dense(64*8*8 → 128) → ReLU
        Dense(128 → 10) → Softmax
    """
    def __init__(self):
        self.conv1 = Conv2D(3, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2)
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2)
        
        self.flatten = Flatten()
        self.fc1 = Linear(64 * 8 * 8, 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(128, 10)
    
    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu1(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool2(self.relu2(self.conv2(x)))  # (B, 64, 8, 8)
        
        # Classification
        x = self.flatten(x)                        # (B, 4096)
        x = self.relu3(self.fc1(x))               # (B, 128)
        x = self.fc2(x)                           # (B, 10)
        return x
```

### Step-by-Step Implementation

1. **Implement Conv2D Forward Pass**
   - Create sliding window iteration over spatial dimensions
   - Apply weight sharing: same filter at all positions
   - Handle batch processing efficiently
   - Verify output shape calculation

2. **Build Pooling Operations**
   - Implement MaxPool2D with maximum extraction
   - Add AvgPool2D for average pooling
   - Handle stride and kernel size correctly
   - Test spatial dimension reduction

3. **Create Flatten Layer**
   - Convert (B, C, H, W) to (B, C*H*W)
   - Prepare spatial features for dense layers
   - Preserve batch dimension
   - Enable gradient flow backward

4. **Design Complete CNN**
   - Stack Conv → ReLU → Pool blocks for feature extraction
   - Add Flatten → Dense for classification
   - Calculate dimensions at each layer
   - Test end-to-end forward pass

5. **Train on CIFAR-10**
   - Load CIFAR-10 using Module 08's DataLoader
   - Train with cross-entropy loss and SGD
   - Track accuracy on test set
   - Achieve >75% accuracy

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/09_spatial
python spatial_dev.py
```

Expected output:
```
Unit Test: Conv2D implementation...
✅ Sliding window convolution works correctly
✅ Weight sharing applied at all positions
✅ Output shapes match expected dimensions
Progress: Conv2D ✓

Unit Test: MaxPool2D implementation...
✅ Maximum extraction works correctly
✅ Spatial dimensions reduced properly
✅ Translation invariance verified
Progress: Pooling ✓

Unit Test: Complete CNN architecture...
✅ Forward pass through all layers successful
✅ Output shape: (32, 10) for 10 classes
✅ Parameter count reasonable: ~500K parameters
Progress: CNN Architecture ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 09_spatial

# Run integration tests
tito test 09_spatial
```

### CIFAR-10 Training Test

```bash
# Train simple CNN on CIFAR-10
python tests/integration/test_cnn_cifar10.py

Expected results:
- Epoch 1: 35% accuracy
- Epoch 5: 60% accuracy
- Epoch 10: 75% accuracy
```

## Where This Code Lives

```
tinytorch/
├── nn/
│   └── spatial.py              # Conv2D, MaxPool2D, etc.
└── __init__.py                 # Exposes CNN components

Usage in other modules:
>>> from tinytorch.nn import Conv2D, MaxPool2D
>>> conv = Conv2D(3, 32, kernel_size=3)
>>> pool = MaxPool2D(kernel_size=2)
```

## Systems Thinking Questions

1. **Parameter Efficiency**: A Conv2D(3, 32, 3) has ~900 parameters. How many parameters would a Dense layer need to connect a 32x32 image to 32 outputs? Why is this difference critical for scaling?

2. **Translation Invariance**: Why does a CNN detect a cat regardless of whether it's in the top-left or bottom-right of an image? How does weight sharing enable this property?

3. **Hierarchical Features**: Early CNN layers detect edges and textures. Later layers detect objects and faces. How does this emerge from stacking convolutions? Why doesn't this happen in dense networks?

4. **Receptive Field Growth**: A single Conv2D(kernel=3) sees a 3x3 region. After two Conv2D layers, what region does each output see? How do deep CNNs build global context from local operations?

5. **Compute vs Memory Trade-offs**: Large kernel sizes (7x7) have more parameters but fewer operations. Small kernels (3x3) stacked deeply have opposite trade-offs. Which is better and why?

## Real-World Connections

### Industry Applications

**Autonomous Vehicles (Tesla, Waymo)**
- Multi-camera CNN systems process 30 FPS at 1920x1200 resolution
- Feature maps from CNNs feed into object detection and segmentation
- Real-time requirements demand efficient Conv2D implementations

**Medical Imaging (PathAI, Zebra Medical)**
- CNNs analyze X-rays and CT scans for diagnostic assistance
- Achieve superhuman performance on specific tasks (diabetic retinopathy detection)
- Architecture design critical for accuracy-interpretability trade-off

**Face Recognition (Apple Face ID, Facebook DeepFace)**
- CNN embeddings enable accurate face matching at billion-user scale
- Lightweight CNN architectures run on mobile devices in real-time
- Privacy concerns drive on-device processing

### Research Impact

This module implements patterns from:
- LeNet-5 (1998): First successful CNN for digit recognition
- AlexNet (2012): Sparked deep learning revolution with CNNs + GPUs
- VGG (2014): Showed deeper is better with simple 3x3 convolutions
- ResNet (2015): Enabled training 152-layer CNNs with skip connections

## What's Next?

In **Module 10: Tokenization**, you'll shift from processing images to processing text:

- Learn how to convert text into numerical representations
- Implement tokenization strategies (character, word, subword)
- Build vocabulary management systems
- Prepare text data for transformers in Module 13

This completes the vision half of the Architecture Tier. Next, you'll tackle language!

---

**Ready to build CNNs from scratch?** Open `modules/09_spatial/spatial_dev.py` and start implementing.

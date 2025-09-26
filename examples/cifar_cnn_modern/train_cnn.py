#!/usr/bin/env python3
"""
Clean CIFAR-10 CNN Example - What Students Built
===============================================

After completing modules 02-10, students can build CNNs for real image classification.
This demonstrates how convolution + pooling creates spatial feature hierarchies.

MODULES EXERCISED IN THIS EXAMPLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : Data structure with gradient tracking
  Module 03 (Activations)   : ReLU activation throughout the network
  Module 04 (Layers)        : Linear layers for classification head
  Module 05 (Networks)      : Module base class for CNN architecture
  Module 06 (Autograd)      : Backprop through conv and dense layers
  Module 07 (Spatial)       : Conv2d, MaxPool2d, Flatten operations
  Module 08 (Optimizers)    : Adam optimizer with momentum
  Module 09 (DataLoader)    : CIFAR10Dataset and batch processing
  Module 10 (Training)      : CrossEntropy loss for multi-class
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CNN Architecture:
    ┌─────────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  ┌─────────┐
    │ Input Image │  │ Conv2d  │  │ MaxPool │  │ Conv2d      │  │ MaxPool │
    │ (32×32×3)   │─▶│ 3→32    │─▶│ (2×2)   │─▶│ 32→64      │─▶│ (2×2)   │
    │ RGB Pixels  │  │ Module  │  │ Module  │  │ Module 07   │  │ Module  │
    └─────────────┘  │   07    │  │   07    │  └─────────────┘  │   07    │
                     └─────────┘  └─────────┘                   └─────────┘
                           │                                           │
                           ▼                                           ▼
                     ┌─────────┐                              ┌─────────────┐
                     │  ReLU   │                              │   Flatten   │
                     │ Module  │                              │  → Dense    │
                     │   03    │                              │ Module 04   │
                     └─────────┘                              └─────────────┘
                                                                     │
                     ┌─────────────────────────────────────────────▼─┐
                     │ Dense Classifier: 1600 → 256 → 10 classes     │
                     │ Module 04: Linear layers + ReLU               │
                     └───────────────────────────────────────────────┘

Feature Hierarchy: Pixels → Edges → Shapes → Objects → Classes
"""

from tinytorch import nn, optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import to_numpy
import numpy as np

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()  # Module 05: You built Module base class!
        # Convolutional feature extraction 
        self.conv1 = nn.Conv2d(3, 32, (3, 3))      # Module 07: You built 2D convolution!
        self.conv2 = nn.Conv2d(32, 64, (3, 3))     # Module 07: You built filter sliding!
        
        # Dense classification
        # After conv1(32x32→30x30) → pool(15x15) → conv2(13x13) → pool(6x6)
        # Final feature size: 64 channels * 6 * 6 = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 256)      # Module 04: You built Linear layers!
        self.fc2 = nn.Linear(256, 10)              # Module 04: Your weight matrices!
    
    def forward(self, x):
        # First conv block: extract low-level features (edges, textures)
        x = self.conv1(x)           # Module 07: Your Conv2d sliding filters!
        x = nn.F.relu(x)            # Module 03: You built ReLU activation!
        x = nn.F.max_pool2d(x, 2)   # Module 07: You built max pooling!
        
        # Second conv block: extract higher-level features (shapes, patterns)
        x = self.conv2(x)           # Module 07: Your deeper convolutions!
        x = nn.F.relu(x)            # Module 03: Your non-linearity!
        x = nn.F.max_pool2d(x, 2)   # Module 07: Your spatial reduction!
        
        # Classification head
        x = nn.F.flatten(x, start_dim=1)  # Module 07: You built flatten operation!
        x = self.fc1(x)             # Module 04: Your Linear layer!
        x = nn.F.relu(x)            # Module 03: Your activation!
        return self.fc2(x)          # Module 04: Your final classification!

def main():
    # For validation testing, test architecture only (no training)
    print("🖼️  Testing CIFAR-10 CNN Architecture...")
    
    model = CIFARCNN()
    
    print("🚀 CNN Architecture Validation!")
    print("   Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck")
    print("   Architecture: Conv → Pool → Conv → Pool → Dense → Classify")
    print(f"   Parameters: {sum(p.data.size for p in model.parameters()):,} weights")
    print()
    
    # Test forward pass with small input
    test_input = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
    print("   Testing forward pass with single 32x32 RGB image...")
    
    try:
        output = model(test_input)
        print(f"   ✅ Forward pass successful! Output shape: {to_numpy(output).shape}")
        print(f"   ✅ Output contains {to_numpy(output).shape[1]} class predictions")
        print()
        print("   CNN architecture validated:")
        print("   • Conv2d layers process spatial features")
        print("   • MaxPool2d reduces spatial dimensions")
        print("   • Flatten converts 2D to 1D for classification")
        print("   • Linear layers perform final classification")
        print()
        print("✅ Success! CNN architecture works correctly")
    except Exception as e:
        print(f"   ❌ Error in forward pass: {e}")
        return
    
    print("\n🎯 What You Learned by Building:")
    print("   • How convolutions detect local features (edges, textures)")
    print("   • Why pooling reduces computation while preserving information")
    print("   • How spatial feature hierarchies enable object recognition")
    print("   • Complete computer vision pipeline from pixels to predictions")

if __name__ == "__main__":
    main()
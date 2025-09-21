# CIFAR-10 Image Recognition Examples

Train neural networks to classify real RGB images from CIFAR-10!

## Examples in this Directory

### 🧪 `test_quick.py` - Pipeline Verification 
Quick test to verify CIFAR-10 → MLP pipeline works without training.
Tests data loading, model architecture, and forward pass.

### 🎯 `train_mlp.py` - Milestone 1: "Machines Can See"
Multi-Layer Perceptron training on CIFAR-10 for **Milestone 1**.
- **Target**: 45%+ accuracy (proves framework works on real data)
- **Architecture**: 3072 → 512 → 256 → 10 (MLP)
- **Learning**: Real data complexity, scaling challenges

### 🏆 `train.py` - Milestone 2: "I Can Train Real AI"
Convolutional Neural Network training on CIFAR-10 for **Milestone 2**.

## What This Demonstrates

- **Convolutional Neural Networks** with spatial operations
- **Batch normalization** for training stability  
- **Real-world computer vision** on natural images
- **Production-level CNN architecture** built from scratch
- **65%+ accuracy** on challenging dataset

## The CIFAR-10 Dataset

- 50,000 training images
- 10,000 test images
- 32×32 RGB color images
- 10 real-world classes:
  - airplane, automobile, bird, cat, deer
  - dog, frog, horse, ship, truck

## Running the Example

```bash
python train.py
```

Expected output:
```
📚 Loading CIFAR-10 dataset...
  Training samples: 50,000
  Test samples: 10,000

🎯 Training CNN...
Epoch 1/20
  Batch   0/782 | Loss: 2.3026 | Acc: 10.9%
  Batch 100/782 | Loss: 1.8234 | Acc: 32.1%
  ...
  
📊 Final Results:
Overall Test Accuracy: 68.5%

Per-Class Accuracy:
  airplane    : 72.3% ████████████████████████████████████
  automobile  : 78.1% ███████████████████████████████████████
  bird        : 58.4% █████████████████████████████
  ...
  
🎉 SUCCESS! Your CNN achieves strong real-world performance!
```

## Architecture

```
Input (32×32×3 RGB)
    ↓
Conv(3→32) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv(32→64) → BatchNorm → ReLU → MaxPool(2×2)  
    ↓
Conv(64→128) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Flatten → Dense(2048→256) → BatchNorm → ReLU
    ↓
Dense(256→10) → Softmax
```

## Key Achievements

- **Real CNN**: Not a toy - this is production architecture
- **Spatial operations**: Conv2D, MaxPool2D you built work!
- **Batch normalization**: Training stability at scale
- **Competitive accuracy**: 65%+ rivals early deep learning papers

## Training Tips

- Start with learning rate 0.001
- Reduce to 0.0001 after epoch 10
- Batch size 64 works well
- 20 epochs should reach 65%+

## Requirements

- Module 06 (Spatial/CNN) for Conv2D, MaxPool2D
- Module 08 (DataLoader) for CIFAR-10 dataset
- Module 10 (Optimizers) for Adam
- Module 11 (Training) for complete training
- TinyTorch package fully exported
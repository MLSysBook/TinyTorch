# CIFAR-10 Image Classification

Train a neural network on real-world 32×32 color images from the CIFAR-10 dataset.

## 🎯 Performance

| Model | Accuracy | Description |
|-------|----------|-------------|
| **Untrained Network** | ~10% | Random weights baseline |
| **Trained Network** | 53-55% | After 15 epochs of training |
| **Improvement** | 5.5× | Learning real patterns from data |

## 📁 Files

- **`untrained_baseline.py`** - Shows what an untrained network achieves (~10%)
- **`train_cifar10.py`** - Training with Rich progress display (55% accuracy)
- **`train_simple.py`** - Simple training without UI dependencies (55% accuracy)

## 🚀 Quick Start

### 1. See Random Baseline
```bash
python untrained_baseline.py
```
Shows that an untrained network gets ~10% (random chance for 10 classes).

### 2. Train the Network

**With Rich UI (recommended):**
```bash
python train_cifar10.py
```

**Simple version (no dependencies):**
```bash
python train_simple.py
```

Both achieve ~55% accuracy in about 2 minutes.

## 📊 Understanding the Results

- **Random chance**: 10% (guessing randomly among 10 classes)
- **Our network**: 55% (5.5× better than random!)
- **What this means**: The network learned real patterns from the images

## 🏗️ Architecture

```
Input: 3072 (32×32×3 flattened image)
   ↓
Hidden Layer 1: 1024 neurons (ReLU)
   ↓
Hidden Layer 2: 512 neurons (ReLU)
   ↓
Hidden Layer 3: 256 neurons (ReLU)
   ↓
Output: 10 neurons (one per class)
```

## 📈 Training Details

- **Optimizer**: Adam (learning rate: 0.002)
- **Loss function**: Cross Entropy
- **Batch size**: 64
- **Epochs**: 15
- **Data augmentation**: Horizontal flips
- **Learning rate schedule**: Decay at epochs 8 and 12

## 🎯 Classes in CIFAR-10

The network learns to classify images into 10 categories:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 💡 Key Insights

1. **MLPs can learn complex patterns**: 55% accuracy on real images is impressive for a simple MLP
2. **Training works**: Starting from 10% (random) and reaching 55% proves learning
3. **Architecture matters**: CNNs would achieve 70-80%+, showing the importance of architecture choice
4. **Your code works**: This is running on 100% student-built TinyTorch code!

## 🔧 Technical Notes

- Dataset is automatically downloaded on first run (~170MB)
- Training takes ~2 minutes on a modern CPU
- The model has ~3.8M parameters
- Memory usage: ~500MB during training
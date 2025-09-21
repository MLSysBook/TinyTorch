# TinyTorch Examples 🔥

Real-world examples showing what you can build with TinyTorch!

## What Are These Examples?

These are **real ML applications** written using TinyTorch just like you would use PyTorch. Each example:
- Uses `import tinytorch` as a real package
- Shows professional ML code patterns
- Demonstrates actual capabilities you've built
- Can be run by anyone to see TinyTorch in action

## Running Examples

```bash
# After installing/building TinyTorch:
cd examples/xornet/
python train.py

# Or for image classification:
cd examples/cifar10/
python train_cifar10_mlp.py
```

## Available Examples

### 🧠 **`xornet/`** - Neural Network Fundamentals
- Classic XOR problem with hidden layers
- Clean implementation showing autograd and training basics
- Architecture: 2 → 4 → 1 with ReLU and Sigmoid
- **Achieves 100% accuracy** on XOR truth table

### 👁️ **`cifar10/`** - Real-World Computer Vision
- Real-world object classification
- **ACHIEVEMENT: 57.2% accuracy** - exceeds typical ML course benchmarks!
- Multiple architectures: MLP, LeNet-5, and optimized models
- Data augmentation, proper initialization, Adam optimization
- Real dataset: 50,000 training images, 10,000 test images

## Example Structure

Each example directory contains:
```
example_name/
├── train.py          # Main training script
├── README.md         # What this example demonstrates
└── data/            # Datasets (downloaded automatically)
```

## Learning Progression

After completing each module, examples become functional:
- **Module 05** → `xornet/` works (Dense layers + activations)
- **Module 11** → `cifar10/` works with training loops

## Quick Demo

Want to see TinyTorch in action? Try these:

```bash
# See a neural network learn XOR (30 seconds):
python examples/xornet/train.py

# Train on real images (5 minutes, 57% accuracy):
python examples/cifar10/train_cifar10_mlp.py --epochs 10
```

## Performance Achievements

- **XORnet**: 100% accuracy (perfect solution)
- **CIFAR-10**: 57.2% accuracy (exceeds typical course benchmarks)

---

**These aren't toy demos - they're real ML applications achieving competitive results with a framework built from scratch!**
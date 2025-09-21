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
cd examples/xor_network/
python train.py

# Or for image classification:
cd examples/cifar10_classifier/
python train_cifar10_mlp.py
```

## Available Examples

### 🧠 Neural Network Fundamentals
- **`xor_network/`** - Classic XOR problem with hidden layers
  - Clean implementation showing autograd and training basics
  - Architecture: 2 → 4 → 1 with ReLU and Sigmoid
  - Achieves 100% accuracy on XOR truth table

### 👁️ Computer Vision  
- **`cifar10_classifier/`** - Real-world object classification
  - **ACHIEVEMENT: 57.2% accuracy** - exceeds typical ML course benchmarks!
  - Multiple architectures: MLP, LeNet-5, and optimized models
  - Data augmentation, proper initialization, Adam optimization
  - Real dataset: 50,000 training images, 10,000 test images

### 🤖 Language & Generation
- **`text_generation/`** - Generate text with TinyGPT (Module 16)
  - Transformer architecture built from scratch
  - Character-level text generation
  - Attention mechanisms and positional encoding

### 📊 Optimization & Analysis
- **`optimization_comparison/`** - SGD vs Adam comparison
  - Side-by-side optimizer performance analysis
  - Visualization of convergence patterns
  - Memory usage and computational efficiency

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
- **Module 05** → `xor_network/` works (Dense layers + activations)
- **Module 11** → `cifar10_classifier/` works with training loops
- **Module 16** → `text_generation/` works (TinyGPT)

## Quick Demo

Want to see TinyTorch in action? Try these:

```bash
# See a neural network learn XOR (30 seconds):
python examples/xor_network/train.py

# Train on real images (5 minutes, 57% accuracy):
python examples/cifar10_classifier/train_cifar10_mlp.py --epochs 10

# Compare optimizers (2 minutes):
python examples/optimization_comparison/compare.py
```

## Performance Achievements

- **XOR Network**: 100% accuracy (perfect solution)
- **CIFAR-10 MLP**: 57.2% accuracy (exceeds typical course benchmarks)
- **Optimization**: Adam 3.2x faster convergence than SGD

---

**These aren't toy demos - they're real ML applications achieving competitive results with a framework built from scratch!**
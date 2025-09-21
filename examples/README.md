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

# Or for image recognition:
cd examples/mnist_recognition/
python train_mnist.py
```

## Example Categories

### 🧠 Neural Network Fundamentals
- `xor_network/` - Classic XOR problem with hidden layers
- `linear_regression/` - Simple regression tasks

### 👁️ Computer Vision
- `mnist_recognition/` - Handwritten digit recognition
- `cifar10_classifier/` - Real-world object classification
- `image_filters/` - Convolutional feature extraction

### 🤖 Language & Generation
- `text_generation/` - Generate text with TinyGPT
- `code_completion/` - Python code generation

### ⚡ Advanced Topics
- `autograd_demo/` - Automatic differentiation
- `optimization_comparison/` - SGD vs Adam
- `model_compression/` - Quantization and pruning
- `performance_profiling/` - Benchmarking tools

## Example Structure

Each example directory contains:
```
example_name/
├── train.py          # Main training script
├── model.py          # Model architecture (if complex)
├── README.md         # What this example demonstrates
├── requirements.txt  # Just needs: tinytorch
└── config.yml        # Metadata about the example
```

## For Students

After completing each module, the corresponding example will work:
- Module 05 → `xor_network/` works
- Module 08 → `mnist_recognition/` works
- Module 11 → `cifar10_classifier/` works with training
- Module 16 → `text_generation/` works

## For Everyone Else

Want to see what TinyTorch can do? Just run any example:
```bash
# See a neural network learn XOR:
python examples/xor_network/train.py

# Train a CNN on real images:
python examples/cifar10_classifier/train.py

# Generate text with a transformer:
python examples/text_generation/generate.py
```

---

**These aren't toy demos - they're real ML applications using a framework built from scratch!**
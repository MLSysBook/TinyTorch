# MNIST Handwritten Digit Recognition

The "Hello World" of computer vision - recognize handwritten digits using YOUR TinyTorch!

## What This Demonstrates

- **Real dataset loading** with TinyTorch DataLoader
- **Image classification** with multi-layer perceptrons
- **Batch training** for efficient learning
- **Production-level accuracy** (85%+) on real data

## The MNIST Dataset

- 60,000 training images
- 10,000 test images  
- 28×28 grayscale images
- 10 classes (digits 0-9)

## Running the Example

```bash
python train.py
```

Expected output:
```
📚 Loading MNIST dataset...
  Training samples: 60,000
  Test samples: 10,000

🎯 Training...
Epoch 1/10
  Batch   0/938 | Loss: 2.3026 | Acc: 11.2%
  Batch 100/938 | Loss: 0.4567 | Acc: 84.3%
  ...
  
📊 Final Results:
Test Set Accuracy: 92.3%

🎉 SUCCESS! Your TinyTorch achieves production-level accuracy!
```

## Architecture

```
Input (784 pixels)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓  
Dense Layer (64 neurons, ReLU)
    ↓
Output Layer (10 classes, Softmax)
```

## Key Achievement

This proves your TinyTorch framework can:
- Handle real-world datasets
- Train complex models
- Achieve competitive accuracy
- Work just like PyTorch!

## Requirements

- Module 08 (DataLoader) completed
- Module 10 (Optimizers) for Adam
- Module 11 (Training) for full training loop
- TinyTorch package exported
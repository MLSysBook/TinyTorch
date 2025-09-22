# TinyTorch 15-Module Structure

## Three-Part Journey: MLPs → CNNs → Transformers

### Part I: Multi-Layer Perceptrons (Modules 1-5)
**Goal**: Build neural networks that can solve XOR

| Module | Topic | What You Build |
|--------|-------|----------------|
| 01 | Setup | Development environment |
| 02 | Tensors | N-dimensional arrays |
| 03 | Activations | ReLU, Sigmoid, Softmax |
| 04 | Layers | Dense layers |
| 05 | Networks | Sequential models |

**Capstone**: XORNet - Proves neural networks can learn non-linear functions

---

### Part II: Convolutional Neural Networks (Modules 6-10)
**Goal**: Build CNNs for image classification

| Module | Topic | What You Build |
|--------|-------|----------------|
| 06 | Spatial | Conv2D, MaxPool2D |
| 07 | DataLoader | Efficient data pipelines |
| 08 | Autograd | Automatic differentiation |
| 09 | Optimizers | SGD, Adam |
| 10 | Training | Complete training loops |

**Capstone**: CIFAR-10 with three approaches:
1. **Random Baseline**: ~10% accuracy (chance)
2. **MLP Approach**: ~55% accuracy (no convolutions)
3. **CNN Approach**: ~60%+ accuracy (WITH Conv2D!)

This progression shows WHY convolutions matter for vision!

---

### Part III: Transformers (Modules 11-15)
**Goal**: Build transformers for text generation

| Module | Topic | What You Build |
|--------|-------|----------------|
| 11 | Embeddings | Token & positional encoding |
| 12 | Attention | Multi-head attention |
| 13 | Normalization | LayerNorm for stable training |
| 14 | Transformers | Complete transformer blocks |
| 15 | Generation | Autoregressive decoding |

**Capstone**: TinyGPT - Character-level text generation

---

## Why This Structure Works

### Pedagogical Excellence
- **Each part introduces ONE major innovation**:
  - Part I: Fully connected networks (the foundation)
  - Part II: Convolutions (spatial processing)
  - Part III: Attention (sequence processing)

### Historical Accuracy
- **Follows ML evolution**:
  - 1980s-90s: MLPs dominate
  - 2012: AlexNet shows CNNs beat MLPs on ImageNet
  - 2017: Transformers revolutionize NLP

### Dependency-Driven Design
- **Nothing unnecessary**: Each module is needed for its capstone
- **Progressive complexity**: Each part builds on the previous
- **Clear motivation**: Students see WHY each innovation matters

## Module Dependencies

```
Part I: Foundations
├── 02_tensor (required by everything)
├── 03_activations (required by 04)
├── 04_layers (required by 05)
└── 05_networks (combines all above)
    └── ✅ XORNet works!

Part II: Computer Vision
├── 06_spatial (Conv2D - THE KEY!)
├── 07_dataloader (handle real data)
├── 08_autograd (enable learning)
├── 09_optimizers (gradient descent)
└── 10_training (put it all together)
    └── ✅ CIFAR-10 CNN works!

Part III: Language Models
├── 11_embeddings (discrete → continuous)
├── 12_attention (THE KEY!)
├── 13_normalization (stable training)
├── 14_transformers (attention + FFN)
└── 15_generation (sampling strategies)
    └── ✅ TinyGPT works!
```

## What We Dropped
- **Module 16 (Regularization)**: Important but not essential for capstones
- **Module 17 (Systems)**: Kernels, benchmarking - advanced optimization

These could be bonus content or a separate "Production ML" course.

## The Beauty of 15 Modules
- **3 parts × 5 modules = 15**: Perfect symmetry!
- **Each part is self-contained**: Students can stop after any part
- **Clear progression**: MLP → CNN → Transformer
- **Manageable scope**: Achievable in one semester
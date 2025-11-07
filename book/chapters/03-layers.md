---
title: "Layers"
description: "Neural network layers (Linear, Sequential)"
module_number: 3
tier: "foundation"
difficulty: "intermediate"
time_estimate: "4-5 hours"
prerequisites: ["01. Tensor", "02. Activations"]
next_module: "04. Losses"
learning_objectives:
  - "Understand layers as composable functions that transform tensor representations"
  - "Implement Linear layers with weight matrices and bias vectors (y = Wx + b)"
  - "Build Sequential containers for composing multiple layers into networks"
  - "Recognize parameter initialization strategies and their impact on training"
  - "Analyze memory usage and computational complexity of layer operations"
---

# 03. Layers

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 03 of 20
</div>

**Build composable building blocks - Linear layers and Sequential containers.**

Difficulty: Intermediate | Time: 4-5 hours | Prerequisites: Modules 01-02

---

## What You'll Build

Layers are the building blocks of neural networks. Each layer is a function that transforms tensors, and networks are compositions of these functions.

By the end of this module, you'll have implemented:

- **Linear Layer** - Affine transformation `y = Wx + b` (also called Dense or Fully Connected)
- **Sequential Container** - Chains multiple layers into a network
- **Parameter Management** - Weight initialization and storage

### Example Usage

```python
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU
from tinytorch.core.tensor import Tensor

# Single linear layer
layer = Linear(in_features=784, out_features=128)
x = Tensor(np.random.randn(32, 784))  # Batch of 32 images
output = layer.forward(x)  # Shape: (32, 128)

# Multi-layer network
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])

logits = model.forward(x)  # Shape: (32, 10)
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement the Linear layer with weight matrix initialization and forward pass computation. Build a Sequential container to chain layers.

### 2. Use
Create multi-layer networks, transform data through layers, and observe how representations change across network depth.

### 3. Understand
Grasp how simple layers compose into powerful networks, why parameter initialization matters, and how memory/computation scales with layer size.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize layers as modular transformations that compose into networks, mirroring Unix philosophy of composable tools

2. **Core Implementation**: Build Linear layers with proper weight initialization and Sequential containers for network composition

3. **Pattern Recognition**: Understand the affine transformation pattern (linear + bias) and how it appears everywhere in neural networks

4. **Framework Connection**: See how your Linear mirrors `torch.nn.Linear` and Sequential mirrors `torch.nn.Sequential`

5. **Performance Trade-offs**: Analyze computational cost (O(batch × in × out) for Linear) and memory usage (parameters + activations)

---

## Why This Matters

### Production Context

Linear layers are ubiquitous in production ML:

- **MLPs**: Stack of Linear + ReLU layers for tabular data (recommendation systems, fraud detection)
- **Vision**: Linear layers in classifier heads (ResNet's final layer: 2048 → 1000 classes)
- **NLP**: Linear projections in transformers (attention Q/K/V projections, feedforward layers)
- **RL**: Policy/value networks use Linear layers to map states to actions

Every neural network contains Linear layers. Understanding their implementation is fundamental.

### Systems Reality Check

**Performance Note**: A single Linear layer with 1000 × 1000 weights performs 1M multiply-adds per sample. Batch size 32 = 32M operations. Matrix multiplication dominates training time.

**Memory Note**: Parameters are persistent (stored across batches), but activations are temporary (one batch at a time). A 1000 × 1000 layer uses 4MB for FP32 weights, but activations can be deallocated after backprop.

---

## Implementation Guide

### Prerequisites Check

```bash
tito test 01 02
```

### Development Workflow

```bash
cd modules/source/03_layers/
jupyter lab layers_dev.py
```

### Step-by-Step Build

#### Step 1: Linear Layer Foundation

Implement the core affine transformation:

```python
class Linear:
    def __init__(self, in_features, out_features):
        """Initialize linear layer: y = Wx + b"""
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features))
        )
        self.bias = Tensor(np.zeros(out_features))
        
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        """Forward pass: y = xW + b"""
        return x.matmul(self.weight) + self.bias
```

**Why Xavier initialization**: Random uniform initialization with variance scaled by layer size prevents vanishing/exploding gradients. Critical for deep network training.

**Design decision**: Weight shape is `(in, out)` so we compute `x @ W` (not `W @ x`). This matches PyTorch conventions and enables clean batched computation.

#### Step 2: Parameter Management

Track parameters for optimization:

```python
def parameters(self):
    """Return list of trainable parameters"""
    return [self.weight, self.bias]
```

**Why this matters**: Optimizers need access to all parameters to update them during training. This method provides that access.

#### Step 3: Sequential Container

Chain layers into networks:

```python
class Sequential:
    def __init__(self, layers):
        """Container for sequential layer execution"""
        self.layers = layers
    
    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def parameters(self):
        """Collect parameters from all layers"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
```

**Pattern insight**: Sequential treats activation functions and layers uniformly—anything with a `forward()` method works. This composability is powerful.

#### Step 4: Network Building

Combine components:

```python
# Two-layer MLP
model = Sequential([
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
])

# Access all parameters
all_params = model.parameters()  # [W1, b1, W2, b2]
```

---

## Testing Your Implementation

### Inline Tests

```python
# Test Linear layer shape
layer = Linear(10, 5)
x = Tensor(np.random.randn(3, 10))  # Batch of 3
y = layer.forward(x)
assert y.shape == (3, 5)
print("✓ Linear shape correct")

# Test parameter count
params = layer.parameters()
assert len(params) == 2  # weight + bias
assert params[0].shape == (10, 5)
assert params[1].shape == (5,)
print("✓ Parameters correct")

# Test Sequential
model = Sequential([Linear(10, 5), ReLU(), Linear(5, 2)])
output = model.forward(x)
assert output.shape == (3, 2)
print("✓ Sequential working")
```

### Module Export & Validation

```bash
tito export 03
tito test 03
```

**Expected output**:
```
✓ All tests passed! [18/18]
✓ Module 03 complete!
```

---

## Where This Code Lives

Your layers are used everywhere in TinyTorch:

```python
# Networks use your Linear layers:
from tinytorch.nn.networks import MLP
from tinytorch.core.layers import Linear, Sequential

# Example: MNIST classifier
mnist_model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])

# This is how all neural networks are built!
```

**Package structure**:
```
tinytorch/
├── core/
│   ├── layers.py  ← YOUR implementations
│   ├── tensor.py
│   ├── activations.py
├── nn/
│   ├── networks.py  (uses your layers)
```

---

## Systems Thinking Questions

1. **Complexity Analysis**: A Linear(1000, 1000) layer performs 1M multiply-adds per sample. How does batch size affect total computation? How does this scale to GPT-3's Linear(12288, 49152) layers?

2. **Memory Trade-offs**: Why store both parameters (weights/biases) and activations? Can activations be deallocated after forward pass? What about during training?

3. **Initialization Strategies**: Why not initialize all weights to zero? What would happen? Why does Xavier initialization use `sqrt(6 / (in + out))`?

4. **Design Decisions**: Why is weight shape `(in, out)` instead of `(out, in)`? How does this affect batching and GPU performance?

5. **Framework Comparison**: PyTorch's Linear supports `bias=False` for layers without bias. When would this be useful? (Hint: LayerNorm, BatchNorm)

---

## Real-World Connections

### Industry Applications

- **ResNet Final Layer**: Linear(2048, 1000) maps features to ImageNet classes
- **BERT Embeddings**: Linear(768, 30522) projects to vocabulary for masked LM
- **GPT-3 Feedforward**: Linear(12288, 49152) in transformer MLP blocks
- **Recommendation Systems**: Linear layers map user/item embeddings to scores

### Architecture Patterns

- **MLPs**: Stack of Linear + activation (used in tabular data, RL)
- **Residual Connections**: `x + Linear(ReLU(Linear(x)))` (ResNet pattern)
- **Attention Projections**: Q = Linear(x), K = Linear(x), V = Linear(x)

---

## What's Next?

**Great progress!** You can now build multi-layer networks. Next, you'll implement loss functions to measure how well networks perform.

**Module 04: Losses** - Implement MSE, Cross-Entropy, and Binary Cross-Entropy loss functions for training

[Continue to Module 04: Losses →](04-losses.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Layers API Reference](../appendices/api-reference.html#layers)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)

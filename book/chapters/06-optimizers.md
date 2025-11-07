---
title: "Optimizers"
description: "Gradient-based optimization algorithms (SGD, Momentum, Adam)"
module_number: 6
tier: "foundation"
difficulty: "intermediate"
time_estimate: "4-5 hours"
prerequisites: ["01. Tensor", "05. Autograd"]
next_module: "07. Training"
learning_objectives:
  - "Understand gradient descent as iterative parameter updates guided by loss gradients"
  - "Implement SGD with momentum and Adam with adaptive learning rates"
  - "Build learning rate schedulers for training stability and convergence"
  - "Recognize optimizer design patterns from PyTorch and TensorFlow"
  - "Analyze memory usage and convergence behavior of different optimizers"
---

# 06. Optimizers

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 06 of 20
</div>

**Build algorithms that use gradients to train neural networks.**

Difficulty: Intermediate | Time: 4-5 hours | Prerequisites: Modules 01, 05

---

## What You'll Build

Optimizers are algorithms that update model parameters using gradients. They're the "learning" in machine learning—the mechanism that improves models over time.

By the end of this module, you'll have implemented three essential optimizers:

- **SGD** (Stochastic Gradient Descent) - The foundation of neural network training
- **SGD with Momentum** - Accelerated convergence with velocity accumulation
- **Adam** (Adaptive Moment Estimation) - Adaptive learning rates per parameter

### Example Usage

```python
from tinytorch.optim import SGD, Momentum, Adam
from tinytorch.nn import MLP

# Create model
model = MLP([784, 128, 10])

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    # Forward pass
    predictions = model.forward(batch_x)
    loss = loss_fn.forward(predictions, batch_y)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement gradient descent, momentum, and Adam optimizers following their mathematical update rules.

### 2. Use
Apply optimizers to train neural networks, observing convergence behavior and training dynamics.

### 3. Understand
Grasp why different optimizers converge at different rates, how adaptive learning rates help, and when to use each optimizer.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize optimizers as iterative algorithms that navigate loss landscapes using gradient information

2. **Core Implementation**: Build SGD, Momentum, and Adam with proper parameter update logic and state management

3. **Pattern Recognition**: Understand the progression from SGD → Momentum (add velocity) → Adam (add adaptive learning rates)

4. **Framework Connection**: See how your optimizers mirror PyTorch's `torch.optim.SGD` and `torch.optim.Adam`

5. **Performance Trade-offs**: Analyze memory overhead (Adam uses 2x parameter memory for momentum/velocity) and convergence speed

---

## Why This Matters

### Production Context

Every trained model uses an optimizer:

- **Computer Vision**: SGD with momentum is standard for training CNNs (ResNet, YOLO)
- **NLP**: Adam is preferred for transformers (BERT, GPT) due to adaptive learning rates
- **Reinforcement Learning**: Adam with gradient clipping for policy optimization
- **Recommendation Systems**: Adam for faster convergence on large datasets

Choosing the right optimizer and learning rate can be the difference between convergence and failure.

### Systems Reality Check

**Performance Note**: SGD is O(n) for n parameters (just one update per parameter). Adam is also O(n) but requires 2x memory to store momentum and velocity states.

**Memory Note**: Adam stores first and second moments for each parameter. For GPT-3 (175B parameters × 4 bytes each), Adam requires 700GB additional memory just for optimizer state!

---

## Implementation Guide

### Prerequisites Check

```bash
tito test 01 05
```

### Development Workflow

```bash
cd modules/source/06_optimizers/
jupyter lab optimizers_dev.py
```

### Step-by-Step Build

#### Step 1: Stochastic Gradient Descent (SGD)

The foundation of optimization:

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        """
        SGD: θ = θ - lr * ∇L
        
        Args:
            parameters: List of Tensors to optimize
            lr: Learning rate (step size)
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Update parameters using gradients"""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data
    
    def zero_grad(self):
        """Clear accumulated gradients"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
```

**Key insight**: SGD moves parameters in the direction that reduces loss (negative gradient direction).

**Learning rate**: Too large causes divergence, too small is slow. Typical range: 0.1 to 0.0001.

#### Step 2: SGD with Momentum

Accelerate optimization using velocity:

```python
class Momentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        Momentum: v = β*v + ∇L, θ = θ - lr*v
        
        Args:
            momentum: Velocity decay factor (typically 0.9)
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]
    
    def step(self):
        """Update with momentum"""
        for param, velocity in zip(self.parameters, self.velocities):
            if param.grad is not None:
                # Update velocity
                velocity[:] = (self.momentum * velocity + 
                              param.grad.data)
                # Update parameter
                param.data -= self.lr * velocity
```

**Why momentum helps**: Accumulates gradients over time, smoothing noisy updates and accelerating convergence in consistent directions.

**Physics analogy**: Like a ball rolling downhill, building momentum in steep directions.

#### Step 3: Adam (Adaptive Moment Estimation)

Adaptive per-parameter learning rates:

```python
class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Adam: Combines momentum + RMSprop
        
        Args:
            betas: (β1, β2) for momentum and RMSprop
            eps: Small constant for numerical stability
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # State for each parameter
        self.m = [np.zeros_like(p.data) for p in parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in parameters]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        """Adam update with bias correction"""
        self.t += 1
        
        for param, m_t, v_t in zip(self.parameters, self.m, self.v):
            if param.grad is not None:
                g = param.grad.data
                
                # Update biased moments
                m_t[:] = self.beta1 * m_t + (1 - self.beta1) * g
                v_t[:] = self.beta2 * v_t + (1 - self.beta2) * (g ** 2)
                
                # Bias correction
                m_hat = m_t / (1 - self.beta1 ** self.t)
                v_hat = v_t / (1 - self.beta2 ** self.t)
                
                # Update parameter
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**Key innovation**: Adapts learning rate per parameter based on gradient history. Parameters with large gradients get smaller effective learning rates.

**Why transformers use Adam**: Sparse gradients (many zeros) benefit from adaptive rates. Adam handles varying gradient scales across parameters.

---

## Testing Your Implementation

### Inline Tests

```python
# Test SGD update
params = [Tensor([1.0, 2.0], requires_grad=True)]
params[0].grad = Tensor([0.1, 0.2])
sgd = SGD(params, lr=0.1)
sgd.step()
assert abs(params[0].data[0] - 0.99) < 1e-6  # 1.0 - 0.1*0.1
print("✓ SGD working")

# Test Momentum
params = [Tensor([1.0], requires_grad=True)]
params[0].grad = Tensor([1.0])
momentum = Momentum(params, lr=0.1, momentum=0.9)
momentum.step()  # First step
params[0].grad = Tensor([1.0])
momentum.step()  # Second step accumulates
print("✓ Momentum working")

# Test Adam
params = [Tensor([1.0], requires_grad=True)]
params[0].grad = Tensor([1.0])
adam = Adam(params, lr=0.001)
adam.step()
assert params[0].data[0] < 1.0  # Should decrease
print("✓ Adam working")
```

### Module Export & Validation

```bash
tito export 06
tito test 06
```

**Expected output**:
```
✓ All tests passed! [18/18]
✓ Module 06 complete!
```

---

## Where This Code Lives

Optimizers drive all training:

```python
from tinytorch.optim import Adam
from tinytorch.nn import MLP
from tinytorch.core.losses import CrossEntropyLoss

# Complete training setup
model = MLP([784, 128, 10])
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        predictions = model.forward(batch_x)
        loss = loss_fn.forward(predictions, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Package structure**:
```
tinytorch/
├── optim/
│   ├── optimizers.py  ← YOUR implementations
├── core/
│   ├── autograd.py  (computes gradients)
```

---

## Systems Thinking Questions

1. **Learning Rate Impact**: What happens if learning rate is too large? Too small? How would you detect each case during training?

2. **Momentum Trade-off**: Momentum accelerates convergence but can overshoot minima. When would you use lower momentum (< 0.9)? Higher momentum (> 0.9)?

3. **Adam Memory Cost**: Adam stores two states per parameter (m, v). For GPT-3 with 175B parameters, how much additional memory does Adam require? Is this significant?

4. **Optimizer Selection**: SGD generalizes better than Adam on some vision tasks. Why might adaptive learning rates hurt generalization? When would you choose SGD over Adam?

5. **Learning Rate Scheduling**: Why do practitioners decay learning rate during training (e.g., reduce by 10x at epochs 30, 60, 90)? What problem does this solve?

---

## Real-World Connections

### Industry Applications

- **ResNet Training**: SGD with momentum (0.9), learning rate 0.1 → 0.01 → 0.001
- **BERT Pre-training**: Adam with lr=1e-4, β1=0.9, β2=0.999
- **GPT-3 Training**: Adam with gradient clipping (norm < 1.0)
- **Stable Diffusion**: AdamW (Adam + weight decay) with lr=1e-4

### Optimization Challenges

- **Saddle Points**: Momentum helps escape flat regions
- **Gradient Noise**: Batch size affects gradient variance; larger batches → more stable
- **Exploding Gradients**: Adam's adaptive rates provide robustness

---

## What's Next?

**Excellent progress!** You've built the algorithms that enable learning. Now you'll tie everything together into a complete training loop.

**Module 07: Training** - Build end-to-end training loops with data loading, validation, and checkpointing

[Continue to Module 07: Training →](07-training.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Optimizers API Reference](../appendices/api-reference.html#optimizers)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)

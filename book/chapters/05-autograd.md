---
title: "Autograd"
description: "Automatic differentiation engine for gradient computation"
module_number: 5
tier: "foundation"
difficulty: "intermediate"
time_estimate: "6-8 hours"
prerequisites: ["01. Tensor", "02. Activations", "03. Layers", "04. Losses"]
next_module: "06. Optimizers"
learning_objectives:
  - "Understand computational graphs and how they enable automatic differentiation"
  - "Implement gradient tracking and backward propagation through operations"
  - "Build differentiable versions of tensor operations using the chain rule"
  - "Recognize how autograd mirrors PyTorch's tape-based differentiation"
  - "Analyze memory usage and computational cost of gradient tracking"
---

# 05. Autograd

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 05 of 20
</div>

**Build automatic differentiation - the engine that makes backpropagation work.**

Difficulty: Intermediate | Time: 6-8 hours | Prerequisites: Modules 01-04

---

## What You'll Build

Autograd (automatic differentiation) is the "magic" behind neural network training. It automatically computes gradients for any differentiable function, eliminating manual derivative calculations.

By the end of this module, you'll have implemented:

- **Computational Graphs** - Track operation dependencies for gradient computation
- **Backward Pass** - Propagate gradients from outputs to inputs using chain rule
- **Gradient Accumulation** - Sum gradients from multiple paths through the graph
- **Differentiable Operations** - Extend all tensor operations to support autograd

### Example Usage

```python
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.tensor import Tensor

# Enable gradient tracking
enable_autograd()

# Create tensors with gradient tracking
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Forward pass - build computational graph
z = x * y + x ** 2  # z = 2*3 + 2^2 = 10

# Backward pass - compute gradients
z.backward()

print(x.grad)  # dz/dx = y + 2x = 3 + 4 = 7
print(y.grad)  # dz/dy = x = 2
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement computational graph construction during forward pass and gradient propagation during backward pass using the chain rule.

### 2. Use
Apply autograd to neural network training - compute loss gradients with respect to all parameters automatically.

### 3. Understand
Grasp how frameworks like PyTorch/TensorFlow compute gradients efficiently, and understand memory/performance trade-offs of gradient tracking.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize autograd as reverse-mode automatic differentiation—a tape-based system that builds computational graphs dynamically

2. **Core Implementation**: Build gradient tracking, backward propagation, and differentiable operations following mathematical chain rule

3. **Pattern Recognition**: Understand the computation-then-differentiation pattern that enables "define-by-run" frameworks

4. **Framework Connection**: See how your autograd mirrors PyTorch's `torch.autograd` and TensorFlow's `tf.GradientTape`

5. **Performance Trade-offs**: Analyze memory overhead of gradient tracking (2x memory for forward + backward) and computational cost (backward pass typically ~2x forward pass time)

---

## Why This Matters

### Production Context

Autograd enables all modern deep learning:

- **Training**: All framework optimizers rely on autograd to compute parameter gradients
- **Backprop**: Neural networks train via gradient descent, which requires gradients
- **Research**: New architectures can be trained without deriving gradients manually
- **Debugging**: Gradient checks validate implementations by comparing numerical vs automatic gradients

Without autograd, every new layer would require manual derivative implementation—infeasible for complex models.

### Systems Reality Check

**Performance Note**: Backward pass is typically 2-3x slower than forward pass. A forward pass computes outputs; backward computes gradients for all intermediate operations.

**Memory Note**: Autograd stores all intermediate tensors from forward pass for backward pass. A 100-layer network stores 100+ activation tensors during training. This is why training uses more memory than inference.

---

## Implementation Guide

### Prerequisites Check

```bash
tito test 01 02 03 04
```

### Development Workflow

```bash
cd modules/source/05_autograd/
jupyter lab autograd_dev.py
```

### Step-by-Step Build

#### Step 1: Computational Graph Node

Track operation dependencies:

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None  # Accumulated gradient
        self.grad_fn = None  # Backward function
        self._prev = []  # Parent tensors in computation graph
```

**Design insight**: `grad_fn` stores the backward function. `_prev` stores parent tensors for graph traversal.

#### Step 2: Operation Backward Functions

Implement chain rule for each operation:

```python
class AddBackward:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def backward(self, grad_output):
        """
        z = x + y
        dL/dx = dL/dz * dz/dx = grad_output * 1
        dL/dy = dL/dz * dz/dy = grad_output * 1
        """
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = Tensor(grad_output.data)
            else:
                self.x.grad = self.x.grad + grad_output
        
        if self.y.requires_grad:
            if self.y.grad is None:
                self.y.grad = Tensor(grad_output.data)
            else:
                self.y.grad = self.y.grad + grad_output
```

**Chain rule**: Gradient flows backward by multiplying output gradient by local derivative.

#### Step 3: Differentiable Operations

Make operations gradient-aware:

```python
# Original (no gradients)
def add(x, y):
    return Tensor(x.data + y.data)

# Differentiable (tracks gradients)
def add(x, y):
    result = Tensor(x.data + y.data, requires_grad=True)
    if x.requires_grad or y.requires_grad:
        result.grad_fn = AddBackward(x, y)
        result._prev = [x, y]
    return result
```

**Pattern**: Every operation creates backward function if inputs require gradients.

#### Step 4: Backward Pass

Traverse graph in reverse topological order:

```python
def backward(tensor):
    """Compute gradients via reverse-mode autodiff"""
    # Topological sort
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v._prev:
                build_topo(parent)
            topo.append(v)
    
    build_topo(tensor)
    
    # Initialize gradient
    tensor.grad = Tensor(np.ones_like(tensor.data))
    
    # Backpropagate
    for node in reversed(topo):
        if node.grad_fn is not None:
            node.grad_fn.backward(node.grad)
```

**Key insight**: Topological sort ensures gradients propagate in correct order (outputs to inputs).

---

## Testing Your Implementation

### Inline Tests

```python
# Test simple gradient
x = Tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4
y.backward()
assert abs(x.grad.data[0] - 4.0) < 1e-6  # dy/dx = 2x = 4
print("✓ Basic gradient working")

# Test chain rule
x = Tensor([3.0], requires_grad=True)
y = x * 2
z = y + 5
z.backward()
assert abs(x.grad.data[0] - 2.0) < 1e-6  # dz/dx = 2
print("✓ Chain rule working")

# Test gradient accumulation
x = Tensor([1.0], requires_grad=True)
y = x + x  # Two paths from x
y.backward()
assert abs(x.grad.data[0] - 2.0) < 1e-6  # Gradients sum
print("✓ Accumulation working")
```

### Module Export & Validation

```bash
tito export 05
tito test 05
```

**Expected output**:
```
✓ All tests passed! [25/25]
✓ Module 05 complete!
```

---

## Where This Code Lives

Autograd powers all training:

```python
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.nn import MLP

# Enable autograd
enable_autograd()

# Training step
model = MLP([784, 128, 10])
loss_fn = CrossEntropyLoss()

predictions = model.forward(x)
loss = loss_fn.forward(predictions, y)
loss.backward()  # ← YOUR autograd computes all gradients!

# Now model parameters have .grad populated
for param in model.parameters():
    print(param.grad)  # Ready for optimizer
```

**Package structure**:
```
tinytorch/
├── core/
│   ├── autograd.py  ← YOUR implementation
│   ├── tensor.py    (uses autograd)
│   ├── losses.py    (differentiable via autograd)
```

---

## Systems Thinking Questions

1. **Memory Trade-off**: Autograd stores all intermediate tensors during forward pass. For a 100-layer ResNet, how much memory overhead does this add? Can this be reduced?

2. **Computational Cost**: Why is backward pass typically 2-3x slower than forward? Which operations are more expensive in backward (matmul, softmax, conv2d)?

3. **Gradient Accumulation**: Why do gradients sum when multiple paths lead to the same parameter? Give an example from a neural network (hint: weight sharing).

4. **Numerical Stability**: What happens if gradients overflow during backprop? How do frameworks handle this (gradient clipping)?

5. **Framework Comparison**: PyTorch uses "define-by-run" (dynamic graphs). TensorFlow 1.x used "define-then-run" (static graphs). What are the trade-offs?

---

## Real-World Connections

### Industry Applications

- **All Neural Network Training**: Every trained model uses autograd for gradient computation
- **Neural Architecture Search**: AutoML systems explore architectures without manual derivative implementation
- **Physics Simulations**: Differentiable rendering, fluid dynamics use autograd for inverse problems
- **Scientific Computing**: JAX enables automatic differentiation for scientific code

### Advanced Patterns

- **Second-order Gradients**: Autograd can differentiate gradients (for meta-learning, Hessian computation)
- **Gradient Checkpointing**: Trade computation for memory by recomputing activations during backward
- **Mixed Precision**: Use FP16 for forward pass, FP32 for gradients to prevent underflow

---

## What's Next?

**Amazing work!** You've built the engine that makes neural network training possible. Next, you'll implement optimizers that use these gradients to update parameters.

**Module 06: Optimizers** - Build SGD, Momentum, and Adam optimizers that use gradients to train networks

[Continue to Module 06: Optimizers →](06-optimizers.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Autograd API Reference](../appendices/api-reference.html#autograd)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)

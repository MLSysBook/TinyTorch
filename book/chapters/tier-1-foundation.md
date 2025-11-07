# Foundation Tier: Build the Engine

**Modules 01-07 | Estimated Time: 25-35 hours**

---

## Overview

The Foundation Tier teaches you to build the mathematical infrastructure of neural networks from first principles. These seven modules implement the core components that power every modern ML framework.

By completing Foundation, you'll understand not just how to *use* PyTorch or TensorFlow, but how these frameworks work internally—from tensor operations to automatic differentiation to training loops.

---

## What You'll Build

### Core Mathematical Infrastructure

**Module 01: Tensor**  
N-dimensional arrays and operations - the foundation of all ML computations

**Module 02: Activations**  
Nonlinear functions that enable neural networks to learn complex patterns

**Module 03: Layers**  
Composable building blocks (Linear, Sequential) for network architectures

**Module 04: Losses**  
Objective functions that measure model performance

**Module 05: Autograd**  
Automatic differentiation - the engine that makes backpropagation work

**Module 06: Optimizers**  
Learning algorithms (SGD, Adam) that update model parameters

**Module 07: Training**  
Complete training loops that tie everything together

---

## Learning Approach

### Build → Use → Understand

In Foundation Tier, you'll follow this pattern for each module:

1. **Build** - Implement the component from scratch using NumPy
2. **Use** - Apply it to real problems with real data
3. **Understand** - Grasp the systems-level implications and design trade-offs

This approach ensures you gain both practical implementation skills and deep conceptual understanding.

---

## Why This Matters

### Industry Relevance

Every ML framework (PyTorch, TensorFlow, JAX) provides these same primitives. Understanding how they work internally enables you to:

- Debug when models don't train correctly
- Optimize bottlenecks in training pipelines
- Extend frameworks with custom operations
- Make informed architecture decisions

### Systems Understanding

Building these components yourself reveals insights impossible to gain from high-level APIs:

- Why certain operations dominate compute time
- How memory usage scales with model size
- What optimizers do beyond calling `.step()`
- How gradients flow through computation graphs

---

## Module Roadmap

### Beginner (Modules 01-02)

**01. Tensor** - 4-6 hours  
Build N-dimensional arrays with arithmetic operations, shape manipulation, and broadcasting

**02. Activations** - 3-4 hours  
Implement ReLU, Sigmoid, Tanh, Softmax, and GELU activation functions

### Intermediate (Modules 03-07)

**03. Layers** - 4-5 hours  
Create Linear layers and Sequential containers for composable architectures

**04. Losses** - 3-4 hours  
Build MSE, Cross-Entropy, and Binary Cross-Entropy loss functions

**05. Autograd** - 6-8 hours  
Implement automatic differentiation with computational graphs and backward passes

**06. Optimizers** - 4-5 hours  
Build SGD, Momentum, and Adam optimizers with learning rate schedules

**07. Training** - 4-5 hours  
Create complete training loops with batching, validation, and checkpointing

---

## Tier Milestone

**After completing Module 07**, you'll unlock the **1957: Rosenblatt's Perceptron** milestone.

This historical demonstration uses *your* implementations to recreate the first trainable neural network. You'll classify handwritten digits using only the components you built - proving that your code works!

```bash
python milestones/01_1957_perceptron/perceptron_digits.py
```

**Expected Results:**
- Training accuracy: ~85-90%
- Test accuracy: ~82-87%
- Training time: 2-3 minutes

This milestone validates that you've successfully built production-quality ML infrastructure.

---

## Prerequisites

**Before starting Foundation Tier:**

- Python 3.8+ installed
- Basic NumPy knowledge (arrays, indexing, broadcasting)
- TinyTorch environment set up

**Verify your setup:**

```bash
tito system doctor
```

All checks should pass before beginning Module 01.

---

## Getting Started

**Ready to build the foundation of ML systems?**

Begin with Module 01: Tensor - the data structure that powers all neural network computations.

[Start Module 01: Tensor →](01-tensor.html)

---

## Additional Resources

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Quick Start Guide](../quickstart-guide.html)
- [Read Course Introduction](00-introduction.html)

**Related Content:**
- [Historical Milestones](milestones.html) - See how your work connects to ML history
- [Testing Framework](../testing-framework.html) - Understand how to validate your implementations
- [Progress Tracking](../learning-progress.html) - Monitor your learning journey


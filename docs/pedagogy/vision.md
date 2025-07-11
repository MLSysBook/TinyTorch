# Tiny🔥Torch: A Systems Engineering Lab for ML Frameworks

**TinyTorch** is the hands-on companion to the [*Machine Learning Systems*](https://mlsysbook.ai) textbook. It adopts a **systems-first approach** to machine learning—not just teaching what ML models are, but how they execute on real infrastructure. Inspired by operating systems and compiler courses that build systems from first principles, TinyTorch guides students through the construction of a complete ML training and inference runtime, from scratch.

TinyTorch is not about using PyTorch—it’s about writing the components that PyTorch is built on. Students implement tensors, autograd engines, data pipelines, optimizers, profilers, and deployment scaffolding, all while confronting the design trade-offs that shape modern ML systems.

---

## Core Philosophy

ML is often introduced as a series of mathematical abstractions: matrix operations, activation functions, gradients. But systems engineers know that models don’t run equations—they run **loops over data**, under real-world constraints like memory hierarchies, compute bottlenecks, and hardware interfaces.

TinyTorch starts at the loop level—where theory meets implementation—and builds upward to full system complexity. At each step, it emphasizes the **systems engineering challenges** introduced by ML concepts, helping students understand not just what ML does, but how it works under the hood.

---

## Learning Approach: “Loops to Systems”

### The Central Insight

The real systems challenges emerge when students ask:

* **How do these loops actually execute?**
* **What are the memory access patterns?**
* **Where are the performance bottlenecks?**
* **How do we make this scale to large workloads?**

### Example: DL Primer → DNN Architectures

In the *DL Primer* module, the focus is not on the math of backpropagation, but on:

* **Data layout**: How tensors flow through memory
* **Compute patterns**: How nested loops affect system performance
* **Interface design**: How layers compose into extensible abstractions

In the *DNN Architectures* module (MLPs, CNNs), the systems lens shifts to:

* **Modular abstractions**: Composable, pluggable layer design
* **Memory locality**: Efficient storage and reuse of activations and weights
* **Performance trade-offs**: How architectural design choices shape system behavior

---

## Module Design Principles

### 1. Systems Engineering Focus

Every module begins by asking: *“What systems challenges does this concept introduce?”* Students learn to frame ML operations in terms of memory hierarchies, computational throughput, interface stability, and scalability.

### 2. Progressive Complexity

Each module builds on the previous one:

* Start with forward pass execution (data flow)
* Add gradient computation (graph tracking and backpropagation)
* Introduce parameter updates (optimizers and training loops)
* Scale to large data (I/O pipelines, batching, parallelism)

### 3. Hands-On Implementation

Students write the systems themselves—not wrappers or stubs. They make real decisions about:

* Memory layout and management
* API design and abstraction boundaries
* Performance bottlenecks and instrumentation
* Debugging multi-stage systems under constraints

### 4. Real-World Constraints

Every implementation addresses:

* **Performance**: Runtime efficiency, vectorization, and bottlenecks
* **Memory**: Allocation, reuse, and growth under training workloads
* **Scalability**: Behavior on large models and datasets
* **Maintainability**: Clean abstractions and extensibility for new components

---

## Target Learning Outcomes

By the end of TinyTorch, students should be able to:

1. **Understand the internal structure** of ML frameworks like PyTorch and TensorFlow
2. **Debug performance and correctness issues** in ML pipelines
3. **Design modular ML infrastructure components** with clean interfaces
4. **Analyze trade-offs** in accuracy, runtime, and memory footprint
5. **Apply systems thinking** to the design and evaluation of ML code

---

## Module → Package Export Conventions

**🎓 Teaching vs. 🔧 Building Structure**: TinyTorch separates learning organization from production code organization:

| **Learning Module** | **Export Target** | **Purpose** |
|---------------------|-------------------|-------------|
| `modules/setup/` | `tinytorch/core/utils.py` | System utilities, environment setup |
| `modules/tensor/` | `tinytorch/core/tensor.py` | Core tensor data structure |
| `modules/mlp/` | `tinytorch/nn/linear.py` | Linear layers and MLPs |
| `modules/cnn/` | `tinytorch/nn/conv.py` | Convolutional layers |
| `modules/autograd/` | `tinytorch/core/autograd.py` | Automatic differentiation |
| `modules/dataloader/` | `tinytorch/core/dataloader` | Data loading and processing |
| `modules/training/` | `tinytorch/optim/` | Optimizers and training loops |
| `modules/profiling/` | `tinytorch/profiling/` | Performance analysis tools |

**Why this separation?**
- **Learning**: Modules progress conceptually (setup → tensors → neural networks)
- **Building**: Package organized functionally (core, nn, data, optim)
- **Real-world**: Mirrors professional ML framework structure (like PyTorch)

---

## Module Progression Logic

TinyTorch aligns with Chapters 5 through 13 of the *Machine Learning Systems* textbook. While each module can be tackled independently, the design encourages a sequential path that mirrors the increasing complexity of ML systems development.

### Foundation (Modules 0–1)

* **Setup**: The development environment as a system
* **Tensor**: Core data structures and memory management primitives

### Core ML Systems (Modules 2–4)

* **MLP**: Forward pass mechanics and loop construction
* **CNN**: Convolution as a system design challenge (locality, reuse)
* **Autograd**: Graph construction, gradient tracking, and memory reuse

### Data & Training Systems (Modules 5–6)

* **Data**: Input pipelines, prefetching, and transformation as systems problems
* **Training**: Coordinating forward, backward, update, and checkpoint stages

### Production Systems (Modules 7–12)

* **Config**: Managing experimental scaffolding
* **Profiling**: Instrumentation and performance diagnostics
* **Compression**: Quantization and pruning from a systems perspective
* **Kernels**: Custom kernel interfaces and tuning
* **Benchmarking**: Building metrics pipelines and evaluation frameworks
* **MLOps**: Deployment, monitoring, and system lifecycle concerns

---

## Success Metrics

Students succeed when they can:

* **Explain** design choices in frameworks like PyTorch
* **Predict** where performance bottlenecks will occur
* **Implement** new layer types or optimization algorithms
* **Debug** issues in multi-stage training systems
* **Reason** about trade-offs across the full ML systems stack

---

## Course Development Ground Truth

This document serves as the canonical guide for TinyTorch module development. When in doubt about what a module should cover, how it should be structured, or how deeply it should engage with implementation, refer back to these guiding principles.

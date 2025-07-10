# Tiny🔥Torch Vision & Pedagogical Approach

## Core Philosophy

TinyTorch is a **systems-first approach** to understanding machine learning frameworks. Rather than teaching ML theory in isolation, we focus on the **systems engineering challenges** that arise when implementing ML concepts at scale.

## Learning Approach: "Loops to Systems"

### The Central Insight
Students often learn ML algorithms as mathematical abstractions (matrix multiplications, activation functions, gradients). But the real systems challenges emerge when you ask:

- **How do these loops actually execute?**
- **What are the memory access patterns?**
- **Where are the performance bottlenecks?**
- **How do you make this scale?**

### Example: DL Primer → DNN Architectures
When we teach **DL Primer** concepts (forward pass, backprop), the focus isn't on the math - it's on:
- **Data layout**: How tensors flow through memory
- **Compute patterns**: How nested loops translate to system performance
- **Interface design**: How layers compose and communicate

When we teach **DNN Architectures** (MLPs, CNNs), we're really teaching:
- **System abstractions**: How to design modular, composable components
- **Memory management**: How activation maps and weights are stored/accessed
- **Computational efficiency**: Why certain architectural choices matter for systems

## Module Design Principles

### 1. Systems Engineering Focus
Every module should answer: "What systems challenges does this concept introduce?"

### 2. Progressive Complexity
Start with simple forward passes, then add complexity:
- Forward pass (understand data flow)
- Backward pass (understand gradient flow)
- Optimization (understand parameter updates)
- Scale (understand performance)

### 3. Hands-On Implementation
Students build the actual systems, not just use them. This forces confrontation with:
- Memory layout decisions
- API design choices
- Performance trade-offs
- Debugging complex systems

### 4. Real-World Constraints
Every implementation should consider:
- **Performance**: How fast does it run?
- **Memory**: How much RAM does it use?
- **Scalability**: How does it handle larger problems?
- **Maintainability**: How easy is it to extend?

## Target Learning Outcomes

By the end of TinyTorch, students should be able to:

1. **Read PyTorch/TensorFlow source code** and understand the systems decisions
2. **Debug performance issues** in ML training pipelines
3. **Design new ML system components** with proper abstractions
4. **Make informed trade-offs** between accuracy, speed, and memory
5. **Think like a systems engineer** when approaching ML problems

## Module Progression Logic

### Foundation (Modules 0-1)
- **Setup**: Development environment as a system
- **Tensor**: Core data structures and memory management

### Core ML Systems (Modules 2-4)
- **MLP**: Basic forward pass systems, loop structures
- **CNN**: Advanced memory patterns, convolution as systems challenge
- **Autograd**: Complex graph computation, memory management for gradients

### Data & Training Systems (Modules 5-6)
- **Data**: I/O pipelines, batching, preprocessing as systems problems
- **Training**: Putting it all together, optimization loops, checkpointing

### Production Systems (Modules 7-12)
- **Config**: Experiment management systems
- **Profiling**: Performance analysis tools
- **Compression**: Model optimization techniques
- **Kernels**: Custom compute optimization
- **Benchmarking**: Measurement and comparison frameworks
- **MLOps**: Deployment and monitoring systems

## Success Metrics

Students succeed when they can:
- **Explain** why PyTorch made certain design decisions
- **Predict** where performance bottlenecks will occur
- **Implement** new layer types or optimization algorithms
- **Debug** complex training issues by understanding the underlying systems

---

*This document serves as our "ground truth" for course design decisions. When in doubt about module content or progression, refer back to these principles.* 
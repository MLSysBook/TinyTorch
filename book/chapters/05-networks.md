---
title: "Networks"
description: "Neural network architectures and composition patterns"
difficulty: "⭐⭐⭐"
time_estimate: "5-6 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# Module: Networks

```{div} badges
⭐⭐⭐ | ⏱️ 5-6 hours
```


## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 5-7 hours
- **Prerequisites**: Tensor, Activations, Layers modules
- **Next Steps**: CNN, Training modules

Compose layers into complete neural network architectures with powerful visualizations. This module teaches you that neural networks are function composition at scale—taking simple building blocks and combining them into systems capable of learning complex patterns and making intelligent decisions.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Master function composition**: Understand how networks are built as `f(x) = layer_n(...layer_2(layer_1(x)))`
- **Design neural architectures**: Build MLPs, classifiers, and regressors from compositional principles
- **Visualize network behavior**: Use advanced plotting to understand data flow and architectural decisions
- **Analyze architectural trade-offs**: Compare depth vs width, activation choices, and design patterns
- **Apply networks to real tasks**: Create appropriate architectures for classification and regression problems

## 🧠 Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Compose layers into complete network architectures using function composition principles
2. **Use**: Apply networks to classification and regression tasks, visualizing behavior and data flow
3. **Optimize**: Analyze architectural choices, compare design patterns, and understand performance trade-offs

## 📚 What You'll Build

### Sequential Network Architecture
```python
# Function composition in action
network = Sequential([
    Dense(784, 128),    # Input transformation
    ReLU(),             # Nonlinearity
    Dense(128, 64),     # Feature compression
    ReLU(),             # More nonlinearity
    Dense(64, 10),      # Classification head
    Sigmoid()           # Probability outputs
])

# Single forward pass processes entire batch
x = Tensor([[...]])  # Input batch
predictions = network(x)  # End-to-end inference
```

### Specialized Network Builders
```python
# MLP for multi-class classification
classifier = create_mlp(
    input_size=784,           # Flattened 28x28 images
    hidden_sizes=[256, 128],  # Two hidden layers
    output_size=10,           # 10 digit classes
    activation=ReLU,          # Hidden layer activation
    output_activation=Sigmoid  # Probability outputs
)

# Regression network for continuous prediction
regressor = create_regression_network(
    input_size=13,       # Housing features
    hidden_sizes=[64, 32], # Progressive compression
    output_size=1        # Single price prediction
)

# Binary classification with appropriate architecture
binary_classifier = create_classification_network(
    input_size=100,
    num_classes=2,
    architecture='deep'  # Optimized for binary tasks
)
```

### Advanced Network Analysis
```python
# Comprehensive architecture visualization
visualize_network_architecture(network)
# Shows: layer types, connections, parameter counts, data flow

# Behavior analysis with real data
analyze_network_behavior(network, sample_data)
# Shows: activation patterns, layer statistics, transformation analysis

# Architectural comparison
compare_networks([shallow_net, deep_net, wide_net])
# Shows: performance characteristics, complexity trade-offs
```

## 🚀 Getting Started

### Prerequisites
Ensure you have mastered the foundational building blocks:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify all prerequisite modules
tito test --module tensor
tito test --module activations
tito test --module layers
```

### Development Workflow
1. **Open the development file**: `modules/source/05_networks/networks_dev.py`
2. **Implement Sequential class**: Build the composition framework for chaining layers
3. **Create network builders**: Implement MLPs and specialized architectures
4. **Add visualization tools**: Build plotting functions for network analysis
5. **Test with real scenarios**: Apply networks to classification and regression tasks
6. **Export and verify**: `tito export --module networks && tito test --module networks`

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify architectural correctness:

```bash
# TinyTorch CLI (recommended)
tito test --module networks

# Direct pytest execution
python -m pytest tests/ -k networks -v
```

### Test Coverage Areas
- ✅ **Sequential Composition**: Verify layers chain correctly with proper data flow
- ✅ **Network Builders**: Test MLP and specialized network creation functions
- ✅ **Shape Consistency**: Ensure networks handle various input shapes and batch sizes
- ✅ **Visualization Functions**: Verify plotting and analysis tools work correctly
- ✅ **Real-world Applications**: Test networks on classification and regression tasks

### Inline Testing & Visualization
The module includes comprehensive educational feedback and visual analysis:
```python
# Example inline test output
🔬 Unit Test: Sequential network composition...
✅ Layers chain correctly with proper data flow
✅ Forward pass produces expected output shapes
✅ Network handles batch processing correctly
📈 Progress: Sequential Networks ✓

# Visualization feedback
📊 Generating network architecture visualization...
📈 Showing data flow through 3-layer MLP
📊 Layer analysis: 784→128→64→10 parameter flow
```

### Manual Testing Examples
```python
from tinytorch.core.tensor import Tensor
from networks_dev import Sequential, create_mlp
from layers_dev import Dense
from activations_dev import ReLU, Sigmoid

# Test network composition
network = Sequential([
    Dense(10, 5),
    ReLU(),
    Dense(5, 2),
    Sigmoid()
])

# Forward pass
x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
output = network(x)
print(f"Network output: {output.data}, Shape: {output.shape}")

# Test MLP builder
mlp = create_mlp(input_size=4, hidden_sizes=[8, 4], output_size=2)
test_input = Tensor([[1.0, 2.0, 3.0, 4.0]])
prediction = mlp(test_input)
print(f"MLP prediction: {prediction.data}")
```

## 🎯 Key Concepts

### Real-World Applications
- **Image Classification**: ResNet and VGG architectures use sequential composition of convolutional and dense layers
- **Natural Language Processing**: Transformer architectures compose attention layers with feed-forward networks
- **Recommendation Systems**: Deep collaborative filtering uses MLPs to learn user-item interactions
- **Autonomous Systems**: Neural networks in self-driving cars compose perception, planning, and control layers

### Function Composition Theory
- **Mathematical Foundation**: Networks implement nested function composition `f_n(f_{n-1}(...f_1(x)))`
- **Universal Approximation**: MLPs with sufficient width can approximate any continuous function
- **Depth vs Width Trade-offs**: Deep networks learn hierarchical features, wide networks increase expressivity
- **Architectural Inductive Biases**: Network structure encodes assumptions about the problem domain

### Visualization and Analysis
- **Architecture Visualization**: Understand network structure through visual representation
- **Data Flow Analysis**: Track how information transforms through each layer
- **Activation Pattern Analysis**: Visualize what each layer learns to represent
- **Comparative Analysis**: Understand trade-offs between different architectural choices

### Design Patterns and Best Practices
- **Progressive Dimensionality**: Common pattern of gradually reducing dimensions toward output
- **Activation Placement**: Standard practice of activation after each linear transformation
- **Output Layer Design**: Task-specific final layers (sigmoid for binary, softmax for multi-class)
- **Network Depth Guidelines**: Balance between expressivity and training difficulty

## 🎉 Ready to Build?

You're about to master the art of neural architecture design! This is where the magic happens—taking simple mathematical building blocks and composing them into systems capable of recognizing images, understanding language, and making intelligent decisions.

Every breakthrough in AI, from AlexNet to GPT, started with someone thoughtfully composing layers into powerful architectures. You're about to learn those same composition principles and build networks that can solve real problems!

 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/05_networks/networks_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/05_networks/networks_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/05_networks/networks_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/04_layers.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/06_cnn.html" title="next page">Next Module →</a>
</div>

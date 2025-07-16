---
title: "Networks"
description: "Neural network architectures and composition patterns"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# 🧠 Module 3: Networks - Neural Network Architectures
---
**Course Navigation:** [Home](../intro.html) → [Module 5: 05 Networks](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐⭐⭐ | <strong>Time:</strong> 5-6 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 5-7 hours
- **Prerequisites**: Tensor, Activations, Layers modules
- **Next Steps**: Training, CNN modules

**Compose layers into complete neural network architectures with powerful visualizations**

## 🎯 Learning Objectives

After completing this module, you will:
- Understand networks as function composition: `f(x) = layer_n(...layer_2(layer_1(x)))`
- Build common architectures (MLP, CNN) from layers
- Visualize network structure and data flow
- See how architecture affects capability
- Master forward pass inference (no training yet!)

> **Note:**
> **MLP (Multi-Layer Perceptron) is not a fundamental building block, but a use case of composing Dense layers and activations in sequence.**
> In TinyTorch, you will learn to build MLPs by composing primitives, not as a separate module. This approach helps you see that all architectures (MLP, CNN, etc.) are just patterns of composition, not new primitives.

## 🧠 Build → Use → Understand

This module follows the TinyTorch pedagogical framework:

1. **Build**: Compose layers into complete networks
2. **Use**: Create different architectures and run inference
3. **Understand**: How architecture design affects network behavior

## 📚 What You'll Build

### **Sequential Network**
```python
# Basic network composition
network = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
    Sigmoid()
])

# Forward pass
x = Tensor([[1.0, 2.0, 3.0, ...]])  # Input data
output = network(x)  # Network prediction
```

### **MLP (Multi-Layer Perceptron)**
```python
# Create MLP for classification
mlp = create_mlp(
    input_size=784,      # 28x28 image
    hidden_sizes=[128, 64],  # Hidden layers
    output_size=10,      # 10 classes
    activation=ReLU,
    output_activation=Sigmoid
)
```

### **Specialized Networks**
```python
# Classification network
classifier = create_classification_network(
    input_size=100, num_classes=2
)

# Regression network  
regressor = create_regression_network(
    input_size=13, output_size=1
)
```

## 🎨 Visualization Features

This module includes powerful visualizations to help you understand:

### **Network Architecture Visualization**
- **Layer-by-layer structure**: See how layers connect
- **Color-coded layers**: Different colors for Dense, ReLU, Sigmoid, etc.
- **Connection arrows**: Visualize data flow between layers
- **Layer details**: Input/output sizes and parameters

### **Data Flow Visualization**
- **Shape transformations**: See how tensor shapes change through the network
- **Activation patterns**: Visualize intermediate layer outputs
- **Statistics tracking**: Mean, std, and distribution of activations
- **Layer analysis**: Understand what each layer learns

### **Network Comparison**
- **Side-by-side analysis**: Compare different architectures
- **Performance metrics**: Output distributions and statistics
- **Architectural insights**: Layer type distributions and complexity

### **Behavior Analysis**
- **Input-output relationships**: How inputs map to outputs
- **Activation patterns**: Layer-by-layer activation analysis
- **Network depth**: Understanding the role of depth vs width
- **Practical insights**: Real-world application considerations

## 🚀 Getting Started

### Prerequisites
- Complete Module 1: Tensor ✅
- Complete Module 2: Layers ✅
- Understand basic function composition
- Familiar with matplotlib for visualizations

### Quick Start
```bash
# Navigate to the networks module
cd modules/networks

# Work in the development notebook
jupyter notebook networks_dev.ipynb

# Or work in the Python file
code networks_dev.py
```

## 📖 Module Structure

```
modules/networks/
├── networks_dev.py           # Main development file (work here!)
├── networks_dev.ipynb        # Jupyter notebook version
├── tests/
│   └── test_networks.py      # Comprehensive tests
├── README.md                # This file
└── solutions/               # Reference implementations (if stuck)
```

## 🎓 Learning Path

### Step 1: Sequential Network (Function Composition)
- Understand `f(x) = layer_n(...layer_1(x))`
- Implement basic network composition
- Test with simple examples

### Step 2: Network Visualization
- Visualize network architectures
- Understand data flow through networks
- Compare different network designs

### Step 3: Common Architectures
- Build MLPs for different tasks
- Create classification networks
- Design regression networks

### Step 4: Behavior Analysis
- Analyze network behavior with different inputs
- Understand architectural trade-offs
- See how design affects capability

### Step 5: Practical Applications
- Build networks for real problems
- Understand classification vs regression
- See how architecture matches task

## 🧪 Testing Your Implementation

### Module-Level Tests
```bash
# Run comprehensive tests
python -m pytest tests/test_networks.py -v

# Quick test
python -c "from networks_dev import Sequential; print('✅ Networks working!')"
```

### Package-Level Tests
```bash
# Export to package
python ../../bin/tito sync

# Test integration
python ../../bin/tito test --module networks
```

## 🎯 Key Concepts

### **Function Composition**
- Networks as `f(x) = g(h(x))`
- Each layer is a function
- Composition creates complex behavior

### **Architecture Design**
- **Depth**: Number of layers
- **Width**: Number of neurons per layer
- **Activation**: Nonlinearity choices
- **Output**: Task-specific final layer

### **Visualization Benefits**
- **Debugging**: See where things go wrong
- **Understanding**: Visualize complex transformations
- **Design**: Compare different architectures
- **Intuition**: Build mental models of networks

### **Practical Considerations**
- **Input size**: Must match your data
- **Output size**: Must match your task
- **Hidden layers**: Balance complexity vs overfitting
- **Activation functions**: Choose based on task

## 🔍 Common Issues

### **Import Errors**
```python
# Make sure you're in the right directory
import sys
sys.path.append('../../')
from modules.layers.layers_dev import Dense
from modules.activations.activations_dev import ReLU, Sigmoid
```

### **Shape Mismatches**
```python
# Check layer sizes match
layer1 = Dense(3, 4)    # 3 inputs, 4 outputs
layer2 = Dense(4, 2)    # 4 inputs (matches layer1 output), 2 outputs
```

### **Visualization Issues**
```python
# Make sure matplotlib is installed
pip install matplotlib seaborn

# Check if plots are disabled during testing
if _should_show_plots():
    # Your visualization code
    pass
```

## 🎉 Success Criteria

You've successfully completed this module when:
- ✅ All tests pass (`pytest tests/test_networks.py`)
- ✅ You can build and visualize different network architectures
- ✅ You understand how architecture affects network behavior
- ✅ You can create networks for classification and regression tasks
- ✅ Package export works (`tito test --module networks`)

## 🚀 What's Next

After completing this module, you're ready for:
- **Module 4: Training** - Learn how networks learn from data
- **Module 5: Data** - Work with real datasets
- **Module 6: Applications** - Solve real-world problems

## 🤝 Getting Help

- Check the tests for examples of expected behavior
- Look at the solutions/ directory if you're stuck
- Review the pedagogical principles in `docs/pedagogy/`
- Remember: Build → Use → Understand!

## 🎨 Visualization Examples

### Network Architecture
```
Input → Dense(784,128) → ReLU → Dense(128,64) → ReLU → Dense(64,10) → Sigmoid → Output
```

### Data Flow
```
(1,784) → (1,128) → (1,128) → (1,64) → (1,64) → (1,10) → (1,10)
```

### Layer Analysis
- **Dense layers**: Linear transformations
- **ReLU**: Introduces nonlinearity
- **Sigmoid**: Outputs probabilities

**Build powerful neural networks with beautiful visualizations!** 🚀 

---

## 🚀 Ready to Build?

**🚀 [Launch in Binder](https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/05_networks/networks_dev.ipynb)** *(Live Jupyter environment)*

**📓 [Open in Colab](https://colab.research.google.com/github/MLSysBook/TinyTorch/blob/main/modules/source/05_networks/networks_dev.ipynb)** *(Google's cloud environment)*

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/04_layers.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/06_cnn.html" title="next page">Next Module →</a>
</div>

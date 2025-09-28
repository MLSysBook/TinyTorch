# 🔥 Module: Layers

## 📊 Module Info
- **Difficulty**: ⭐⭐ Intermediate
- **Time Estimate**: 4-5 hours
- **Prerequisites**: Tensor, Activations modules
- **Next Steps**: Loss Functions module

Build the fundamental transformations that compose into neural networks. This module teaches you that layers are simply functions that transform tensors, and neural networks are just sophisticated function composition using these building blocks.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Understand layers as mathematical functions**: Recognize that layers transform tensors through well-defined mathematical operations
- **Implement Linear layers + Module base + Flatten**: Complete neural network building blocks
- **Integrate activation functions**: Combine linear layers with nonlinear activations to enable complex pattern learning
- **Compose simple building blocks**: Chain layers together to create complete neural network architectures
- **Debug layer implementations**: Use shape analysis and mathematical properties to verify correct implementation

## 🧠 Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement Linear layers, Module base class, and Flatten operation
2. **Use**: Build complete neural networks with parameter tracking
3. **Reflect**: Understand how Module base enables automatic parameter management

## 📚 What You'll Build

### 🎯 **COMPLETE BUILDING BLOCKS: Everything You Need**
```python
# Linear layer: fundamental building block
class MLP(Module):  # Module base provides parameter tracking!
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)  # Linear transformation
        self.fc2 = Linear(128, 10)   # Output layer
    
    def forward(self, x):
        x = flatten(x, start_dim=1)  # Flatten: 2D images → 1D vectors
        x = self.fc1(x)              # Linear: matrix multiply + bias
        x = relu(x)                  # Activation (from Module 03)
        return self.fc2(x)           # Final prediction

# Automatic parameter collection!
model = MLP()
params = model.parameters()  # Gets all Linear layer weights/biases automatically!
optimizer = SGD(params)      # Ready for training!
```

### Linear Layer (renamed from Dense)
- **Mathematical foundation**: Linear transformation `y = Wx + b`
- **Weight initialization**: Xavier/Glorot uniform initialization for stable gradients
- **Bias handling**: Optional bias terms for translation invariance
- **Shape management**: Automatic handling of batch dimensions and matrix operations

### Module Base Class - **GAME CHANGER**
- **Automatic parameter tracking**: Collects all trainable weights recursively
- **Nested module support**: Handles complex architectures automatically
- **Clean interface**: Standard `forward()` method for all layers
- **Production pattern**: Same design as PyTorch nn.Module

### Flatten Operation - **ESSENTIAL FOR VISION**
- **Shape transformation**: Convert 2D/3D tensors to 1D for Linear layers
- **Batch preservation**: Keeps batch dimension, flattens the rest
- **Vision pipeline**: Connect CNNs to fully-connected layers
- **Memory efficient**: View operation, no data copying

## 🚀 Getting Started

### Prerequisites
Ensure you have completed the foundational modules:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify prerequisite modules
tito test --module tensor
tito test --module activations
```

### Development Workflow
1. **Open the development file**: `modules/source/04_layers/layers_dev.py`
2. **Implement Linear layer**: Matrix multiplication + bias (`y = Wx + b`)
3. **Build Module base class**: Automatic parameter collection infrastructure
4. **Add Flatten operation**: Essential for connecting CNNs to Linear layers
5. **Build complete networks**: Use Module base to create complex architectures
6. **Export and verify**: `tito module complete 04_layers` (includes testing)

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify mathematical correctness:

```bash
# TinyTorch CLI (recommended)
tito test --module layers

# Direct pytest execution
python -m pytest tests/ -k layers -v
```

### Test Coverage Areas
- ✅ **Layer Functionality**: Verify Dense layers perform correct linear transformations
- ✅ **Weight Initialization**: Ensure proper weight initialization for training stability
- ✅ **Shape Preservation**: Confirm layers handle batch dimensions correctly
- ✅ **Activation Integration**: Test seamless combination with activation functions
- ✅ **Network Composition**: Verify layers can be chained into complete networks

### Inline Testing & Development
The module includes educational feedback during development:
```python
# Example inline test output
🔬 Unit Test: Dense layer functionality...
✅ Dense layer computes y = Wx + b correctly
✅ Weight initialization within expected range
✅ Output shape matches expected dimensions
📈 Progress: Dense Layer ✓

# Integration testing
🔬 Unit Test: Layer composition...
✅ Multiple layers chain correctly
✅ Activations integrate seamlessly
📈 Progress: Layer Composition ✓
```

### Manual Testing Examples
```python
from tinytorch.core.tensor import Tensor
from layers_dev import Dense
from activations_dev import ReLU

# Test basic layer functionality
layer = Dense(input_size=3, output_size=2)
x = Tensor([[1.0, 2.0, 3.0]])
y = layer(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")

# Test layer composition
layer1 = Dense(3, 4)
layer2 = Dense(4, 2)
relu = ReLU()

# Forward pass
h1 = relu(layer1(x))
output = layer2(h1)
print(f"Final output: {output.data}")
```

## 🎯 Key Concepts

### Real-World Applications
- **Computer Vision**: Dense layers process flattened image features in CNNs (like VGG, ResNet final layers)
- **Natural Language Processing**: Dense layers transform word embeddings in transformers and RNNs
- **Recommendation Systems**: Dense layers combine user and item features for preference prediction
- **Scientific Computing**: Dense layers approximate complex functions in physics simulations and engineering

### Mathematical Foundations
- **Linear Transformation**: `y = Wx + b` where W is the weight matrix and b is the bias vector
- **Matrix Multiplication**: Efficient batch processing through vectorized operations
- **Weight Initialization**: Xavier/Glorot initialization prevents vanishing/exploding gradients
- **Function Composition**: Networks as nested function calls: `f3(f2(f1(x)))`

### Neural Network Building Blocks
- **Modularity**: Layers as reusable components that can be combined in different ways
- **Standardized Interface**: All layers follow the same input/output pattern for easy composition
- **Shape Consistency**: Automatic handling of batch dimensions and shape transformations
- **Nonlinearity**: Activation functions between layers enable learning of complex patterns

### Implementation Patterns
- **Class-based Design**: Layers as objects with state (weights) and behavior (forward pass)
- **Initialization Strategy**: Proper weight initialization for stable training dynamics
- **Error Handling**: Graceful handling of shape mismatches and invalid inputs
- **Testing Philosophy**: Comprehensive testing of mathematical properties and edge cases

## 🎉 Ready to Build?

You're about to build the fundamental building blocks that power every neural network! Dense layers might seem simple, but they're the workhorses of deep learning—from the final layers of image classifiers to the core components of language models.

Understanding how these simple linear transformations compose into complex intelligence is one of the most beautiful insights in machine learning. Take your time, understand the mathematics, and enjoy building the foundation of artificial intelligence!

```{grid} 3
:gutter: 3
:margin: 2

{grid-item-card} 🚀 Launch Builder
:link: https://mybinder.org/v2/gh/VJProductions/TinyTorch/main?filepath=modules/source/04_layers/layers_dev.py
:class-title: text-center
:class-body: text-center

Interactive development environment

{grid-item-card} 📓 Open in Colab  
:link: https://colab.research.google.com/github/VJProductions/TinyTorch/blob/main/modules/source/04_layers/layers_dev.ipynb
:class-title: text-center
:class-body: text-center

Google Colab notebook

{grid-item-card} 👀 View Source
:link: https://github.com/VJProductions/TinyTorch/blob/main/modules/source/04_layers/layers_dev.py  
:class-title: text-center
:class-body: text-center

Browse the code on GitHub
``` 
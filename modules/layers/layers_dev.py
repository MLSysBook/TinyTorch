# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Module 2: Layers - Neural Network Building Blocks

Welcome to the Layers module! This is where neural networks begin. You'll implement the fundamental building blocks that transform tensors.

## Learning Goals
- Understand layers as functions that transform tensors: `y = f(x)`
- Implement Dense layers with linear transformations: `y = Wx + b`
- Add activation functions for nonlinearity (ReLU, Sigmoid, Tanh)
- See how neural networks are just function composition
- Build intuition before diving into training

## Build → Use → Understand
1. **Build**: Dense layers and activation functions
2. **Use**: Transform tensors and see immediate results
3. **Understand**: How neural networks transform information

## Module → Package Structure
**🎓 Teaching vs. 🔧 Building**: 
- **Learning side**: Work in `modules/layers/layers_dev.py`  
- **Building side**: Exports to `tinytorch/core/layers.py`

This module builds the fundamental transformations that compose into neural networks.
"""

# %%
#| default_exp core.layers

# Setup and imports
import numpy as np
import sys
from typing import Union, Optional, Callable
import math

# %%
#| export
import numpy as np
import math
import sys
from typing import Union, Optional, Callable
from tinytorch.core.tensor import Tensor

# Import our Tensor class
# sys.path.append('../../')
# from modules.tensor.tensor_dev import Tensor

# print("🔥 TinyTorch Layers Module")
# print(f"NumPy version: {np.__version__}")
# print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
# print("Ready to build neural network layers!")

# %% [markdown]
"""
## Step 1: What is a Layer?

A **layer** is a function that transforms tensors. Think of it as:
- **Input**: Tensor with some shape
- **Transformation**: Mathematical operation (linear, nonlinear, etc.)
- **Output**: Tensor with possibly different shape

**The fundamental insight**: Neural networks are just function composition!
```
x → Layer1 → Layer2 → Layer3 → y
```

**Why layers matter**:
- They're the building blocks of all neural networks
- Each layer learns a different transformation
- Composing layers creates complex functions
- Understanding layers = understanding neural networks

Let's start with the most important layer: **Dense** (also called Linear or Fully Connected).
"""

# %%
#| export
class Dense:
    """
    Dense (Linear) Layer: y = Wx + b
    
    The fundamental building block of neural networks.
    Performs linear transformation: matrix multiplication + bias addition.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        use_bias: Whether to include bias term (default: True)
        
    TODO: Implement the Dense layer with weight initialization and forward pass.
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True):
        """
        Initialize Dense layer with random weights.
        
        TODO: 
        1. Store layer parameters (input_size, output_size, use_bias)
        2. Initialize weights with small random values
        3. Initialize bias to zeros (if use_bias=True)
        """
        raise NotImplementedError("Student implementation required")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = Wx + b
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
            
        TODO: Implement matrix multiplication and bias addition
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Dense:
    """
    Dense (Linear) Layer: y = Wx + b
    
    The fundamental building block of neural networks.
    Performs linear transformation: matrix multiplication + bias addition.
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True):
        """Initialize Dense layer with random weights."""
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights with Xavier/Glorot initialization
        # This helps with gradient flow during training
        limit = math.sqrt(6.0 / (input_size + output_size))
        self.weights = Tensor(
            np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        )
        
        # Initialize bias to zeros
        if use_bias:
            self.bias = Tensor(np.zeros(output_size, dtype=np.float32))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = Wx + b"""
        # Matrix multiplication: x @ weights
        # x shape: (batch_size, input_size)
        # weights shape: (input_size, output_size)
        # result shape: (batch_size, output_size)
        output = Tensor(x.data @ self.weights.data)
        
        # Add bias if present
        if self.bias is not None:
            output = Tensor(output.data + self.bias.data)
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Dense Layer

Once you implement the Dense layer above, run this cell to test it:
"""

# %%
# Test the Dense layer
try:
    print("=== Testing Dense Layer ===")
    
    # Create a simple Dense layer: 3 inputs → 2 outputs
    layer = Dense(input_size=3, output_size=2)
    print(f"Created Dense layer: {layer.input_size} → {layer.output_size}")
    print(f"Weights shape: {layer.weights.shape}")
    print(f"Bias shape: {layer.bias.shape if layer.bias else 'No bias'}")
    
    # Test with a single example
    x = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input: {x.data}")
    print(f"Output: {y.data}")
    
    # Test with batch of examples
    x_batch = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    y_batch = layer(x_batch)
    print(f"\nBatch input shape: {x_batch.shape}")
    print(f"Batch output shape: {y_batch.shape}")
    print(f"Batch output: {y_batch.data}")
    
    print("✅ Dense layer working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Dense layer above!")

# %% [markdown]
"""
## Step 2: Activation Functions

Dense layers alone can only learn **linear** transformations. But most real-world problems need **nonlinear** transformations.

**Activation functions** add nonlinearity:
- **ReLU**: `max(0, x)` - Most common, simple and effective
- **Sigmoid**: `1 / (1 + e^(-x))` - Squashes to (0, 1)
- **Tanh**: `tanh(x)` - Squashes to (-1, 1)

**Why nonlinearity matters**: Without it, stacking layers is pointless!
```
Linear → Linear → Linear = Just one big Linear transformation
Linear → NonLinear → Linear = Can learn complex patterns
```
"""

# %%
#| export
class ReLU:
    """
    ReLU Activation: f(x) = max(0, x)
    
    The most popular activation function in deep learning.
    Simple, effective, and computationally efficient.
    
    TODO: Implement ReLU activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU: f(x) = max(0, x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU applied element-wise
            
        TODO: Implement element-wise max(0, x) operation
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make activation callable: relu(x) same as relu.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class ReLU:
    """ReLU Activation: f(x) = max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU: f(x) = max(0, x)"""
        return Tensor(np.maximum(0, x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| export
class Sigmoid:
    """
    Sigmoid Activation: f(x) = 1 / (1 + e^(-x))
    
    Squashes input to range (0, 1). Often used for binary classification.
    
    TODO: Implement Sigmoid activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid: f(x) = 1 / (1 + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid applied element-wise
            
        TODO: Implement sigmoid function (be careful with numerical stability!)
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Sigmoid:
    """Sigmoid Activation: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid with numerical stability"""
        # Use the numerically stable version to avoid overflow
        # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
        # For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
        x_data = x.data
        result = np.zeros_like(x_data)
        
        # Stable computation
        positive_mask = x_data >= 0
        result[positive_mask] = 1.0 / (1.0 + np.exp(-x_data[positive_mask]))
        result[~positive_mask] = np.exp(x_data[~positive_mask]) / (1.0 + np.exp(x_data[~positive_mask]))
        
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| export
class Tanh:
    """
    Tanh Activation: f(x) = tanh(x)
    
    Squashes input to range (-1, 1). Zero-centered output.
    
    TODO: Implement Tanh activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh: f(x) = tanh(x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh applied element-wise
            
        TODO: Implement tanh function
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Tanh:
    """Tanh Activation: f(x) = tanh(x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh"""
        return Tensor(np.tanh(x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Activation Functions

Once you implement the activation functions above, run this cell to test them:
"""

# %%
# Test activation functions
try:
    print("=== Testing Activation Functions ===")
    
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    print(f"Input: {x.data}")
    
    # Test ReLU
    relu = ReLU()
    y_relu = relu(x)
    print(f"ReLU output: {y_relu.data}")
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    y_sigmoid = sigmoid(x)
    print(f"Sigmoid output: {y_sigmoid.data}")
    
    # Test Tanh
    tanh = Tanh()
    y_tanh = tanh(x)
    print(f"Tanh output: {y_tanh.data}")
    
    print("✅ Activation functions working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the activation functions above!")

# %% [markdown]
"""
## Step 3: Layer Composition - Building Neural Networks

Now comes the magic! We can **compose** layers to build neural networks:

```
Input → Dense → ReLU → Dense → Sigmoid → Output
```

This is a 2-layer neural network that can learn complex nonlinear patterns!
"""

# %%
# Build a simple 2-layer neural network
try:
    print("=== Building a 2-Layer Neural Network ===")
    
    # Network architecture: 3 → 4 → 2
    # Input: 3 features
    # Hidden: 4 neurons with ReLU
    # Output: 2 neurons with Sigmoid
    
    layer1 = Dense(input_size=3, output_size=4)
    activation1 = ReLU()
    layer2 = Dense(input_size=4, output_size=2)
    activation2 = Sigmoid()
    
    print("Network architecture:")
    print(f"  Input: 3 features")
    print(f"  Hidden: {layer1.input_size} → {layer1.output_size} (Dense + ReLU)")
    print(f"  Output: {layer2.input_size} → {layer2.output_size} (Dense + Sigmoid)")
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 examples, 3 features each
    print(f"\nInput shape: {x.shape}")
    print(f"Input data: {x.data}")
    
    # Forward pass through the network
    h1 = layer1(x)           # Dense layer 1
    h1_activated = activation1(h1)  # ReLU activation
    h2 = layer2(h1_activated)       # Dense layer 2  
    output = activation2(h2)        # Sigmoid activation
    
    print(f"\nAfter layer 1: {h1.shape}")
    print(f"After ReLU: {h1_activated.shape}")
    print(f"After layer 2: {h2.shape}")
    print(f"Final output: {output.shape}")
    print(f"Output values: {output.data}")
    
    print("\n🎉 Neural network working! You just built your first neural network!")
    print("Notice how the network transforms 3D input into 2D output through learned transformations.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the layers and activations above!")

# %% [markdown]
"""
## Step 4: Understanding What We Built

Congratulations! You just implemented the fundamental building blocks of neural networks:

### 🧱 **What You Built**
1. **Dense Layer**: Linear transformation `y = Wx + b`
2. **Activation Functions**: Nonlinear transformations (ReLU, Sigmoid, Tanh)
3. **Layer Composition**: Chaining layers to build networks

### 🎯 **Key Insights**
- **Layers are functions**: They transform tensors from one space to another
- **Composition creates complexity**: Simple layers → complex networks
- **Nonlinearity is crucial**: Without it, deep networks are just linear transformations
- **Neural networks are function approximators**: They learn to map inputs to outputs

### 🚀 **What's Next**
In the next modules, you'll learn:
- **Training**: How networks learn from data (backpropagation, optimizers)
- **Architectures**: Specialized layers for different problems (CNNs, RNNs)
- **Applications**: Using networks for real problems

### 🔧 **Export to Package**
Run this to export your layers to the TinyTorch package:
```bash
python bin/tito.py sync
```

Then test your implementation:
```bash
python bin/tito.py test --module layers
```

**Great job! You've built the foundation of neural networks!** 🎉
"""

# %%
# Final demonstration: A more complex example
try:
    print("=== Final Demo: Image Classification Network ===")
    
    # Simulate a small image: 28x28 pixels flattened to 784 features
    # This is like a tiny MNIST digit
    image_size = 28 * 28  # 784 pixels
    num_classes = 10      # 10 digits (0-9)
    
    # Build a 3-layer network for digit classification
    # 784 → 128 → 64 → 10
    layer1 = Dense(input_size=image_size, output_size=128)
    relu1 = ReLU()
    layer2 = Dense(input_size=128, output_size=64)
    relu2 = ReLU()
    layer3 = Dense(input_size=64, output_size=num_classes)
    softmax = Sigmoid()  # Using Sigmoid as a simple "probability-like" output
    
    print(f"Image classification network:")
    print(f"  Input: {image_size} pixels (28x28 image)")
    print(f"  Hidden 1: {layer1.input_size} → {layer1.output_size} (Dense + ReLU)")
    print(f"  Hidden 2: {layer2.input_size} → {layer2.output_size} (Dense + ReLU)")
    print(f"  Output: {layer3.input_size} → {layer3.output_size} (Dense + Sigmoid)")
    
    # Simulate a batch of 5 images
    batch_size = 5
    fake_images = Tensor(np.random.randn(batch_size, image_size).astype(np.float32))
    
    # Forward pass
    h1 = relu1(layer1(fake_images))
    h2 = relu2(layer2(h1))
    predictions = softmax(layer3(h2))
    
    print(f"\nBatch processing:")
    print(f"  Input batch shape: {fake_images.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions.data[0]}")  # First image predictions
    
    print("\n🎉 You built a neural network that could classify images!")
    print("With training, this network could learn to recognize handwritten digits!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your layer implementations!") 
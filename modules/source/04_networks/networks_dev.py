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
# Module 4: Networks - Neural Network Architectures

Welcome to the Networks module! This is where we compose layers into complete neural network architectures.

## Learning Goals
- Understand networks as function composition: `f(x) = layer_n(...layer_2(layer_1(x)))`
- Build the Sequential network architecture for composing layers
- Create common network patterns like MLPs (Multi-Layer Perceptrons)
- Visualize network architectures and understand their capabilities
- Master forward pass inference through complete networks

## Build → Use → Understand
1. **Build**: Sequential networks that compose layers into complete architectures
2. **Use**: Create different network patterns and run inference
3. **Understand**: How architecture design affects network behavior and capability
"""

# %% nbgrader={"grade": false, "grade_id": "networks-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.networks

#| export
import numpy as np
import sys
import os
from typing import List, Union, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Import all the building blocks we need - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_activations'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_layers'))
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    from layers_dev import Dense

# %% nbgrader={"grade": false, "grade_id": "networks-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% nbgrader={"grade": false, "grade_id": "networks-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Networks Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network architectures!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/04_networks/networks_dev.py`  
**Building Side:** Code exports to `tinytorch.core.networks`

```python
# Final package structure:
from tinytorch.core.networks import Sequential, MLP  # Network architectures!
from tinytorch.core.layers import Dense, Conv2D  # Building blocks
from tinytorch.core.activations import ReLU, Sigmoid, Tanh  # Nonlinearity
from tinytorch.core.tensor import Tensor  # Foundation
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.Sequential`
- **Consistency:** All network architectures live together in `core.networks`
- **Integration:** Works seamlessly with layers, activations, and tensors
"""

# %% [markdown]
"""
## 🧠 The Mathematical Foundation of Neural Networks

### Function Composition at Scale
Neural networks are fundamentally about **function composition**:

```
f(x) = f_n(f_{n-1}(...f_2(f_1(x))))
```

Each layer is a function, and the network is the composition of all these functions.

### Why Function Composition is Powerful
- **Modularity**: Each layer has a specific purpose
- **Composability**: Simple functions combine to create complex behaviors
- **Universal approximation**: Deep compositions can approximate any function
- **Hierarchical learning**: Early layers learn simple features, later layers learn complex patterns

### The Architecture Design Space
Different arrangements of layers create different capabilities:
- **Depth**: More layers → more complex representations
- **Width**: More neurons per layer → more capacity per layer
- **Connections**: How layers connect affects information flow
- **Activation functions**: Add nonlinearity for complex patterns

### Connection to Real ML Systems
Every framework uses sequential composition:
- **PyTorch**: `torch.nn.Sequential([layer1, layer2, layer3])`
- **TensorFlow**: `tf.keras.Sequential([layer1, layer2, layer3])`
- **JAX**: `jax.nn.Sequential([layer1, layer2, layer3])`
- **TinyTorch**: `tinytorch.core.networks.Sequential([layer1, layer2, layer3])` (what we're building!)

### Performance and Design Considerations
- **Forward pass efficiency**: Sequential computation through layers
- **Memory management**: Intermediate activations storage
- **Gradient flow**: How information flows backward (for training)
- **Architecture search**: Finding optimal network structures
"""

# %% [markdown]
"""
## Step 1: What is a Network?

### Definition
A **network** is a composition of layers that transforms input data into output predictions. Think of it as a pipeline of transformations:

```
Input → Layer1 → Layer2 → Layer3 → Output
```

### Why Networks Matter
- **Function composition**: Complex behavior from simple building blocks
- **Learnable parameters**: Each layer has weights that can be learned
- **Architecture design**: Different layouts solve different problems
- **Real-world applications**: Classification, regression, generation, etc.

### The Fundamental Insight
**Neural networks are just function composition!**
- Each layer is a function: `f_i(x)`
- The network is: `f(x) = f_n(...f_2(f_1(x)))`
- Complex behavior emerges from simple building blocks

### Real-World Examples
- **MLP (Multi-Layer Perceptron)**: Classic feedforward network
- **CNN (Convolutional Neural Network)**: For image processing
- **RNN (Recurrent Neural Network)**: For sequential data
- **Transformer**: For attention-based processing

### Visual Intuition
```
Input: [1, 2, 3] (3 features)
Layer1: [1.4, 2.8] (linear transformation)
Layer2: [1.4, 2.8] (nonlinearity)
Layer3: [0.7] (final prediction)
```

Let's start by building the most fundamental network: **Sequential**.
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Sequential:
    """
    Sequential Network: Composes layers in sequence
    
    The most fundamental network architecture.
    Applies layers in order: f(x) = layer_n(...layer_2(layer_1(x)))
    """
    
    def __init__(self, layers: List):
        """
        Initialize Sequential network with layers.
        
        Args:
            layers: List of layers to compose in order
            
        TODO: Store the layers and implement forward pass
        
        APPROACH:
        1. Store the layers list as an instance variable
        2. This creates the network architecture ready for forward pass
        
        EXAMPLE:
        Sequential([Dense(3,4), ReLU(), Dense(4,2)])
        creates a 3-layer network: Dense → ReLU → Dense
        
        HINTS:
        - Store layers in self.layers
        - This is the foundation for all network architectures
        """
        ### BEGIN SOLUTION
        self.layers = layers
        ### END SOLUTION
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all layers
            
        TODO: Implement sequential forward pass through all layers
        
        APPROACH:
        1. Start with the input tensor
        2. Apply each layer in sequence
        3. Each layer's output becomes the next layer's input
        4. Return the final output
        
        EXAMPLE:
        Input: Tensor([[1, 2, 3]])
        Layer1 (Dense): Tensor([[1.4, 2.8]])
        Layer2 (ReLU): Tensor([[1.4, 2.8]])
        Layer3 (Dense): Tensor([[0.7]])
        Output: Tensor([[0.7]])
        
        HINTS:
        - Use a for loop: for layer in self.layers:
        - Apply each layer: x = layer(x)
        - The output of one layer becomes input to the next
        - Return the final result
        """
        ### BEGIN SOLUTION
        # Apply each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make network callable: network(x) same as network.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
## Step 2: Building Multi-Layer Perceptrons (MLPs)

### What is an MLP?
A **Multi-Layer Perceptron** is the classic neural network architecture:

```
Input → Dense → Activation → Dense → Activation → ... → Dense → Output
```

### Why MLPs are Important
- **Universal approximation**: Can approximate any continuous function
- **Foundation**: Basis for understanding all neural networks
- **Versatile**: Works for classification, regression, and more
- **Simple**: Easy to understand and implement

### MLP Architecture Pattern
```
create_mlp(3, [4, 2], 1) creates:
Dense(3→4) → ReLU → Dense(4→2) → ReLU → Dense(2→1) → Sigmoid
```

### Real-World Applications
- **Tabular data**: Customer analytics, financial modeling
- **Feature learning**: Learning representations from raw data
- **Classification**: Spam detection, medical diagnosis
- **Regression**: Price prediction, time series forecasting
"""

# %% nbgrader={"grade": false, "grade_id": "create-mlp", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def create_mlp(input_size: int, hidden_sizes: List[int], output_size: int, 
               activation=ReLU, output_activation=Sigmoid) -> Sequential:
    """
    Create a Multi-Layer Perceptron (MLP) network.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output features
        activation: Activation function for hidden layers (default: ReLU)
        output_activation: Activation function for output layer (default: Sigmoid)
        
    Returns:
        Sequential network with MLP architecture
        
    TODO: Implement MLP creation with alternating Dense and activation layers.
    
    APPROACH:
    1. Start with an empty list of layers
    2. Add layers in this pattern:
       - Dense(input_size → first_hidden_size)
       - Activation()
       - Dense(first_hidden_size → second_hidden_size)
       - Activation()
       - ...
       - Dense(last_hidden_size → output_size)
       - Output_activation()
    3. Return Sequential(layers)
    
    EXAMPLE:
    create_mlp(3, [4, 2], 1) creates:
    Dense(3→4) → ReLU → Dense(4→2) → ReLU → Dense(2→1) → Sigmoid
    
    HINTS:
    - Start with layers = []
    - Track current_size starting with input_size
    - For each hidden_size: add Dense(current_size, hidden_size), then activation
    - Finally add Dense(last_hidden_size, output_size), then output_activation
    - Return Sequential(layers)
    """
    ### BEGIN SOLUTION
    layers = []
    current_size = input_size
    
    # Add hidden layers with activations
    for hidden_size in hidden_sizes:
        layers.append(Dense(current_size, hidden_size))
        layers.append(activation())
        current_size = hidden_size
    
    # Add output layer with output activation
    layers.append(Dense(current_size, output_size))
    layers.append(output_activation())
    
    return Sequential(layers)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Network Implementations

Once you implement the functions above, run these cells to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-sequential", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test the Sequential network
print("Testing Sequential network...")

# Create a simple 2-layer network: 3 → 4 → 2
network = Sequential([
    Dense(input_size=3, output_size=4),
    ReLU(),
    Dense(input_size=4, output_size=2),
    Sigmoid()
])

print(f"Network created with {len(network.layers)} layers")

# Test with sample data
x = Tensor([[1.0, 2.0, 3.0]])
print(f"Input: {x}")

# Forward pass
y = network(x)
print(f"Output: {y}")
print(f"Output shape: {y.shape}")

# Verify the network works
assert y.shape == (1, 2), f"Expected shape (1, 2), got {y.shape}"
assert np.all(y.data >= 0) and np.all(y.data <= 1), "Sigmoid output should be between 0 and 1"

print("✅ Sequential network tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-mlp", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test MLP creation
print("Testing MLP creation...")

# Create a simple MLP: 3 → 4 → 2 → 1
mlp = create_mlp(input_size=3, hidden_sizes=[4, 2], output_size=1)

print(f"MLP created with {len(mlp.layers)} layers")

# Test the structure
expected_layers = [
    Dense,  # 3 → 4
    ReLU,   # activation
    Dense,  # 4 → 2
    ReLU,   # activation
    Dense,  # 2 → 1
    Sigmoid # output activation
]

assert len(mlp.layers) == 6, f"Expected 6 layers, got {len(mlp.layers)}"

# Test with sample data
x = Tensor([[1.0, 2.0, 3.0]])
y = mlp(x)
print(f"MLP output: {y}")
print(f"MLP output shape: {y.shape}")

# Verify the output
assert y.shape == (1, 1), f"Expected shape (1, 1), got {y.shape}"
assert np.all(y.data >= 0) and np.all(y.data <= 1), "Sigmoid output should be between 0 and 1"

print("✅ MLP creation tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-network-comparison", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test different network architectures
print("Testing different network architectures...")

# Create networks with different architectures
shallow_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1)
deep_net = create_mlp(input_size=3, hidden_sizes=[4, 4, 4], output_size=1)
wide_net = create_mlp(input_size=3, hidden_sizes=[10], output_size=1)

# Test input
x = Tensor([[1.0, 2.0, 3.0]])

# Test all networks
shallow_out = shallow_net(x)
deep_out = deep_net(x)
wide_out = wide_net(x)

print(f"Shallow network output: {shallow_out}")
print(f"Deep network output: {deep_out}")
print(f"Wide network output: {wide_out}")

# Verify all outputs are valid
for name, output in [("Shallow", shallow_out), ("Deep", deep_out), ("Wide", wide_out)]:
    assert output.shape == (1, 1), f"{name} network output shape should be (1, 1), got {output.shape}"
    assert np.all(output.data >= 0) and np.all(output.data <= 1), f"{name} network output should be between 0 and 1"

print("✅ Network architecture comparison tests passed!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented complete neural network architectures:

### What You've Accomplished
✅ **Sequential Networks**: The fundamental architecture for composing layers  
✅ **Function Composition**: Understanding how layers combine to create complex behaviors  
✅ **MLP Creation**: Building Multi-Layer Perceptrons with flexible architectures  
✅ **Architecture Patterns**: Creating shallow, deep, and wide networks  
✅ **Forward Pass**: Complete inference through multi-layer networks  

### Key Concepts You've Learned
- **Networks are function composition**: Complex behavior from simple building blocks
- **Sequential architecture**: The foundation of most neural networks
- **MLP patterns**: Dense → Activation → Dense → Activation → Output
- **Architecture design**: How depth and width affect network capability
- **Forward pass**: How data flows through complete networks

### Mathematical Foundations
- **Function composition**: f(x) = f_n(...f_2(f_1(x)))
- **Universal approximation**: MLPs can approximate any continuous function
- **Hierarchical learning**: Early layers learn simple features, later layers learn complex patterns
- **Nonlinearity**: Activation functions enable complex decision boundaries

### Real-World Applications
- **Classification**: Image recognition, spam detection, medical diagnosis
- **Regression**: Price prediction, time series forecasting
- **Feature learning**: Extracting meaningful representations from raw data
- **Transfer learning**: Using pre-trained networks for new tasks

### Next Steps
1. **Export your code**: `tito package nbdev --export 04_networks`
2. **Test your implementation**: `tito module test 04_networks`
3. **Use your networks**: 
   ```python
   from tinytorch.core.networks import Sequential, create_mlp
   from tinytorch.core.layers import Dense
   from tinytorch.core.activations import ReLU
   
   # Create custom network
   network = Sequential([Dense(10, 5), ReLU(), Dense(5, 1)])
   
   # Create MLP
   mlp = create_mlp(10, [20, 10], 1)
   ```
4. **Move to Module 5**: Start building convolutional networks for images!

**Ready for the next challenge?** Let's add convolutional layers for image processing and build CNNs!
""" 
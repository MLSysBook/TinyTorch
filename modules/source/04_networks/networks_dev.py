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

### The Mathematical Foundation: Function Composition Theory

#### **Function Composition in Mathematics**
In mathematics, function composition combines simple functions to create complex ones:

```python
# Mathematical composition: (f ∘ g)(x) = f(g(x))
def compose(f, g):
    return lambda x: f(g(x))

# Neural network composition: h(x) = f_n(f_{n-1}(...f_2(f_1(x))))
def network(layers):
    return lambda x: reduce(lambda acc, layer: layer(acc), layers, x)
```

#### **Why Composition is Powerful**
1. **Modularity**: Each layer has a specific, well-defined purpose
2. **Composability**: Simple functions combine to create arbitrarily complex behaviors
3. **Hierarchical learning**: Early layers learn simple features, later layers learn complex patterns
4. **Universal approximation**: Deep compositions can approximate any continuous function

#### **The Emergence of Intelligence**
Complex behavior emerges from simple layer composition:

```python
# Example: Image classification
raw_pixels → [Edge detectors] → [Shape detectors] → [Object detectors] → [Class predictor]
     ↓              ↓                    ↓                    ↓                 ↓
  [28x28]      [64 features]      [128 features]      [256 features]      [10 classes]
```

### Architectural Design Principles

#### **1. Depth vs. Width Trade-offs**
- **Deep networks**: More layers → more complex representations
  - **Advantages**: Better feature hierarchies, parameter efficiency
  - **Disadvantages**: Harder to train, gradient problems
- **Wide networks**: More neurons per layer → more capacity per layer
  - **Advantages**: Easier to train, parallel computation
  - **Disadvantages**: More parameters, potential overfitting

#### **2. Information Flow Patterns**
```python
# Sequential flow (what we're building):
x → layer1 → layer2 → layer3 → output

# Residual flow (advanced):
x → layer1 → layer2 + x → layer3 → output

# Attention flow (transformers):
x → attention(x, x, x) → feedforward → output
```

#### **3. Activation Function Placement**
```python
# Standard pattern:
linear_transformation → nonlinear_activation → next_layer

# Why this works:
# Linear + Linear = Linear (no increase in expressiveness)
# Linear + Nonlinear + Linear = Nonlinear (exponential increase in expressiveness)
```

### Real-World Architecture Examples

#### **Multi-Layer Perceptron (MLP)**
```python
# Classic feedforward network
input → dense(512) → relu → dense(256) → relu → dense(10) → softmax
```
- **Use cases**: Tabular data, feature learning, classification
- **Strengths**: Universal approximation, well-understood
- **Weaknesses**: Doesn't exploit spatial/temporal structure

#### **Convolutional Neural Network (CNN)**
```python
# Exploits spatial structure
input → conv2d → relu → pool → conv2d → relu → pool → dense → softmax
```
- **Use cases**: Image processing, computer vision
- **Strengths**: Translation invariance, parameter sharing
- **Weaknesses**: Fixed receptive field, not great for sequences

#### **Recurrent Neural Network (RNN)**
```python
# Processes sequences
input_t → rnn_cell(hidden_{t-1}) → hidden_t → output_t
```
- **Use cases**: Natural language processing, time series
- **Strengths**: Variable length sequences, memory
- **Weaknesses**: Sequential computation, gradient problems

#### **Transformer**
```python
# Attention-based processing
input → attention → feedforward → attention → feedforward → output
```
- **Use cases**: Language models, machine translation
- **Strengths**: Parallelizable, long-range dependencies
- **Weaknesses**: Quadratic complexity, large memory requirements

### The Network Design Process

#### **1. Problem Analysis**
- **Data type**: Images, text, tabular, time series?
- **Task type**: Classification, regression, generation?
- **Constraints**: Latency, memory, accuracy requirements?

#### **2. Architecture Selection**
- **Start simple**: Begin with basic MLP
- **Add structure**: Incorporate domain-specific inductive biases
- **Scale up**: Increase depth/width as needed

#### **3. Component Design**
- **Input layer**: Match data dimensions
- **Hidden layers**: Gradual dimension reduction typical
- **Output layer**: Match task requirements (classes, regression targets)
- **Activation functions**: ReLU for hidden, task-specific for output

#### **4. Optimization Considerations**
- **Gradient flow**: Ensure gradients can flow through the network
- **Computational efficiency**: Balance expressiveness with speed
- **Memory usage**: Consider intermediate activation storage

### Performance Characteristics

#### **Forward Pass Complexity**
For a network with L layers, each with n neurons:
- **Time complexity**: O(L × n²) for dense layers
- **Space complexity**: O(L × n) for activations
- **Parallelization**: Each layer can be parallelized

#### **Memory Management**
```python
# Memory usage during forward pass:
input_memory = batch_size × input_size
hidden_memory = batch_size × hidden_size × num_layers
output_memory = batch_size × output_size
total_memory = input_memory + hidden_memory + output_memory
```

#### **Computational Optimization**
- **Batch processing**: Process multiple samples simultaneously
- **Vectorization**: Use optimized matrix operations
- **Hardware acceleration**: Leverage GPUs/TPUs for parallel computation

### Connection to Previous Modules

#### **From Module 1 (Tensor)**
- **Data flow**: Tensors flow through the network
- **Shape management**: Ensure compatible dimensions between layers

#### **From Module 2 (Activations)**
- **Nonlinearity**: Activation functions between layers enable complex learning
- **Function choice**: Different activations for different purposes

#### **From Module 3 (Layers)**
- **Building blocks**: Layers are the fundamental components
- **Composition**: Networks compose layers into complete architectures

### Why Networks Matter: The Scaling Laws

#### **Empirical Observations**
- **More parameters**: Generally better performance (up to a point)
- **More data**: Enables training of larger networks
- **More compute**: Allows exploration of larger architectures

#### **The Deep Learning Revolution**
```python
# Pre-2012: Shallow networks
input → hidden(100) → output

# Post-2012: Deep networks
input → hidden(512) → hidden(512) → hidden(512) → ... → output
```

The key insight: **Depth enables hierarchical feature learning**

Let's start building our Sequential network architecture!
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
### 🧪 Unit Test: Sequential Network

Let's test your Sequential network implementation! This is the foundation of all neural network architectures.

**This is a unit test** - it tests one specific class (Sequential network) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-sequential-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test Sequential network immediately after implementation
print("🔬 Unit Test: Sequential Network...")

# Create a simple 2-layer network: 3 → 4 → 2
try:
    network = Sequential([
        Dense(input_size=3, output_size=4),
        ReLU(),
        Dense(input_size=4, output_size=2),
        Sigmoid()
    ])
    
    print(f"Network created with {len(network.layers)} layers")
    print("✅ Sequential network creation successful")
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0]])
    print(f"Input: {x}")
    
    # Forward pass
    y = network(x)
    print(f"Output: {y}")
    print(f"Output shape: {y.shape}")
    
    # Verify the network works
    assert y.shape == (1, 2), f"Expected shape (1, 2), got {y.shape}"
    print("✅ Sequential network produces correct output shape")
    
    # Test that sigmoid output is in valid range
    assert np.all(y.data >= 0) and np.all(y.data <= 1), "Sigmoid output should be between 0 and 1"
    print("✅ Sequential network output is in valid range")
    
    # Test that layers are stored correctly
    assert len(network.layers) == 4, f"Expected 4 layers, got {len(network.layers)}"
    print("✅ Sequential network stores layers correctly")
    
except Exception as e:
    print(f"❌ Sequential network test failed: {e}")
    raise

# Show the network architecture
print("🎯 Sequential network behavior:")
print("   Applies layers in sequence: f(g(h(x)))")
print("   Input flows through each layer in order")
print("   Output of layer i becomes input of layer i+1")
print("📈 Progress: Sequential network ✓")

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
### 🧪 Unit Test: MLP Creation

Let's test your MLP creation function! This builds complete neural networks with a single function call.

**This is a unit test** - it tests one specific function (create_mlp) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-mlp-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test MLP creation immediately after implementation
print("🔬 Unit Test: MLP Creation...")

# Create a simple MLP: 3 → 4 → 2 → 1
try:
    mlp = create_mlp(input_size=3, hidden_sizes=[4, 2], output_size=1)
    
    print(f"MLP created with {len(mlp.layers)} layers")
    print("✅ MLP creation successful")
    
    # Test the structure - should have 6 layers: Dense, ReLU, Dense, ReLU, Dense, Sigmoid
    expected_layers = 6  # 3 Dense + 2 ReLU + 1 Sigmoid
    assert len(mlp.layers) == expected_layers, f"Expected {expected_layers} layers, got {len(mlp.layers)}"
    print("✅ MLP has correct number of layers")
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0]])
    y = mlp(x)
    print(f"MLP input: {x}")
    print(f"MLP output: {y}")
    print(f"MLP output shape: {y.shape}")
    
    # Verify the output
    assert y.shape == (1, 1), f"Expected shape (1, 1), got {y.shape}"
    print("✅ MLP produces correct output shape")
    
    # Test that sigmoid output is in valid range
    assert np.all(y.data >= 0) and np.all(y.data <= 1), "Sigmoid output should be between 0 and 1"
    print("✅ MLP output is in valid range")
    
except Exception as e:
    print(f"❌ MLP creation test failed: {e}")
    raise

# Test different architectures
try:
    # Test shallow network
    shallow_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1)
    assert len(shallow_net.layers) == 4, f"Shallow network should have 4 layers, got {len(shallow_net.layers)}"
    
    # Test deep network  
    deep_net = create_mlp(input_size=3, hidden_sizes=[4, 4, 4], output_size=1)
    assert len(deep_net.layers) == 8, f"Deep network should have 8 layers, got {len(deep_net.layers)}"
    
    # Test wide network
    wide_net = create_mlp(input_size=3, hidden_sizes=[10], output_size=1)
    assert len(wide_net.layers) == 4, f"Wide network should have 4 layers, got {len(wide_net.layers)}"
    
    print("✅ Different MLP architectures work correctly")
    
except Exception as e:
    print(f"❌ MLP architecture test failed: {e}")
    raise

# Show the MLP pattern
print("🎯 MLP creation pattern:")
print("   Input → Dense → Activation → Dense → Activation → ... → Dense → Output_Activation")
print("   Automatically creates the complete architecture")
print("   Handles any number of hidden layers")
print("📈 Progress: Sequential network ✓, MLP creation ✓")
print("🚀 Complete neural networks ready!")

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

# %% [markdown]
"""
## 🧪 Comprehensive Testing: Neural Network Architectures

Let's thoroughly test your network implementations to ensure they work correctly in all scenarios.
This comprehensive testing ensures your networks are robust and ready for real ML applications.
"""

# %% nbgrader={"grade": true, "grade_id": "test-networks-comprehensive", "locked": true, "points": 30, "schema_version": 3, "solution": false, "task": false}
def test_networks_comprehensive():
    """Comprehensive test of Sequential networks and MLP creation."""
    print("🔬 Testing neural network architectures comprehensively...")
    
    tests_passed = 0
    total_tests = 10
    
    # Test 1: Sequential Network Creation and Structure
    try:
        # Create a simple 2-layer network
        network = Sequential([
            Dense(input_size=3, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        assert len(network.layers) == 4, f"Expected 4 layers, got {len(network.layers)}"
        
        # Test layer types
        assert isinstance(network.layers[0], Dense), "First layer should be Dense"
        assert isinstance(network.layers[1], ReLU), "Second layer should be ReLU"
        assert isinstance(network.layers[2], Dense), "Third layer should be Dense"
        assert isinstance(network.layers[3], Sigmoid), "Fourth layer should be Sigmoid"
        
        print("✅ Sequential network creation and structure")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sequential network creation failed: {e}")
    
    # Test 2: Sequential Network Forward Pass
    try:
        network = Sequential([
            Dense(input_size=3, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        # Test single sample
        x_single = Tensor([[1.0, 2.0, 3.0]])
        y_single = network(x_single)
        
        assert y_single.shape == (1, 2), f"Single sample output should be (1, 2), got {y_single.shape}"
        assert np.all((y_single.data >= 0) & (y_single.data <= 1)), "Sigmoid output should be in [0,1]"
        
        # Test batch processing
        x_batch = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_batch = network(x_batch)
        
        assert y_batch.shape == (3, 2), f"Batch output should be (3, 2), got {y_batch.shape}"
        assert np.all((y_batch.data >= 0) & (y_batch.data <= 1)), "All batch outputs should be in [0,1]"
        
        print("✅ Sequential network forward pass: single and batch")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sequential network forward pass failed: {e}")
    
    # Test 3: MLP Creation Basic Functionality
    try:
        # Create simple MLP: 3 → 4 → 2 → 1
        mlp = create_mlp(input_size=3, hidden_sizes=[4, 2], output_size=1)
        
        # Should have 6 layers: Dense, ReLU, Dense, ReLU, Dense, Sigmoid
        expected_layers = 6
        assert len(mlp.layers) == expected_layers, f"Expected {expected_layers} layers, got {len(mlp.layers)}"
        
        # Test layer pattern
        layer_types = [type(layer).__name__ for layer in mlp.layers]
        expected_pattern = ['Dense', 'ReLU', 'Dense', 'ReLU', 'Dense', 'Sigmoid']
        assert layer_types == expected_pattern, f"Expected pattern {expected_pattern}, got {layer_types}"
        
        # Test forward pass
        x = Tensor([[1.0, 2.0, 3.0]])
        y = mlp(x)
        
        assert y.shape == (1, 1), f"MLP output should be (1, 1), got {y.shape}"
        assert np.all((y.data >= 0) & (y.data <= 1)), "MLP output should be in [0,1]"
        
        print("✅ MLP creation basic functionality")
        tests_passed += 1
    except Exception as e:
        print(f"❌ MLP creation basic failed: {e}")
    
    # Test 4: Different MLP Architectures
    try:
        # Test shallow network (1 hidden layer)
        shallow_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1)
        assert len(shallow_net.layers) == 4, f"Shallow network should have 4 layers, got {len(shallow_net.layers)}"
        
        # Test deep network (3 hidden layers)
        deep_net = create_mlp(input_size=3, hidden_sizes=[4, 4, 4], output_size=1)
        assert len(deep_net.layers) == 8, f"Deep network should have 8 layers, got {len(deep_net.layers)}"
        
        # Test wide network (1 large hidden layer)
        wide_net = create_mlp(input_size=3, hidden_sizes=[20], output_size=1)
        assert len(wide_net.layers) == 4, f"Wide network should have 4 layers, got {len(wide_net.layers)}"
        
        # Test very deep network
        very_deep_net = create_mlp(input_size=3, hidden_sizes=[5, 5, 5, 5, 5], output_size=1)
        assert len(very_deep_net.layers) == 12, f"Very deep network should have 12 layers, got {len(very_deep_net.layers)}"
        
        # Test all networks work
        x = Tensor([[1.0, 2.0, 3.0]])
        for name, net in [("Shallow", shallow_net), ("Deep", deep_net), ("Wide", wide_net), ("Very Deep", very_deep_net)]:
            y = net(x)
            assert y.shape == (1, 1), f"{name} network output shape should be (1, 1), got {y.shape}"
            assert np.all((y.data >= 0) & (y.data <= 1)), f"{name} network output should be in [0,1]"
        
        print("✅ Different MLP architectures: shallow, deep, wide, very deep")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Different MLP architectures failed: {e}")
    
    # Test 5: MLP with Different Activation Functions
    try:
        # Test with Tanh activation
        mlp_tanh = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=Tanh, output_activation=Sigmoid)
        
        # Check layer types
        layer_types = [type(layer).__name__ for layer in mlp_tanh.layers]
        expected_pattern = ['Dense', 'Tanh', 'Dense', 'Sigmoid']
        assert layer_types == expected_pattern, f"Tanh MLP pattern should be {expected_pattern}, got {layer_types}"
        
        # Test forward pass
        x = Tensor([[1.0, 2.0, 3.0]])
        y = mlp_tanh(x)
        assert y.shape == (1, 1), "Tanh MLP should work correctly"
        
        # Test with different output activation
        mlp_tanh_out = create_mlp(input_size=3, hidden_sizes=[4], output_size=3, activation=ReLU, output_activation=Softmax)
        y_multi = mlp_tanh_out(x)
        assert y_multi.shape == (1, 3), "Multi-output MLP should work"
        
        # Check softmax properties
        assert abs(np.sum(y_multi.data) - 1.0) < 1e-6, "Softmax outputs should sum to 1"
        
        print("✅ MLP with different activation functions: Tanh, Softmax")
        tests_passed += 1
    except Exception as e:
        print(f"❌ MLP with different activations failed: {e}")
    
    # Test 6: Network Layer Composition
    try:
        # Test that network correctly chains layers
        network = Sequential([
            Dense(input_size=4, output_size=3),
            ReLU(),
            Dense(input_size=3, output_size=2),
            Tanh(),
            Dense(input_size=2, output_size=1),
            Sigmoid()
        ])
        
        x = Tensor([[1.0, -1.0, 2.0, -2.0]])
        
        # Manual forward pass to verify composition
        h1 = network.layers[0](x)  # Dense
        h2 = network.layers[1](h1)  # ReLU
        h3 = network.layers[2](h2)  # Dense
        h4 = network.layers[3](h3)  # Tanh
        h5 = network.layers[4](h4)  # Dense
        h6 = network.layers[5](h5)  # Sigmoid
        
        # Compare with network forward pass
        y_network = network(x)
        
        assert np.allclose(h6.data, y_network.data), "Manual and network forward pass should match"
        
        # Check intermediate shapes
        assert h1.shape == (1, 3), f"h1 shape should be (1, 3), got {h1.shape}"
        assert h2.shape == (1, 3), f"h2 shape should be (1, 3), got {h2.shape}"
        assert h3.shape == (1, 2), f"h3 shape should be (1, 2), got {h3.shape}"
        assert h4.shape == (1, 2), f"h4 shape should be (1, 2), got {h4.shape}"
        assert h5.shape == (1, 1), f"h5 shape should be (1, 1), got {h5.shape}"
        assert h6.shape == (1, 1), f"h6 shape should be (1, 1), got {h6.shape}"
        
        # Check activation effects
        assert np.all(h2.data >= 0), "ReLU should produce non-negative values"
        assert np.all((h4.data >= -1) & (h4.data <= 1)), "Tanh should produce values in [-1,1]"
        assert np.all((h6.data >= 0) & (h6.data <= 1)), "Sigmoid should produce values in [0,1]"
        
        print("✅ Network layer composition: correct chaining and shapes")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Network layer composition failed: {e}")
    
    # Test 7: Edge Cases and Robustness
    try:
        # Test with minimal network (1 layer)
        minimal_net = Sequential([Dense(input_size=2, output_size=1)])
        x_minimal = Tensor([[1.0, 2.0]])
        y_minimal = minimal_net(x_minimal)
        assert y_minimal.shape == (1, 1), "Minimal network should work"
        
        # Test with single neuron layers
        single_neuron_net = create_mlp(input_size=1, hidden_sizes=[1], output_size=1)
        x_single = Tensor([[5.0]])
        y_single_neuron = single_neuron_net(x_single)
        assert y_single_neuron.shape == (1, 1), "Single neuron network should work"
        
        # Test with large batch
        large_net = create_mlp(input_size=10, hidden_sizes=[5], output_size=1)
        x_large_batch = Tensor(np.random.randn(100, 10))
        y_large_batch = large_net(x_large_batch)
        assert y_large_batch.shape == (100, 1), "Large batch should work"
        assert not np.any(np.isnan(y_large_batch.data)), "Should not produce NaN"
        assert not np.any(np.isinf(y_large_batch.data)), "Should not produce Inf"
        
        print("✅ Edge cases: minimal networks, single neurons, large batches")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Edge cases failed: {e}")
    
    # Test 8: Multi-class Classification Networks
    try:
        # Create multi-class classifier
        classifier = create_mlp(input_size=4, hidden_sizes=[8, 6], output_size=3, output_activation=Softmax)
        
        # Test with batch of samples
        x_multi = Tensor(np.random.randn(5, 4))
        y_multi = classifier(x_multi)
        
        assert y_multi.shape == (5, 3), f"Multi-class output should be (5, 3), got {y_multi.shape}"
        
        # Check softmax properties for each sample
        row_sums = np.sum(y_multi.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each sample should have probabilities summing to 1"
        assert np.all(y_multi.data > 0), "All probabilities should be positive"
        
        # Test that argmax gives valid class predictions
        predictions = np.argmax(y_multi.data, axis=1)
        assert np.all((predictions >= 0) & (predictions < 3)), "Predictions should be valid class indices"
        
        print("✅ Multi-class classification: softmax probabilities, valid predictions")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Multi-class classification failed: {e}")
    
    # Test 9: Real ML Scenarios
    try:
        # Scenario 1: Binary classification (like spam detection)
        spam_classifier = create_mlp(input_size=100, hidden_sizes=[50, 20], output_size=1, output_activation=Sigmoid)
        
        # Simulate email features
        email_features = Tensor(np.random.randn(10, 100))
        spam_probabilities = spam_classifier(email_features)
        
        assert spam_probabilities.shape == (10, 1), "Spam classifier should output probabilities for each email"
        assert np.all((spam_probabilities.data >= 0) & (spam_probabilities.data <= 1)), "Should output valid probabilities"
        
        # Scenario 2: Image classification (like MNIST)
        mnist_classifier = create_mlp(input_size=784, hidden_sizes=[256, 128], output_size=10, output_activation=Softmax)
        
        # Simulate flattened images
        images = Tensor(np.random.randn(32, 784))  # Batch of 32 images
        class_probabilities = mnist_classifier(images)
        
        assert class_probabilities.shape == (32, 10), "MNIST classifier should output 10 class probabilities"
        
        # Check softmax properties
        batch_sums = np.sum(class_probabilities.data, axis=1)
        assert np.allclose(batch_sums, 1.0), "Each image should have class probabilities summing to 1"
        
        # Scenario 3: Regression (like house price prediction)
        price_predictor = Sequential([
            Dense(input_size=8, output_size=16),
            ReLU(),
            Dense(input_size=16, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=1)  # No activation for regression
        ])
        
        # Simulate house features
        house_features = Tensor(np.random.randn(5, 8))
        predicted_prices = price_predictor(house_features)
        
        assert predicted_prices.shape == (5, 1), "Price predictor should output one price per house"
        
        print("✅ Real ML scenarios: spam detection, image classification, price prediction")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Real ML scenarios failed: {e}")
    
    # Test 10: Network Comparison and Analysis
    try:
        # Create networks with same total parameters but different architectures
        x_test = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Wide network: 4 → 20 → 1 (parameters: 4*20 + 20 + 20*1 + 1 = 121)
        wide_network = create_mlp(input_size=4, hidden_sizes=[20], output_size=1)
        
        # Deep network: 4 → 10 → 10 → 1 (parameters: 4*10 + 10 + 10*10 + 10 + 10*1 + 1 = 171)
        deep_network = create_mlp(input_size=4, hidden_sizes=[10, 10], output_size=1)
        
        # Test both networks
        wide_output = wide_network(x_test)
        deep_output = deep_network(x_test)
        
        assert wide_output.shape == (1, 1), "Wide network should produce correct output"
        assert deep_output.shape == (1, 1), "Deep network should produce correct output"
        
        # Both should be valid but potentially different
        assert np.all((wide_output.data >= 0) & (wide_output.data <= 1)), "Wide network output should be valid"
        assert np.all((deep_output.data >= 0) & (deep_output.data <= 1)), "Deep network output should be valid"
        
        # Test network complexity
        def count_parameters(network):
            total = 0
            for layer in network.layers:
                if isinstance(layer, Dense):
                    total += layer.weights.size
                    if layer.bias is not None:
                        total += layer.bias.size
            return total
        
        wide_params = count_parameters(wide_network)
        deep_params = count_parameters(deep_network)
        
        assert wide_params > 0, "Wide network should have parameters"
        assert deep_params > 0, "Deep network should have parameters"
        
        print(f"✅ Network comparison: wide ({wide_params} params) vs deep ({deep_params} params)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Network comparison failed: {e}")
    
    # Results summary
    print(f"\n📊 Networks Module Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All network tests passed! Your implementations support:")
        print("  • Sequential networks: layer composition and chaining")
        print("  • MLP creation: flexible multi-layer perceptron architectures")
        print("  • Different architectures: shallow, deep, wide networks")
        print("  • Multiple activation functions: ReLU, Tanh, Sigmoid, Softmax")
        print("  • Multi-class classification: softmax probability distributions")
        print("  • Real ML scenarios: spam detection, image classification, regression")
        print("  • Network analysis: parameter counting and architecture comparison")
        print("📈 Progress: All Network Functionality ✓")
        return True
    else:
        print("⚠️  Some network tests failed. Common issues:")
        print("  • Check Sequential class layer composition")
        print("  • Verify create_mlp function layer creation pattern")
        print("  • Ensure proper activation function integration")
        print("  • Test forward pass through complete networks")
        print("  • Verify shape handling across all layers")
        return False

# Run the comprehensive test
success = test_networks_comprehensive()

# %% [markdown]
"""
### 🧪 Integration Test: Complete Neural Network Applications

Let's test your networks in realistic machine learning applications.
"""

# %% nbgrader={"grade": true, "grade_id": "test-networks-integration", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_networks_integration():
    """Integration test with complete neural network applications."""
    print("🔬 Testing networks in complete ML applications...")
    
    try:
        print("🧠 Building complete ML applications with neural networks...")
        
        # Application 1: Iris Classification
        print("\n🌸 Application 1: Iris Classification (Multi-class)")
        iris_classifier = create_mlp(
            input_size=4,      # 4 flower measurements
            hidden_sizes=[8, 6], # Hidden layers
            output_size=3,     # 3 iris species
            output_activation=Softmax
        )
        
        # Simulate iris data
        iris_samples = Tensor([
            [5.1, 3.5, 1.4, 0.2],  # Setosa-like
            [7.0, 3.2, 4.7, 1.4],  # Versicolor-like
            [6.3, 3.3, 6.0, 2.5]   # Virginica-like
        ])
        
        iris_predictions = iris_classifier(iris_samples)
        
        assert iris_predictions.shape == (3, 3), "Should predict 3 classes for 3 samples"
        
        # Check that predictions are valid probabilities
        row_sums = np.sum(iris_predictions.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each prediction should sum to 1"
        
        # Get predicted classes
        predicted_classes = np.argmax(iris_predictions.data, axis=1)
        print(f"  Predicted classes: {predicted_classes}")
        print(f"  Confidence scores: {np.max(iris_predictions.data, axis=1)}")
        
        print("✅ Iris classification: valid multi-class predictions")
        
        # Application 2: Housing Price Prediction
        print("\n🏠 Application 2: Housing Price Prediction (Regression)")
        price_predictor = Sequential([
            Dense(input_size=8, output_size=16),  # 8 house features
            ReLU(),
            Dense(input_size=16, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=1)    # 1 price output (no activation for regression)
        ])
        
        # Simulate house features: [size, bedrooms, bathrooms, age, location_score, etc.]
        house_data = Tensor([
            [2000, 3, 2, 5, 8.5, 1, 0, 1],    # Large, new house
            [1200, 2, 1, 20, 6.0, 0, 1, 0],   # Small, older house
            [1800, 3, 2, 10, 7.5, 1, 0, 0]    # Medium house
        ])
        
        predicted_prices = price_predictor(house_data)
        
        assert predicted_prices.shape == (3, 1), "Should predict 1 price for each house"
        assert not np.any(np.isnan(predicted_prices.data)), "Prices should not be NaN"
        
        print(f"  Predicted prices: {predicted_prices.data.flatten()}")
        print("✅ Housing price prediction: valid regression outputs")
        
        # Application 3: Sentiment Analysis
        print("\n💭 Application 3: Sentiment Analysis (Binary Classification)")
        sentiment_analyzer = create_mlp(
            input_size=100,    # 100 text features (like TF-IDF)
            hidden_sizes=[50, 25], # Deep network for text
            output_size=1,     # Binary sentiment (positive/negative)
            output_activation=Sigmoid
        )
        
        # Simulate text features for different reviews
        review_features = Tensor(np.random.randn(5, 100))  # 5 reviews
        sentiment_scores = sentiment_analyzer(review_features)
        
        assert sentiment_scores.shape == (5, 1), "Should predict sentiment for each review"
        assert np.all((sentiment_scores.data >= 0) & (sentiment_scores.data <= 1)), "Sentiment scores should be probabilities"
        
        # Convert to sentiment labels
        sentiment_labels = (sentiment_scores.data > 0.5).astype(int)
        print(f"  Sentiment predictions: {sentiment_labels.flatten()}")
        print(f"  Confidence scores: {sentiment_scores.data.flatten()}")
        
        print("✅ Sentiment analysis: valid binary classification")
        
        # Application 4: MNIST-like Digit Recognition
        print("\n🔢 Application 4: Digit Recognition (Image Classification)")
        digit_classifier = create_mlp(
            input_size=784,     # 28x28 flattened images
            hidden_sizes=[256, 128, 64], # Deep network for images
            output_size=10,     # 10 digits (0-9)
            output_activation=Softmax
        )
        
        # Simulate flattened digit images
        digit_images = Tensor(np.random.randn(8, 784))  # 8 digit images
        digit_predictions = digit_classifier(digit_images)
        
        assert digit_predictions.shape == (8, 10), "Should predict 10 classes for each image"
        
        # Check softmax properties
        row_sums = np.sum(digit_predictions.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each prediction should sum to 1"
        
        # Get predicted digits
        predicted_digits = np.argmax(digit_predictions.data, axis=1)
        confidence_scores = np.max(digit_predictions.data, axis=1)
        
        print(f"  Predicted digits: {predicted_digits}")
        print(f"  Confidence scores: {confidence_scores}")
        
        print("✅ Digit recognition: valid multi-class image classification")
        
        # Application 5: Network Architecture Comparison
        print("\n📊 Application 5: Architecture Comparison Study")
        
        # Create different architectures for same task
        architectures = {
            "Shallow": create_mlp(4, [16], 3, output_activation=Softmax),
            "Medium": create_mlp(4, [12, 8], 3, output_activation=Softmax),
            "Deep": create_mlp(4, [8, 8, 8], 3, output_activation=Softmax),
            "Wide": create_mlp(4, [24], 3, output_activation=Softmax)
        }
        
        # Test all architectures on same data
        test_data = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        for name, network in architectures.items():
            prediction = network(test_data)
            assert prediction.shape == (1, 3), f"{name} network should output 3 classes"
            assert abs(np.sum(prediction.data) - 1.0) < 1e-6, f"{name} network should output valid probabilities"
            
            # Count parameters
            param_count = sum(layer.weights.size + (layer.bias.size if hasattr(layer, 'bias') and layer.bias is not None else 0) 
                            for layer in network.layers if hasattr(layer, 'weights'))
            
            print(f"  {name} network: {param_count} parameters, prediction: {prediction.data.flatten()}")
        
        print("✅ Architecture comparison: all networks work with different complexities")
        
        # Application 6: Transfer Learning Simulation
        print("\n🔄 Application 6: Transfer Learning Simulation")
        
        # Create "pre-trained" feature extractor
        feature_extractor = Sequential([
            Dense(input_size=100, output_size=50),
            ReLU(),
            Dense(input_size=50, output_size=25),
            ReLU()
        ])
        
        # Create task-specific classifier
        classifier_head = Sequential([
            Dense(input_size=25, output_size=10),
            ReLU(),
            Dense(input_size=10, output_size=2),
            Softmax()
        ])
        
        # Simulate transfer learning pipeline
        raw_data = Tensor(np.random.randn(3, 100))
        
        # Extract features
        features = feature_extractor(raw_data)
        assert features.shape == (3, 25), "Feature extractor should output 25 features"
        
        # Classify using extracted features
        final_predictions = classifier_head(features)
        assert final_predictions.shape == (3, 2), "Classifier should output 2 classes"
        
        row_sums = np.sum(final_predictions.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Transfer learning predictions should be valid"
        
        print("✅ Transfer learning simulation: modular network composition")
        
        print("\n🎉 Integration test passed! Your networks work correctly in:")
        print("  • Multi-class classification (Iris flowers)")
        print("  • Regression tasks (housing prices)")
        print("  • Binary classification (sentiment analysis)")
        print("  • Image classification (digit recognition)")
        print("  • Architecture comparison studies")
        print("  • Transfer learning scenarios")
        print("📈 Progress: Networks ready for real ML applications!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("\n💡 This suggests an issue with:")
        print("  • Network architecture composition")
        print("  • Forward pass through complete networks")
        print("  • Shape compatibility between layers")
        print("  • Activation function integration")
        print("  • Check your Sequential and create_mlp implementations")
        return False

# Run the integration test
success = test_networks_integration() and success

# Print final summary
print(f"\n{'='*60}")
print("🎯 NETWORKS MODULE TESTING COMPLETE")
print(f"{'='*60}")

if success:
    print("🎉 CONGRATULATIONS! All network tests passed!")
    print("\n✅ Your networks module successfully implements:")
    print("  • Sequential networks: flexible layer composition")
    print("  • MLP creation: automated multi-layer perceptron building")
    print("  • Architecture flexibility: shallow, deep, wide networks")
    print("  • Multiple activations: ReLU, Tanh, Sigmoid, Softmax")
    print("  • Real ML applications: classification, regression, image recognition")
    print("  • Network analysis: parameter counting and architecture comparison")
    print("  • Transfer learning: modular network composition")
    print("\n🚀 You're ready to tackle any neural network architecture!")
    print("📈 Final Progress: Networks Module ✓ COMPLETE")
else:
    print("⚠️  Some tests failed. Please review the error messages above.")
    print("\n🔧 To fix issues:")
    print("  1. Check your Sequential class implementation")
    print("  2. Verify create_mlp function layer creation")
    print("  3. Ensure proper forward pass through all layers")
    print("  4. Test shape compatibility between layers")
    print("  5. Verify activation function integration")
    print("\n💪 Keep building! These networks are the foundation of modern AI.")

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
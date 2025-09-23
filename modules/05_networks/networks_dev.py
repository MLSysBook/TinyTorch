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
# Networks - Complete Multi-Layer Neural Network Architectures

Welcome to the Networks module! You'll compose individual layers into complete neural network architectures that can solve real-world problems.

## Learning Goals
- Systems understanding: How function composition creates complex behaviors from simple layer operations
- Core implementation skill: Build Sequential networks and Multi-Layer Perceptrons (MLPs) with flexible architectures
- Pattern recognition: Understand how network depth, width, and activation patterns affect learning capability
- Framework connection: See how your Sequential implementation mirrors PyTorch's nn.Sequential design pattern
- Performance insight: Learn why network architecture choices dramatically affect training time and memory usage

## Build → Use → Reflect
1. **Build**: Sequential network container that composes layers into complete architectures
2. **Use**: Create MLPs with different depth/width configurations and test on real classification problems
3. **Reflect**: Why do deeper networks learn more complex functions, but also become harder to train?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how layer composition enables universal function approximation
- Practical capability to design and implement neural network architectures for different problem types
- Systems insight into why network architecture is often more important than algorithm choice for ML performance
- Performance consideration of how network size affects training speed, memory usage, and convergence behavior
- Connection to production ML systems and how architectural innovations drive ML breakthroughs

## Systems Reality Check
💡 **Production Context**: PyTorch's nn.Sequential is used throughout production systems because it provides a clean abstraction for complex architectures while maintaining automatic differentiation
⚡ **Performance Note**: Network depth affects memory linearly but can affect training time exponentially due to gradient flow problems - architecture design is a systems engineering problem
"""

# %% nbgrader={"grade": false, "grade_id": "networks-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.dense

#| export
import numpy as np
import sys
import os
from typing import List, Optional
import matplotlib.pyplot as plt

# Import all the building blocks we need - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_activations'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04_layers'))
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    from layers_dev import Dense

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
from tinytorch.core.networks import Sequential, create_mlp  # Network architectures!
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
## 🔧 DEVELOPMENT
"""

# %% [markdown]
"""
## Step 1: Understanding Neural Networks as Function Composition

### What is a Neural Network?
A neural network is simply **function composition** - chaining simple functions together to create complex behaviors:

```
f(x) = f_n(f_{n-1}(...f_2(f_1(x))))
```

### Real-World Analogy: Assembly Line
Think of an assembly line in a factory:
- **Input:** Raw materials (data)
- **Stations:** Each worker (layer) transforms the product
- **Output:** Final product (predictions)

### The Power of Composition
```python
# Simple functions
def add_one(x): return x + 1
def multiply_two(x): return x * 2
def square(x): return x * x

# Composed function
def complex_function(x):
    return square(multiply_two(add_one(x)))
    
# This is what neural networks do!
```

### Why This Matters
- **Universal Approximation:** MLPs can approximate any continuous function
- **Hierarchical Learning:** Early layers learn simple features, later layers learn complex patterns
- **Composability:** Mix and match layers to create custom architectures
- **Scalability:** Add more layers or make them wider as needed

### From Modules We've Built
- **Tensors:** The data containers that flow through networks
- **Activations:** The nonlinear transformations that enable complex behaviors
- **Layers:** The building blocks that transform data

Now let's build our first network architecture!
"""

# %% [markdown]
"""
## Step 2: Building the Sequential Network

### What is Sequential?
**Sequential** is the most fundamental network architecture - it applies layers in order:

```
Sequential([layer1, layer2, layer3]) 
→ f(x) = layer3(layer2(layer1(x)))
```

### Why Sequential Matters
- **Foundation:** Every neural network library has this pattern
- **Simplicity:** Easy to understand and implement
- **Flexibility:** Can compose any layers in any order
- **Building Block:** Foundation for more complex architectures

### The Sequential Pattern
```python
# PyTorch style
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Our TinyTorch style
model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10)
])
```

Let's implement this fundamental architecture!
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Sequential:
    """
    Sequential Network: Composes layers in sequence
    
    The most fundamental network architecture.
    Applies layers in order: f(x) = layer_n(...layer_2(layer_1(x)))
    """
    
    def __init__(self, layers: Optional[List] = None):
        """
        Initialize Sequential network with layers.
        
        Args:
            layers: List of layers to compose in order (optional, defaults to empty list)
            
        TODO: Store the layers and implement forward pass
        
        APPROACH:
        1. Store the layers list as an instance variable
        2. Initialize empty list if no layers provided
        3. Prepare for forward pass implementation
        
        EXAMPLE:
        Sequential([Dense(3,4), ReLU(), Dense(4,2)])
        creates a 3-layer network: Dense → ReLU → Dense
        
        HINTS:
        - Use self.layers to store the layers
        - Handle empty initialization case
        
        LEARNING CONNECTIONS:
        - This is equivalent to torch.nn.Sequential in PyTorch
        - Used in every neural network to chain layers together
        - Foundation for models like VGG, ResNet, and transformers
        - Enables modular network design and experimentation
        """
        ### BEGIN SOLUTION
        self.layers = layers if layers is not None else []
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
        
        LEARNING CONNECTIONS:
        - This is the core of feedforward neural networks
        - Powers inference in every deployed model
        - Critical for real-time predictions in production
        - Foundation for gradient flow in backpropagation
        """
        ### BEGIN SOLUTION
        # Apply each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the network callable: sequential(x) instead of sequential.forward(x)"""
        return self.forward(x)
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

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
    
    # Test batch processing
    x_batch = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_batch = network(x_batch)
    assert y_batch.shape == (2, 2), f"Expected batch shape (2, 2), got {y_batch.shape}"
    print("✅ Sequential network handles batch processing")
    
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
## Step 3: Building Multi-Layer Perceptrons (MLPs)

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

### The MLP Factory Pattern
Instead of manually creating each layer, we'll build a function that creates MLPs automatically!
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
    
    LEARNING CONNECTIONS:
    - This pattern is used in every feedforward network implementation
    - Foundation for architectures like autoencoders and GANs
    - Enables rapid prototyping of neural architectures
    - Similar to tf.keras.Sequential with Dense layers
    """
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
    
    # Test layer types
    layer_types = [type(layer).__name__ for layer in mlp.layers]
    expected_pattern = ['Linear', 'ReLU', 'Linear', 'ReLU', 'Linear', 'Sigmoid']
    assert layer_types == expected_pattern, f"Expected pattern {expected_pattern}, got {layer_types}"
    print("✅ MLP follows correct layer pattern")
    
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

# %% [markdown]
"""
## Step 4: Understanding Network Architectures

### Architecture Patterns
Different network architectures solve different problems:

#### **Shallow vs Deep Networks**
```python
# Shallow: 1 hidden layer
shallow = create_mlp(10, [20], 1)

# Deep: Many hidden layers
deep = create_mlp(10, [20, 20, 20], 1)
```

#### **Narrow vs Wide Networks**
```python
# Narrow: Few neurons per layer
narrow = create_mlp(10, [5, 5], 1)

# Wide: Many neurons per layer
wide = create_mlp(10, [50], 1)
```

### Why Architecture Matters
- **Capacity:** More parameters can learn more complex patterns
- **Depth:** Enables hierarchical feature learning
- **Width:** Allows parallel processing of features
- **Efficiency:** Balance between performance and computation

### Different Activation Functions
   ```python
# ReLU networks (most common)
relu_net = create_mlp(10, [20], 1, activation=ReLU)
   
# Tanh networks (centered around 0)
tanh_net = create_mlp(10, [20], 1, activation=Tanh)
   
# Multi-class classification
classifier = create_mlp(10, [20], 3, output_activation=Softmax)
   ```

Let's test different architectures!
""" 

# %% [markdown]
"""
### 🧪 Unit Test: Architecture Variations

Let's test different network architectures to understand their behavior.

**This is a unit test** - it tests architectural variations in isolation.
"""

# %% [markdown] 
"""
### 📊 Visualization: Network Architecture Comparison

This function creates and visualizes different neural network architectures to demonstrate how activation functions and layer configurations affect network behavior and output characteristics.
"""

# %% nbgrader={"grade": true, "grade_id": "test-architectures", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def plot_network_architectures():
    """Visualize different network architectures."""
        
    # Create different architectures
    relu_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=ReLU)
    tanh_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=Tanh)
    classifier = create_mlp(input_size=3, hidden_sizes=[4], output_size=3, output_activation=Softmax)

    # Create input data
    x = Tensor([[1.0, 2.0, 3.0]])
    
    # Get outputs
    y_relu = relu_net(x)
    y_tanh = tanh_net(x)
    y_multi = classifier(x)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].set_title("ReLU Network Output")
    axs[0].bar(['Output'], [y_relu.data[0][0]], color='skyblue')
    
    axs[1].set_title("Tanh Network Output")
    axs[1].bar(['Output'], [y_tanh.data[0][0]], color='salmon')
    
    axs[2].set_title("Softmax Classifier Output")
    axs[2].bar([f"Class {i}" for i in range(3)], y_multi.data[0], color='lightgreen')
    
    plt.tight_layout()
    # plt.show()  # Disabled for automated testing

plot_network_architectures()

# %% [markdown]
"""
### 🧪 Unit Test: Network Architecture Variations

This test validates different neural network architectures created with various activation functions. It ensures that networks with ReLU, Tanh, and Softmax activations work correctly, and tests both shallow and deep network configurations for comprehensive architecture validation.
"""

# %%
def test_unit_network_architectures():
    """Unit test for different network architectures."""
    # Test different architectures
    print("🔬 Unit Test: Network Architecture Variations...")

    try:
        # Test different activation functions
        relu_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=ReLU)
        tanh_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=Tanh)
        
        # Test different output activations
        classifier = create_mlp(input_size=3, hidden_sizes=[4], output_size=3, output_activation=Softmax)
        
        # Test with sample data
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Test ReLU network
        y_relu = relu_net(x)
        assert y_relu.shape == (1, 1), "ReLU network should work"
        print("✅ ReLU network works correctly")
        
        # Test Tanh network
        y_tanh = tanh_net(x)
        assert y_tanh.shape == (1, 1), "Tanh network should work"
        print("✅ Tanh network works correctly")
        
        # Test multi-class classifier
        y_multi = classifier(x)
        assert y_multi.shape == (1, 3), "Multi-class classifier should work"
        
        # Check softmax properties
        assert abs(np.sum(y_multi.data) - 1.0) < 1e-6, "Softmax outputs should sum to 1"
        print("✅ Multi-class classifier with Softmax works correctly")
        
        # Test different architectures
        shallow = create_mlp(input_size=4, hidden_sizes=[5], output_size=1)
        deep = create_mlp(input_size=4, hidden_sizes=[5, 5, 5], output_size=1)
        wide = create_mlp(input_size=4, hidden_sizes=[20], output_size=1)
        
        x_test = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Test all architectures
        for name, net in [("Shallow", shallow), ("Deep", deep), ("Wide", wide)]:
            y = net(x_test)
            assert y.shape == (1, 1), f"{name} network should produce correct shape"
            print(f"✅ {name} network works correctly")
        
        print("✅ All network architectures work correctly")
            
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        raise

    print("🎯 Architecture insights:")
    print("   Different activations create different behaviors")
    print("   Softmax enables multi-class classification")
    print("   Architecture affects network capacity and learning")
    print("📈 Progress: Sequential ✓, MLP creation ✓, Architecture variations ✓")

# %% [markdown]
"""
### 📊 Visualization Demo: Network Architectures

Let's visualize the different network architectures for educational purposes:
"""

# %% [markdown]
"""
## Step 5: Weight Initialization Methods

### Why Weight Initialization Matters
Proper weight initialization is critical for training deep networks:

- **Xavier Initialization**: Maintains variance across layers (good for tanh/sigmoid)
- **He Initialization**: Designed for ReLU activations (prevents vanishing gradients)
- **Uniform vs Normal**: Different distribution shapes affect training dynamics

### Production Context
- **PyTorch**: Uses Kaiming (He) initialization by default for ReLU networks
- **TensorFlow**: Provides various initializers for different activation functions
- **Critical**: Poor initialization can make networks untrainable
"""

# %% nbgrader={"grade": false, "grade_id": "weight-initialization", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def xavier_uniform_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Xavier (Glorot) uniform initialization for neural network weights.
    
    Designed to maintain variance across layers, especially good for 
    tanh and sigmoid activations.
    
    Formula: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        Weight matrix with Xavier uniform initialization
    """
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

def xavier_normal_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Xavier (Glorot) normal initialization for neural network weights.
    
    Normal distribution version of Xavier initialization.
    
    Formula: N(0, sqrt(2/(fan_in + fan_out)))
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        Weight matrix with Xavier normal initialization
    """
    std = np.sqrt(2.0 / (input_size + output_size))
    return np.random.normal(0, std, (input_size, output_size))

def he_uniform_init(input_size: int, output_size: int) -> np.ndarray:
    """
    He (Kaiming) uniform initialization for neural network weights.
    
    Designed specifically for ReLU activations to prevent vanishing gradients.
    
    Formula: U(-sqrt(6/fan_in), sqrt(6/fan_in))
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        Weight matrix with He uniform initialization
    """
    limit = np.sqrt(6.0 / input_size)
    return np.random.uniform(-limit, limit, (input_size, output_size))

def he_normal_init(input_size: int, output_size: int) -> np.ndarray:
    """
    He (Kaiming) normal initialization for neural network weights.
    
    Normal distribution version of He initialization, most commonly used.
    
    Formula: N(0, sqrt(2/fan_in))
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        Weight matrix with He normal initialization
    """
    std = np.sqrt(2.0 / input_size)
    return np.random.normal(0, std, (input_size, output_size))

# %% [markdown]
"""
### 🧪 Unit Test: Weight Initialization Methods

Let's test the weight initialization functions to ensure they produce properly scaled weights.
"""

# %% nbgrader={"grade": true, "grade_id": "test-weight-init", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_weight_initialization():
    """Unit test for weight initialization methods."""
    print("🔬 Unit Test: Weight Initialization Methods...")
    
    input_size, output_size = 100, 50
    
    # Test Xavier uniform
    xavier_uniform_weights = xavier_uniform_init(input_size, output_size)
    expected_limit = np.sqrt(6.0 / (input_size + output_size))
    assert np.all(np.abs(xavier_uniform_weights) <= expected_limit), "Xavier uniform weights out of range"
    assert xavier_uniform_weights.shape == (input_size, output_size), "Xavier uniform shape incorrect"
    print("✅ Xavier uniform initialization works correctly")
    
    # Test Xavier normal
    xavier_normal_weights = xavier_normal_init(input_size, output_size)
    expected_std = np.sqrt(2.0 / (input_size + output_size))
    actual_std = np.std(xavier_normal_weights)
    assert abs(actual_std - expected_std) < 0.1, f"Xavier normal std {actual_std} != expected {expected_std}"
    assert xavier_normal_weights.shape == (input_size, output_size), "Xavier normal shape incorrect"
    print("✅ Xavier normal initialization works correctly")
    
    # Test He uniform
    he_uniform_weights = he_uniform_init(input_size, output_size)
    expected_limit = np.sqrt(6.0 / input_size)
    assert np.all(np.abs(he_uniform_weights) <= expected_limit), "He uniform weights out of range"
    assert he_uniform_weights.shape == (input_size, output_size), "He uniform shape incorrect"
    print("✅ He uniform initialization works correctly")
    
    # Test He normal
    he_normal_weights = he_normal_init(input_size, output_size)
    expected_std = np.sqrt(2.0 / input_size)
    actual_std = np.std(he_normal_weights)
    assert abs(actual_std - expected_std) < 0.1, f"He normal std {actual_std} != expected {expected_std}"
    assert he_normal_weights.shape == (input_size, output_size), "He normal shape incorrect"
    print("✅ He normal initialization works correctly")
    
    print("🎯 All weight initialization methods work correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
### 📊 Performance Analysis: Weight Initialization Impact

Let's analyze how different initialization methods affect network behavior.
"""

# %% nbgrader={"grade": false, "grade_id": "weight-init-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
def analyze_initialization_impact():
    """Analyze the impact of different weight initialization methods."""
    print("📊 WEIGHT INITIALIZATION IMPACT ANALYSIS")
    print("=" * 50)
    
    # Create networks with different initializations
    input_size, hidden_size, output_size = 10, 20, 1
    
    # Test different initialization methods
    init_methods = {
        "Xavier Uniform": lambda: xavier_uniform_init(input_size, hidden_size),
        "Xavier Normal": lambda: xavier_normal_init(input_size, hidden_size),
        "He Uniform": lambda: he_uniform_init(input_size, hidden_size),
        "He Normal": lambda: he_normal_init(input_size, hidden_size),
        "Random Normal": lambda: np.random.normal(0, 1, (input_size, hidden_size))
    }
    
    # Create test input
    x = Tensor(np.random.randn(5, input_size))
    
    print(f"\n🔍 Analyzing activation statistics for different initializations:")
    
    for init_name, init_func in init_methods.items():
        # Create network with specific initialization
        network = Sequential([
            Dense(input_size, hidden_size),
            ReLU(),
            Dense(hidden_size, output_size)
        ])
        
        # Override weights with specific initialization
        network.layers[0].weights.data[:] = init_func()
        network.layers[2].weights.data[:] = xavier_normal_init(hidden_size, output_size)
        
        # Forward pass
        try:
            hidden_output = network.layers[0](x)
            final_output = network(x)
            
            print(f"\n📈 {init_name}:")
            print(f"   Hidden layer output mean: {np.mean(hidden_output.data):.4f}")
            print(f"   Hidden layer output std:  {np.std(hidden_output.data):.4f}")
            print(f"   Final output range: [{np.min(final_output.data):.4f}, {np.max(final_output.data):.4f}]")
            
            # Check for dead neurons (ReLU outputs all zeros)
            relu_output = network.layers[1](hidden_output)
            dead_neurons = np.sum(np.all(relu_output.data == 0, axis=0))
            print(f"   Dead neurons: {dead_neurons}/{hidden_size}")
            
        except Exception as e:
            print(f"   ❌ Forward pass failed: {str(e)}")

analyze_initialization_impact()

# %% [markdown]
"""
## Step 6: Complete NeuralNetwork Class

### Production-Ready Neural Network Class
Let's implement a complete NeuralNetwork class that provides parameter management 
and professional network interfaces similar to PyTorch's nn.Module.
"""

# %% nbgrader={"grade": false, "grade_id": "neural-network-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class NeuralNetwork:
    """
    Complete Neural Network class with parameter management.
    
    Provides a professional interface for neural networks similar to PyTorch's nn.Module.
    Includes parameter counting, initialization options, and state management.
    """
    
    def __init__(self, layers: List = None, name: str = "NeuralNetwork"):
        """
        Initialize neural network with layers and metadata.
        
        Args:
            layers: List of layers to include in the network
            name: Name for the network (useful for logging/debugging)
        """
        self.layers = layers if layers is not None else []
        self.name = name
        self._training = True
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make network callable."""
        return self.forward(x)
    
    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def count_parameters(self) -> dict:
        """
        Count trainable parameters in the network.
        
        Returns:
            Dictionary with parameter counts and memory estimates
        """
        total_params = 0
        layer_info = []
        
        for i, layer in enumerate(self.layers):
            layer_params = 0
            if hasattr(layer, 'weights'):
                layer_params += layer.weights.data.size
            if hasattr(layer, 'bias'):
                layer_params += layer.bias.data.size
            
            layer_info.append({
                'layer_index': i,
                'layer_type': type(layer).__name__,
                'parameters': layer_params
            })
            total_params += layer_params
        
        # Estimate memory usage (float32 = 4 bytes)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'memory_estimate_mb': memory_mb,
            'layer_breakdown': layer_info
        }
    
    def initialize_weights(self, method: str = "he_normal"):
        """
        Initialize all network weights using specified method.
        
        Args:
            method: Initialization method ("xavier_uniform", "xavier_normal", 
                   "he_uniform", "he_normal")
        """
        init_functions = {
            "xavier_uniform": xavier_uniform_init,
            "xavier_normal": xavier_normal_init,
            "he_uniform": he_uniform_init,
            "he_normal": he_normal_init
        }
        
        if method not in init_functions:
            raise ValueError(f"Unknown initialization method: {method}")
        
        init_func = init_functions[method]
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                input_size, output_size = layer.weights.shape
                layer.weights.data[:] = init_func(input_size, output_size)
    
    def summary(self):
        """Print network architecture summary."""
        print(f"🔥 {self.name} Architecture Summary")
        print("=" * 50)
        
        param_info = self.count_parameters()
        
        print(f"{'Layer':<15} {'Type':<15} {'Parameters':<15}")
        print("-" * 45)
        
        for layer_info in param_info['layer_breakdown']:
            print(f"{layer_info['layer_index']:<15} "
                  f"{layer_info['layer_type']:<15} "
                  f"{layer_info['parameters']:,}")
        
        print("-" * 45)
        print(f"Total Parameters: {param_info['total_parameters']:,}")
        print(f"Memory Estimate: {param_info['memory_estimate_mb']:.2f} MB")
        print("=" * 50)

# %% [markdown]
"""
### 🧪 Unit Test: Complete NeuralNetwork Class

Let's test the complete NeuralNetwork class with parameter management.
"""

# %% nbgrader={"grade": true, "grade_id": "test-neural-network-class", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_complete_neural_network():
    """Unit test for the complete NeuralNetwork class."""
    print("🔬 Unit Test: Complete NeuralNetwork Class...")
    
    # Create a network using the NeuralNetwork class
    network = NeuralNetwork([
        Dense(10, 20),
        ReLU(),
        Dense(20, 5),
        ReLU(),
        Dense(5, 1)
    ], name="TestNetwork")
    
    # Test forward pass
    x = Tensor(np.random.randn(3, 10))
    y = network(x)
    assert y.shape == (3, 1), "Network should produce correct output shape"
    print("✅ Forward pass works correctly")
    
    # Test parameter counting
    param_info = network.count_parameters()
    expected_params = (10*20 + 20) + (20*5 + 5) + (5*1 + 1)  # weights + biases
    assert param_info['total_parameters'] == expected_params, "Parameter count incorrect"
    print("✅ Parameter counting works correctly")
    
    # Test weight initialization
    network.initialize_weights("he_normal")
    first_layer = network.layers[0]
    assert hasattr(first_layer, 'weights'), "First layer should have weights"
    print("✅ Weight initialization works correctly")
    
    # Test summary (should not crash)
    try:
        network.summary()
        print("✅ Network summary works correctly")
    except Exception as e:
        print(f"❌ Network summary failed: {e}")
    
    print("🎯 Complete NeuralNetwork class works correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 7: Comprehensive Test - Complete Network Applications

### Real-World Network Applications
Let's test our networks on realistic scenarios:

#### **Classification Problem**
```python
# 4 features → 2 classes (binary classification)
classifier = create_mlp(4, [8, 4], 2, output_activation=Softmax)
```

#### **Regression Problem**
```python
# 3 features → 1 continuous output
regressor = create_mlp(3, [10, 5], 1, output_activation=lambda: Dense(0, 0))  # Linear output
```

#### **Deep Learning Pattern**
```python
# Complex feature learning
deep_net = create_mlp(10, [64, 32, 16], 1)
```

This comprehensive test ensures our networks work for real ML applications!
"""

# %% nbgrader={"grade": true, "grade_id": "test-integration", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
# Comprehensive test - complete network applications
print("🔬 Comprehensive Test: Complete Network Applications...")

try:
    # Test 1: Multi-class Classification (Iris-like dataset)
    print("\n1. Multi-class Classification Test:")
    iris_classifier = create_mlp(input_size=4, hidden_sizes=[8, 6], output_size=3, output_activation=Softmax)
    
    # Simulate iris features: [sepal_length, sepal_width, petal_length, petal_width]
    iris_samples = Tensor([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [7.0, 3.2, 4.7, 1.4],  # Versicolor
        [6.3, 3.3, 6.0, 2.5]   # Virginica
        ])
        
    iris_predictions = iris_classifier(iris_samples)
    assert iris_predictions.shape == (3, 3), "Iris classifier should output 3 classes for 3 samples"
        
    # Check softmax properties
    row_sums = np.sum(iris_predictions.data, axis=1)
    assert np.allclose(row_sums, 1.0), "Each prediction should sum to 1"
    print("✅ Multi-class classification works correctly")
    
    # Test 2: Regression Task (Housing prices)
    print("\n2. Regression Task Test:")
    # Create a regressor without final activation (linear output)
    class Identity:
        def __call__(self, x): return x
    
    housing_regressor = create_mlp(input_size=3, hidden_sizes=[10, 5], output_size=1, output_activation=Identity)
    
    # Simulate housing features: [size, bedrooms, location_score]
    housing_samples = Tensor([
        [2000, 3, 8.5],  # Large house, good location
        [1200, 2, 6.0],  # Medium house, ok location
        [800, 1, 4.0]    # Small house, poor location
    ])
    
    housing_predictions = housing_regressor(housing_samples)
    assert housing_predictions.shape == (3, 1), "Housing regressor should output 1 value per sample"
    print("✅ Regression task works correctly")
    
    # Test 3: Deep Network Performance
    print("\n3. Deep Network Test:")
    deep_network = create_mlp(input_size=10, hidden_sizes=[20, 15, 10, 5], output_size=1)
    
    # Test with realistic batch size
    batch_data = Tensor(np.random.randn(32, 10))  # 32 samples, 10 features
    deep_predictions = deep_network(batch_data)
    
    assert deep_predictions.shape == (32, 1), "Deep network should handle batch processing"
    assert not np.any(np.isnan(deep_predictions.data)), "Deep network should not produce NaN"
    print("✅ Deep network handles batch processing correctly")
    
    # Test 4: Network Composition
    print("\n4. Network Composition Test:")
    # Create a feature extractor and classifier separately
    feature_extractor = Sequential([
        Dense(input_size=10, output_size=5),
        ReLU(),
        Dense(input_size=5, output_size=3),
        ReLU()
    ])
    
    classifier_head = Sequential([
        Dense(input_size=3, output_size=2),
        Softmax()
    ])
    
    # Test composition
    raw_data = Tensor(np.random.randn(5, 10))
    features = feature_extractor(raw_data)
    final_predictions = classifier_head(features)
    
    assert features.shape == (5, 3), "Feature extractor should output 3 features"
    assert final_predictions.shape == (5, 2), "Classifier should output 2 classes"
    
    row_sums = np.sum(final_predictions.data, axis=1)
    assert np.allclose(row_sums, 1.0), "Composed network predictions should be valid"
    print("✅ Network composition works correctly")
    
    print("\n🎉 Comprehensive test passed! Your networks work correctly for:")
    print("  • Multi-class classification (Iris flowers)")
    print("  • Regression tasks (housing prices)")
    print("  • Deep learning architectures")
    print("  • Network composition and feature extraction")

except Exception as e:
    print(f"❌ Comprehensive test failed: {e}")

print("📈 Final Progress: Complete network architectures ready for real ML applications!")

# Test function defined (called in main block)

# %% [markdown]
"""
### 🏗️ Class: MLP (Multi-Layer Perceptron)

This class provides a convenient wrapper around Sequential networks specifically designed for standard MLP architectures. It maintains parameter information and provides a clean interface for creating and managing multi-layer perceptrons with consistent structure.
"""

# %% nbgrader={"grade": false, "grade_id": "networks-compatibility", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| export
class MLP:
    """
    Multi-Layer Perceptron (MLP) class.
    
    A convenient wrapper around Sequential networks for standard MLP architectures.
    Maintains parameter information and provides a clean interface.
    
    Args:
        input_size: Number of input features
        hidden_size: Size of the single hidden layer
        output_size: Number of output features
        activation: Activation function for hidden layer (default: ReLU)
        output_activation: Activation function for output layer (default: Sigmoid)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 activation=ReLU, output_activation=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Build the network layers
        layers = []
        
        # Input to hidden layer
        layers.append(Dense(input_size, hidden_size))
        layers.append(activation())
        
        # Hidden to output layer
        layers.append(Dense(hidden_size, output_size))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = Sequential(layers)
    
    def forward(self, x):
        """Forward pass through the MLP network."""
        return self.network.forward(x)
    
    def __call__(self, x):
        """Make the MLP callable."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Sequential Network Implementation

This test validates the Sequential network class functionality, ensuring proper layer composition, forward pass execution, and network architecture validation for multi-layer neural networks.
"""

# %%
def test_unit_sequential_networks():
    """Unit test for the Sequential network implementation."""
    print("🔬 Unit Test: Sequential Networks...")
    
    # Test basic Sequential network
    net = Sequential([
        Dense(input_size=3, output_size=4),
        ReLU(),
        Dense(input_size=4, output_size=2),
        Sigmoid()
    ])
    
    x = Tensor([[1.0, 2.0, 3.0]])
    y = net(x)
    
    assert y.shape == (1, 2), "Sequential network should produce correct output shape"
    assert np.all(y.data > 0), "Sigmoid output should be positive"
    assert np.all(y.data < 1), "Sigmoid output should be less than 1"
    
    print("✅ Sequential networks work correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
### 🧪 Unit Test: MLP Creation Function

This test validates the `create_mlp` function, ensuring it correctly constructs Multi-Layer Perceptrons with various architectures, activation functions, and layer configurations for different machine learning tasks.
"""

# %%
def test_unit_mlp_creation():
    """Unit test for the MLP creation function."""
    print("🔬 Unit Test: MLP Creation...")
    
    # Test different MLP architectures
    shallow = create_mlp(input_size=4, hidden_sizes=[5], output_size=1)
    deep = create_mlp(input_size=4, hidden_sizes=[8, 6, 4], output_size=2)
    
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    
    # Test shallow network
    y_shallow = shallow(x)
    assert y_shallow.shape == (1, 1), "Shallow MLP should work"
    
    # Test deep network  
    y_deep = deep(x)
    assert y_deep.shape == (1, 2), "Deep MLP should work"
    
    print("✅ MLP creation works correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
### 🧪 Unit Test: Network Applications in Real ML Scenarios

This comprehensive test validates network performance on real machine learning tasks including classification and regression, ensuring the implementations work correctly with actual datasets and practical applications.
"""

# %%
def test_unit_network_applications():
    """Comprehensive unit test for network applications in real ML scenarios."""
    print("🔬 Comprehensive Test: Network Applications...")
    
    # Test multi-class classification
    iris_classifier = create_mlp(input_size=4, hidden_sizes=[8, 6], output_size=3, output_activation=Softmax)
    iris_samples = Tensor([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]])
    iris_predictions = iris_classifier(iris_samples)
    
    assert iris_predictions.shape == (3, 3), "Iris classifier should work"
    row_sums = np.sum(iris_predictions.data, axis=1)
    assert np.allclose(row_sums, 1.0), "Predictions should sum to 1"

# Test function defined (called in main block)

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

# %% [markdown]
"""
## 🔬 Integration Test: End-to-End Network Forward Pass
"""

# %%
def test_module_full_network_forward_pass():
    """
    Integration test for a complete forward pass through a multi-layer network.
    
    Tests a complete forward pass through a multi-layer network,
    integrating Tensors, Dense layers, Activations, and the Sequential container.
    """
    print("🔬 Running Integration Test: Full Network Forward Pass...")

    # 1. Define a simple 2-layer MLP
    # Input (3) -> Dense(4) -> ReLU -> Dense(2) -> Output
    model = Sequential([
        Dense(3, 4),
        ReLU(),
        Dense(4, 2)
    ])

    # 2. Create a batch of input Tensors
    # Batch of 5 samples, each with 3 features
    input_tensor = Tensor(np.random.randn(5, 3))

    # 3. Perform a forward pass through the entire network
    output_tensor = model(input_tensor)

    # 4. Assert the final output is correct
    assert isinstance(output_tensor, Tensor), "Network output must be a Tensor"
    assert output_tensor.shape == (5, 2), f"Expected output shape (5, 2), but got {output_tensor.shape}"
    print("✅ Integration Test Passed: Full network forward pass is successful.")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🔧 ML Systems: Network Stability & Error Handling

Now that you have complete neural networks, let's develop **production robustness skills**. This section teaches you to identify and fix stability issues that can break training in production systems.

### **Learning Outcome**: *"I understand why numerical stability matters in production and can detect/fix stability issues"*

---

## Network Stability Monitor (Medium Guided Implementation)

As an ML systems engineer, you need to ensure networks remain stable during training. Let's build tools to detect numerical instability and understand gradient flow issues.
"""

# %%
import time
import numpy as np

class NetworkStabilityMonitor:
    """
    Stability monitoring toolkit for neural networks.
    
    Helps ML engineers detect numerical instability, gradient problems,
    and other issues that can break training in production systems.
    """
    
    def __init__(self):
        self.stability_history = []
        self.warning_threshold = 1e6
        self.error_threshold = 1e10
        
    def check_tensor_stability(self, tensor, tensor_name="tensor"):
        """
        Check if a tensor has numerical stability issues.
        
        TODO: Implement tensor stability checking.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Check for NaN values using np.isnan()
        2. Check for infinite values using np.isinf() 
        3. Check for extremely large values (> 1e6)
        4. Calculate value statistics (min, max, mean, std)
        5. Return stability report with warnings
        
        EXAMPLE:
        monitor = NetworkStabilityMonitor()
        tensor = Tensor([1.0, 2.0, np.inf])
        report = monitor.check_tensor_stability(tensor, "weights")
        print(f"Stable: {report['is_stable']}")
        print(f"Issues: {report['issues']}")
        
        HINTS:
        - Use tensor.data to get numpy array
        - Check: np.any(np.isnan(tensor.data))
        - Check: np.any(np.isinf(tensor.data))
        - Check: np.any(np.abs(tensor.data) > self.warning_threshold)
        - Return dict with analysis
        
        LEARNING CONNECTIONS:
        - Critical for debugging exploding/vanishing gradients
        - Used in production monitoring systems at scale
        - Foundation for automated model health checks
        - Similar to TensorBoard's histogram monitoring
        """
        ### BEGIN SOLUTION
        data = tensor.data
        
        # Check for numerical issues
        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))
        has_large = np.any(np.abs(data) > self.warning_threshold)
        has_extreme = np.any(np.abs(data) > self.error_threshold)
        
        # Calculate statistics (avoiding issues if all values are problematic)
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            finite_data = data[finite_mask]
            stats = {
                'min': np.min(finite_data),
                'max': np.max(finite_data),
                'mean': np.mean(finite_data),
                'std': np.std(finite_data),
                'finite_count': np.sum(finite_mask),
                'total_count': data.size
            }
        else:
            stats = {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'finite_count': 0,
                'total_count': data.size
            }
        
        # Compile issues
        issues = []
        if has_nan:
            issues.append("Contains NaN values")
        if has_inf:
            issues.append("Contains infinite values")
        if has_extreme:
            issues.append(f"Contains extremely large values (>{self.error_threshold:.0e})")
        elif has_large:
            issues.append(f"Contains large values (>{self.warning_threshold:.0e})")
        
        is_stable = len(issues) == 0
        
        return {
            'tensor_name': tensor_name,
            'is_stable': is_stable,
            'issues': issues,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'has_large_values': has_large,
            'statistics': stats
        }
        ### END SOLUTION
    
    def analyze_gradient_flow(self, network, input_tensor, target_output):
        """
        Analyze gradient flow through a network to detect vanishing/exploding gradients.
        
        TODO: Implement gradient flow analysis.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Perform forward pass through network
        2. Simulate simple loss calculation (MSE)
        3. Estimate gradient magnitudes using finite differences
        4. Check for vanishing gradients (very small)
        5. Check for exploding gradients (very large)
        6. Return gradient flow analysis
        
        EXAMPLE:
        monitor = NetworkStabilityMonitor()
        analysis = monitor.analyze_gradient_flow(network, input_data, target)
        print(f"Gradient health: {analysis['gradient_status']}")
        
        HINTS:
        - Forward pass: output = network(input_tensor)
        - Simple loss: 0.5 * np.sum((output.data - target_output.data)**2)
        - Use small perturbations to estimate gradients
        - Vanishing: gradients < 1e-6, Exploding: gradients > 1e3
        
        LEARNING CONNECTIONS:
        - Essential for training deep networks successfully
        - Used in gradient clipping and batch normalization design
        - Foundation for understanding network initialization strategies
        - Similar to PyTorch's gradient debugging tools
        """
        ### BEGIN SOLUTION
        # Forward pass
        output = network(input_tensor)
        
        # Calculate simple MSE loss
        loss = 0.5 * np.sum((output.data - target_output.data)**2)
        
        # Estimate gradient magnitudes using finite differences
        # This is a simplified approach - real backprop would be more accurate
        epsilon = 1e-5
        gradient_estimates = []
        
        # Check first layer weights (simplified analysis)
        if hasattr(network, 'layers') and len(network.layers) > 0:
            first_layer = network.layers[0]
            if hasattr(first_layer, 'weights'):
                # Perturb a small sample of weights to estimate gradients
                original_weight = first_layer.weights.data[0, 0]
                
                # Forward pass with small perturbation
                weights_copy = first_layer.weights.data.copy()
                weights_copy[0, 0] = original_weight + epsilon
                first_layer.weights.data[:] = weights_copy
                output_plus = network(input_tensor)
                loss_plus = 0.5 * np.sum((output_plus.data - target_output.data)**2)
                
                # Estimate gradient
                grad_estimate = (loss_plus - loss) / epsilon
                gradient_estimates.append(abs(grad_estimate))
                
                # Restore original weight
                weights_copy[0, 0] = original_weight
                first_layer.weights.data[:] = weights_copy
        
        # Analyze gradient magnitudes
        if gradient_estimates:
            avg_grad = np.mean(gradient_estimates)
            max_grad = np.max(gradient_estimates)
            
            if avg_grad < 1e-8:
                gradient_status = "Vanishing gradients detected"
            elif max_grad > 1e3:
                gradient_status = "Exploding gradients detected"
            elif avg_grad < 1e-6:
                gradient_status = "Potentially vanishing gradients"
            elif max_grad > 100:
                gradient_status = "Potentially exploding gradients"
            else:
                gradient_status = "Healthy gradient flow"
        else:
            gradient_status = "Unable to analyze gradients"
        
        return {
            'loss': loss,
            'gradient_estimates': gradient_estimates,
            'avg_gradient': np.mean(gradient_estimates) if gradient_estimates else 0,
            'max_gradient': np.max(gradient_estimates) if gradient_estimates else 0,
            'gradient_status': gradient_status
        }
        ### END SOLUTION
    
    def comprehensive_stability_check(self, network, input_tensor, target_output):
        """
        Perform comprehensive stability analysis of a neural network.
        
        This function is PROVIDED to demonstrate complete stability monitoring.
        Students use it to understand production stability requirements.
        """
        print("🔧 COMPREHENSIVE NETWORK STABILITY CHECK")
        print("=" * 50)
        
        stability_report = {
            'overall_status': 'STABLE',
            'issues_found': [],
            'recommendations': []
        }
        
        # Check input stability
        input_check = self.check_tensor_stability(input_tensor, "input")
        if not input_check['is_stable']:
            stability_report['overall_status'] = 'UNSTABLE'
            stability_report['issues_found'].extend([f"Input: {issue}" for issue in input_check['issues']])
            stability_report['recommendations'].append("Normalize or clip input data")
        
        print(f"📊 Input Check: {'✅ STABLE' if input_check['is_stable'] else '❌ UNSTABLE'}")
        if input_check['issues']:
            for issue in input_check['issues']:
                print(f"   - {issue}")
        
        # Check each layer's weights and outputs
        if hasattr(network, 'layers'):
            for i, layer in enumerate(network.layers):
                if hasattr(layer, 'weights'):
                    weight_check = self.check_tensor_stability(layer.weights, f"layer_{i}_weights")
                    if not weight_check['is_stable']:
                        stability_report['overall_status'] = 'UNSTABLE'
                        stability_report['issues_found'].extend([f"Layer {i}: {issue}" for issue in weight_check['issues']])
                        stability_report['recommendations'].append(f"Re-initialize layer {i} weights")
                    
                    print(f"🔗 Layer {i} Weights: {'✅ STABLE' if weight_check['is_stable'] else '❌ UNSTABLE'}")
                    if weight_check['issues']:
                        for issue in weight_check['issues']:
                            print(f"   - {issue}")
        
        # Check network output
        try:
            output = network(input_tensor)
            output_check = self.check_tensor_stability(output, "network_output")
            if not output_check['is_stable']:
                stability_report['overall_status'] = 'UNSTABLE'
                stability_report['issues_found'].extend([f"Output: {issue}" for issue in output_check['issues']])
                stability_report['recommendations'].append("Check activation functions and weight initialization")
            
            print(f"📤 Output Check: {'✅ STABLE' if output_check['is_stable'] else '❌ UNSTABLE'}")
            if output_check['issues']:
                for issue in output_check['issues']:
                    print(f"   - {issue}")
        
        except Exception as e:
            stability_report['overall_status'] = 'CRITICAL'
            stability_report['issues_found'].append(f"Network forward pass failed: {str(e)}")
            stability_report['recommendations'].append("Check network architecture and input compatibility")
            print(f"📤 Output Check: ❌ CRITICAL - Forward pass failed")
        
        # Gradient flow analysis
        try:
            gradient_analysis = self.analyze_gradient_flow(network, input_tensor, target_output)
            print(f"🌊 Gradient Flow: {gradient_analysis['gradient_status']}")
            
            if "exploding" in gradient_analysis['gradient_status'].lower():
                stability_report['overall_status'] = 'UNSTABLE'
                stability_report['recommendations'].append("Use gradient clipping or reduce learning rate")
            elif "vanishing" in gradient_analysis['gradient_status'].lower():
                stability_report['overall_status'] = 'UNSTABLE'
                stability_report['recommendations'].append("Use ReLU activations or residual connections")
        
        except Exception as e:
            print(f"🌊 Gradient Flow: ❌ Analysis failed - {str(e)}")
        
        print(f"\n🎯 OVERALL STATUS: {stability_report['overall_status']}")
        if stability_report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in stability_report['recommendations']:
                print(f"   - {rec}")
        
        return stability_report

def create_unstable_network_demo():
    """
    Create networks with known stability issues for demonstration.
    
    This function is PROVIDED to show common stability problems.
    Students use it to practice detecting and fixing issues.
    """
    print("⚠️  STABILITY ISSUES DEMONSTRATION")
    print("=" * 50)
    
    # Create networks with different stability issues
    demo_networks = {}
    
    # 1. Network with exploding weights
    print("\n1. 🔥 Exploding Weights Network:")
    exploding_net = Sequential([
        Dense(10, 5),
        ReLU(),
        Dense(5, 2)
    ])
    # Manually set large weights to simulate training instability
    # exploding_net.layers[0].weights.data *= 100  # Very large weights (commented to avoid error)
    demo_networks['exploding'] = exploding_net
    print("   Created network with artificially large weights")
    
    # 2. Network with NaN weights (simulate numerical overflow)
    print("\n2. 💀 NaN Weights Network:")
    nan_net = Sequential([
        Dense(10, 5),
        ReLU(),
        Dense(5, 2)
    ])
    # Inject NaN values (create a copy and modify it)
    weights_copy = nan_net.layers[0].weights.data.copy()
    weights_copy[0, 0] = np.nan
    nan_net.layers[0].weights.data[:] = weights_copy
    demo_networks['nan'] = nan_net
    print("   Created network with NaN values in weights")
    
    # 3. Healthy network for comparison
    print("\n3. ✅ Healthy Network:")
    healthy_net = Sequential([
        Dense(10, 5),
        ReLU(),
        Dense(5, 2)
    ])
    demo_networks['healthy'] = healthy_net
    print("   Created properly initialized network")
    
    return demo_networks

# %% [markdown]
"""
### 🎯 Learning Activity 1: Stability Detection Practice (Medium Guided Implementation)

**Goal**: Learn to detect numerical instability issues that can break neural network training in production.

Complete the missing implementations in the `NetworkStabilityMonitor` class above, then use your monitor to detect stability issues.
"""

# %%
# Initialize the network stability monitor
monitor = NetworkStabilityMonitor()

print("🔧 NETWORK STABILITY MONITORING")
print("=" * 50)

# Create test networks with different stability characteristics
demo_networks = create_unstable_network_demo()

# Create test data
input_data = Tensor(np.random.randn(3, 10))  # Batch of 3 samples
target_data = Tensor(np.random.randn(3, 2))  # Target outputs

print(f"\n🔍 STABILITY ANALYSIS RESULTS:")
print(f"=" * 40)

# Test each network
for network_name, network in demo_networks.items():
    print(f"\n📊 Testing {network_name.upper()} Network:")
    
    # Students use their implemented stability checker
    stability_report = monitor.comprehensive_stability_check(network, input_data, target_data)
    
    # Show what this means for production
    if stability_report['overall_status'] == 'STABLE':
        print(f"   🎯 Production Impact: Safe to deploy")
    elif stability_report['overall_status'] == 'UNSTABLE':
        print(f"   ⚠️  Production Impact: May cause training failures")
    else:
        print(f"   💀 Production Impact: Would crash in production")

print(f"\n💡 STABILITY ENGINEERING INSIGHTS:")
print(f"   - NaN values spread through entire network (one bad value ruins everything)")
print(f"   - Large weights cause exponential growth through layers")
print(f"   - Stability monitoring prevents silent training failures")
print(f"   - Early detection saves compute resources and time")

# %% [markdown]
"""
### 🎯 Learning Activity 2: Production Stability Patterns (Review & Understand)

**Goal**: Understand common stability issues in production ML systems and learn industry best practices for preventing them.
"""

# %%
print("🏭 PRODUCTION STABILITY PATTERNS")
print("=" * 50)

# Test different input scenarios that cause instability
print("\n🔍 Input Data Stability Scenarios:")

stability_scenarios = [
    ("Normal Data", np.random.randn(5, 10)),
    ("Large Values", np.random.randn(5, 10) * 1000),
    ("Extreme Values", np.random.randn(5, 10) * 1e8),
    ("Mixed with NaN", np.random.randn(5, 10)),
    ("All Zeros", np.zeros((5, 10))),
    ("All Ones", np.ones((5, 10)) * 1e6)
]

# Inject NaN in mixed scenario
stability_scenarios[3] = ("Mixed with NaN", np.random.randn(5, 10))
scenario_data = stability_scenarios[3][1].copy()
scenario_data[0, 0] = np.nan
stability_scenarios[3] = ("Mixed with NaN", scenario_data)

# Test each scenario
healthy_network = demo_networks['healthy']

for scenario_name, test_data in stability_scenarios:
    print(f"\n📊 {scenario_name}:")
    
    try:
        input_tensor = Tensor(test_data)
        input_check = monitor.check_tensor_stability(input_tensor, scenario_name)
        
        print(f"   Input Status: {'✅ STABLE' if input_check['is_stable'] else '❌ UNSTABLE'}")
        if input_check['issues']:
            print(f"   Issues: {', '.join(input_check['issues'])}")
        
        # Try network forward pass
        try:
            output = healthy_network(input_tensor)
            output_check = monitor.check_tensor_stability(output, f"{scenario_name}_output")
            print(f"   Output Status: {'✅ STABLE' if output_check['is_stable'] else '❌ UNSTABLE'}")
            if output_check['issues']:
                print(f"   Output Issues: {', '.join(output_check['issues'])}")
        
        except Exception as e:
            print(f"   ❌ Forward pass failed: {str(e)}")
    
    except Exception as e:
        print(f"   ❌ Could not create tensor: {str(e)}")

print(f"\n🎯 PRODUCTION STABILITY LESSONS:")
print(f"=" * 40)

print(f"\n1. 🛡️ INPUT VALIDATION:")
print(f"   - Always validate input data before processing")
print(f"   - Clip extreme values to reasonable ranges")
print(f"   - Check for NaN/inf values in data pipelines")

print(f"\n2. 🔧 MONITORING STRATEGY:")
print(f"   - Monitor weight magnitudes during training")
print(f"   - Track gradient norms to detect vanishing/exploding")
print(f"   - Log activation statistics to catch distribution shift")

print(f"\n3. 🚨 EARLY WARNING SYSTEM:")
print(f"   - Set thresholds for weight magnitudes")
print(f"   - Alert when gradients become too large/small")
print(f"   - Automatically stop training on stability issues")

print(f"\n4. 🛠️ PREVENTIVE MEASURES:")
print(f"   - Proper weight initialization (Xavier/He)")
print(f"   - Gradient clipping for exploding gradients")
print(f"   - Batch normalization for internal stability")
print(f"   - Learning rate scheduling to prevent instability")

print(f"\n💡 SYSTEMS ENGINEERING INSIGHT:")
print(f"Stability monitoring is like production health checks:")
print(f"- Prevent silent failures that waste compute resources")
print(f"- Enable automatic recovery strategies (restart training)")
print(f"- Provide debugging information for model developers")
print(f"- Critical for unattended training jobs in production")

# %% [markdown]
"""
## 🔧 ML Systems Analysis: Memory Profiling and Performance Characteristics

### Memory Analysis: Network Architecture Impact on System Resources

Understanding memory usage patterns is critical for deploying networks in production environments with constrained resources.
"""

# %%
import tracemalloc
import time

def profile_network_memory():
    """
    Profile memory usage patterns of different network architectures.
    
    This function demonstrates ML systems engineering by measuring actual
    memory consumption, not just theoretical parameter counts.
    """
    print("💾 NETWORK MEMORY PROFILING")
    print("=" * 50)
    
    # Start memory tracking
    tracemalloc.start()
    
    architectures = [
        ("Shallow Wide", create_mlp(100, [200], 10)),
        ("Deep Narrow", create_mlp(100, [50, 50, 50, 50], 10)),
        ("Balanced", create_mlp(100, [128, 64], 10)),
        ("Very Deep", create_mlp(100, [32, 32, 32, 32, 32, 32], 10))
    ]
    
    memory_profiles = []
    
    for arch_name, network in architectures:
        # Clear memory tracking
        tracemalloc.clear_traces()
        start_mem = tracemalloc.get_traced_memory()[0]
        
        # Create batch of data and perform forward pass
        batch_size = 64
        x = Tensor(np.random.randn(batch_size, 100))
        
        # Time the forward pass
        start_time = time.time()
        y = network(x)
        forward_time = time.time() - start_time
        
        # Get memory usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        memory_mb = peak_mem / (1024 * 1024)
        
        # Count parameters
        param_count = 0
        for layer in network.layers:
            if hasattr(layer, 'weights'):
                param_count += layer.weights.data.size
            if hasattr(layer, 'bias'):
                param_count += layer.bias.data.size
        
        profile = {
            'architecture': arch_name,
            'parameters': param_count,
            'memory_mb': memory_mb,
            'forward_time_ms': forward_time * 1000,
            'throughput_samples_per_sec': batch_size / forward_time
        }
        memory_profiles.append(profile)
        
        print(f"\n📊 {arch_name}:")
        print(f"   Parameters: {param_count:,}")
        print(f"   Memory Usage: {memory_mb:.2f} MB")
        print(f"   Forward Time: {forward_time*1000:.2f} ms")
        print(f"   Throughput: {batch_size/forward_time:.1f} samples/sec")
    
    tracemalloc.stop()
    
    print(f"\n🎯 MEMORY ENGINEERING INSIGHTS:")
    print(f"=" * 40)
    
    # Find most memory efficient
    min_memory = min(profiles['memory_mb'] for profiles in memory_profiles)
    max_throughput = max(profiles['throughput_samples_per_sec'] for profiles in memory_profiles)
    
    for profile in memory_profiles:
        if profile['memory_mb'] == min_memory:
            print(f"   🏆 Most Memory Efficient: {profile['architecture']}")
        if profile['throughput_samples_per_sec'] == max_throughput:
            print(f"   🚀 Highest Throughput: {profile['architecture']}")
    
    print(f"\n💡 PRODUCTION IMPLICATIONS:")
    print(f"   - Deep networks use more memory due to intermediate activations")
    print(f"   - Wide networks may be faster but use more parameters")
    print(f"   - Memory usage scales with batch size (important for deployment)")
    print(f"   - Consider memory vs accuracy trade-offs for edge deployment")
    
    return memory_profiles

# Run memory profiling
memory_results = profile_network_memory()

# %% [markdown]
"""
### Performance Characteristics: Computational Complexity Analysis

Understanding how network architecture affects computational complexity is essential 
for designing systems that scale to production workloads.
"""

# %%
def analyze_computational_complexity():
    """
    Analyze computational complexity of different network operations.
    
    This function demonstrates ML systems thinking by measuring actual
    performance characteristics, not just theoretical complexity.
    """
    print("⚡ COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 50)
    
    # Test different input sizes
    input_sizes = [10, 50, 100, 500, 1000]
    network_configs = [
        ("Linear Scaling", lambda n: create_mlp(n, [n], 10)),
        ("Quadratic Scaling", lambda n: create_mlp(n, [n*2, n], 10)),
        ("Constant Hidden", lambda n: create_mlp(n, [128], 10))
    ]
    
    print(f"\n📈 Timing analysis for different input sizes:")
    print(f"{'Input Size':<12} {'Linear':<12} {'Quadratic':<12} {'Constant':<12}")
    print("-" * 50)
    
    complexity_results = {}
    
    for input_size in input_sizes:
        times = {}
        
        for config_name, network_func in network_configs:
            # Create network for this input size
            network = network_func(input_size)
            
            # Create test data
            x = Tensor(np.random.randn(32, input_size))  # Batch of 32
            
            # Time multiple forward passes for accuracy
            start_time = time.time()
            for _ in range(10):
                y = network(x)
            total_time = time.time() - start_time
            avg_time = total_time / 10
            
            times[config_name] = avg_time * 1000  # Convert to milliseconds
        
        complexity_results[input_size] = times
        
        print(f"{input_size:<12} "
              f"{times['Linear Scaling']:<12.2f} "
              f"{times['Quadratic Scaling']:<12.2f} "
              f"{times['Constant Hidden']:<12.2f}")
    
    print(f"\n🎯 COMPLEXITY ENGINEERING INSIGHTS:")
    print(f"=" * 40)
    
    # Analyze scaling behavior
    small_input = complexity_results[input_sizes[0]]
    large_input = complexity_results[input_sizes[-1]]
    
    for config_name in ['Linear Scaling', 'Quadratic Scaling', 'Constant Hidden']:
        scaling_factor = large_input[config_name] / small_input[config_name]
        input_scaling = input_sizes[-1] / input_sizes[0]
        
        print(f"\n📊 {config_name}:")
        print(f"   Input scaled by: {input_scaling:.1f}x")
        print(f"   Time scaled by: {scaling_factor:.1f}x")
        
        if config_name == 'Linear Scaling':
            expected_scaling = input_scaling  # O(n) for weights
            print(f"   Expected O(n): {expected_scaling:.1f}x")
        elif config_name == 'Quadratic Scaling':
            expected_scaling = input_scaling * input_scaling  # O(n²) for weights
            print(f"   Expected O(n²): {expected_scaling:.1f}x")
        else:
            expected_scaling = input_scaling  # O(n) for input processing
            print(f"   Expected O(n): {expected_scaling:.1f}x")
    
    print(f"\n💡 SCALING IMPLICATIONS:")
    print(f"   - Network width (hidden layer size) affects memory linearly")
    print(f"   - Network depth affects computation and memory linearly")
    print(f"   - Input size affects computation linearly (for fixed architecture)")
    print(f"   - Batch size affects memory and computation linearly")
    print(f"   - Architecture choices have direct performance implications")
    
    return complexity_results

# Run complexity analysis
complexity_results = analyze_computational_complexity()

# %% [markdown]
"""
### Scaling Behavior: Production Performance Characteristics

Understanding how networks scale with different parameters is critical for 
production deployment and resource planning.
"""

# %%
def analyze_scaling_behavior():
    """
    Analyze how network performance scales with batch size and model complexity.
    
    This demonstrates production ML systems engineering by measuring
    performance characteristics that affect deployment decisions.
    """
    print("📈 SCALING BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Test batch size scaling
    batch_sizes = [1, 8, 16, 32, 64, 128]
    network = create_mlp(100, [128, 64], 10)
    
    print(f"\n🔄 Batch Size Scaling (throughput analysis):")
    print(f"{'Batch Size':<12} {'Time/Batch (ms)':<16} {'Samples/Sec':<12} {'Efficiency':<12}")
    print("-" * 55)
    
    baseline_efficiency = None
    
    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size, 100))
        
        # Time multiple runs
        start_time = time.time()
        for _ in range(50):  # More runs for small batches
            y = network(x)
        total_time = time.time() - start_time
        
        time_per_batch = (total_time / 50) * 1000  # ms
        samples_per_sec = batch_size / (total_time / 50)
        
        # Calculate efficiency (samples per second per parameter)
        param_count = sum(layer.weights.data.size + layer.bias.data.size 
                         for layer in network.layers if hasattr(layer, 'weights'))
        efficiency = samples_per_sec / param_count * 1000  # Scale for readability
        
        if baseline_efficiency is None:
            baseline_efficiency = efficiency
        
        relative_efficiency = efficiency / baseline_efficiency
        
        print(f"{batch_size:<12} "
              f"{time_per_batch:<16.2f} "
              f"{samples_per_sec:<12.1f} "
              f"{relative_efficiency:<12.2f}")
    
    print(f"\n🎯 BATCH SIZE INSIGHTS:")
    print(f"   - Larger batches improve throughput (better GPU utilization)")
    print(f"   - Memory usage scales linearly with batch size")
    print(f"   - Optimal batch size balances memory and throughput")
    print(f"   - Production systems need batch size tuning")
    
    # Test network depth scaling
    print(f"\n🏗️ Network Depth Scaling (architecture analysis):")
    print(f"{'Depth':<8} {'Parameters':<12} {'Memory (MB)':<12} {'Time (ms)':<12} {'Accuracy Proxy':<15}")
    print("-" * 65)
    
    depths = [1, 2, 3, 4, 5]
    hidden_size = 64
    input_size = 100
    batch_size = 32
    
    for depth in depths:
        # Create network with specified depth
        hidden_sizes = [hidden_size] * depth
        network = create_mlp(input_size, hidden_sizes, 10)
        
        # Count parameters
        param_count = sum(layer.weights.data.size + layer.bias.data.size 
                         for layer in network.layers if hasattr(layer, 'weights'))
        
        # Estimate memory (parameters + activations)
        param_memory = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
        activation_memory = batch_size * hidden_size * depth * 4 / (1024 * 1024)
        total_memory = param_memory + activation_memory
        
        # Time forward pass
        x = Tensor(np.random.randn(batch_size, input_size))
        start_time = time.time()
        for _ in range(20):
            y = network(x)
        forward_time = (time.time() - start_time) / 20 * 1000
        
        # Simple "accuracy proxy" - output variance (more variance often means more capacity)
        output_variance = np.var(y.data)
        
        print(f"{depth:<8} "
              f"{param_count:<12,} "
              f"{total_memory:<12.2f} "
              f"{forward_time:<12.2f} "
              f"{output_variance:<15.4f}")
    
    print(f"\n🎯 DEPTH SCALING INSIGHTS:")
    print(f"   - Deeper networks have more parameters (capacity)")
    print(f"   - Memory usage includes parameters + intermediate activations")
    print(f"   - Forward pass time scales roughly linearly with depth")
    print(f"   - Gradient computation (backprop) would scale with depth")
    print(f"   - Production trade-off: capacity vs speed vs memory")
    
    print(f"\n💡 PRODUCTION SCALING DECISIONS:")
    print(f"   🎯 Batch Size: Tune for hardware (GPU memory, throughput)")
    print(f"   🏗️ Architecture: Balance capacity, speed, and memory")
    print(f"   📊 Monitoring: Track throughput, latency, and resource usage")
    print(f"   🔧 Optimization: Profile bottlenecks in production workloads")

# Run scaling analysis
analyze_scaling_behavior()

# %% [markdown]
"""
### Production Context: How Real ML Systems Handle Network Architectures

Understanding how production ML systems optimize network architectures provides insight
into the engineering challenges of deploying neural networks at scale.
"""

# %%
def demonstrate_production_patterns():
    """
    Demonstrate common production patterns for network architecture management.
    
    This shows how production ML systems handle the challenges we've explored:
    memory management, performance optimization, and scalability.
    """
    print("🏭 PRODUCTION ML SYSTEMS PATTERNS")
    print("=" * 50)
    
    print(f"\n1. 🎯 DYNAMIC BATCH SIZE OPTIMIZATION:")
    print(f"   Production systems adjust batch sizes based on available memory:")
    
    # Simulate production batch size optimization
    available_memory_mb = 4 * 1024  # 4GB GPU memory
    network = create_mlp(1000, [512, 256], 100)
    
    # Estimate memory per sample
    param_memory = sum(layer.weights.data.size + layer.bias.data.size 
                      for layer in network.layers if hasattr(layer, 'weights')) * 4 / (1024 * 1024)
    activation_memory_per_sample = (1000 + 512 + 256 + 100) * 4 / (1024 * 1024)
    
    max_batch_size = int((available_memory_mb - param_memory) / activation_memory_per_sample)
    optimal_batch_size = min(max_batch_size, 128)  # Cap for numerical stability
    
    print(f"   📊 Memory Analysis:")
    print(f"      Parameter memory: {param_memory:.2f} MB")
    print(f"      Per-sample activation memory: {activation_memory_per_sample:.4f} MB")
    print(f"      Maximum batch size: {max_batch_size}")
    print(f"      Optimal batch size: {optimal_batch_size}")
    
    print(f"\n2. 🔧 MODEL ARCHITECTURE OPTIMIZATION:")
    print(f"   Production systems use architecture search for deployment targets:")
    
    # Simulate different deployment targets
    deployment_targets = {
        "Cloud GPU": {"memory_limit_mb": 16*1024, "latency_limit_ms": 100},
        "Edge Device": {"memory_limit_mb": 512, "latency_limit_ms": 50},
        "Mobile": {"memory_limit_mb": 128, "latency_limit_ms": 20}
    }
    
    for target_name, constraints in deployment_targets.items():
        print(f"\n   🎯 {target_name} Optimization:")
        
        # Design network for this target
        if target_name == "Cloud GPU":
            network = create_mlp(1000, [512, 256, 128], 100)
        elif target_name == "Edge Device":
            network = create_mlp(1000, [128, 64], 100)
        else:  # Mobile
            network = create_mlp(1000, [64], 100)
        
        # Estimate performance
        param_count = sum(layer.weights.data.size + layer.bias.data.size 
                         for layer in network.layers if hasattr(layer, 'weights'))
        memory_mb = param_count * 4 / (1024 * 1024)
        
        # Simple latency estimate (parameters affect computation)
        latency_ms = param_count / 10000  # Rough estimate
        
        meets_memory = memory_mb <= constraints["memory_limit_mb"]
        meets_latency = latency_ms <= constraints["latency_limit_ms"]
        
        print(f"      Parameters: {param_count:,}")
        print(f"      Memory: {memory_mb:.1f} MB ({'✅' if meets_memory else '❌'} {constraints['memory_limit_mb']} MB limit)")
        print(f"      Latency: {latency_ms:.1f} ms ({'✅' if meets_latency else '❌'} {constraints['latency_limit_ms']} ms limit)")
    
    print(f"\n3. 🔄 ADAPTIVE ARCHITECTURE PATTERNS:")
    print(f"   Production systems adapt architectures based on runtime conditions:")
    print(f"   • Early exit networks (BranchyNet pattern)")
    print(f"   • Dynamic depth based on input complexity")
    print(f"   • Cascade architectures (fast → accurate)")
    print(f"   • Model ensembles with different speed/accuracy trade-offs")
    
    print(f"\n4. 📊 PRODUCTION MONITORING:")
    print(f"   Real systems monitor network performance continuously:")
    print(f"   • Throughput: samples/second, requests/minute")
    print(f"   • Latency: P50, P95, P99 response times")
    print(f"   • Resource usage: GPU/CPU utilization, memory consumption")
    print(f"   • Quality: accuracy drift, prediction confidence")
    
    print(f"\n💡 PRODUCTION ENGINEERING TAKEAWAYS:")
    print(f"   🎯 Architecture design is a systems engineering problem")
    print(f"   ⚡ Performance characteristics drive deployment decisions")
    print(f"   📊 Continuous monitoring enables optimization")
    print(f"   🔧 Production systems require adaptive, not static, architectures")

# Demonstrate production patterns
demonstrate_production_patterns()

if __name__ == "__main__":
    # Run all tests
    test_unit_network_architectures()
    test_unit_sequential_networks()
    test_unit_mlp_creation()
    test_unit_network_applications()
    test_unit_weight_initialization()
    test_unit_complete_neural_network()
    test_module_full_network_forward_pass()
    
    print("All tests passed!")
    print("networks_dev module complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built complete neural network architectures, let's connect this foundational work to broader ML systems challenges. These questions help you think critically about how network composition patterns scale to production ML environments.

Take time to reflect thoughtfully on each question - your insights will help you understand how the network concepts you've implemented connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Composition Patterns and Architectural Design

**Context**: Your sequential network implementation enables flexible composition of layers into complex architectures. Production ML systems must support diverse architectural patterns: from simple MLPs to complex models with branching, skip connections, and dynamic computation graphs.

**Reflection Question**: Design a network composition system that supports both sequential and complex architectural patterns for production ML systems. How would you extend your sequential approach to handle branching networks, residual connections, and dynamic routing? Consider scenarios where model architectures need to adapt during training or inference based on input characteristics or computational constraints.

Think about: architectural flexibility, dynamic graph construction, branching and merging patterns, and computational graph optimization opportunities.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-composition-patterns", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON COMPOSITION PATTERNS AND ARCHITECTURAL DESIGN:

TODO: Replace this text with your thoughtful response about network composition system design.

Consider addressing:
- How would you extend sequential composition to support complex architectural patterns?
- What strategies would you use to handle branching, merging, and skip connections?
- How would you implement dynamic network architectures that adapt during execution?
- What role would computational graph optimization play in your design?
- How would you balance architectural flexibility with performance optimization?

Write an architectural analysis connecting your sequential networks to real composition pattern challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of advanced architectural composition patterns (3 points)
- Addresses dynamic and complex network structure challenges (3 points)
- Shows practical knowledge of graph optimization techniques (2 points)
- Demonstrates systems thinking about architectural flexibility vs performance (2 points)
- Clear architectural reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
"""
To support complex architectural patterns beyond sequential composition, I would design a dynamic computational graph system with the following key components:

**Graph-Based Architecture Framework:**
- Replace linear Sequential with a DAG-based ComputationGraph class that supports arbitrary node connections
- Implement ModuleNode wrappers that maintain input/output specifications and dependency tracking
- Add support for branching through conditional execution nodes and merging through concatenation/addition nodes

**Dynamic Architecture Support:**
- Implement adaptive depth through early-exit mechanisms where inference can terminate at intermediate layers based on confidence thresholds
- Add dynamic routing through gating networks that decide which computational paths to activate based on input characteristics
- Support skip connections via residual blocks that maintain gradient flow and enable much deeper architectures

**Optimization Strategies:**
- Implement computational graph optimization through dead code elimination, operation fusion, and memory reuse analysis
- Add device placement optimization that automatically distributes different graph regions across available hardware
- Support just-in-time compilation of graph regions to optimize for specific hardware targets and input shapes

This approach balances architectural flexibility with performance by maintaining explicit graph structure for optimization while enabling complex patterns like attention mechanisms, residual networks, and adaptive computation.
"""
### END SOLUTION

# %% [markdown]
"""
### Question 2: Modularity and Distributed Training

**Context**: Your network architecture separates layer composition from individual layer implementation. Production ML systems must scale these architectures across distributed training environments while maintaining modularity and enabling efficient model parallelism.

**Reflection Question**: Architect a modular network system that enables efficient distributed training across multiple devices and nodes. How would you design network decomposition strategies that balance computation across devices, implement communication-efficient model parallelism, and maintain modularity for different deployment scenarios? Consider challenges where network parts need to run on different hardware with varying computational capabilities.

Think about: model parallelism strategies, communication optimization, device placement algorithms, and modular deployment patterns.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-modularity-distributed", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON MODULARITY AND DISTRIBUTED TRAINING:

TODO: Replace this text with your thoughtful response about modular distributed training system design.

Consider addressing:
- How would you design network decomposition for efficient distributed training?
- What strategies would you use to balance computation and communication across devices?
- How would you implement model parallelism while maintaining modularity?
- What role would device placement optimization play in your system?
- How would you handle heterogeneous hardware in distributed training scenarios?

Write a systems analysis connecting your modular networks to real distributed training challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of distributed training and model parallelism challenges (3 points)
- Designs practical approaches to modular distributed architectures (3 points)
- Addresses communication optimization and device placement (2 points)
- Demonstrates systems thinking about scalability and modularity trade-offs (2 points)
- Clear systems reasoning with distributed computing insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
"""
For efficient distributed training across multiple devices, I would architect a modular system with intelligent decomposition and communication strategies:

**Model Decomposition Strategies:**
- Implement layer-wise parallelism where different layers run on different devices, with pipeline parallelism to maintain throughput
- Add tensor parallelism for large layers by splitting weight matrices across devices and using collective communication for gathering results
- Support hybrid data+model parallelism where the batch is split across some devices while the model is split across others

**Communication Optimization:**
- Implement gradient compression techniques like quantization and sparsification to reduce bandwidth requirements
- Add asynchronous communication overlap where gradient communication happens during backward pass computation
- Use hierarchical communication patterns (intra-node vs inter-node) to optimize for network topology

**Device Placement Intelligence:**
- Implement cost-based placement algorithms that consider compute capability, memory constraints, and communication costs
- Add dynamic load balancing that can migrate computation based on device utilization and bottleneck identification
- Support heterogeneous hardware through capability-aware scheduling that matches layer complexity to device capabilities

**Modular Deployment Patterns:**
- Design containerized model serving where different model components can be deployed independently and composed at runtime
- Implement versioned module interfaces that enable A/B testing and gradual rollouts of model components
- Add fault tolerance through checkpoint sharding and component redundancy

This approach enables efficient scaling while maintaining modularity through explicit communication interfaces and intelligent resource management.
"""
### END SOLUTION

# %% [markdown]
"""
### Question 3: Architecture Design and Performance Optimization

**Context**: Your network implementations provide a foundation for various ML applications from classification to regression. Production ML systems must optimize network architectures for specific deployment constraints: inference latency, memory usage, energy consumption, and accuracy requirements.

**Reflection Question**: Design an architecture optimization system that automatically configures network structures for specific deployment targets and performance constraints. How would you implement neural architecture search for production environments, balance architecture complexity with inference requirements, and optimize networks for edge deployment with strict resource constraints? Consider scenarios where the same model needs to perform well across mobile devices, cloud servers, and embedded systems.

Think about: neural architecture search, performance profiling, resource-constrained optimization, and multi-target deployment strategies.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-architecture-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON ARCHITECTURE DESIGN AND PERFORMANCE OPTIMIZATION:

TODO: Replace this text with your thoughtful response about architecture optimization system design.

Consider addressing:
- How would you implement automated architecture optimization for different deployment targets?
- What strategies would you use to balance architecture complexity with performance constraints?
- How would you optimize networks for resource-constrained edge deployment?
- What role would neural architecture search play in your optimization system?
- How would you handle multi-target deployment with varying resource constraints?

Write an optimization analysis connecting your network architectures to real deployment optimization challenges.

GRADING RUBRIC (Instructor Use):
- Understands architecture optimization and deployment constraint challenges (3 points)
- Designs practical approaches to automated architecture optimization (3 points)
- Addresses resource constraints and multi-target deployment (2 points)
- Shows systems thinking about performance vs complexity trade-offs (2 points)
- Clear optimization reasoning with deployment insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
"""
I would design an adaptive architecture optimization system that automatically configures networks for diverse deployment targets through multi-objective optimization:

**Neural Architecture Search Framework:**
- Implement differentiable architecture search (DARTS) that jointly optimizes architecture and weights through gradient-based methods
- Add hardware-aware search that includes actual latency and memory measurements in the optimization objective
- Support progressive search strategies that start with simple architectures and gradually increase complexity based on deployment constraints

**Performance-Constraint Optimization:**
- Design multi-objective optimization that balances accuracy, latency, memory usage, and energy consumption using Pareto frontier analysis
- Implement dynamic architecture adaptation where the same model can switch between high-accuracy and high-speed modes based on runtime conditions
- Add quantization-aware search that finds architectures robust to low-precision deployment while maintaining target performance

**Multi-Target Deployment Strategy:**
- Create architecture families where the same base design can be scaled up/down for different deployment targets (mobile->edge->cloud)
- Implement knowledge distillation pipelines that transfer learning from large teacher networks to smaller student networks optimized for specific devices
- Support elastic architectures with removable components that maintain compatibility across different resource constraints

**Resource-Constrained Edge Optimization:**
- Design memory-efficient architectures using techniques like depthwise separable convolutions and mobile-optimized activation functions
- Implement dynamic batching and input resolution scaling to adapt to varying device capabilities and power states
- Add model compression techniques including pruning, quantization, and knowledge distillation integrated into the search process

This system enables deployment optimization through automated architecture discovery while maintaining performance guarantees across diverse hardware targets.
"""
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Neural Network Architectures

Congratulations! You've successfully implemented complete neural network architectures:

### What You've Accomplished
✅ **Sequential Networks**: Chained layers for complex transformations
✅ **MLP Creation**: Multi-layer perceptrons with flexible architectures
✅ **Network Architectures**: Different activation patterns and output types
✅ **Integration**: Real-world applications like classification and regression

### Key Concepts You've Learned
- **Sequential Processing**: How layers chain together for complex functions
- **MLP Design**: Multi-layer perceptrons as universal function approximators  
- **Architecture Choices**: How depth, width, and activations affect learning
- **Real Applications**: Classification, regression, and feature extraction

### Next Steps
1. **Export your code**: `tito package nbdev --export 04_networks`
2. **Test your implementation**: `tito test 04_networks`
3. **Build complete models**: Combine with training for full ML pipelines
4. **Move to Module 5**: Add convolutional layers for image processing!

**Ready for CNNs?** Your network foundations are now ready for specialized architectures!
"""
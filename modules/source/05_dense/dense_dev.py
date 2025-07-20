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
# Networks - Neural Network Architectures

Welcome to the Networks module! This is where we compose layers into complete neural network architectures.

## Learning Goals
- Understand networks as function composition: `f(x) = layer_n(...layer_2(layer_1(x)))`
- Build the Sequential network architecture for composing layers
- Create common network patterns like MLPs (Multi-Layer Perceptrons)
- Visualize network architectures and understand their capabilities
- Master forward pass inference through complete networks

## Build → Use → Reflect
1. **Build**: Sequential networks that compose layers into complete architectures
2. **Use**: Create different network patterns and run inference
3. **Reflect**: How architecture design affects network behavior and capability

## What You'll Learn
By the end of this module, you'll understand:
- How simple layers combine to create complex behaviors
- The fundamental Sequential architecture pattern
- How to build MLPs with any number of layers
- Different network architectures (shallow, deep, wide)
- How neural networks approximate complex functions
"""

# %% nbgrader={"grade": false, "grade_id": "networks-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.dense

#| export
import numpy as np
import sys
import os
from typing import List, Union, Optional, Callable
import matplotlib.pyplot as plt

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
    expected_pattern = ['Dense', 'ReLU', 'Dense', 'ReLU', 'Dense', 'Sigmoid']
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

# %% nbgrader={"grade": true, "grade_id": "test-architectures", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def plot_network_architectures():
    """Visualize different network architectures."""
    if not _should_show_plots():
        return
        
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
    plt.show()

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
    
    # Plot the architectures if not in test mode
    plot_network_architectures()
        
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
## Step 5: Comprehensive Test - Complete Network Applications

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

def test_unit_network_architectures():
    """Unit test for different network architectures."""
    print("🔬 Unit Test: Network Architectures...")
    
    # Test different activation functions
    relu_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=ReLU)
    tanh_net = create_mlp(input_size=3, hidden_sizes=[4], output_size=1, activation=Tanh)
    
    # Test multi-class classifier
    classifier = create_mlp(input_size=3, hidden_sizes=[4], output_size=3, output_activation=Softmax)
    
    x = Tensor([[1.0, 2.0, 3.0]])
    
    # Test all architectures
    y_relu = relu_net(x)
    y_tanh = tanh_net(x)
    y_multi = classifier(x)
    
    assert y_relu.shape == (1, 1), "ReLU network should work"
    assert y_tanh.shape == (1, 1), "Tanh network should work"
    assert y_multi.shape == (1, 3), "Multi-class classifier should work"
    assert abs(np.sum(y_multi.data) - 1.0) < 1e-6, "Softmax outputs should sum to 1"
    
    print("✅ Network architectures work correctly")

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

# %% [markdown]
"""
## 🤖 AUTO TESTING
"""

# %%
if __name__ == "__main__":
    test_module_full_network_forward_pass()
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Networks")

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
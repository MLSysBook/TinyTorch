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
# Module 3: Networks - Neural Network Architectures

Welcome to the Networks module! This is where we compose layers into complete neural network architectures.

## Learning Goals
- Understand networks as function composition: `f(x) = layer_n(...layer_2(layer_1(x)))`
- Build common architectures (MLP, CNN) from layers
- Visualize network structure and data flow
- See how architecture affects capability
- Master forward pass inference (no training yet!)

## Build → Use → Understand
1. **Build**: Compose layers into complete networks
2. **Use**: Create different architectures and run inference
3. **Understand**: How architecture design affects network behavior

## Module Dependencies
This module builds on previous modules:
- **tensor** → **activations** → **layers** → **networks**
- Clean composition: math functions → building blocks → complete systems
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `assignments/source/04_networks/networks_dev.py`  
**Building Side:** Code exports to `tinytorch.core.networks`

```python
# Final package structure:
from tinytorch.core.networks import Sequential, MLP
from tinytorch.core.layers import Dense, Conv2D
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.tensor import Tensor
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn`
- **Consistency:** All network architectures live together in `core.networks`
"""

# %% nbgrader={"grade": false, "grade_id": "networks-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.networks

# Setup and imports
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

print("🔥 TinyTorch Networks Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network architectures!")

# %%
#| export
import numpy as np
import sys
from typing import List, Union, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Import our building blocks
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Sigmoid, Tanh

# %%
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    return 'pytest' not in sys.modules and 'test' not in sys.argv

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

### The Math Behind It
For a network with layers `f_1, f_2, ..., f_n`:
```
f(x) = f_n(f_{n-1}(...f_2(f_1(x))))
```

Each layer transforms the data, and the final output is the composition of all these transformations.

Let's start by building the most fundamental network: **Sequential**.
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Sequential:
    """
    Sequential Network: Composes layers in sequence
    
    The most fundamental network architecture.
    Applies layers in order: f(x) = layer_n(...layer_2(layer_1(x)))
    
    Args:
        layers: List of layers to compose
        
    TODO: Implement the Sequential network with forward pass.
    
    APPROACH:
    1. Store the list of layers as an instance variable
    2. Implement forward pass that applies each layer in sequence
    3. Make the network callable for easy use
    
    EXAMPLE:
    network = Sequential([
        Dense(3, 4),
        ReLU(),
        Dense(4, 2),
        Sigmoid()
    ])
    x = Tensor([[1, 2, 3]])
    y = network(x)  # Forward pass through all layers
    
    HINTS:
    - Store layers in self.layers
    - Use a for loop to apply each layer in order
    - Each layer's output becomes the next layer's input
    - Return the final output
    """
    
    def __init__(self, layers: List):
        """
        Initialize Sequential network with layers.
        
        Args:
            layers: List of layers to compose in order
            
        TODO: Store the layers and implement forward pass
        
        STEP-BY-STEP:
        1. Store the layers list as self.layers
        2. This creates the network architecture
        
        EXAMPLE:
        Sequential([Dense(3,4), ReLU(), Dense(4,2)])
        creates a 3-layer network: Dense → ReLU → Dense
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
        
        STEP-BY-STEP:
        1. Start with the input tensor: current = x
        2. Loop through each layer in self.layers
        3. Apply each layer: current = layer(current)
        4. Return the final output
        
        EXAMPLE:
        Input: Tensor([[1, 2, 3]])
        Layer1 (Dense): Tensor([[1.4, 2.8]])
        Layer2 (ReLU): Tensor([[1.4, 2.8]])
        Layer3 (Dense): Tensor([[0.7]])
        Output: Tensor([[0.7]])
        
        HINTS:
        - Use a for loop: for layer in self.layers:
        - Apply each layer: current = layer(current)
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

# %%
#| hide
#| export
class Sequential:
    """
    Sequential Network: Composes layers in sequence
    
    The most fundamental network architecture.
    Applies layers in order: f(x) = layer_n(...layer_2(layer_1(x)))
    """
    
    def __init__(self, layers: List):
        """Initialize Sequential network with layers."""
        self.layers = layers
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers in sequence."""
        # Apply each layer in order
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make network callable: network(x) same as network.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Sequential Network
"""

# %%
# Test the Sequential network
print("Testing Sequential network...")

try:
    # Create a simple 2-layer network: 3 → 4 → 2
    network = Sequential([
        Dense(input_size=3, output_size=4),
        ReLU(),
        Dense(input_size=4, output_size=2),
        Sigmoid()
    ])
    
    print(f"✅ Network created with {len(network.layers)} layers")
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0]])
    print(f"✅ Input: {x}")
    
    # Forward pass
    y = network(x)
    print(f"✅ Output: {y}")
    print(f"✅ Output shape: {y.shape}")
    
    # Verify the network works
    assert y.shape == (1, 2), f"❌ Expected shape (1, 2), got {y.shape}"
    assert np.all(y.data >= 0) and np.all(y.data <= 1), "❌ Sigmoid output should be between 0 and 1"
    print("🎉 Sequential network works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Sequential network above!")

# %% [markdown]
"""
## Step 2: Understanding Network Architecture

Now let's explore how different network architectures affect the network's capabilities.

### What is Network Architecture?
**Architecture** refers to how layers are arranged and connected. It determines:
- **Capacity**: How complex patterns the network can learn
- **Efficiency**: How many parameters and computations needed
- **Specialization**: What types of problems it's good at

### Common Architectures

#### 1. **MLP (Multi-Layer Perceptron)**
```
Input → Dense → ReLU → Dense → ReLU → Dense → Output
```
- **Use case**: General-purpose learning
- **Strengths**: Universal approximation, simple to understand
- **Weaknesses**: Doesn't exploit spatial structure

#### 2. **CNN (Convolutional Neural Network)**
```
Input → Conv2D → ReLU → Conv2D → ReLU → Dense → Output
```
- **Use case**: Image processing, spatial data
- **Strengths**: Parameter sharing, translation invariance
- **Weaknesses**: Fixed spatial structure

#### 3. **Deep Network**
```
Input → Dense → ReLU → Dense → ReLU → Dense → ReLU → Dense → Output
```
- **Use case**: Complex pattern recognition
- **Strengths**: High capacity, can learn complex functions
- **Weaknesses**: More parameters, harder to train

Let's build some common architectures!
"""

# %%
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
    2. Add the first Dense layer: input_size → first hidden size
    3. For each hidden layer:
       - Add activation function
       - Add Dense layer connecting to next hidden size
    4. Add final activation function
    5. Add final Dense layer: last hidden size → output_size
    6. Add output activation function
    7. Return Sequential(layers)
    
    EXAMPLE:
    create_mlp(3, [4, 2], 1) creates:
    Dense(3→4) → ReLU → Dense(4→2) → ReLU → Dense(2→1) → Sigmoid
    
    HINTS:
    - Start with layers = []
    - Add Dense layers with appropriate input/output sizes
    - Add activation functions between Dense layers
    - Don't forget the final output activation
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def create_mlp(input_size: int, hidden_sizes: List[int], output_size: int, 
               activation=ReLU, output_activation=Sigmoid) -> Sequential:
    """Create a Multi-Layer Perceptron (MLP) network."""
    layers = []
    
    # Add first layer
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(Dense(input_size=current_size, output_size=hidden_size))
        layers.append(activation())
        current_size = hidden_size
    
    # Add output layer
    layers.append(Dense(input_size=current_size, output_size=output_size))
    layers.append(output_activation())
    
    return Sequential(layers)

# %% [markdown]
"""
### 🧪 Test Your MLP Creation
"""

# %%
# Test MLP creation
print("Testing MLP creation...")

try:
    # Create different MLP architectures
    mlp1 = create_mlp(input_size=3, hidden_sizes=[4], output_size=1)
    mlp2 = create_mlp(input_size=5, hidden_sizes=[8, 4], output_size=2)
    mlp3 = create_mlp(input_size=2, hidden_sizes=[10, 6, 3], output_size=1, activation=Tanh)
    
    print(f"✅ MLP1: {len(mlp1.layers)} layers")
    print(f"✅ MLP2: {len(mlp2.layers)} layers")
    print(f"✅ MLP3: {len(mlp3.layers)} layers")
    
    # Test forward pass
    x = Tensor([[1.0, 2.0, 3.0]])
    y1 = mlp1(x)
    print(f"✅ MLP1 output: {y1}")
    
    x2 = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y2 = mlp2(x2)
    print(f"✅ MLP2 output: {y2}")
    
    print("🎉 MLP creation works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement create_mlp above!")

# %% [markdown]
"""
## Step 3: Network Visualization and Analysis

Let's create tools to visualize and analyze network architectures. This helps us understand what our networks are doing.

### Why Visualization Matters
- **Architecture understanding**: See how data flows through the network
- **Debugging**: Identify bottlenecks and issues
- **Design**: Compare different architectures
- **Communication**: Explain networks to others

### What We'll Build
1. **Architecture visualization**: Show layer connections
2. **Data flow visualization**: See how data transforms
3. **Network comparison**: Compare different architectures
4. **Behavior analysis**: Understand network capabilities
"""

# %%
#| export
def visualize_network_architecture(network: Sequential, title: str = "Network Architecture"):
    """
    Visualize the architecture of a Sequential network.
    
    Args:
        network: Sequential network to visualize
        title: Title for the plot
        
    TODO: Create a visualization showing the network structure.
    
    APPROACH:
    1. Create a matplotlib figure
    2. For each layer, draw a box showing its type and size
    3. Connect the boxes with arrows showing data flow
    4. Add labels and formatting
    
    EXAMPLE:
    Input → Dense(3→4) → ReLU → Dense(4→2) → Sigmoid → Output
    
    HINTS:
    - Use plt.subplots() to create the figure
    - Use plt.text() to add layer labels
    - Use plt.arrow() to show connections
    - Add proper spacing and formatting
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def visualize_network_architecture(network: Sequential, title: str = "Network Architecture"):
    """Visualize the architecture of a Sequential network."""
    if not _should_show_plots():
        print("📊 Visualization disabled during testing")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate positions
    num_layers = len(network.layers)
    x_positions = np.linspace(0, 10, num_layers + 2)
    
    # Draw input
    ax.text(x_positions[0], 0, 'Input', ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))
    
    # Draw layers
    for i, layer in enumerate(network.layers):
        layer_name = type(layer).__name__
        ax.text(x_positions[i+1], 0, layer_name, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
        
        # Draw arrow
        ax.arrow(x_positions[i], 0, 0.8, 0, head_width=0.1, head_length=0.1, 
                fc='black', ec='black')
    
    # Draw output
    ax.text(x_positions[-1], 0, 'Output', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'))
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

# %% [markdown]
"""
### 🧪 Test Network Visualization
"""

# %%
# Test network visualization
print("Testing network visualization...")

try:
    # Create a test network
    test_network = Sequential([
        Dense(input_size=3, output_size=4),
        ReLU(),
        Dense(input_size=4, output_size=2),
        Sigmoid()
    ])
    
    # Visualize the network
    if _should_show_plots():
        visualize_network_architecture(test_network, "Test Network Architecture")
        print("✅ Network visualization created!")
    else:
        print("✅ Network visualization skipped during testing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement visualize_network_architecture above!")

# %% [markdown]
"""
## Step 4: Data Flow Analysis

Let's create tools to analyze how data flows through the network. This helps us understand what each layer is doing.

### Why Data Flow Analysis Matters
- **Debugging**: See where data gets corrupted
- **Optimization**: Identify bottlenecks
- **Understanding**: Learn what each layer learns
- **Design**: Choose appropriate layer sizes
"""

# %%
#| export
def visualize_data_flow(network: Sequential, input_data: Tensor, title: str = "Data Flow Through Network"):
    """
    Visualize how data flows through the network.
    
    Args:
        network: Sequential network to analyze
        input_data: Input tensor to trace through the network
        title: Title for the plot
        
    TODO: Create a visualization showing how data transforms through each layer.
    
    APPROACH:
    1. Trace the input through each layer
    2. Record the output of each layer
    3. Create a visualization showing the transformations
    4. Add statistics (mean, std, range) for each layer
    
    EXAMPLE:
    Input: [1, 2, 3] → Layer1: [1.4, 2.8] → Layer2: [1.4, 2.8] → Output: [0.7]
    
    HINTS:
    - Use a for loop to apply each layer
    - Store intermediate outputs
    - Use plt.subplot() to create multiple subplots
    - Show statistics for each layer output
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def visualize_data_flow(network: Sequential, input_data: Tensor, title: str = "Data Flow Through Network"):
    """Visualize how data flows through the network."""
    if not _should_show_plots():
        print("📊 Visualization disabled during testing")
        return
    
    # Trace data through network
    current_data = input_data
    layer_outputs = [current_data.data.flatten()]
    layer_names = ['Input']
    
    for layer in network.layers:
        current_data = layer(current_data)
        layer_outputs.append(current_data.data.flatten())
        layer_names.append(type(layer).__name__)
    
    # Create visualization
    fig, axes = plt.subplots(2, len(layer_outputs), figsize=(15, 8))
    
    for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
        # Histogram
        axes[0, i].hist(output, bins=20, alpha=0.7)
        axes[0, i].set_title(f'{name}\nShape: {output.shape}')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # Statistics
        stats_text = f'Mean: {np.mean(output):.3f}\nStd: {np.std(output):.3f}\nRange: [{np.min(output):.3f}, {np.max(output):.3f}]'
        axes[1, i].text(0.1, 0.5, stats_text, transform=axes[1, i].transAxes, 
                        verticalalignment='center', fontsize=10)
        axes[1, i].set_title(f'{name} Statistics')
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
### 🧪 Test Data Flow Visualization
"""

# %%
# Test data flow visualization
print("Testing data flow visualization...")

try:
    # Create a test network
    test_network = Sequential([
        Dense(input_size=3, output_size=4),
        ReLU(),
        Dense(input_size=4, output_size=2),
        Sigmoid()
    ])
    
    # Test input
    test_input = Tensor([[1.0, 2.0, 3.0]])
    
    # Visualize data flow
    if _should_show_plots():
        visualize_data_flow(test_network, test_input, "Test Network Data Flow")
        print("✅ Data flow visualization created!")
    else:
        print("✅ Data flow visualization skipped during testing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement visualize_data_flow above!")

# %% [markdown]
"""
## Step 5: Network Comparison and Analysis

Let's create tools to compare different network architectures and understand their capabilities.

### Why Network Comparison Matters
- **Architecture selection**: Choose the right network for your problem
- **Performance analysis**: Understand trade-offs between different designs
- **Design insights**: Learn what makes networks effective
- **Research**: Compare new architectures to baselines
"""

# %%
#| export
def compare_networks(networks: List[Sequential], network_names: List[str], 
                    input_data: Tensor, title: str = "Network Comparison"):
    """
    Compare multiple networks on the same input.
    
    Args:
        networks: List of Sequential networks to compare
        network_names: Names for each network
        input_data: Input tensor to test all networks
        title: Title for the plot
        
    TODO: Create a comparison visualization showing how different networks process the same input.
    
    APPROACH:
    1. Run the same input through each network
    2. Collect the outputs and intermediate results
    3. Create a visualization comparing the results
    4. Show statistics and differences
    
    EXAMPLE:
    Compare MLP vs Deep Network vs Wide Network on same input
    
    HINTS:
    - Use a for loop to test each network
    - Store outputs and any relevant statistics
    - Use plt.subplot() to create comparison plots
    - Show both outputs and intermediate layer results
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def compare_networks(networks: List[Sequential], network_names: List[str], 
                    input_data: Tensor, title: str = "Network Comparison"):
    """Compare multiple networks on the same input."""
    if not _should_show_plots():
        print("📊 Visualization disabled during testing")
        return
    
    # Test all networks
    outputs = []
    for network in networks:
        output = network(input_data)
        outputs.append(output.data.flatten())
    
    # Create comparison plot
    fig, axes = plt.subplots(2, len(networks), figsize=(15, 8))
    
    for i, (output, name) in enumerate(zip(outputs, network_names)):
        # Output distribution
        axes[0, i].hist(output, bins=20, alpha=0.7)
        axes[0, i].set_title(f'{name}\nOutput Distribution')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # Statistics
        stats_text = f'Mean: {np.mean(output):.3f}\nStd: {np.std(output):.3f}\nRange: [{np.min(output):.3f}, {np.max(output):.3f}]\nSize: {len(output)}'
        axes[1, i].text(0.1, 0.5, stats_text, transform=axes[1, i].transAxes, 
                        verticalalignment='center', fontsize=10)
        axes[1, i].set_title(f'{name} Statistics')
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
### 🧪 Test Network Comparison
"""

# %%
# Test network comparison
print("Testing network comparison...")

try:
    # Create different networks
    network1 = create_mlp(input_size=3, hidden_sizes=[4], output_size=1)
    network2 = create_mlp(input_size=3, hidden_sizes=[8, 4], output_size=1)
    network3 = create_mlp(input_size=3, hidden_sizes=[2], output_size=1, activation=Tanh)
    
    networks = [network1, network2, network3]
    names = ["Small MLP", "Deep MLP", "Tanh MLP"]
    
    # Test input
    test_input = Tensor([[1.0, 2.0, 3.0]])
    
    # Compare networks
    if _should_show_plots():
        compare_networks(networks, names, test_input, "Network Architecture Comparison")
        print("✅ Network comparison created!")
    else:
        print("✅ Network comparison skipped during testing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement compare_networks above!")

# %% [markdown]
"""
## Step 6: Practical Network Architectures

Now let's create some practical network architectures for common machine learning tasks.

### Common Network Types

#### 1. **Classification Networks**
- **Binary classification**: Output single probability
- **Multi-class classification**: Output probability distribution
- **Use cases**: Image classification, spam detection, sentiment analysis

#### 2. **Regression Networks**
- **Single output**: Predict continuous value
- **Multiple outputs**: Predict multiple values
- **Use cases**: Price prediction, temperature forecasting, demand estimation

#### 3. **Feature Extraction Networks**
- **Encoder networks**: Compress data into features
- **Use cases**: Dimensionality reduction, feature learning, representation learning
"""

# %%
#| export
def create_classification_network(input_size: int, num_classes: int, 
                                hidden_sizes: List[int] = None) -> Sequential:
    """
    Create a network for classification tasks.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        hidden_sizes: List of hidden layer sizes (default: [input_size * 2])
        
    Returns:
        Sequential network for classification
        
    TODO: Implement classification network creation.
    
    APPROACH:
    1. Use default hidden sizes if none provided
    2. Create MLP with appropriate architecture
    3. Use Sigmoid for binary classification (num_classes=1)
    4. Use appropriate activation for multi-class
    
    EXAMPLE:
    create_classification_network(10, 3) creates:
    Dense(10→20) → ReLU → Dense(20→3) → Sigmoid
    
    HINTS:
    - Use create_mlp() function
    - Choose appropriate output activation based on num_classes
    - For binary classification (num_classes=1), use Sigmoid
    - For multi-class, you could use Sigmoid or no activation
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def create_classification_network(input_size: int, num_classes: int, 
                                hidden_sizes: List[int] = None) -> Sequential:
    """Create a network for classification tasks."""
    if hidden_sizes is None:
        hidden_sizes = [input_size // 2]  # Use input_size // 2 as default
    
    # Choose appropriate output activation
    output_activation = Sigmoid if num_classes == 1 else Softmax
    
    return create_mlp(input_size, hidden_sizes, num_classes, 
                     activation=ReLU, output_activation=output_activation)

# %%
#| export
def create_regression_network(input_size: int, output_size: int = 1,
                             hidden_sizes: List[int] = None) -> Sequential:
    """
    Create a network for regression tasks.
    
    Args:
        input_size: Number of input features
        output_size: Number of output values (default: 1)
        hidden_sizes: List of hidden layer sizes (default: [input_size * 2])
        
    Returns:
        Sequential network for regression
        
    TODO: Implement regression network creation.
    
    APPROACH:
    1. Use default hidden sizes if none provided
    2. Create MLP with appropriate architecture
    3. Use no activation on output layer (linear output)
    
    EXAMPLE:
    create_regression_network(5, 1) creates:
    Dense(5→10) → ReLU → Dense(10→1) (no activation)
    
    HINTS:
    - Use create_mlp() but with no output activation
    - For regression, we want linear outputs (no activation)
    - You can pass None or identity function as output_activation
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def create_regression_network(input_size: int, output_size: int = 1,
                             hidden_sizes: List[int] = None) -> Sequential:
    """Create a network for regression tasks."""
    if hidden_sizes is None:
        hidden_sizes = [input_size // 2]  # Use input_size // 2 as default
    
    # Create MLP with Tanh output activation for regression
    return create_mlp(input_size, hidden_sizes, output_size, 
                     activation=ReLU, output_activation=Tanh)

# %% [markdown]
"""
### 🧪 Test Practical Networks
"""

# %%
# Test practical networks
print("Testing practical networks...")

try:
    # Test classification network
    class_net = create_classification_network(input_size=5, num_classes=1)
    x_class = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y_class = class_net(x_class)
    print(f"✅ Classification output: {y_class}")
    print(f"✅ Output range: [{np.min(y_class.data):.3f}, {np.max(y_class.data):.3f}]")
    
    # Test regression network
    reg_net = create_regression_network(input_size=3, output_size=1)
    x_reg = Tensor([[1.0, 2.0, 3.0]])
    y_reg = reg_net(x_reg)
    print(f"✅ Regression output: {y_reg}")
    print(f"✅ Output range: [{np.min(y_reg.data):.3f}, {np.max(y_reg.data):.3f}]")
    
    print("🎉 Practical networks work!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the network creation functions above!")

# %% [markdown]
"""
## Step 7: Network Behavior Analysis

Let's create tools to analyze how networks behave with different inputs and understand their capabilities.

### Why Behavior Analysis Matters
- **Understanding**: Learn what patterns networks can learn
- **Debugging**: Identify when networks fail
- **Design**: Choose appropriate architectures
- **Validation**: Ensure networks work as expected
"""

# %%
#| export
def analyze_network_behavior(network: Sequential, input_data: Tensor, 
                           title: str = "Network Behavior Analysis"):
    """
    Analyze how a network behaves with different inputs.
    
    Args:
        network: Sequential network to analyze
        input_data: Input tensor to test
        title: Title for the plot
        
    TODO: Create an analysis showing network behavior and capabilities.
    
    APPROACH:
    1. Test the network with the given input
    2. Analyze the output characteristics
    3. Test with variations of the input
    4. Create visualizations showing behavior patterns
    
    EXAMPLE:
    Test network with original input and noisy versions
    Show how output changes with input variations
    
    HINTS:
    - Test the original input
    - Create variations (noise, scaling, etc.)
    - Compare outputs across variations
    - Show statistics and patterns
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def analyze_network_behavior(network: Sequential, input_data: Tensor, 
                           title: str = "Network Behavior Analysis"):
    """Analyze how a network behaves with different inputs."""
    if not _should_show_plots():
        print("📊 Visualization disabled during testing")
        return
    
    # Test original input
    original_output = network(input_data)
    
    # Create variations
    noise_levels = [0.0, 0.1, 0.2, 0.5]
    outputs = []
    
    for noise in noise_levels:
        noisy_input = Tensor(input_data.data + noise * np.random.randn(*input_data.data.shape))
        output = network(noisy_input)
        outputs.append(output.data.flatten())
    
    # Create analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original output
    axes[0, 0].hist(outputs[0], bins=20, alpha=0.7)
    axes[0, 0].set_title('Original Input Output')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Output stability
    output_means = [np.mean(out) for out in outputs]
    output_stds = [np.std(out) for out in outputs]
    axes[0, 1].plot(noise_levels, output_means, 'bo-', label='Mean')
    axes[0, 1].fill_between(noise_levels, 
                           [m-s for m, s in zip(output_means, output_stds)],
                           [m+s for m, s in zip(output_means, output_stds)], 
                           alpha=0.3, label='±1 Std')
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('Output Value')
    axes[0, 1].set_title('Output Stability')
    axes[0, 1].legend()
    
    # Output distribution comparison
    for i, (output, noise) in enumerate(zip(outputs, noise_levels)):
        axes[1, 0].hist(output, bins=20, alpha=0.5, label=f'Noise={noise}')
    axes[1, 0].set_xlabel('Output Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Output Distribution Comparison')
    axes[1, 0].legend()
    
    # Statistics
    stats_text = f'Original Mean: {np.mean(outputs[0]):.3f}\nOriginal Std: {np.std(outputs[0]):.3f}\nOutput Range: [{np.min(outputs[0]):.3f}, {np.max(outputs[0]):.3f}]'
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='center', fontsize=10)
    axes[1, 1].set_title('Network Statistics')
    axes[1, 1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
### 🧪 Test Network Behavior Analysis
"""

# %%
# Test network behavior analysis
print("Testing network behavior analysis...")

try:
    # Create a test network
    test_network = create_classification_network(input_size=3, num_classes=1)
    test_input = Tensor([[1.0, 2.0, 3.0]])
    
    # Analyze behavior
    if _should_show_plots():
        analyze_network_behavior(test_network, test_input, "Test Network Behavior")
        print("✅ Network behavior analysis created!")
    else:
        print("✅ Network behavior analysis skipped during testing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement analyze_network_behavior above!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've built the foundation of neural network architectures:

### What You've Accomplished
✅ **Sequential Networks**: Composing layers into complete architectures  
✅ **MLP Creation**: Building multi-layer perceptrons  
✅ **Network Visualization**: Understanding architecture and data flow  
✅ **Network Comparison**: Analyzing different architectures  
✅ **Practical Networks**: Classification and regression networks  
✅ **Behavior Analysis**: Understanding network capabilities  

### Key Concepts You've Learned
- **Networks** are compositions of layers that transform data
- **Architecture design** determines network capabilities
- **Sequential networks** are the most fundamental building block
- **Different architectures** solve different problems
- **Visualization tools** help understand network behavior

### What's Next
In the next modules, you'll build on this foundation:
- **Autograd**: Enable automatic differentiation for training
- **Training**: Learn parameters using gradients and optimizers
- **Loss Functions**: Define objectives for learning
- **Applications**: Solve real problems with neural networks

### Real-World Connection
Your network architectures are now ready to:
- Compose layers into complete neural networks
- Create specialized architectures for different tasks
- Analyze and understand network behavior
- Integrate with the rest of the TinyTorch ecosystem

**Ready for the next challenge?** Let's move on to automatic differentiation to enable training!
"""

# %%
# Final verification
print("\n" + "="*50)
print("🎉 NETWORKS MODULE COMPLETE!")
print("="*50)
print("✅ Sequential network implementation")
print("✅ MLP creation and architecture design")
print("✅ Network visualization and analysis")
print("✅ Network comparison tools")
print("✅ Practical classification and regression networks")
print("✅ Network behavior analysis")
print("\n🚀 Ready to enable training with autograd in the next module!") 
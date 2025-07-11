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

## Module → Package Structure
**🎓 Teaching vs. 🔧 Building**: 
- **Learning side**: Work in `modules/networks/networks_dev.py`  
- **Building side**: Exports to `tinytorch/core/networks.py`

This module teaches how to compose layers into complete neural network architectures.
"""

# %%
#| default_exp core.networks

# Setup and imports
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

A **network** is a composition of layers that transforms input data into output predictions. Think of it as:

```
Input → Layer1 → Layer2 → Layer3 → Output
```

**The fundamental insight**: Neural networks are just function composition!
- Each layer is a function: `f_i(x)`
- The network is: `f(x) = f_n(...f_2(f_1(x)))`
- Complex behavior emerges from simple building blocks

**Why networks matter**:
- They solve real problems (classification, regression, etc.)
- Architecture determines what problems you can solve
- Understanding networks = understanding deep learning
- They're the foundation for all modern AI

Let's start by building the most fundamental network: **Sequential**.
"""

# %%
#| export
class Sequential:
    """
    Sequential Network: Composes layers in sequence
    
    The most fundamental network architecture.
    Applies layers in order: f(x) = layer_n(...layer_2(layer_1(x)))
    
    Args:
        layers: List of layers to compose
        
    TODO: Implement the Sequential network with forward pass.
    """
    
    def __init__(self, layers: List):
        """
        Initialize Sequential network with layers.
        
        Args:
            layers: List of layers to compose in order
            
        TODO: Store the layers and implement forward pass
        """
        raise NotImplementedError("Student implementation required")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all layers
            
        TODO: Implement sequential forward pass through all layers
        """
        raise NotImplementedError("Student implementation required")
    
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

Once you implement the Sequential network above, run this cell to test it:
"""

# %%
# Test the Sequential network
try:
    print("=== Testing Sequential Network ===")
    
    # Create a simple 2-layer network: 3 → 4 → 2
    network = Sequential([
        Dense(3, 4),
        ReLU(),
        Dense(4, 2),
        Sigmoid()
    ])
    
    # Test with sample data
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Input shape: {x.shape}")
    print(f"Input data: {x.data}")
    
    # Forward pass
    output = network(x)
    print(f"Output shape: {output.shape}")
    print(f"Output data: {output.data}")
    
    print("✅ Sequential network working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Sequential network!")

# %% [markdown]
"""
## Step 2: Network Visualization

Now let's create powerful visualizations to understand what our networks look like and how they work!
"""

# %%
#| export
def visualize_network_architecture(network: Sequential, title: str = "Network Architecture"):
    """
    Create a visual representation of network architecture.
    
    Args:
        network: Sequential network to visualize
        title: Title for the plot
    """
    if not _should_show_plots():
        print("📊 Plots disabled during testing - this is normal!")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Network parameters
    layer_count = len(network.layers)
    layer_height = 0.8
    layer_spacing = 1.2
    
    # Colors for different layer types
    colors = {
        'Dense': '#4CAF50',      # Green
        'ReLU': '#2196F3',       # Blue
        'Sigmoid': '#FF9800',    # Orange
        'Tanh': '#9C27B0',       # Purple
        'default': '#757575'      # Gray
    }
    
    # Draw layers
    for i, layer in enumerate(network.layers):
        # Determine layer type and color
        layer_type = type(layer).__name__
        color = colors.get(layer_type, colors['default'])
        
        # Layer position
        x = i * layer_spacing
        y = 0
        
        # Create layer box
        layer_box = FancyBboxPatch(
            (x - 0.3, y - layer_height/2),
            0.6, layer_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(layer_box)
        
        # Add layer label
        ax.text(x, y, layer_type, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Add layer details
        if hasattr(layer, 'input_size') and hasattr(layer, 'output_size'):
            details = f"{layer.input_size}→{layer.output_size}"
            ax.text(x, y - 0.3, details, ha='center', va='center',
                   fontsize=8, color='white')
        
        # Draw connections to next layer
        if i < layer_count - 1:
            next_x = (i + 1) * layer_spacing
            connection = ConnectionPatch(
                (x + 0.3, y), (next_x - 0.3, y),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="black", lw=2
            )
            ax.add_patch(connection)
    
    # Formatting
    ax.set_xlim(-0.5, (layer_count - 1) * layer_spacing + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = []
    for layer_type, color in colors.items():
        if layer_type != 'default':
            legend_elements.append(patches.Patch(color=color, label=layer_type))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()

# %%
#| export
def visualize_data_flow(network: Sequential, input_data: Tensor, title: str = "Data Flow Through Network"):
    """
    Visualize how data flows through the network.
    
    Args:
        network: Sequential network
        input_data: Input tensor
        title: Title for the plot
    """
    if not _should_show_plots():
        print("📊 Plots disabled during testing - this is normal!")
        return
    
    # Get intermediate outputs
    intermediate_outputs = []
    x = input_data
    
    for i, layer in enumerate(network.layers):
        x = layer(x)
        intermediate_outputs.append({
            'layer': network.layers[i],
            'output': x,
            'layer_index': i
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, len(network.layers), figsize=(4*len(network.layers), 8))
    if len(network.layers) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (layer, output) in enumerate(zip(network.layers, intermediate_outputs)):
        # Top row: Layer information
        ax_top = axes[0, i] if len(network.layers) > 1 else axes[0]
        
        # Layer type and details
        layer_type = type(layer).__name__
        ax_top.text(0.5, 0.8, layer_type, ha='center', va='center',
                   fontsize=12, fontweight='bold')
        
        if hasattr(layer, 'input_size') and hasattr(layer, 'output_size'):
            ax_top.text(0.5, 0.6, f"{layer.input_size} → {layer.output_size}", 
                       ha='center', va='center', fontsize=10)
        
        # Output shape
        ax_top.text(0.5, 0.4, f"Shape: {output['output'].shape}", 
                   ha='center', va='center', fontsize=9)
        
        # Output statistics
        output_data = output['output'].data
        ax_top.text(0.5, 0.2, f"Mean: {np.mean(output_data):.3f}", 
                   ha='center', va='center', fontsize=9)
        ax_top.text(0.5, 0.1, f"Std: {np.std(output_data):.3f}", 
                   ha='center', va='center', fontsize=9)
        
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.axis('off')
        
        # Bottom row: Output visualization
        ax_bottom = axes[1, i] if len(network.layers) > 1 else axes[1]
        
        # Show output as heatmap or histogram
        output_data = output['output'].data.flatten()
        
        if len(output_data) <= 20:  # Small output - show as bars
            ax_bottom.bar(range(len(output_data)), output_data, alpha=0.7)
            ax_bottom.set_title(f"Layer {i+1} Output")
            ax_bottom.set_xlabel("Output Index")
            ax_bottom.set_ylabel("Value")
        else:  # Large output - show histogram
            ax_bottom.hist(output_data, bins=20, alpha=0.7, edgecolor='black')
            ax_bottom.set_title(f"Layer {i+1} Output Distribution")
            ax_bottom.set_xlabel("Value")
            ax_bottom.set_ylabel("Frequency")
        
        ax_bottom.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%
#| export
def compare_networks(networks: List[Sequential], network_names: List[str], 
                    input_data: Tensor, title: str = "Network Comparison"):
    """
    Compare different network architectures side-by-side.
    
    Args:
        networks: List of networks to compare
        network_names: Names for each network
        input_data: Input tensor to test with
        title: Title for the plot
    """
    if not _should_show_plots():
        print("📊 Plots disabled during testing - this is normal!")
        return
    
    fig, axes = plt.subplots(2, len(networks), figsize=(6*len(networks), 10))
    if len(networks) == 1:
        axes = axes.reshape(2, -1)
    
    for i, (network, name) in enumerate(zip(networks, network_names)):
        # Get network output
        output = network(input_data)
        
        # Top row: Architecture visualization
        ax_top = axes[0, i] if len(networks) > 1 else axes[0]
        
        # Count layer types
        layer_types = {}
        for layer in network.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        # Create pie chart of layer types
        if layer_types:
            labels = list(layer_types.keys())
            sizes = list(layer_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax_top.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax_top.set_title(f"{name}\nLayer Distribution")
        
        # Bottom row: Output comparison
        ax_bottom = axes[1, i] if len(networks) > 1 else axes[1]
        
        output_data = output.data.flatten()
        
        # Show output statistics
        ax_bottom.hist(output_data, bins=20, alpha=0.7, edgecolor='black')
        ax_bottom.axvline(np.mean(output_data), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(output_data):.3f}')
        ax_bottom.axvline(np.median(output_data), color='green', linestyle='--',
                         label=f'Median: {np.median(output_data):.3f}')
        
        ax_bottom.set_title(f"{name} Output Distribution")
        ax_bottom.set_xlabel("Output Value")
        ax_bottom.set_ylabel("Frequency")
        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
## Step 3: Building Common Architectures

Now let's build some common neural network architectures and visualize them!
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
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        
    Returns:
        Sequential network
    """
    layers = []
    
    # Input layer
    if hidden_sizes:
        layers.append(Dense(input_size, hidden_sizes[0]))
        layers.append(activation())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(Dense(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation())
        
        # Output layer
        layers.append(Dense(hidden_sizes[-1], output_size))
    else:
        # Direct input to output
        layers.append(Dense(input_size, output_size))
    
    layers.append(output_activation())
    
    return Sequential(layers)

# %%
# Test MLP creation and visualization
try:
    print("=== Testing MLP Creation and Visualization ===")
    
    # Create different MLP architectures
    mlp_small = create_mlp(input_size=3, hidden_sizes=[4], output_size=2)
    mlp_medium = create_mlp(input_size=10, hidden_sizes=[16, 8], output_size=3)
    mlp_large = create_mlp(input_size=784, hidden_sizes=[128, 64, 32], output_size=10)
    
    print("Created MLP architectures:")
    print(f"  Small: 3 → 4 → 2")
    print(f"  Medium: 10 → 16 → 8 → 3")
    print(f"  Large: 784 → 128 → 64 → 32 → 10")
    
    # Test with sample data
    x = Tensor(np.random.randn(5, 3).astype(np.float32))
    
    # Visualize architectures
    visualize_network_architecture(mlp_small, "Small MLP Architecture")
    visualize_network_architecture(mlp_medium, "Medium MLP Architecture")
    visualize_network_architecture(mlp_large, "Large MLP Architecture")
    
    # Visualize data flow
    visualize_data_flow(mlp_small, x, "Data Flow Through Small MLP")
    
    # Compare networks
    networks = [mlp_small, mlp_medium]
    names = ["Small MLP", "Medium MLP"]
    compare_networks(networks, names, x, "MLP Architecture Comparison")
    
    print("✅ MLP creation and visualization working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the visualization functions!")

# %% [markdown]
"""
## Step 4: Understanding Network Behavior

Let's analyze how different network architectures behave with different types of input data.
"""

# %%
#| export
def analyze_network_behavior(network: Sequential, input_data: Tensor, 
                           title: str = "Network Behavior Analysis"):
    """
    Analyze how a network behaves with different types of input.
    
    Args:
        network: Network to analyze
        input_data: Input tensor
        title: Title for the plot
    """
    if not _should_show_plots():
        print("📊 Plots disabled during testing - this is normal!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Input vs Output relationship
    ax1 = axes[0, 0]
    input_flat = input_data.data.flatten()
    output = network(input_data)
    output_flat = output.data.flatten()
    
    ax1.scatter(input_flat, output_flat, alpha=0.6)
    ax1.plot([input_flat.min(), input_flat.max()], 
             [input_flat.min(), input_flat.max()], 'r--', alpha=0.5, label='y=x')
    ax1.set_xlabel('Input Values')
    ax1.set_ylabel('Output Values')
    ax1.set_title('Input vs Output')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Output distribution
    ax2 = axes[0, 1]
    ax2.hist(output_flat, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(output_flat), color='red', linestyle='--', 
                label=f'Mean: {np.mean(output_flat):.3f}')
    ax2.set_xlabel('Output Values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Output Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Layer-by-layer activation patterns
    ax3 = axes[0, 2]
    activations = []
    x = input_data
    
    for layer in network.layers:
        x = layer(x)
        if hasattr(layer, 'input_size'):  # Dense layer
            activations.append(np.mean(x.data))
        else:  # Activation layer
            activations.append(np.mean(x.data))
    
    ax3.plot(range(len(activations)), activations, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Mean Activation')
    ax3.set_title('Layer-by-Layer Activations')
    ax3.grid(True, alpha=0.3)
    
    # 4. Network depth analysis
    ax4 = axes[1, 0]
    layer_types = [type(layer).__name__ for layer in network.layers]
    layer_counts = {}
    for layer_type in layer_types:
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    if layer_counts:
        ax4.bar(layer_counts.keys(), layer_counts.values(), alpha=0.7)
        ax4.set_xlabel('Layer Type')
        ax4.set_ylabel('Count')
        ax4.set_title('Layer Type Distribution')
        ax4.grid(True, alpha=0.3)
    
    # 5. Shape transformation
    ax5 = axes[1, 1]
    shapes = [input_data.shape]
    x = input_data
    
    for layer in network.layers:
        x = layer(x)
        shapes.append(x.shape)
    
    layer_indices = range(len(shapes))
    shape_sizes = [np.prod(shape) for shape in shapes]
    
    ax5.plot(layer_indices, shape_sizes, 'go-', linewidth=2, markersize=8)
    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Tensor Size')
    ax5.set_title('Shape Transformation')
    ax5.grid(True, alpha=0.3)
    
    # 6. Network summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
Network Summary:
• Total Layers: {len(network.layers)}
• Input Shape: {input_data.shape}
• Output Shape: {output.shape}
• Parameters: {sum(np.prod(layer.weights.data.shape) if hasattr(layer, 'weights') else 0 for layer in network.layers)}
• Architecture: {' → '.join([type(layer).__name__ for layer in network.layers])}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%
# Test network behavior analysis
try:
    print("=== Testing Network Behavior Analysis ===")
    
    # Create a network for analysis
    network = create_mlp(input_size=5, hidden_sizes=[8, 4], output_size=2)
    
    # Test with different types of input
    x_normal = Tensor(np.random.randn(10, 5).astype(np.float32))
    x_uniform = Tensor(np.random.uniform(-1, 1, (10, 5)).astype(np.float32))
    x_zeros = Tensor(np.zeros((10, 5)).astype(np.float32))
    
    print("Analyzing network behavior with different inputs...")
    
    # Analyze behavior
    analyze_network_behavior(network, x_normal, "Network Behavior: Normal Input")
    analyze_network_behavior(network, x_uniform, "Network Behavior: Uniform Input")
    analyze_network_behavior(network, x_zeros, "Network Behavior: Zero Input")
    
    print("✅ Network behavior analysis working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the behavior analysis function!")

# %% [markdown]
"""
## Step 5: Practical Applications

Let's see how our networks can be applied to real-world problems!
"""

# %%
#| export
def create_classification_network(input_size: int, num_classes: int, 
                                hidden_sizes: List[int] = None) -> Sequential:
    """
    Create a network for classification problems.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        hidden_sizes: List of hidden layer sizes (default: [input_size//2])
        
    Returns:
        Sequential network for classification
    """
    if hidden_sizes is None:
        hidden_sizes = [input_size // 2]
    
    return create_mlp(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        activation=ReLU,
        output_activation=Sigmoid
    )

# %%
#| export
def create_regression_network(input_size: int, output_size: int = 1,
                             hidden_sizes: List[int] = None) -> Sequential:
    """
    Create a network for regression problems.
    
    Args:
        input_size: Number of input features
        output_size: Number of output values (default: 1)
        hidden_sizes: List of hidden layer sizes (default: [input_size//2])
        
    Returns:
        Sequential network for regression
    """
    if hidden_sizes is None:
        hidden_sizes = [input_size // 2]
    
    return create_mlp(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=ReLU,
        output_activation=Tanh  # No activation for regression
    )

# %%
# Test practical applications
try:
    print("=== Testing Practical Applications ===")
    
    # Create networks for different tasks
    digit_classifier = create_classification_network(
        input_size=784,  # 28x28 image
        num_classes=10,  # 10 digits
        hidden_sizes=[128, 64]
    )
    
    sentiment_analyzer = create_classification_network(
        input_size=100,  # 100-dimensional word embeddings
        num_classes=2,   # Positive/Negative
        hidden_sizes=[32, 16]
    )
    
    house_price_predictor = create_regression_network(
        input_size=13,   # 13 house features
        output_size=1,   # 1 price prediction
        hidden_sizes=[8, 4]
    )
    
    print("Created networks for different applications:")
    print(f"  Digit Classifier: 784 → 128 → 64 → 10")
    print(f"  Sentiment Analyzer: 100 → 32 → 16 → 2")
    print(f"  House Price Predictor: 13 → 8 → 4 → 1")
    
    # Test with sample data
    digit_input = Tensor(np.random.randn(1, 784).astype(np.float32))
    sentiment_input = Tensor(np.random.randn(1, 100).astype(np.float32))
    house_input = Tensor(np.random.randn(1, 13).astype(np.float32))
    
    # Get predictions
    digit_pred = digit_classifier(digit_input)
    sentiment_pred = sentiment_analyzer(sentiment_input)
    house_pred = house_price_predictor(house_input)
    
    print(f"\nSample predictions:")
    print(f"  Digit classifier output: {digit_pred.data[0]}")
    print(f"  Sentiment analyzer output: {sentiment_pred.data[0]}")
    print(f"  House price predictor output: {house_pred.data[0]}")
    
    # Visualize architectures
    visualize_network_architecture(digit_classifier, "Digit Classification Network")
    visualize_network_architecture(sentiment_analyzer, "Sentiment Analysis Network")
    visualize_network_architecture(house_price_predictor, "House Price Prediction Network")
    
    print("✅ Practical applications working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the application functions!")

# %% [markdown]
"""
## 🎓 Module Summary

### What You Learned
1. **Network Composition**: Building complete networks from layers
2. **Architecture Design**: How to choose network structures
3. **Visualization**: Understanding networks through visual analysis
4. **Practical Applications**: Real-world network use cases

### Key Architectural Insights
- **Function Composition**: Networks as `f(x) = layer_n(...layer_1(x))`
- **Modular Design**: Clean separation between layers and networks
- **Visual Understanding**: How to analyze network behavior
- **Application Patterns**: Classification vs regression architectures

### Network Design Principles
- **Depth vs Width**: Trade-offs in network architecture
- **Activation Functions**: How they affect network behavior
- **Shape Management**: Understanding tensor transformations
- **Practical Considerations**: Choosing architectures for specific tasks

### Next Steps
- **Training**: Learn how networks learn from data (autograd, optimization)
- **Advanced Architectures**: CNNs, RNNs, Transformers
- **Real Data**: Working with actual datasets
- **Production**: Deploying networks in real applications

**Congratulations on mastering neural network architectures!** 🚀
""" 
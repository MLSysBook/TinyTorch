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
# Layers - Building Neural Network Architectures

Welcome to Layers! You'll implement the essential building blocks that compose into complete neural network architectures.

## LINK Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): N-dimensional arrays with shape management and broadcasting
- Module 03 (Activations): ReLU and Softmax functions providing nonlinear intelligence

**What's Working**: You can create tensors and apply nonlinear transformations for complex pattern learning!

**The Gap**: You have data structures and nonlinear functions, but no way to combine them into trainable neural network architectures.

**This Module's Solution**: Implement Linear layers, Module composition patterns, and Sequential networks - the architectural foundations enabling everything from MLPs to transformers.

**Connection Map**:
```
Activations -> Layers -> Training
(intelligence)  (architecture)  (learning)
```

## Learning Objectives

By completing this module, you will:

1. **Build layer abstractions** - Create the building blocks that compose into neural networks
2. **Implement Linear layers** - The fundamental operation that transforms data between dimensions
3. **Create Sequential networks** - Chain layers together to build complete neural networks
4. **Manage parameters** - Handle weights and biases in an organized way
5. **Foundation for architectures** - Enable building everything from simple MLPs to complex models

## Build -> Use -> Reflect
1. **Build**: Module base class, Linear layers, and Sequential composition
2. **Use**: Combine layers into complete neural networks with real data
3. **Reflect**: Understand how simple building blocks enable complex architectures
"""

# In[ ]:

#| default_exp core.layers

#| export
import numpy as np
import sys
import os

# Smart import system: works both during development and in production
# This pattern allows the same code to work in two scenarios:
# 1. During development: imports from local module files (tensor_dev.py)
# 2. In production: imports from installed tinytorch package
# This flexibility is essential for educational development workflows

if 'tinytorch' in sys.modules:
    # Production: Import from installed package
    # When tinytorch is installed as a package, use the packaged version
    from tinytorch.core.tensor import Tensor
else:
    # Development: Import from local module files
    # During development, we need to import directly from the source files
    # This allows us to work with modules before they're packaged
    tensor_module_path = os.path.join(os.path.dirname(__file__), '..', '01_tensor')
    sys.path.insert(0, tensor_module_path)
    try:
        from tensor_dev import Tensor
    finally:
        sys.path.pop(0)  # Always clean up path to avoid side effects

# REMOVED: Parameter class - now using Tensor directly with requires_grad=True
#
# This creates a clean evolution pattern:
# - Module 01-04: Use Tensor(data, requires_grad=True) directly
# - Module 05: Tensor gains full autograd capabilities
# - No more hasattr() hacks or wrapper classes needed

# In[ ]:

print("FIRE TinyTorch Layers Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network layers!")

# %% [markdown]
"""
## Visual Guide: Understanding Neural Network Architecture Through Diagrams

### Neural Network Layers: From Components to Systems

```
Individual Neuron:                Neural Network Layer:
    x₁ --○ w₁                    +---------------------+
          \\                     |   Input Vector      |
    x₂ --○ w₂ --> Sum --> f() --> y |   [x₁, x₂, x₃]    |
          /                     +---------------------+
    x₃ --○ w₃                              v
       + bias                    +---------------------+
                                 |  Weight Matrix W    |
One computation unit             |  +w₁₁ w₁₂ w₁₃+     |
                                 |  |w₂₁ w₂₂ w₂₃|     |
                                 |  +w₃₁ w₃₂ w₃₃+     |
                                 +---------------------+
                                             v
                                   Matrix multiplication
                                     Y = X @ W + b
                                             v
                                 +---------------------+
                                 |  Output Vector      |
                                 |   [y₁, y₂, y₃]     |
                                 +---------------------+

Parallel processing of many neurons!
```

### Layer Composition: Building Complex Architectures

```
Multi-Layer Perceptron (MLP) Architecture:

   Input        Hidden Layer 1    Hidden Layer 2     Output
 (784 dims)      (256 neurons)     (128 neurons)    (10 classes)
+---------+     +-------------+   +-------------+   +---------+
|  Image  |----▶|    ReLU     |--▶|    ReLU     |--▶| Softmax |
| 28*28px |     | Activations |   | Activations |   | Probs   |
+---------+     +-------------+   +-------------+   +---------+
     v                v                 v               v
200,960 params   32,896 params    1,290 params   Total: 235,146

Parameter calculation for Linear(input_size, output_size):
• Weights: input_size * output_size matrix
• Biases:  output_size vector
• Total:   (input_size * output_size) + output_size

Memory scaling pattern:
Layer width doubles -> Parameters quadruple -> Memory quadruples
```

### Module System: Automatic Parameter Management

```
Parameter Collection Hierarchy:

Model (Sequential)
+-- Layer1 (Linear)
|   +-- weights [784 * 256]  --+
|   +-- bias [256]           --┤
+-- Layer2 (Linear)           +--▶ model.parameters()
|   +-- weights [256 * 128]  --┤   Automatically collects
|   +-- bias [128]           --┤   all parameters for
+-- Layer3 (Linear)           +--▶ optimizer.step()
    +-- weights [128 * 10]   --┤
    +-- bias [10]            --+

Before Module system:        With Module system:
manually track params   ->    automatic collection
params = [w1, b1, w2,...]    params = model.parameters()

Enables: optimizer = Adam(model.parameters())
```

### Memory Layout and Performance Implications

```
Tensor Memory Access Patterns:

Matrix Multiplication: A @ B = C

Efficient (Row-major access):    Inefficient (Column-major):
A: --------------▶               A: | | | | | ▶
   Cache-friendly                    | | | | |
   Sequential reads                  v v v v v
                                     Cache misses
B: |                             B: --------------▶
   |
   v

Performance impact:
• Good memory layout: 100% cache hit ratio
• Poor memory layout: 10-50% cache hit ratio
• 10-100x performance difference in practice

Why contiguous tensors matter in production!
```
"""

# %% [markdown]
"""
## Part 1: Module Base Class - The Foundation of Neural Network Architecture
"""

# %% nbgrader={"grade": false, "grade_id": "module-base", "solution": true}

# Before building specific layers, we need a base class that enables clean composition and automatic parameter management.

#| export
class Module:
    """
    Base class for all neural network modules.
    
    Provides automatic parameter collection, forward pass management,
    and clean composition patterns. All layers (Dense, Conv2d, etc.)
    inherit from this class.
    
    Key Features:
    - Automatic parameter registration when you assign parameter Tensors (weights, bias)
    - Recursive parameter collection from sub-modules
    - Clean __call__ interface: model(x) instead of model.forward(x)
    - Extensible for custom layers
    
    Example Usage:
        class MLP(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(784, 128)  # Auto-registered!
                self.layer2 = Linear(128, 10)   # Auto-registered!
                
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
                
        model = MLP()
        params = model.parameters()  # Gets all parameters automatically!
        output = model(input)        # Clean interface!
    """
    
    def __init__(self):
        """Initialize module with empty parameter and sub-module storage."""
        self._parameters = []
        self._modules = []
    
    def __setattr__(self, name, value):
        """
        Intercept attribute assignment to auto-register parameters and modules.
        
        When you do self.weight = Parameter(...), this automatically adds
        the parameter to our collection for easy optimization.
        """
        # Step 1: Check if this looks like a parameter (Tensor with parameter naming)
        # Pure tensor evolution: identify parameters by naming convention
        is_tensor_type = isinstance(value, Tensor)
        is_parameter_name = name in ['weights', 'weight', 'bias']

        if is_tensor_type and is_parameter_name:
            # Step 2: Add to our parameter list for optimization
            self._parameters.append(value)
        
        # Step 3: Check if it's a sub-module (another neural network layer)
        elif isinstance(value, Module):
            # Step 4: Add to module list for recursive parameter collection
            self._modules.append(value)
        
        # Step 5: Always set the actual attribute (this is essential!)
        super().__setattr__(name, value)
    
    def parameters(self):
        """
        Recursively collect all parameters from this module and sub-modules.
        
        Returns:
            List of all parameters (Tensors containing weights and biases)
            
        This enables: optimizer = Adam(model.parameters()) (when optimizers are available)
        """
        # Start with our own parameters
        params = list(self._parameters)
        
        # Add parameters from sub-modules recursively
        for module in self._modules:
            params.extend(module.parameters())
            
        return params
    
    def __call__(self, *args, **kwargs):
        """
        Makes modules callable: model(x) instead of model.forward(x).
        
        This is the magic that enables clean syntax like:
            output = model(input)
        instead of:
            output = model.forward(input)
        """
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass - must be implemented by subclasses.
        
        This is where the actual computation happens. Every layer
        defines its own forward() method.
        """
        raise NotImplementedError("Subclasses must implement forward()")

# In[ ]:

# PASS IMPLEMENTATION CHECKPOINT: Basic Module class complete

# THINK PREDICTION: How many parameters would a simple 3-layer network have?
# Write your guess here: _______

# 🔍 SYSTEMS ANALYSIS: Layer Performance and Scaling
def analyze_layer_performance():
    """Analyze layer performance and scaling characteristics."""
    print("📊 LAYER SYSTEMS ANALYSIS")
    print("Understanding how neural network layers scale and perform...")

    try:
        # Parameter scaling analysis
        print("\n1. Parameter Scaling:")
        layer_sizes = [(784, 256), (256, 128), (128, 10)]
        total_params = 0

        for i, (input_size, output_size) in enumerate(layer_sizes):
            weights = input_size * output_size
            biases = output_size
            layer_params = weights + biases
            total_params += layer_params
            print(f"   Layer {i+1} ({input_size}→{output_size}): {layer_params:,} params")

        print(f"   Total network: {total_params:,} parameters")
        print(f"   Memory usage: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

        # Computational complexity
        print("\n2. Computational Complexity:")
        batch_size = 32
        total_flops = 0

        for i, (input_size, output_size) in enumerate(layer_sizes):
            matmul_flops = 2 * batch_size * input_size * output_size
            bias_flops = batch_size * output_size
            layer_flops = matmul_flops + bias_flops
            total_flops += layer_flops
            print(f"   Layer {i+1}: {layer_flops:,} FLOPs ({matmul_flops:,} matmul + {bias_flops:,} bias)")

        print(f"   Total forward pass: {total_flops:,} FLOPs")

        # Scaling patterns
        print("\n3. Scaling Insights:")
        print("   • Parameter growth: O(input_size × output_size) - quadratic")
        print("   • Computation: O(batch × input × output) - linear in each dimension")
        print("   • Memory: Parameters + activations scale differently")
        print("   • Bottlenecks: Large layers dominate both memory and compute")

        print("\n💡 KEY INSIGHT: Layer size quadratically affects parameters but linearly affects computation per sample")

    except Exception as e:
        print(f"⚠️ Analysis error: {e}")

# In[ ]:

# %% [markdown]
"""
### ✅ IMPLEMENTATION CHECKPOINT: Module Base Class Complete

You've built the foundation that enables automatic parameter management across all neural network components!

🤔 **PREDICTION**: How many parameters would a simple 3-layer network have?
Network: 784 → 256 → 128 → 10
Your guess: _______
"""

# %% [markdown]
"""
## Part 2: Linear Layer - The Fundamental Neural Network Component

Linear layers (also called Dense or Fully Connected layers) are the building blocks of neural networks.
"""

# %% nbgrader={"grade": false, "grade_id": "linear-layer", "solution": true}

#| export
class Linear(Module):
    """
    Linear (Fully Connected) Layer implementation.
    
    Applies the transformation: output = input @ weights + bias
    
    Inherits from Module for automatic parameter management and clean API.
    This is PyTorch's nn.Linear equivalent with the same name for familiarity.
    
    Features:
    - Automatic parameter registration (weights and bias)
    - Clean call interface: layer(input) instead of layer.forward(input)
    - Works with optimizers via model.parameters()
    """
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True):
        """
        Initialize Linear layer with random weights and optional bias.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features  
            use_bias: Whether to include bias term
        
        TODO: Implement Linear layer initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store input_size and output_size as instance variables
        2. Initialize weights as Tensor with shape (input_size, output_size)
        3. Use small random values: np.random.randn(...) * 0.1
        4. Initialize bias as Tensor with shape (output_size,) if use_bias is True
        5. Set bias to None if use_bias is False
        
        LEARNING CONNECTIONS:
        - Small random initialization prevents symmetry breaking
        - Weight shape (input_size, output_size) enables matrix multiplication
        - Bias allows shifting the output (like y-intercept in linear regression)
        - PyTorch uses more sophisticated initialization (Xavier, Kaiming)
        
        IMPLEMENTATION HINTS:
        - Use np.random.randn() for Gaussian random numbers
        - Scale by 0.1 to keep initial values small
        - Remember to wrap numpy arrays in Tensor()
        - Store use_bias flag for forward pass logic
        """
        ### BEGIN SOLUTION
        super().__init__()  # Initialize Module base class
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights with small random values using Parameter
        # Shape: (input_size, output_size) for matrix multiplication
        #
        # MAGNIFY WEIGHT INITIALIZATION CONTEXT:
        # Weight initialization is critical for training deep networks successfully.
        # Our simple approach (small random * 0.1) works for shallow networks, but
        # deeper networks require more sophisticated initialization strategies:
        #
        # • Xavier/Glorot: scale = sqrt(1/fan_in) - good for tanh/sigmoid activations
        # • Kaiming/He: scale = sqrt(2/fan_in) - optimized for ReLU activations
        # • Our approach: scale = 0.1 - simple but effective for basic networks
        #
        # Why proper initialization matters:
        # - Prevents vanishing gradients (weights too small -> signals disappear)
        # - Prevents exploding gradients (weights too large -> signals blow up)
        # - Enables stable training in deeper architectures (Module 11 training)
        # - Affects convergence speed and final model performance
        #
        # Production frameworks automatically choose initialization based on layer type!
        weight_data = np.random.randn(input_size, output_size) * 0.1
        self.weights = Tensor(weight_data)  # Pure tensor - will become trainable in Module 05
        
        # Initialize bias if requested
        if use_bias:
            # MAGNIFY GRADIENT FLOW PREPARATION:
            # Clean parameter management is essential for backpropagation (Module 09).
            # When we implement autograd, the optimizer needs to find ALL trainable
            # parameters automatically. Our Module base class ensures that:
            #
            # • Parameters are automatically registered when assigned
            # • Recursive parameter collection works through network hierarchies
            # • Gradient updates can flow to all learnable weights and biases
            # • Memory management handles parameter lifecycle correctly
            #
            # This design enables the autograd system to:
            # - Track computational graphs through all layers
            # - Accumulate gradients for each parameter during backpropagation
            # - Support optimizers that update parameters based on gradients
            # - Scale to arbitrarily deep and complex network architectures
            #
            # Bias also uses small random initialization (could be zeros, but small random works well)
            bias_data = np.random.randn(output_size) * 0.1
            self.bias = Tensor(bias_data)  # Pure tensor - will become trainable in Module 05
        else:
            self.bias = None
        ### END SOLUTION
    
    def forward(self, x):
        """
        Forward pass through the Linear layer with automatic differentiation.

        Args:
            x: Input Variable (shape: ..., input_size)

        Returns:
            Output Variable (shape: ..., output_size) with gradient tracking

        CRITICAL FIX: This method now properly uses autograd operations
        to ensure gradients flow through parameters during backpropagation.

        TODO: Implement the linear transformation using autograd operations

        STEP-BY-STEP IMPLEMENTATION:
        1. Convert input to Variable if needed (with gradient tracking)
        2. Use autograd matrix multiplication: matmul(x, weights)
        3. Add bias using autograd addition if it exists: add(result, bias)
        4. Return Variable with gradient tracking enabled

        LEARNING CONNECTIONS:
        - Uses autograd operations instead of raw numpy for gradient flow
        - Parameters (weights/bias) are Variables with requires_grad=True
        - Matrix multiplication and addition maintain computational graph
        - This enables backpropagation through all parameters

        IMPLEMENTATION HINTS:
        - Import autograd operations locally to avoid circular imports
        - Ensure result Variable has proper gradient tracking
        - Handle both Tensor and Variable inputs gracefully
        """
        ### BEGIN SOLUTION
        # Clean Tensor Evolution Pattern:
        # - Modules 01-04: Use basic Tensor operations (@, +)
        # - Module 05+: Tensor gains full autograd capabilities automatically

        # Ensure input is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Matrix multiplication: input @ weights
        # Uses Tensor's built-in @ operator (will be autograd-capable after Module 05)
        result = x @ self.weights

        # Add bias if it exists
        if self.bias is not None:
            result = result + self.bias

        # Result is automatically a Variable with gradient tracking
        return result
        ### END SOLUTION

# In[ ]:

# %% [markdown]
"""
### 🧪 Unit Test: Linear Layer
This test validates our Linear layer implementation with matrix multiplication and parameter management.

**What we're testing**: Linear layer transforms input dimensions correctly
**Why it matters**: Linear layers are the fundamental building blocks of neural networks
**Expected**: Correct output shapes, parameter handling, and batch processing

### Linear Layer Computation Visualization

```
Forward Pass: y = x @ W + b

Input Batch:          Weight Matrix:        Bias Vector:         Output:
┌─────────────┐      ┌───────────────┐     ┌─────────┐         ┌──────────┐
│ [1, 2, 3]   │      │ w₁₁  w₁₂     │     │   b₁    │         │ [y₁, y₂] │
│ [4, 5, 6]   │  @   │ w₂₁  w₂₂     │  +  │   b₂    │    =    │ [y₃, y₄] │
└─────────────┘      │ w₃₁  w₃₂     │     └─────────┘         └──────────┘
  Batch(2,3)         └───────────────┘        (2,)               Batch(2,2)
                        Weights(3,2)

Memory Layout:
• Input: [batch_size, input_features]
• Weights: [input_features, output_features]
• Bias: [output_features]
• Output: [batch_size, output_features]
```
"""

def test_unit_linear():
    """Test Linear layer implementation."""
    print("🔬 Unit Test: Linear Layer...")
    
    # Test case 1: Basic functionality
    layer = Linear(input_size=3, output_size=2)
    input_tensor = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    output = layer.forward(input_tensor)
    
    # Check output shape
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print("PASS Output shape correct")
    
    # Test case 2: No bias
    layer_no_bias = Linear(input_size=2, output_size=3, use_bias=False)
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("PASS No bias option works")
    
    # Test case 3: Multiple samples (batch processing)
    batch_input = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape: (3, 2)
    layer_batch = Linear(input_size=2, output_size=2)
    batch_output = layer_batch.forward(batch_input)
    
    assert batch_output.shape == (3, 2), f"Expected shape (3, 2), got {batch_output.shape}"
    print("PASS Batch processing works")
    
    # Test case 4: Callable interface
    callable_output = layer_batch(batch_input)
    assert np.allclose(callable_output.data, batch_output.data), "Callable interface should match forward()"
    print("PASS Callable interface works")
    
    # Test case 5: Parameter initialization
    layer_init = Linear(input_size=10, output_size=5)
    assert layer_init.weights.shape == (10, 5), f"Expected weights shape (10, 5), got {layer_init.weights.shape}"
    assert layer_init.bias.shape == (5,), f"Expected bias shape (5,), got {layer_init.bias.shape}"
    
    # Check that weights are reasonably small (good initialization)
    mean_val = np.abs(layer_init.weights.data).mean()
    # Convert to float - mean_val is a numpy scalar from np.abs().mean()
    mean_val = float(mean_val)  # Direct conversion since np.mean returns numpy scalar
    assert mean_val < 1.0, "Weights should be small for good initialization"
    print("PASS Parameter initialization correct")
    
    print("CELEBRATE All Linear layer tests passed!")

test_unit_linear()

# In[ ]:

# TEST Unit Test: Parameter Management
# %% [markdown]
"""
### 🧪 Unit Test: Parameter Management
This test validates automatic parameter collection and module composition.

**What we're testing**: Module system automatically collects parameters from nested layers
**Why it matters**: Enables automatic optimization and parameter management in complex networks
**Expected**: All parameters collected hierarchically, proper parameter counting

### Parameter Management Hierarchy Visualization

```
Network Architecture:           Parameter Collection:

SimpleNetwork                   network.parameters()
├── layer1: Linear(4→3)           ├── layer1.weights [4×3] = 12 params
│   ├── weights: (4,3)            ├── layer1.bias [3] = 3 params
│   └── bias: (3,)                ├── layer2.weights [3×2] = 6 params
└── layer2: Linear(3→2)           └── layer2.bias [2] = 2 params
    ├── weights: (3,2)                              Total: 23 params
    └── bias: (2,)

Manual Tracking:          vs    Automatic Collection:
weights = [                     params = model.parameters()
  layer1.weights,               # Automatically finds ALL
  layer1.bias,                  # parameters in the hierarchy
  layer2.weights,               # No manual bookkeeping!
  layer2.bias,
]
```

### Memory and Parameter Scaling

```
Layer Configuration:        Parameters:              Memory (float32):
Linear(100, 50)          → 100×50 + 50    = 5,050  → ~20KB
Linear(256, 128)         → 256×128 + 128  = 32,896 → ~131KB
Linear(512, 256)         → 512×256 + 256  = 131,328 → ~525KB
Linear(1024, 512)        → 1024×512 + 512 = 524,800 → ~2.1MB

Pattern: O(input_size × output_size) scaling
Large layers dominate memory usage!
```
"""

def test_unit_parameter_management():
    """Test Linear layer parameter management and module composition."""
    print("🔬 Unit Test: Parameter Management...")
    
    # Test case 1: Parameter registration
    layer = Linear(input_size=3, output_size=2)
    params = layer.parameters()
    
    assert len(params) == 2, f"Expected 2 parameters (weights + bias), got {len(params)}"
    assert layer.weights in params, "Weights should be in parameters list"
    assert layer.bias in params, "Bias should be in parameters list"
    print("PASS Parameter registration works")
    
    # Test case 2: Module composition
    class SimpleNetwork(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Linear(4, 3)
            self.layer2 = Linear(3, 2)
        
        def forward(self, x):
            x = self.layer1(x)
            return self.layer2(x)
    
    network = SimpleNetwork()
    all_params = network.parameters()
    
    # Should have 4 parameters: 2 from each layer (weights + bias)
    assert len(all_params) == 4, f"Expected 4 parameters from network, got {len(all_params)}"
    print("PASS Module composition and parameter collection works")
    
    # Test case 3: Forward pass through composed network
    input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0]])
    output = network(input_tensor)
    
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("PASS Network forward pass works")
    
    # Test case 4: No bias option
    layer_no_bias = Linear(input_size=3, output_size=2, use_bias=False)
    params_no_bias = layer_no_bias.parameters()
    
    assert len(params_no_bias) == 1, f"Expected 1 parameter (weights only), got {len(params_no_bias)}"
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("PASS No bias option works")
    
    print("CELEBRATE All parameter management tests passed!")

test_unit_parameter_management()

# In[ ]:

# PASS IMPLEMENTATION CHECKPOINT: Linear layer complete

# THINK PREDICTION: How does memory usage scale with network depth vs width?
# Deeper network (more layers): _______
# Wider network (more neurons per layer): _______

# MAGNIFY SYSTEMS INSIGHT #3: Architecture Memory Analysis
# Architecture analysis consolidated into analyze_layer_performance() above

# Analysis consolidated into analyze_layer_performance() above

# %% [markdown]
"""
## Part 4: Sequential Network Composition
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-composition", "solution": true}

#| export
class Sequential(Module):
    """
    Sequential Network: Composes layers in sequence.
    
    The most fundamental network architecture that applies layers in order:
    f(x) = layer_n(...layer_2(layer_1(x)))
    
    Inherits from Module for automatic parameter collection from all sub-layers.
    This enables optimizers to find all parameters automatically.
    
    Example Usage:
        # Create a 3-layer MLP
        model = Sequential([
            Linear(784, 128),
            ReLU(),
            Linear(128, 64), 
            ReLU(),
            Linear(64, 10)
        ])
        
        # Use the model
        output = model(input_data)  # Clean interface!
        params = model.parameters()  # All parameters from all layers!
    """
    
    def __init__(self, layers=None):
        """
        Initialize Sequential network with layers.
        
        Args:
            layers: List of layers to compose in order (optional)
        """
        super().__init__()  # Initialize Module base class
        self.layers = layers if layers is not None else []
        
        # Register all layers as sub-modules for parameter collection
        for i, layer in enumerate(self.layers):
            # This automatically adds each layer to self._modules
            setattr(self, f'layer_{i}', layer)
    
    def forward(self, x):
        """
        Forward pass through all layers in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        # Register the new layer for parameter collection
        setattr(self, f'layer_{len(self.layers)-1}', layer)

# In[ ]:

# TEST Unit Test: Sequential Networks
def test_unit_sequential():
    """Test Sequential network implementation."""
    print("TEST Testing Sequential Network...")
    
    # Test case 1: Create empty network
    empty_net = Sequential()
    assert len(empty_net.layers) == 0, "Empty Sequential should have no layers"
    print("PASS Empty Sequential network creation")
    
    # Test case 2: Create network with layers
    layers = [Linear(3, 4), Linear(4, 2)]
    network = Sequential(layers)
    assert len(network.layers) == 2, "Network should have 2 layers"
    print("PASS Sequential network with layers")
    
    # Test case 3: Forward pass through network
    input_tensor = Tensor([[1.0, 2.0, 3.0]])
    output = network(input_tensor)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("PASS Forward pass through Sequential network")
    
    # Test case 4: Parameter collection from all layers
    all_params = network.parameters()
    # Should have 4 parameters: 2 weights + 2 biases from 2 Linear layers
    assert len(all_params) == 4, f"Expected 4 parameters from Sequential network, got {len(all_params)}"
    print("PASS Parameter collection from all layers")
    
    # Test case 5: Adding layers dynamically
    network.add(Linear(2, 1))
    assert len(network.layers) == 3, "Network should have 3 layers after adding one"
    
    # Test forward pass after adding layer
    final_output = network(input_tensor)
    assert final_output.shape == (1, 1), f"Expected final output shape (1, 1), got {final_output.shape}"
    print("PASS Dynamic layer addition")
    
    print("CELEBRATE All Sequential network tests passed!")

test_unit_sequential()

# %% [markdown]
"""
## Part 5: Flatten Operation - Connecting Different Layer Types
"""

# %% nbgrader={"grade": false, "grade_id": "flatten-operations", "solution": true}

#| export
def flatten(x, start_dim=1):
    """
    Flatten tensor starting from a given dimension.
    
    This is essential for transitioning from convolutional layers
    (which output 4D tensors) to linear layers (which expect 2D).
    
    Args:
        x: Input tensor (Tensor or any array-like)
        start_dim: Dimension to start flattening from (default: 1 to preserve batch)
        
    Returns:
        Flattened tensor preserving batch dimension
        
    Examples:
        # Flatten CNN output for Linear layer
        conv_output = Tensor(np.random.randn(32, 64, 8, 8))  # (batch, channels, height, width)
        flat = flatten(conv_output)  # (32, 4096) - ready for Linear layer!
        
        # Flatten image for MLP
        images = Tensor(np.random.randn(32, 3, 28, 28))  # CIFAR-10 batch
        flat = flatten(images)  # (32, 2352) - ready for MLP!
    """
    # Get the data (handle both Tensor and numpy arrays)
    if isinstance(x, Tensor):
        data = x.data
    else:
        data = x

    # Calculate new shape
    batch_size = data.shape[0] if start_dim > 0 else 1
    remaining_size = np.prod(data.shape[start_dim:])
    new_shape = (batch_size, remaining_size) if start_dim > 0 else (remaining_size,)

    # Reshape while preserving the original tensor type
    if isinstance(x, Tensor):
        # It's a Tensor - create a new Tensor with flattened data
        flattened_data = data.reshape(new_shape)
        # Create new tensor - pure tensor approach (no gradient tracking yet)
        return Tensor(flattened_data)
    else:
        # It's a numpy array - just reshape and return
        return data.reshape(new_shape)

#| export
class Flatten(Module):
    """
    Flatten layer that reshapes tensors from multi-dimensional to 2D.
    
    Essential for connecting convolutional layers (which output 4D tensors)
    to linear layers (which expect 2D tensors). Preserves the batch dimension.
    
    Example Usage:
        # In a CNN architecture
        model = Sequential([
            Conv2D(3, 16, kernel_size=3),  # Output: (batch, 16, height, width)
            ReLU(),
            Flatten(),                     # Output: (batch, 16*height*width)
            Linear(16*height*width, 10)    # Now compatible!
        ])
    """
    
    def __init__(self, start_dim=1):
        """
        Initialize Flatten layer.
        
        Args:
            start_dim: Dimension to start flattening from (default: 1 to preserve batch)
        """
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x):
        """
        Flatten tensor starting from start_dim.
        
        Args:
            x: Input tensor
            
        Returns:
            Flattened tensor with batch dimension preserved
        """
        return flatten(x, start_dim=self.start_dim)

# In[ ]:

# TEST Unit Test: Flatten Operations
def test_unit_flatten():
    """Test Flatten layer and function implementation."""
    print("TEST Testing Flatten Operations...")
    
    # Test case 1: Flatten function with 2D tensor
    x_2d = Tensor([[1, 2], [3, 4]])
    flattened_func = flatten(x_2d)
    assert flattened_func.shape == (2, 2), f"Expected shape (2, 2), got {flattened_func.shape}"
    print("PASS Flatten function with 2D tensor")
    
    # Test case 2: Flatten function with 4D tensor (simulating CNN output)
    x_4d = Tensor(np.random.randn(2, 3, 4, 4))  # (batch, channels, height, width)
    flattened_4d = flatten(x_4d)
    assert flattened_4d.shape == (2, 48), f"Expected shape (2, 48), got {flattened_4d.shape}"  # 3*4*4 = 48
    print("PASS Flatten function with 4D tensor")
    
    # Test case 3: Flatten layer class
    flatten_layer = Flatten()
    layer_output = flatten_layer(x_4d)
    assert layer_output.shape == (2, 48), f"Expected shape (2, 48), got {layer_output.shape}"
    assert np.allclose(layer_output.data, flattened_4d.data), "Flatten layer should match flatten function"
    print("PASS Flatten layer class")
    
    # Test case 4: Different start dimensions
    flatten_from_0 = Flatten(start_dim=0)
    full_flat = flatten_from_0(x_2d)
    assert len(full_flat.shape) <= 2, "Flattening from dim 0 should create vector"
    print("PASS Different start dimensions")
    
    # Test case 5: Integration with Sequential
    network = Sequential([
        Linear(8, 4),
        Flatten()
    ])
    test_input = Tensor(np.random.randn(2, 8))
    output = network(test_input)
    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
    print("PASS Flatten integration with Sequential")
    
    print("CELEBRATE All Flatten operations tests passed!")

test_unit_flatten()

# In[ ]:

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/03_layers/layers_dev.py
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Module, Linear, Sequential, Flatten  # This module
from tinytorch.core.tensor import Tensor  # Pure tensor foundation (always needed)
```

**Why this matters:**
- **Learning:** Complete layer system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn with all core components together
- **Consistency:** All layer operations and parameter management in core.layers
- **Integration:** Works seamlessly with tensors for complete neural network building
"""

# %%

# %% [markdown]
"""
## Complete Neural Network Demo
"""

def demonstrate_complete_networks():
    """Demonstrate complete neural networks using all implemented components."""
    print("FIRE Complete Neural Network Demo")
    print("=" * 50)
    
    print("\n1. MLP for Classification (MNIST-style):")
    # Multi-layer perceptron for image classification
    mlp = Sequential([
        Flatten(),              # Flatten input images
        Linear(784, 256),       # First hidden layer
        Linear(256, 128),       # Second hidden layer  
        Linear(128, 10)         # Output layer (10 classes)
    ])
    
    # Test with batch of "images"
    batch_images = Tensor(np.random.randn(32, 28, 28))  # 32 MNIST-like images
    mlp_output = mlp(batch_images)
    print(f"   Input: {batch_images.shape} (batch of 28x28 images)")
    print(f"   Output: {mlp_output.shape} (class logits for 32 images)")
    print(f"   Parameters: {len(mlp.parameters())} tensors")
    
    print("\n2. CNN-style Architecture (with Flatten):")
    # Simulate CNN -> Flatten -> Dense pattern
    cnn_style = Sequential([
        # Simulate Conv2D output with random "features"
        Flatten(),              # Flatten spatial features
        Linear(512, 256),       # Dense layer after convolution
        Linear(256, 10)         # Classification head
    ])
    
    # Test with simulated conv output
    conv_features = Tensor(np.random.randn(16, 8, 8, 8))  # Simulated (B,C,H,W)
    cnn_output = cnn_style(conv_features)
    print(f"   Input: {conv_features.shape} (simulated conv features)")
    print(f"   Output: {cnn_output.shape} (class predictions)")
    
    print("\n3. Deep Network with Many Layers:")
    # Demonstrate deep composition
    deep_net = Sequential()
    layer_sizes = [100, 80, 60, 40, 20, 10]
    
    for i in range(len(layer_sizes) - 1):
        deep_net.add(Linear(layer_sizes[i], layer_sizes[i+1]))
        print(f"   Added layer: {layer_sizes[i]} -> {layer_sizes[i+1]}")
    
    # Test deep network
    deep_input = Tensor(np.random.randn(8, 100))
    deep_output = deep_net(deep_input)
    print(f"   Deep network: {deep_input.shape} -> {deep_output.shape}")
    print(f"   Total parameters: {len(deep_net.parameters())} tensors")
    
    print("\n4. Parameter Management Across Networks:")
    networks = {'MLP': mlp, 'CNN-style': cnn_style, 'Deep': deep_net}
    
    for name, net in networks.items():
        params = net.parameters()
        total_params = sum(p.data.size for p in params)
        memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"   {name}: {len(params)} param tensors, {total_params:,} total params, {memory_mb:.2f} MB")
    
    print("\nCELEBRATE All components work together seamlessly!")
    print("   • Module system enables automatic parameter collection")
    print("   • Linear layers handle matrix transformations") 
    print("   • Sequential composes layers into complete architectures")
    print("   • Flatten connects different layer types")
    print("   • Everything integrates for production-ready neural networks!")

demonstrate_complete_networks()

# In[ ]:

# %% [markdown]
"""
## Testing Framework
"""

def test_module():
    """Run complete module validation."""
    print("🧪 TESTING ALL LAYER COMPONENTS")
    print("=" * 40)

    # Call every individual test function
    test_unit_linear()
    test_unit_parameter_management()
    test_unit_sequential()
    test_unit_flatten()

    print("\n✅ ALL TESTS PASSED! Layer module ready for integration.")

# In[ ]:

if __name__ == "__main__":
    print("🚀 TINYTORCH LAYERS MODULE")
    print("=" * 50)

    # Test all components
    test_module()

    # Systems analysis
    print("\n" + "=" * 50)
    analyze_layer_performance()

    # Complete demo
    print("\n" + "=" * 50)
    demonstrate_complete_networks()

    print("\n🎉 LAYERS MODULE COMPLETE!")
    print("✅ Ready for advanced architectures and training!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented all the core neural network components, let's think about their implications for ML systems:

**Question 1: Memory vs Computation Analysis**

You're designing a neural network for deployment on a mobile device with limited memory (1GB RAM) but decent compute power.

You have two architecture options:
A) Wide network: 784 -> 2048 -> 2048 -> 10 (3 layers, wide)
B) Deep network: 784 -> 256 -> 256 -> 256 -> 256 -> 10 (5 layers, narrow)

Calculate the memory requirements for each option and explain which you'd choose for mobile deployment and why.

Consider:
- Parameter storage requirements
- Intermediate activation storage during forward pass
- Training vs inference memory requirements
- How your choice affects model capacity and accuracy

⭐ **Question 2: Production Performance Optimization**

Your Linear layer implementation works correctly, but you notice it's slower than PyTorch's nn.Linear on the same hardware.

Investigate and explain:
1. Why might our implementation be slower? (Hint: think about underlying linear algebra libraries)
2. What optimization techniques do production frameworks use?
3. How would you modify our implementation to approach production performance?
4. When might our simple implementation actually be preferable?

Research areas to consider:
- BLAS (Basic Linear Algebra Subprograms) libraries
- Memory layout and cache efficiency
- Vectorization and SIMD instructions
- GPU kernel optimization

⭐ **Question 3: Systems Architecture Scaling**

Modern transformer models like GPT-3 have billions of parameters, primarily in Linear layers.

Analyze the scaling challenges:
1. How does memory requirement scale with model size? Calculate the memory needed for a 175B parameter model.
2. What are the computational bottlenecks during training vs inference?
3. How do systems like distributed training address these scaling challenges?
4. Why do large models use techniques like gradient checkpointing and model parallelism?

Systems considerations:
- Memory hierarchy (L1/L2/L3 cache, RAM, storage)
- Network bandwidth for distributed training
- GPU memory constraints and model sharding
- Inference optimization for production serving
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Layers - Complete Neural Network Foundation

### What You've Accomplished

You've successfully implemented the complete foundation for neural networks - all the essential components working together:

### ✅ **Complete Core System**
- **Module Base Class**: Parameter management and composition patterns for all neural network components
- **Matrix Multiplication**: The computational primitive underlying all neural network operations
- **Linear (Dense) Layers**: Complete implementation with proper parameter initialization and forward propagation
- **Sequential Networks**: Clean composition system for building complete neural network architectures
- **Flatten Operations**: Tensor reshaping to connect different layer types (essential for CNN->MLP transitions)

### ✅ **Systems Understanding**
- **Architectural Patterns**: How modular design enables everything from MLPs to complex deep networks
- **Memory Analysis**: How layer composition affects memory usage and computational efficiency
- **Performance Characteristics**: Understanding how tensor operations and layer composition affect performance
- **Production Context**: Connection to real-world ML frameworks and their component organization

### ✅ **ML Engineering Skills**
- **Complete Parameter Management**: How neural networks automatically collect parameters from all components
- **Network Composition**: Building complex architectures from simple, reusable components
- **Tensor Operations**: Essential reshaping and transformation operations for different network types
- **Clean Abstraction**: Professional software design patterns that scale to production systems

### 🔗 **Connection to Production ML Systems**

Your unified implementation mirrors the complete component systems used in:
- **PyTorch's nn.Module system**: Same parameter management and composition patterns
- **PyTorch's nn.Sequential**: Identical architecture composition approach
- **All major frameworks**: The same modular design principles that power TensorFlow, JAX, and others
- **Production ML systems**: Clean abstractions that enable complex models while maintaining manageable code

### 🚀 **What's Next**

With your complete layer foundation, you're ready to:
- **Module 05 (Dense)**: Build complete dense networks for classification tasks
- **Module 06 (Spatial)**: Add convolutional layers for computer vision
- **Module 09 (Autograd)**: Enable automatic differentiation for learning
- **Module 10 (Optimizers)**: Implement sophisticated optimization algorithms

### 💡 **Key Systems Insights**

1. **Modular composition is the key to scalable ML systems** - clean interfaces enable complex behaviors
2. **Parameter management must be automatic** - manual parameter tracking doesn't scale to deep networks
3. **Tensor operations like flattening are architectural requirements** - different layer types need different tensor shapes
4. **Clean abstractions enable innovation** - good foundational design supports unlimited architectural experimentation

You now understand how to build complete, production-ready neural network foundations that can scale to any architecture!
"""
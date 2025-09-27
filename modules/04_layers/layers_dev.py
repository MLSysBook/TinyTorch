#!/usr/bin/env python
# coding: utf-8

# # Layers - Building Neural Network Architectures

# Welcome to Layers! You'll implement the essential building blocks that compose into complete neural network architectures.

# ## 🔗 Building on Previous Learning
# **What You Built Before**:
# - Module 02 (Tensor): N-dimensional arrays with shape management and broadcasting
# - Module 03 (Activations): ReLU and Softmax functions providing nonlinear intelligence

# **What's Working**: You can create tensors and apply nonlinear transformations for complex pattern learning!

# **The Gap**: You have data structures and nonlinear functions, but no way to combine them into trainable neural network architectures.

# **This Module's Solution**: Implement Linear layers, Module composition patterns, and Sequential networks - the architectural foundations enabling everything from MLPs to transformers.

# **Connection Map**:
# ```
# Activations → Layers → Training
# (intelligence)  (architecture)  (learning)
# ```

# ## Learning Goals
# - Systems understanding: How layer composition affects memory usage, parameter counts, and computational complexity in neural networks
# - Core implementation skill: Build complete Module system, Linear transformations, and Sequential composition for scalable architectures  
# - Pattern/abstraction mastery: Understand how modular design patterns enable building complex networks from simple, reusable components
# - Framework connections: See how your implementation mirrors PyTorch's nn.Module, nn.Linear, and nn.Sequential - the foundation of all modern ML frameworks
# - Optimization trade-offs: Learn why proper parameter management and clean abstractions are essential for both performance and maintainability in production systems

# ## Build → Use → Reflect
# 1. **Build**: Complete layer system with Module base class, Linear transformations, Sequential composition, and tensor reshaping operations
# 2. **Use**: Compose layers into complete neural networks and analyze architectural trade-offs with real parameter counting
# 3. **Reflect**: How does modular architecture design affect both system scalability and computational efficiency in production ML systems?

# ## Systems Reality Check
# 💡 **Production Context**: PyTorch's nn.Module system enables all modern neural networks through automatic parameter collection and clean composition patterns
# ⚡ **Performance Insight**: Layer composition and parameter management patterns determine training speed and memory efficiency - proper abstraction is a systems requirement, not just good design

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
    from tinytorch.core.tensor import Tensor, Parameter
else:
    # Development: Import from local module files
    # During development, we need to import directly from the source files
    # This allows us to work with modules before they're packaged
    tensor_module_path = os.path.join(os.path.dirname(__file__), '..', '02_tensor')
    sys.path.insert(0, tensor_module_path)
    try:
        from tensor_dev import Tensor, Parameter
    finally:
        sys.path.pop(0)  # Always clean up path to avoid side effects

# In[ ]:

print("🔥 TinyTorch Layers Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network layers!")

# ## Visual Guide: Understanding Neural Network Architecture Through Diagrams

# ### Neural Network Layers: From Components to Systems
# 
# ```
# Individual Neuron:                Neural Network Layer:
#     x₁ ──○ w₁                    ┌─────────────────────┐
#           ╲                     │   Input Vector      │
#     x₂ ──○ w₂ ──> Σ ──> f() ──> y │   [x₁, x₂, x₃]    │
#           ╱                     └─────────────────────┘
#     x₃ ──○ w₃                              ↓
#        + bias                    ┌─────────────────────┐
#                                  │  Weight Matrix W    │
# One computation unit             │  ┌w₁₁ w₁₂ w₁₃┐     │
#                                  │  │w₂₁ w₂₂ w₂₃│     │
#                                  │  └w₃₁ w₃₂ w₃₃┘     │
#                                  └─────────────────────┘
#                                              ↓
#                                    Matrix multiplication
#                                      Y = X @ W + b
#                                              ↓
#                                  ┌─────────────────────┐
#                                  │  Output Vector      │
#                                  │   [y₁, y₂, y₃]     │
#                                  └─────────────────────┘
# 
# Parallel processing of many neurons!
# ```

# ### Layer Composition: Building Complex Architectures
# 
# ```
# Multi-Layer Perceptron (MLP) Architecture:
# 
#    Input        Hidden Layer 1    Hidden Layer 2     Output
#  (784 dims)      (256 neurons)     (128 neurons)    (10 classes)
# ┌─────────┐     ┌─────────────┐   ┌─────────────┐   ┌─────────┐
# │  Image  │────▶│    ReLU     │──▶│    ReLU     │──▶│ Softmax │
# │ 28×28px │     │ Activations │   │ Activations │   │ Probs   │
# └─────────┘     └─────────────┘   └─────────────┘   └─────────┘
#      ↓                ↓                 ↓               ↓
# 200,960 params   32,896 params    1,290 params   Total: 235,146
# 
# Parameter calculation for Linear(input_size, output_size):
# • Weights: input_size × output_size matrix
# • Biases:  output_size vector  
# • Total:   (input_size × output_size) + output_size
# 
# Memory scaling pattern:
# Layer width doubles → Parameters quadruple → Memory quadruples
# ```

# ### Module System: Automatic Parameter Management
# 
# ```
# Parameter Collection Hierarchy:
# 
# Model (Sequential)
# ├── Layer1 (Linear)
# │   ├── weights [784 × 256]  ──┐
# │   └── bias [256]           ──┤
# ├── Layer2 (Linear)           ├──▶ model.parameters()
# │   ├── weights [256 × 128]  ──┤   Automatically collects
# │   └── bias [128]           ──┤   all parameters for
# └── Layer3 (Linear)           ├──▶ optimizer.step()
#     ├── weights [128 × 10]   ──┤
#     └── bias [10]            ──┘
# 
# Before Module system:        With Module system:
# manually track params   →    automatic collection
# params = [w1, b1, w2,...]    params = model.parameters()
# 
# Enables: optimizer = Adam(model.parameters())
# ```

# ### Memory Layout and Performance Implications
# 
# ```
# Tensor Memory Access Patterns:
# 
# Matrix Multiplication: A @ B = C
# 
# Efficient (Row-major access):    Inefficient (Column-major):
# A: ──────────────▶               A: │ │ │ │ │ ▶
#    Cache-friendly                    │ │ │ │ │
#    Sequential reads                  ▼ ▼ ▼ ▼ ▼
#                                      Cache misses
# B: │                             B: ──────────────▶
#    │                                
#    ▼                                
# 
# Performance impact:
# • Good memory layout: 100% cache hit ratio
# • Poor memory layout: 10-50% cache hit ratio  
# • 10-100x performance difference in practice
# 
# Why contiguous tensors matter in production!
# ```

# In[ ]:

# ## Part 1: Module Base Class - The Foundation of Neural Network Architecture

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
        # Step 1: Check if this looks like a parameter (Tensor with data and specific name)
        # Break down the complex boolean logic for clarity:
        is_tensor_like = hasattr(value, 'data') and hasattr(value, 'shape')
        is_tensor_type = isinstance(value, Tensor)
        is_parameter_name = name in ['weights', 'weight', 'bias']
        
        if is_tensor_like and is_tensor_type and is_parameter_name:
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

# ✅ IMPLEMENTATION CHECKPOINT: Basic Module class complete

# 🤔 PREDICTION: How many parameters would a simple 3-layer network have?
# Write your guess here: _______

# 🔍 SYSTEMS INSIGHT #1: Parameter Counter
def analyze_parameter_scaling():
    """Count parameters in networks of different sizes."""
    try:
        print("📊 Parameter Scaling Analysis")
        print("=" * 40)
        
        layer_configs = [
            (100, 50),      # Small network
            (784, 256),     # MNIST-style  
            (1024, 512),    # Medium network
            (2048, 1024),   # Large network
            (4096, 2048),   # Very large
        ]
        
        for input_size, output_size in layer_configs:
            # Calculate parameters for Linear layer
            weight_params = input_size * output_size
            bias_params = output_size
            total_params = weight_params + bias_params
            
            # Memory calculation (float32 = 4 bytes)
            memory_mb = total_params * 4 / (1024 * 1024)
            
            print(f"  {input_size:4d} → {output_size:4d}: {total_params:,} params, {memory_mb:.2f} MB")
        
        print("\n💡 Key Insights:")
        print("  • Parameters scale quadratically with layer width")
        print("  • Doubling width → 4x parameters → 4x memory")
        print("  • Modern networks balance width vs depth carefully")
        print("  • GPT-3 has 175B parameters = ~700GB just for weights!")
        
    except Exception as e:
        print(f"⚠️ Error in parameter analysis: {e}")

# Run the analysis
analyze_parameter_scaling()

# In[ ]:

# ## Part 2: Matrix Multiplication - The Heart of Neural Networks

# Every neural network operation ultimately reduces to matrix multiplication. Let's build the foundation that powers everything from simple perceptrons to transformers.

#| export
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication for tensors using explicit loops.
    
    This implementation uses triple-nested loops for educational understanding
    of the fundamental operations. Module 15 will show the optimization progression
    from loops → blocking → vectorized operations.
    
    Args:
        a: Left tensor (shape: ..., m, k)
        b: Right tensor (shape: ..., k, n)
    
    Returns:
        Result tensor (shape: ..., m, n)
    
    TODO: Implement matrix multiplication using explicit loops.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy arrays from both tensors using .data
    2. Check tensor shapes for compatibility
    3. Use triple-nested loops to show every operation
    4. Wrap result in a new Tensor and return
    
    LEARNING CONNECTIONS:
    - This is the core operation in Dense layers: output = input @ weights
    - Shows the fundamental computation before optimization
    - Module 15 will demonstrate the progression to high-performance implementations
    - Understanding loops helps appreciate vectorization and GPU parallelization
    
    EDUCATIONAL APPROACH:
    - Intentionally simple for understanding, not performance
    - Makes every multiply-add operation explicit
    - Sets up Module 15 to show optimization techniques
    
    EXAMPLE:
    ```python
    a = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
    b = Tensor([[5, 6], [7, 8]])  # shape (2, 2)
    result = matmul(a, b)
    # result.data = [[19, 22], [43, 50]]
    ```
    
    IMPLEMENTATION HINTS:
    - Use explicit loops to show every operation
    - This is educational, not optimized for performance
    - Module 15 will show the progression to fast implementations
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays from tensors
    a_data = a.data
    b_data = b.data
    
    # Get dimensions and validate compatibility
    if len(a_data.shape) != 2 or len(b_data.shape) != 2:
        raise ValueError("matmul requires 2D tensors")
    
    m, k = a_data.shape
    k2, n = b_data.shape
    
    if k != k2:
        raise ValueError(
            f"Matrix multiplication requires inner dimensions to match!\n"
            f"Left matrix: {a_data.shape} (inner dim: {k})\n"
            f"Right matrix: {b_data.shape} (inner dim: {k2})\n"
            f"For A @ B, A's columns must equal B's rows."
        )
    
    # Initialize result matrix
    result = np.zeros((m, n), dtype=a_data.dtype)
    
    # Triple nested loops - educational, shows every operation
    # This is intentionally simple to understand the fundamental computation
    #
    # Matrix multiplication visualization:
    # A (2,3) @ B (3,4) = C (2,4)
    # 
    # A = [[a11, a12, a13],     B = [[b11, b12, b13, b14],
    #      [a21, a22, a23]]          [b21, b22, b23, b24],
    #                                [b31, b32, b33, b34]]
    #
    # C[0,0] = a11*b11 + a12*b21 + a13*b31 (dot product of A's row 0 with B's column 0)
    #
    # Module 15 will show the optimization journey:
    #   Step 1 (here): Educational loops - slow but clear
    #   Step 2: Loop blocking for cache efficiency  
    #   Step 3: Vectorized operations with NumPy
    #   Step 4: GPU acceleration and BLAS libraries
    for i in range(m):                      # For each row in result
        for j in range(n):                  # For each column in result
            for k_idx in range(k):          # Dot product: sum over inner dimension
                result[i, j] += a_data[i, k_idx] * b_data[k_idx, j]
    
    # Return new Tensor with result
    return Tensor(result)
    ### END SOLUTION

# In[ ]:

# 🧪 Unit Test: Matrix Multiplication
def test_unit_matmul():
    """Test matrix multiplication implementation."""
    print("🧪 Testing Matrix Multiplication...")
    
    # Test case 1: Simple 2x2 matrices
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = matmul(a, b)
    expected = np.array([[19, 22], [43, 50]])
    
    assert np.allclose(result.data, expected), f"Expected {expected}, got {result.data}"
    print("✅ 2x2 matrix multiplication")
    
    # Test case 2: Non-square matrices
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    b = Tensor([[7, 8], [9, 10], [11, 12]])  # 3x2
    result = matmul(a, b)
    expected = np.array([[58, 64], [139, 154]])
    
    assert np.allclose(result.data, expected), f"Expected {expected}, got {result.data}"
    print("✅ Non-square matrix multiplication")
    
    # Test case 3: Vector-matrix multiplication
    a = Tensor([[1, 2, 3]])  # 1x3 (row vector)
    b = Tensor([[4], [5], [6]])  # 3x1 (column vector)
    result = matmul(a, b)
    expected = np.array([[32]])  # 1*4 + 2*5 + 3*6 = 32
    
    assert np.allclose(result.data, expected), f"Expected {expected}, got {result.data}"
    print("✅ Vector-matrix multiplication")
    
    print("🎉 All matrix multiplication tests passed!")

test_unit_matmul()

# In[ ]:

# ✅ IMPLEMENTATION CHECKPOINT: Matrix multiplication complete

# 🤔 PREDICTION: How many operations does matrix multiplication take?
# For two N×N matrices, your guess: _______

# 🔍 SYSTEMS INSIGHT #2: FLOPS Analysis
def analyze_matmul_complexity():
    """Analyze computational complexity of matrix multiplication."""
    try:
        print("📊 Matrix Multiplication FLOPS Analysis")
        print("=" * 45)
        
        sizes = [64, 128, 256, 512, 1024]
        
        for size in sizes:
            # For N×N @ N×N matrices:
            # - N³ multiply operations
            # - N³ add operations  
            # - Total: 2N³ FLOPs (Floating Point Operations)
            flops = 2 * size ** 3
            
            # Memory requirements
            memory_elements = 3 * size * size  # A, B, and result matrices
            memory_mb = memory_elements * 4 / (1024 * 1024)  # float32 = 4 bytes
            
            print(f"  {size:4d}×{size:4d}: {flops/1e9:.1f} GFLOPS, {memory_mb:.1f} MB")
        
        print("\n💡 Computational Insights:")
        print("  • FLOPs grow cubically O(N³) - very expensive!")
        print("  • Memory grows quadratically O(N²)")
        print("  • Large matrices become compute-bound")
        print("  • GPU acceleration essential for deep learning")
        print("  • This is why matrix operations dominate ML workloads")
        
    except Exception as e:
        print(f"⚠️ Error in FLOPS analysis: {e}")

# Run the analysis
analyze_matmul_complexity()

# In[ ]:

# ## Part 3: Linear Layer - The Fundamental Neural Network Component

# Linear layers (also called Dense or Fully Connected layers) are the building blocks of neural networks.

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
        # 🔍 WEIGHT INITIALIZATION CONTEXT:
        # Weight initialization is critical for training deep networks successfully.
        # Our simple approach (small random * 0.1) works for shallow networks, but
        # deeper networks require more sophisticated initialization strategies:
        #
        # • Xavier/Glorot: scale = sqrt(1/fan_in) - good for tanh/sigmoid activations
        # • Kaiming/He: scale = sqrt(2/fan_in) - optimized for ReLU activations
        # • Our approach: scale = 0.1 - simple but effective for basic networks
        #
        # Why proper initialization matters:
        # - Prevents vanishing gradients (weights too small → signals disappear)
        # - Prevents exploding gradients (weights too large → signals blow up)
        # - Enables stable training in deeper architectures (Module 11 training)
        # - Affects convergence speed and final model performance
        #
        # Production frameworks automatically choose initialization based on layer type!
        weight_data = np.random.randn(input_size, output_size) * 0.1
        self.weights = Parameter(weight_data)  # Auto-registers for optimization!
        
        # Initialize bias if requested
        if use_bias:
            # 🔍 GRADIENT FLOW PREPARATION:
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
            self.bias = Parameter(bias_data)  # Auto-registers for optimization!
        else:
            self.bias = None
        ### END SOLUTION
    
    def forward(self, x):
        """
        Forward pass through the Linear layer.
        
        Args:
            x: Input tensor (shape: ..., input_size)
        
        Returns:
            Output tensor (shape: ..., output_size)
        
        COMMON PITFALL: Make sure input tensor has shape (..., input_size)
        If you get shape mismatch errors, check that your input's last dimension
        matches the layer's input_size parameter.
        
        TODO: Implement the linear transformation: output = input @ weights + bias
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Extract data from input tensor using x.data
        2. Get weight and bias data using self.weights.data and self.bias.data
        3. Perform matrix multiplication: np.dot(x.data, weights.data)
        4. Add bias if it exists: result + bias.data
        5. Return new Tensor with result
        
        LEARNING CONNECTIONS:
        - This is the core neural network operation: y = Wx + b
        - Matrix multiplication handles batch processing automatically
        - Each row in input produces one row in output
        - This is pure linear algebra - no autograd complexity yet
        
        IMPLEMENTATION HINTS:
        - Use np.dot() for matrix multiplication
        - Handle the case where bias is None
        - Always return a new Tensor object
        - Focus on the mathematical operation, not gradient tracking
        """
        ### BEGIN SOLUTION
        # Extract data from input tensor
        x_data = x.data
        weights_data = self.weights.data
        
        # Matrix multiplication: input @ weights
        output_data = np.dot(x_data, weights_data)
        
        # Add bias if it exists
        if self.bias is not None:
            bias_data = self.bias.data
            output_data = output_data + bias_data
        
        # Return new Tensor with result
        return Tensor(output_data)
        ### END SOLUTION

# In[ ]:

# 🧪 Unit Test: Linear Layer
def test_unit_linear():
    """Test Linear layer implementation."""
    print("🧪 Testing Linear Layer...")
    
    # Test case 1: Basic functionality
    layer = Linear(input_size=3, output_size=2)
    input_tensor = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    output = layer.forward(input_tensor)
    
    # Check output shape
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print("✅ Output shape correct")
    
    # Test case 2: No bias
    layer_no_bias = Linear(input_size=2, output_size=3, use_bias=False)
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("✅ No bias option works")
    
    # Test case 3: Multiple samples (batch processing)
    batch_input = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape: (3, 2)
    layer_batch = Linear(input_size=2, output_size=2)
    batch_output = layer_batch.forward(batch_input)
    
    assert batch_output.shape == (3, 2), f"Expected shape (3, 2), got {batch_output.shape}"
    print("✅ Batch processing works")
    
    # Test case 4: Callable interface
    callable_output = layer_batch(batch_input)
    assert np.allclose(callable_output.data, batch_output.data), "Callable interface should match forward()"
    print("✅ Callable interface works")
    
    # Test case 5: Parameter initialization
    layer_init = Linear(input_size=10, output_size=5)
    assert layer_init.weights.shape == (10, 5), f"Expected weights shape (10, 5), got {layer_init.weights.shape}"
    assert layer_init.bias.shape == (5,), f"Expected bias shape (5,), got {layer_init.bias.shape}"
    
    # Check that weights are reasonably small (good initialization)
    assert np.abs(layer_init.weights.data).mean() < 1.0, "Weights should be small for good initialization"
    print("✅ Parameter initialization correct")
    
    print("🎉 All Linear layer tests passed!")

test_unit_linear()

# In[ ]:

# 🧪 Unit Test: Parameter Management
def test_unit_parameter_management():
    """Test Linear layer parameter management and module composition."""
    print("🧪 Testing Parameter Management...")
    
    # Test case 1: Parameter registration
    layer = Linear(input_size=3, output_size=2)
    params = layer.parameters()
    
    assert len(params) == 2, f"Expected 2 parameters (weights + bias), got {len(params)}"
    assert layer.weights in params, "Weights should be in parameters list"
    assert layer.bias in params, "Bias should be in parameters list"
    print("✅ Parameter registration works")
    
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
    print("✅ Module composition and parameter collection works")
    
    # Test case 3: Forward pass through composed network
    input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0]])
    output = network(input_tensor)
    
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("✅ Network forward pass works")
    
    # Test case 4: No bias option
    layer_no_bias = Linear(input_size=3, output_size=2, use_bias=False)
    params_no_bias = layer_no_bias.parameters()
    
    assert len(params_no_bias) == 1, f"Expected 1 parameter (weights only), got {len(params_no_bias)}"
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("✅ No bias option works")
    
    print("🎉 All parameter management tests passed!")

test_unit_parameter_management()

# In[ ]:

# ✅ IMPLEMENTATION CHECKPOINT: Linear layer complete

# 🤔 PREDICTION: How does memory usage scale with network depth vs width?
# Deeper network (more layers): _______
# Wider network (more neurons per layer): _______

# 🔍 SYSTEMS INSIGHT #3: Architecture Memory Analysis
def analyze_architecture_scaling():
    """Compare memory usage of deep vs wide networks."""
    try:
        print("📊 Architecture Scaling: Deep vs Wide Networks")
        print("=" * 50)
        
        # Compare networks with similar parameter counts
        print("\nDeep Network (8 layers, narrow):")
        deep_layers = [128, 64, 64, 64, 64, 64, 64, 10]
        deep_params = 0
        deep_memory = 0
        
        for i in range(len(deep_layers) - 1):
            layer_params = deep_layers[i] * deep_layers[i+1] + deep_layers[i+1]
            deep_params += layer_params
            layer_memory = layer_params * 4 / (1024 * 1024)  # MB
            deep_memory += layer_memory
            print(f"  Layer {i+1}: {deep_layers[i]:3d} → {deep_layers[i+1]:3d} = {layer_params:,} params")
        
        print(f"  Total: {deep_params:,} params, {deep_memory:.2f} MB")
        
        print("\nWide Network (3 layers, wide):")
        wide_layers = [128, 256, 256, 10]
        wide_params = 0
        wide_memory = 0
        
        for i in range(len(wide_layers) - 1):
            layer_params = wide_layers[i] * wide_layers[i+1] + wide_layers[i+1]
            wide_params += layer_params
            layer_memory = layer_params * 4 / (1024 * 1024)  # MB
            wide_memory += layer_memory
            print(f"  Layer {i+1}: {wide_layers[i]:3d} → {wide_layers[i+1]:3d} = {layer_params:,} params")
        
        print(f"  Total: {wide_params:,} params, {wide_memory:.2f} MB")
        
        print(f"\n💡 Architecture Insights:")
        print(f"  • Deep network: {len(deep_layers)-1} layers, {deep_params:,} params")
        print(f"  • Wide network: {len(wide_layers)-1} layers, {wide_params:,} params")
        print(f"  • Memory ratio: {wide_memory/deep_memory:.1f}x (wide uses more)")
        print(f"  • Deep networks: better feature hierarchies")
        print(f"  • Wide networks: more parallel computation")
        print(f"  • Modern trend: Balance depth + width for best performance")
        
    except Exception as e:
        print(f"⚠️ Error in architecture analysis: {e}")

# Run the analysis
analyze_architecture_scaling()

# In[ ]:

# ## Part 4: Sequential Network Composition

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

# 🧪 Unit Test: Sequential Networks
def test_unit_sequential():
    """Test Sequential network implementation."""
    print("🧪 Testing Sequential Network...")
    
    # Test case 1: Create empty network
    empty_net = Sequential()
    assert len(empty_net.layers) == 0, "Empty Sequential should have no layers"
    print("✅ Empty Sequential network creation")
    
    # Test case 2: Create network with layers
    layers = [Linear(3, 4), Linear(4, 2)]
    network = Sequential(layers)
    assert len(network.layers) == 2, "Network should have 2 layers"
    print("✅ Sequential network with layers")
    
    # Test case 3: Forward pass through network
    input_tensor = Tensor([[1.0, 2.0, 3.0]])
    output = network(input_tensor)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("✅ Forward pass through Sequential network")
    
    # Test case 4: Parameter collection from all layers
    all_params = network.parameters()
    # Should have 4 parameters: 2 weights + 2 biases from 2 Linear layers
    assert len(all_params) == 4, f"Expected 4 parameters from Sequential network, got {len(all_params)}"
    print("✅ Parameter collection from all layers")
    
    # Test case 5: Adding layers dynamically
    network.add(Linear(2, 1))
    assert len(network.layers) == 3, "Network should have 3 layers after adding one"
    
    # Test forward pass after adding layer
    final_output = network(input_tensor)
    assert final_output.shape == (1, 1), f"Expected final output shape (1, 1), got {final_output.shape}"
    print("✅ Dynamic layer addition")
    
    print("🎉 All Sequential network tests passed!")

test_unit_sequential()

# In[ ]:

# ## Part 5: Flatten Operation - Connecting Different Layer Types

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
    if hasattr(x, 'data'):
        data = x.data
    else:
        data = x
    
    # Calculate new shape
    batch_size = data.shape[0] if start_dim > 0 else 1
    remaining_size = np.prod(data.shape[start_dim:])
    new_shape = (batch_size, remaining_size) if start_dim > 0 else (remaining_size,)
    
    # Reshape while preserving the original tensor type
    if hasattr(x, 'data'):
        # It's a Tensor - create a new Tensor with flattened data
        flattened_data = data.reshape(new_shape)
        # Use type(x) to preserve the exact Tensor type (Parameter vs regular Tensor)
        # This ensures that if input was a Parameter, output is also a Parameter
        return type(x)(flattened_data)
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

# 🧪 Unit Test: Flatten Operations
def test_unit_flatten():
    """Test Flatten layer and function implementation."""
    print("🧪 Testing Flatten Operations...")
    
    # Test case 1: Flatten function with 2D tensor
    x_2d = Tensor([[1, 2], [3, 4]])
    flattened_func = flatten(x_2d)
    assert flattened_func.shape == (2, 2), f"Expected shape (2, 2), got {flattened_func.shape}"
    print("✅ Flatten function with 2D tensor")
    
    # Test case 2: Flatten function with 4D tensor (simulating CNN output)
    x_4d = Tensor(np.random.randn(2, 3, 4, 4))  # (batch, channels, height, width)
    flattened_4d = flatten(x_4d)
    assert flattened_4d.shape == (2, 48), f"Expected shape (2, 48), got {flattened_4d.shape}"  # 3*4*4 = 48
    print("✅ Flatten function with 4D tensor")
    
    # Test case 3: Flatten layer class
    flatten_layer = Flatten()
    layer_output = flatten_layer(x_4d)
    assert layer_output.shape == (2, 48), f"Expected shape (2, 48), got {layer_output.shape}"
    assert np.allclose(layer_output.data, flattened_4d.data), "Flatten layer should match flatten function"
    print("✅ Flatten layer class")
    
    # Test case 4: Different start dimensions
    flatten_from_0 = Flatten(start_dim=0)
    full_flat = flatten_from_0(x_2d)
    assert len(full_flat.shape) <= 2, "Flattening from dim 0 should create vector"
    print("✅ Different start dimensions")
    
    # Test case 5: Integration with Sequential
    network = Sequential([
        Linear(8, 4),
        Flatten()
    ])
    test_input = Tensor(np.random.randn(2, 8))
    output = network(test_input)
    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
    print("✅ Flatten integration with Sequential")
    
    print("🎉 All Flatten operations tests passed!")

test_unit_flatten()

# In[ ]:

# ## NBGrader Assessment Questions

# ⭐ QUESTION 1: Parameter Counting Challenge
"""
You're building a Multi-Layer Perceptron (MLP) for MNIST digit classification.

Network architecture:
- Input: 784 features (28×28 pixel images, flattened)
- Hidden layer 1: 256 neurons with ReLU activation
- Hidden layer 2: 128 neurons with ReLU activation  
- Output layer: 10 neurons (one per digit class)

Calculate the total number of trainable parameters in this network.

Show your work:
- Layer 1 parameters: _____ 
- Layer 2 parameters: _____
- Layer 3 parameters: _____
- Total parameters: _____

Hint: Remember that each Linear layer has both weights and biases!
"""

# ### BEGIN SOLUTION
# Layer 1: Linear(784, 256)
# - Weights: 784 × 256 = 200,704
# - Biases: 256
# - Subtotal: 200,960

# Layer 2: Linear(256, 128)  
# - Weights: 256 × 128 = 32,768
# - Biases: 128
# - Subtotal: 32,896

# Layer 3: Linear(128, 10)
# - Weights: 128 × 10 = 1,280
# - Biases: 10
# - Subtotal: 1,290

# Total: 200,960 + 32,896 + 1,290 = 235,146 parameters
# ### END SOLUTION

# ⭐ QUESTION 2: Memory Analysis Challenge
"""
Compare the memory requirements of two different MLP architectures for the same task:

Architecture A (Wide): 784 → 512 → 512 → 10
Architecture B (Deep): 784 → 128 → 128 → 128 → 128 → 10

For each architecture, calculate:
1. Total number of parameters
2. Memory usage for parameters (assume float32 = 4 bytes per parameter)
3. Which architecture would you choose for a mobile device with limited memory?

Architecture A calculations:
- Total parameters: _____
- Memory usage: _____ MB

Architecture B calculations:  
- Total parameters: _____
- Memory usage: _____ MB

Mobile device choice and reasoning: _____
"""

# ### BEGIN SOLUTION
# Architecture A (Wide): 784 → 512 → 512 → 10
# - Layer 1: (784 × 512) + 512 = 401,920
# - Layer 2: (512 × 512) + 512 = 262,656  
# - Layer 3: (512 × 10) + 10 = 5,130
# - Total: 669,706 parameters
# - Memory: 669,706 × 4 bytes = 2.68 MB

# Architecture B (Deep): 784 → 128 → 128 → 128 → 128 → 10
# - Layer 1: (784 × 128) + 128 = 100,480
# - Layer 2: (128 × 128) + 128 = 16,512
# - Layer 3: (128 × 128) + 128 = 16,512  
# - Layer 4: (128 × 128) + 128 = 16,512
# - Layer 5: (128 × 10) + 10 = 1,290
# - Total: 151,306 parameters
# - Memory: 151,306 × 4 bytes = 0.61 MB

# Mobile choice: Architecture B (Deep)
# Reasoning: Uses 4.4x less memory while maintaining similar representational capacity through depth
# ### END SOLUTION

# ⭐ QUESTION 3: FLOPS Calculation Challenge
"""
Calculate the computational cost (in FLOPs) for a forward pass through this network:

Input batch: 32 samples × 784 features
Network: 784 → 256 → 128 → 10

For each layer, calculate:
- Matrix multiplication FLOPs: 2 × batch_size × input_size × output_size
- Bias addition FLOPs: batch_size × output_size
- Total FLOPs per layer

Layer 1 (784 → 256):
- MatMul FLOPs: _____
- Bias FLOPs: _____
- Layer total: _____

Layer 2 (256 → 128):
- MatMul FLOPs: _____  
- Bias FLOPs: _____
- Layer total: _____

Layer 3 (128 → 10):
- MatMul FLOPs: _____
- Bias FLOPs: _____
- Layer total: _____

Network total FLOPs: _____
"""

# ### BEGIN SOLUTION
# Batch size = 32 samples

# Layer 1 (784 → 256):
# - MatMul FLOPs: 2 × 32 × 784 × 256 = 12,582,912
# - Bias FLOPs: 32 × 256 = 8,192
# - Layer total: 12,591,104

# Layer 2 (256 → 128):
# - MatMul FLOPs: 2 × 32 × 256 × 128 = 2,097,152
# - Bias FLOPs: 32 × 128 = 4,096  
# - Layer total: 2,101,248

# Layer 3 (128 → 10):
# - MatMul FLOPs: 2 × 32 × 128 × 10 = 81,920
# - Bias FLOPs: 32 × 10 = 320
# - Layer total: 82,240

# Network total: 12,591,104 + 2,101,248 + 82,240 = 14,774,592 FLOPs (~14.8 MFLOPS)
# ### END SOLUTION

# In[ ]:

# ## Complete Neural Network Demo

def demonstrate_complete_networks():
    """Demonstrate complete neural networks using all implemented components."""
    print("🔥 Complete Neural Network Demo")
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
    # Simulate CNN → Flatten → Dense pattern
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
        print(f"   Added layer: {layer_sizes[i]} → {layer_sizes[i+1]}")
    
    # Test deep network
    deep_input = Tensor(np.random.randn(8, 100))
    deep_output = deep_net(deep_input)
    print(f"   Deep network: {deep_input.shape} → {deep_output.shape}")
    print(f"   Total parameters: {len(deep_net.parameters())} tensors")
    
    print("\n4. Parameter Management Across Networks:")
    networks = {'MLP': mlp, 'CNN-style': cnn_style, 'Deep': deep_net}
    
    for name, net in networks.items():
        params = net.parameters()
        total_params = sum(p.data.size for p in params)
        memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"   {name}: {len(params)} param tensors, {total_params:,} total params, {memory_mb:.2f} MB")
    
    print("\n🎉 All components work together seamlessly!")
    print("   • Module system enables automatic parameter collection")
    print("   • Linear layers handle matrix transformations") 
    print("   • Sequential composes layers into complete architectures")
    print("   • Flatten connects different layer types")
    print("   • Everything integrates for production-ready neural networks!")

demonstrate_complete_networks()

# In[ ]:

# ## Testing Framework

def test_unit_all():
    """Run complete module validation."""
    print("🧪 Running all unit tests...")
    
    # Call every individual test function
    test_unit_matmul()
    test_unit_linear()
    test_unit_parameter_management()
    test_unit_sequential()
    test_unit_flatten()
    
    print("✅ All tests passed! Module ready for integration.")

# In[ ]:

if __name__ == "__main__":
    print("🔥 TinyTorch Layers Module - Complete Foundation Demo")
    print("=" * 60)
    
    # Test all core components
    print("\n🧪 Testing All Core Components:")
    test_unit_all()
    
    print("\n" + "="*60)
    demonstrate_complete_networks()
    
    print("\n🎉 Complete neural network foundation ready!")
    print("   ✅ Module system for parameter management")
    print("   ✅ Linear layers for transformations")
    print("   ✅ Sequential networks for composition")
    print("   ✅ Flatten operations for tensor reshaping")
    print("   ✅ All components tested and integrated!")

# ## 🤔 ML Systems Thinking: Interactive Questions

# Now that you've implemented all the core neural network components, let's think about their implications for ML systems:

# ⭐ QUESTION: Memory vs Computation Trade-offs
"""
🤔 **Question 1: Memory vs Computation Analysis**

You're designing a neural network for deployment on a mobile device with limited memory (1GB RAM) but decent compute power.

You have two architecture options:
A) Wide network: 784 → 2048 → 2048 → 10 (3 layers, wide)
B) Deep network: 784 → 256 → 256 → 256 → 256 → 10 (5 layers, narrow)

Calculate the memory requirements for each option and explain which you'd choose for mobile deployment and why.

Consider:
- Parameter storage requirements
- Intermediate activation storage during forward pass
- Training vs inference memory requirements
- How your choice affects model capacity and accuracy
"""

# ⭐ QUESTION: Performance Optimization
"""
🤔 **Question 2: Production Performance Optimization**

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
"""

# ⭐ QUESTION: Scaling and Architecture Design
"""
🤔 **Question 3: Systems Architecture Scaling**

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

# ## 🎯 MODULE SUMMARY: Layers - Complete Neural Network Foundation

# ## 🎯 What You've Accomplished

# You've successfully implemented the complete foundation for neural networks - all the essential components working together:

# ### ✅ **Complete Core System**
# - **Module Base Class**: Parameter management and composition patterns for all neural network components
# - **Matrix Multiplication**: The computational primitive underlying all neural network operations
# - **Linear (Dense) Layers**: Complete implementation with proper parameter initialization and forward propagation
# - **Sequential Networks**: Clean composition system for building complete neural network architectures
# - **Flatten Operations**: Tensor reshaping to connect different layer types (essential for CNN→MLP transitions)

# ### ✅ **Systems Understanding**
# - **Architectural Patterns**: How modular design enables everything from MLPs to complex deep networks
# - **Memory Analysis**: How layer composition affects memory usage and computational efficiency
# - **Performance Characteristics**: Understanding how tensor operations and layer composition affect performance
# - **Production Context**: Connection to real-world ML frameworks and their component organization

# ### ✅ **ML Engineering Skills**
# - **Complete Parameter Management**: How neural networks automatically collect parameters from all components
# - **Network Composition**: Building complex architectures from simple, reusable components
# - **Tensor Operations**: Essential reshaping and transformation operations for different network types
# - **Clean Abstraction**: Professional software design patterns that scale to production systems

# ## 🔗 **Connection to Production ML Systems**

# Your unified implementation mirrors the complete component systems used in:
# - **PyTorch's nn.Module system**: Same parameter management and composition patterns
# - **PyTorch's nn.Sequential**: Identical architecture composition approach
# - **All major frameworks**: The same modular design principles that power TensorFlow, JAX, and others
# - **Production ML systems**: Clean abstractions that enable complex models while maintaining manageable code

# ## 🚀 **What's Next**

# With your complete layer foundation, you're ready to:
# - **Module 05 (Dense)**: Build complete dense networks for classification tasks
# - **Module 06 (Spatial)**: Add convolutional layers for computer vision
# - **Module 09 (Autograd)**: Enable automatic differentiation for learning
# - **Module 10 (Optimizers)**: Implement sophisticated optimization algorithms

# ## 💡 **Key Systems Insights**

# 1. **Modular composition is the key to scalable ML systems** - clean interfaces enable complex behaviors
# 2. **Parameter management must be automatic** - manual parameter tracking doesn't scale to deep networks
# 3. **Tensor operations like flattening are architectural requirements** - different layer types need different tensor shapes
# 4. **Clean abstractions enable innovation** - good foundational design supports unlimited architectural experimentation

# You now understand how to build complete, production-ready neural network foundations that can scale to any architecture!
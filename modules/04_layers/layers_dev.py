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
# Layers - Neural Network Building Blocks and Composition Patterns

Welcome to the unified Layers module! You'll build all the fundamental components for neural networks: base classes, linear transformations, network composition, and tensor reshaping operations.

## Learning Goals
- Systems understanding: How layer composition creates complex function approximators from simple building blocks
- Core implementation skill: Build Module base class, Linear layers, Sequential networks, and Flatten operations
- Pattern recognition: Understand how different layer types solve different computational problems and compose together
- Framework connection: See how your implementations mirror PyTorch's nn.Module, nn.Linear, nn.Sequential, and nn.Flatten patterns
- Performance insight: Learn why layer composition, memory layout, and tensor operations determine training speed

## Build → Use → Reflect
1. **Build**: Module system, matrix operations, Dense layers, Sequential networks, and tensor flattening
2. **Use**: Compose all components into complete neural networks and observe data flow patterns
3. **Reflect**: Why does proper abstraction enable complex architectures while maintaining clean interfaces?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of neural network component architecture and composition patterns
- Practical capability to build complete neural network systems from fundamental building blocks  
- Systems insight into why modular design is essential for scalable ML systems
- Performance consideration of how tensor operations and memory layout affect computational efficiency
- Connection to production ML systems and how major frameworks organize neural network components

## Systems Reality Check
💡 **Production Context**: PyTorch's nn.Module system enables all modern neural networks through clean composition patterns
⚡ **Performance Note**: Tensor reshape operations and layer composition can create memory bottlenecks - understanding this is key to efficient neural network design
"""

# %% nbgrader={"grade": false, "grade_id": "layers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.layers

#| export
import numpy as np
import sys
import os

# Clean production-style imports - no try/except hacking
if 'tinytorch' in sys.modules:
    # Production: Import from installed package
    from tinytorch.core.tensor import Tensor, Parameter
else:
    # Development: Direct import from local module
    tensor_module_path = os.path.join(os.path.dirname(__file__), '..', '02_tensor')
    sys.path.insert(0, tensor_module_path)
    try:
        from tensor_dev import Tensor, Parameter
    finally:
        sys.path.pop(0)  # Clean up path

# %% nbgrader={"grade": false, "grade_id": "layers-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Layers Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build neural network layers!")

# %% [markdown]
"""
## Module Base Class - Neural Network Foundation

Before building specific layers like Dense and Conv2d, we need a base class that handles parameter management and provides a clean interface. This is the foundation that makes neural networks composable and easy to use.

### Why We Need a Module Base Class

🏗️ **Organization**: Automatic parameter collection across all layers  
🔄 **Composition**: Modules can contain other modules (networks of networks)  
🎯 **Clean API**: Enable `model(input)` instead of `model.forward(input)`  
📦 **PyTorch Compatibility**: Same patterns as `torch.nn.Module`  

Let's build the foundation that will make all our neural network code clean and powerful:
"""

# %% nbgrader={"grade": false, "grade_id": "module-base-class", "locked": false, "schema_version": 3, "solution": false, "task": false}
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
                self.layer1 = Dense(784, 128)  # Auto-registered!
                self.layer2 = Dense(128, 10)   # Auto-registered!
                
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
        # Check if it's a Tensor that looks like a parameter (has .data attribute)
        # Parameters are typically named 'weights', 'bias', 'weight', etc.
        if (hasattr(value, 'data') and hasattr(value, 'shape') and 
            isinstance(value, Tensor) and 
            name in ['weights', 'weight', 'bias']):
            self._parameters.append(value)
        # Check if it's another Module (sub-module)
        elif isinstance(value, Module):
            self._modules.append(value)
        
        # Always call parent to actually set the attribute
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

# %% [markdown]
"""
## Where This Code Lives in the Final Package

**Learning Side:** You work in modules/04_layers/layers_dev.py  
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Module, Linear, Dense, Sequential, Flatten, matmul  # Complete layer system!
from tinytorch.core.tensor import Tensor, Parameter  # The foundation
from tinytorch.core.activations import ReLU, Sigmoid  # Nonlinearity
```

**Why this matters:**
- **Learning:** Complete layer system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn with all core components together
- **Consistency:** All layer types, network composition, and tensor operations in core.layers
- **Integration:** Works seamlessly with tensors and activations for complete neural networks
"""

# %% [markdown]
"""
# Matrix Multiplication - The Heart of Neural Networks

Every neural network operation ultimately reduces to matrix multiplication. Let's build the foundation that powers everything from simple perceptrons to transformers.

## Why Matrix Multiplication Matters

🧠 **Neural Network Core**: Every layer applies: output = input @ weights + bias  
⚡ **Parallel Processing**: Matrix ops utilize vectorized CPU instructions and GPU parallelism  
🏗️ **Scalable Architecture**: Stacking matrix operations creates arbitrarily complex function approximators  
📈 **Performance Critical**: 90%+ of neural network compute time is spent in matrix multiplication  

## Learning Objectives
By implementing matrix multiplication, you'll understand:
- How neural networks transform data through linear algebra
- Why matrix operations are the building blocks of all modern ML frameworks
- How proper implementation affects performance by orders of magnitude
- The connection between mathematical operations and computational efficiency
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
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
        raise ValueError(f"Inner dimensions must match: {k} != {k2}")
    
    # Initialize result matrix
    result = np.zeros((m, n), dtype=a_data.dtype)
    
    # Triple nested loops - educational, shows every operation
    # This is intentionally simple to understand the fundamental computation
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

# %% [markdown]
"""
## Testing Matrix Multiplication

Let's verify our matrix multiplication works correctly with some test cases.
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
def test_matmul():
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

test_matmul()

# %% [markdown]
"""
# Dense Layer - The Fundamental Neural Network Component

Dense layers (also called Linear or Fully Connected layers) are the building blocks of neural networks. They apply the transformation: **output = input @ weights + bias**

## Why Dense Layers Matter

🧠 **Universal Function Approximators**: Dense layers can approximate any continuous function when stacked  
🔧 **Parameter Learning**: Weights and biases are learned through backpropagation  
🏗️ **Modular Design**: Dense layers compose into complex architectures (MLPs, transformers, etc.)  
⚡ **Computational Efficiency**: Matrix operations leverage optimized linear algebra libraries  

## Learning Objectives
By implementing Dense layers, you'll understand:
- How neural networks learn through adjustable parameters
- The mathematical foundation underlying all neural network layers
- Why proper parameter initialization is crucial for training success
- How layer composition enables complex function approximation
"""

# %% nbgrader={"grade": false, "grade_id": "dense-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
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
        weight_data = np.random.randn(input_size, output_size) * 0.1
        self.weights = Parameter(weight_data)  # Auto-registers for optimization!
        
        # Initialize bias if requested
        if use_bias:
            bias_data = np.random.randn(output_size) * 0.1
            self.bias = Parameter(bias_data)  # Auto-registers for optimization!
        else:
            self.bias = None
        ### END SOLUTION
    
    def forward(self, x):
        """
        Forward pass through the Linear layer.
        
        Args:
            x: Input tensor or Variable (shape: ..., input_size)
        
        Returns:
            Output tensor or Variable (shape: ..., output_size)
            Preserves Variable type for gradient tracking in training
        
        TODO: Implement autograd-aware forward pass: output = input @ weights + bias
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Handle both Tensor and Variable inputs seamlessly
        2. Convert Parameters to Variables to maintain gradient connections
        3. Perform matrix multiplication: output = input @ weights
        4. Add bias if it exists: output = output + bias
        5. Return result maintaining Variable chain for training
        
        LEARNING CONNECTIONS:
        - This supports both inference (Tensors) and training (Variables)
        - Parameters are converted to Variables to enable gradient flow
        - Result maintains computational graph for automatic differentiation
        - Works with optimizers that expect Parameter gradients
        
        IMPLEMENTATION HINTS:
        - Import Variable from autograd module
        - Convert self.weights to Variable(self.weights) when needed
        - Use @ operator for matrix multiplication (calls __matmul__)
        - Handle bias addition with + operator
        """
        ### BEGIN SOLUTION
        # Import Variable for gradient tracking
        try:
            from tinytorch.core.autograd import Variable
        except ImportError:
            # Fallback for development
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '06_autograd'))
            from autograd_dev import Variable
        
        # Ensure input supports autograd if it's a Variable
        input_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        
        # Convert parameters to Variables to maintain gradient connections
        weight_var = Variable(self.weights) if not isinstance(self.weights, Variable) else self.weights
        
        # Matrix multiplication: input @ weights using Variable-aware operation
        output = input_var @ weight_var  # Use Variable.__matmul__ which calls matmul_vars
        
        # Add bias if it exists
        if self.bias is not None:
            bias_var = Variable(self.bias) if not isinstance(self.bias, Variable) else self.bias
            output = output + bias_var
        
        return output
        ### END SOLUTION

# Backward compatibility alias
#| export  
Dense = Linear

# %% [markdown]
"""
## Testing Linear Layer

Let's verify our Linear layer works correctly with comprehensive tests.
The tests use Dense for backward compatibility, but Dense is now an alias for Linear.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
def test_dense_layer():
    """Test Dense layer implementation."""
    print("🧪 Testing Dense Layer...")
    
    # Test case 1: Basic functionality
    layer = Dense(input_size=3, output_size=2)
    input_tensor = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    output = layer.forward(input_tensor)
    
    # Check output shape
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print("✅ Output shape correct")
    
    # Test case 2: No bias
    layer_no_bias = Dense(input_size=2, output_size=3, use_bias=False)
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("✅ No bias option works")
    
    # Test case 3: Multiple samples (batch processing)
    batch_input = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape: (3, 2)
    layer_batch = Dense(input_size=2, output_size=2)
    batch_output = layer_batch.forward(batch_input)
    
    assert batch_output.shape == (3, 2), f"Expected shape (3, 2), got {batch_output.shape}"
    print("✅ Batch processing works")
    
    # Test case 4: Callable interface
    callable_output = layer_batch(batch_input)
    assert np.allclose(callable_output.data, batch_output.data), "Callable interface should match forward()"
    print("✅ Callable interface works")
    
    # Test case 5: Parameter initialization
    layer_init = Dense(input_size=10, output_size=5)
    assert layer_init.weights.shape == (10, 5), f"Expected weights shape (10, 5), got {layer_init.weights.shape}"
    assert layer_init.bias.shape == (5,), f"Expected bias shape (5,), got {layer_init.bias.shape}"
    
    # Check that weights are reasonably small (good initialization)
    assert np.abs(layer_init.weights.data).mean() < 1.0, "Weights should be small for good initialization"
    print("✅ Parameter initialization correct")
    
    print("🎉 All Dense layer tests passed!")

test_dense_layer()

# %% [markdown]
"""
## Testing Autograd Integration

Now let's test that our Dense layer works correctly with parameter management and module composition.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dense-parameter-management", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
def test_dense_parameter_management():
    """Test Dense layer parameter management and module composition."""
    print("🧪 Testing Dense Layer Parameter Management...")
    
    # Test case 1: Parameter registration
    layer = Dense(input_size=3, output_size=2)
    params = layer.parameters()
    
    assert len(params) == 2, f"Expected 2 parameters (weights + bias), got {len(params)}"
    assert layer.weights in params, "Weights should be in parameters list"
    assert layer.bias in params, "Bias should be in parameters list"
    print("✅ Parameter registration works")
    
    # Test case 2: Module composition
    class SimpleNetwork(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Dense(4, 3)
            self.layer2 = Dense(3, 2)
        
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
    
    # Test case 4: Parameter shapes
    layer = Dense(input_size=10, output_size=5)
    assert layer.weights.shape == (10, 5), f"Expected weights shape (10, 5), got {layer.weights.shape}"
    assert layer.bias.shape == (5,), f"Expected bias shape (5,), got {layer.bias.shape}"
    print("✅ Parameter shapes correct")
    
    # Test case 5: No bias option
    layer_no_bias = Dense(input_size=3, output_size=2, use_bias=False)
    params_no_bias = layer_no_bias.parameters()
    
    assert len(params_no_bias) == 1, f"Expected 1 parameter (weights only), got {len(params_no_bias)}"
    assert layer_no_bias.bias is None, "Bias should be None when use_bias=False"
    print("✅ No bias option works")
    
    print("🎉 All Dense layer parameter management tests passed!")

test_dense_parameter_management()

# %% [markdown]
"""
# Sequential Network Composition - Building Complete Architectures

Now that we have solid layers, let's build the Sequential network that composes layers into complete neural network architectures. This is the foundation for all neural networks from MLPs to complex deep learning models.

## Why Sequential Networks Matter

🏗️ **Architecture Foundation**: Sequential is the building block for all neural network architectures  
🔄 **Function Composition**: Chain simple functions to create complex behaviors
📦 **Clean Interface**: Write networks as lists of layers - intuitive and maintainable  
⚡ **Production Standard**: Every major framework uses this pattern for neural network construction  

## Learning Objectives
By implementing Sequential networks, you'll understand:
- How function composition enables universal approximation in neural networks
- The architectural patterns that power everything from MLPs to transformers
- Why clean abstractions matter for building complex systems
- How layer composition creates the foundation for all modern deep learning
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
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

# %% [markdown]
"""
## Testing Sequential Networks

Let's verify our Sequential network works correctly with comprehensive tests.
"""

# %% nbgrader={"grade": true, "grade_id": "test-sequential", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
def test_sequential_network():
    """Test Sequential network implementation."""
    print("🧪 Testing Sequential Network...")
    
    # Test case 1: Create empty network
    empty_net = Sequential()
    assert len(empty_net.layers) == 0, "Empty Sequential should have no layers"
    print("✅ Empty Sequential network creation")
    
    # Test case 2: Create network with layers
    layers = [Dense(3, 4), Dense(4, 2)]
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
    # Should have 4 parameters: 2 weights + 2 biases from 2 Dense layers
    assert len(all_params) == 4, f"Expected 4 parameters from Sequential network, got {len(all_params)}"
    print("✅ Parameter collection from all layers")
    
    # Test case 5: Adding layers dynamically
    network.add(Dense(2, 1))
    assert len(network.layers) == 3, "Network should have 3 layers after adding one"
    
    # Test forward pass after adding layer
    final_output = network(input_tensor)
    assert final_output.shape == (1, 1), f"Expected final output shape (1, 1), got {final_output.shape}"
    print("✅ Dynamic layer addition")
    
    print("🎉 All Sequential network tests passed!")

test_sequential_network()

# %% [markdown]
"""
# Flatten Operation - Connecting Different Layer Types

The Flatten operation is essential for connecting convolutional layers to dense layers, or reshaping tensors between different network components. This is a fundamental operation in neural networks.

## Why Flatten Matters

🔗 **Interface Bridge**: Connects spatial layers (Conv2D) to dense layers (Linear)  
📐 **Dimension Management**: Converts multi-dimensional tensors to vectors for different layer types  
🏗️ **Architecture Flexibility**: Enables mixing different layer types in the same network  
⚡ **Memory Efficiency**: Provides clean tensor reshaping without copying data  

## Learning Objectives
By implementing Flatten, you'll understand:
- How neural networks handle tensors of different shapes between layer types
- The critical role of tensor reshaping in network architecture design
- How to preserve batch dimensions while flattening spatial dimensions
- The connection between memory layout and computational efficiency
"""

# %% nbgrader={"grade": false, "grade_id": "flatten-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
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

# %% nbgrader={"grade": false, "grade_id": "flatten-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
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
    
    # Reshape preserving tensor type
    if hasattr(x, 'data'):
        # It's a Tensor - preserve type
        flattened_data = data.reshape(new_shape)
        return type(x)(flattened_data)
    else:
        # It's a numpy array
        return data.reshape(new_shape)

# %% [markdown]
"""
## Testing Flatten Operations

Let's verify our Flatten implementation works correctly with various tensor shapes.
"""

# %% nbgrader={"grade": true, "grade_id": "test-flatten", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
def test_flatten_operations():
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
        Dense(8, 4),
        Flatten()
    ])
    test_input = Tensor(np.random.randn(2, 8))
    output = network(test_input)
    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
    print("✅ Flatten integration with Sequential")
    
    print("🎉 All Flatten operations tests passed!")

test_flatten_operations()

# %% [markdown]
"""
# Systems Analysis: Memory and Performance Characteristics

Let's analyze the memory usage and computational complexity of our layer implementations.

## Memory Analysis
- **Dense Layer Storage**: input_size × output_size weights + output_size bias terms
- **Forward Pass Memory**: Input tensor + weight tensor + output tensor (temporary storage)
- **Scaling Behavior**: Memory grows quadratically with layer size

## Computational Complexity
- **Matrix Multiplication**: O(batch_size × input_size × output_size)
- **Bias Addition**: O(batch_size × output_size)
- **Total**: Dominated by matrix multiplication for large layers

## Production Insights
In production ML systems:
- **Memory Management**: PyTorch uses memory pools to avoid frequent allocation/deallocation
- **Compute Optimization**: BLAS libraries (MKL, OpenBLAS) optimize matrix operations for specific hardware
- **GPU Acceleration**: CUDA kernels parallelize matrix operations across thousands of cores
- **Mixed Precision**: Using float16 instead of float32 can halve memory usage with minimal accuracy loss
"""

# %% nbgrader={"grade": false, "grade_id": "memory-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
def analyze_layer_memory():
    """Analyze memory usage of different layer sizes."""
    print("📊 Layer Memory Analysis")
    print("=" * 40)
    
    layer_sizes = [(10, 10), (100, 100), (1000, 1000), (784, 128), (128, 10)]
    
    for input_size, output_size in layer_sizes:
        # Calculate parameter count
        weight_params = input_size * output_size
        bias_params = output_size
        total_params = weight_params + bias_params
        
        # Calculate memory usage (assuming float32 = 4 bytes)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        print(f"  {input_size:4d} → {output_size:4d}: {total_params:,} params, {memory_mb:.3f} MB")
    
    print("\n🔍 Key Insights:")
    print("  • Memory grows quadratically with layer width")
    print("  • Large layers (1000×1000) use significant memory")
    print("  • Modern networks balance width vs depth for efficiency")

analyze_layer_memory()

# %% [markdown]
"""
# ML Systems Thinking: Interactive Questions

Let's explore the deeper implications of our layer implementations.
"""

# %% nbgrader={"grade": false, "grade_id": "systems-thinking", "locked": false, "schema_version": 3, "solution": false, "task": false}
def explore_layer_scaling():
    """Explore how layer operations scale with size."""
    print("🤔 Scaling Analysis: Matrix Multiplication Performance")
    print("=" * 55)
    
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        # Estimate FLOPs for square matrix multiplication
        flops = 2 * size * size * size  # 2 operations per multiply-add
        
        # Estimate memory bandwidth (reading A, B, writing C)
        memory_ops = 3 * size * size  # Elements read/written
        memory_mb = memory_ops * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        print(f"  Size {size:3d}×{size:3d}: {flops/1e6:.1f} MFLOPS, {memory_mb:.2f} MB transfers")
    
    print("\n💡 Performance Insights:")
    print("  • FLOPs grow cubically (O(n³)) with matrix size")
    print("  • Memory bandwidth grows quadratically (O(n²))")
    print("  • Large matrices become memory-bound, not compute-bound")
    print("  • This is why GPUs excel: high memory bandwidth + parallel compute")

explore_layer_scaling()

# %% [markdown]
"""
# Complete Neural Network Demo - All Components Working Together

Let's demonstrate how all our components work together to build complete neural networks.
"""

# %% nbgrader={"grade": false, "grade_id": "complete-network-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
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

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented all the core neural network components, let's think about their implications for ML systems:
"""

# %% nbgrader={"grade": false, "grade_id": "question-1", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 1: Memory vs Computation Trade-offs
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

# %% nbgrader={"grade": false, "grade_id": "question-2", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 2: Performance Optimization
"""
🤔 **Question 2: Production Performance Optimization**

Your Dense layer implementation works correctly, but you notice it's slower than PyTorch's nn.Linear on the same hardware.

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

# %% nbgrader={"grade": false, "grade_id": "question-3", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 3: Scaling and Architecture Design
"""
🤔 **Question 3: Systems Architecture Scaling**

Modern transformer models like GPT-3 have billions of parameters, primarily in Dense layers.

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
# Comprehensive Testing and Integration

Let's run a comprehensive test suite to verify all our implementations work correctly together.
"""

# %% nbgrader={"grade": false, "grade_id": "comprehensive-tests", "locked": false, "schema_version": 3, "solution": false, "task": false}
def run_comprehensive_tests():
    """Run comprehensive tests of all layer functionality."""
    print("🔬 Comprehensive Layer Testing Suite")
    print("=" * 45)
    
    # Test 1: Matrix multiplication edge cases
    print("\n1. Matrix Multiplication Edge Cases:")
    
    # Single element
    a = Tensor([[5]])
    b = Tensor([[3]])
    result = matmul(a, b)
    assert result.data[0, 0] == 15, "Single element multiplication failed"
    print("   ✅ Single element multiplication")
    
    # Identity matrix
    identity = Tensor([[1, 0], [0, 1]])
    test_matrix = Tensor([[2, 3], [4, 5]])
    result = matmul(test_matrix, identity)
    assert np.allclose(result.data, test_matrix.data), "Identity multiplication failed"
    print("   ✅ Identity matrix multiplication")
    
    # Test 2: Dense layer composition
    print("\n2. Dense Layer Composition:")
    
    # Create a simple 2-layer network
    layer1 = Dense(4, 3)
    layer2 = Dense(3, 2)
    
    # Test data flow
    input_data = Tensor([[1, 2, 3, 4]])
    hidden = layer1(input_data)
    output = layer2(hidden)
    
    assert output.shape == (1, 2), f"Expected final output shape (1, 2), got {output.shape}"
    print("   ✅ Multi-layer composition")
    
    # Test 3: Batch processing
    print("\n3. Batch Processing:")
    
    batch_size = 10
    batch_input = Tensor(np.random.randn(batch_size, 4))
    batch_hidden = layer1(batch_input)
    batch_output = layer2(batch_hidden)
    
    assert batch_output.shape == (batch_size, 2), f"Expected batch output shape ({batch_size}, 2), got {batch_output.shape}"
    print("   ✅ Batch processing")
    
    # Test 4: Parameter access and modification
    print("\n4. Parameter Management:")
    
    layer = Dense(5, 3)
    original_weights = layer.weights.data.copy()
    
    # Simulate parameter update
    layer.weights = Tensor(original_weights + 0.1)
    
    assert not np.allclose(layer.weights.data, original_weights), "Parameter update failed"
    print("   ✅ Parameter modification")
    
    print("\n🎉 All comprehensive tests passed!")
    print("   Your layer implementations are ready for neural network construction!")

run_comprehensive_tests()

# %% [markdown]
"""
## Autograd Integration Demo

Let's demonstrate how layers compose together to build neural networks.
"""

# %% nbgrader={"grade": false, "grade_id": "layer-composition-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
def demonstrate_layer_composition():
    """Demonstrate how layers compose into neural networks."""
    print("🔥 Layer Composition Demo")
    print("=" * 50)
    
    print("\n1. Creating individual layers:")
    layer1 = Dense(input_size=4, output_size=3)
    layer2 = Dense(input_size=3, output_size=2)
    
    print(f"   Layer 1: {layer1.input_size} → {layer1.output_size}")
    print(f"   Layer 2: {layer2.input_size} → {layer2.output_size}")
    
    print("\n2. Manual layer composition:")
    input_data = Tensor([[1.0, 2.0, 3.0, 4.0]])
    print(f"   Input shape: {input_data.shape}")
    
    # Forward pass through each layer
    hidden = layer1(input_data)
    print(f"   After layer 1: {hidden.shape}")
    
    output = layer2(hidden)
    print(f"   Final output: {output.shape}")
    print(f"   Output values: {output.data.tolist()}")
    
    print("\n3. Creating a composed network class:")
    class TwoLayerNetwork(Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layer1 = Dense(input_size, hidden_size)
            self.layer2 = Dense(hidden_size, output_size)
        
        def forward(self, x):
            x = self.layer1(x)
            return self.layer2(x)
    
    network = TwoLayerNetwork(4, 3, 2)
    network_output = network(input_data)
    
    print(f"   Network output shape: {network_output.shape}")
    print(f"   Network output values: {network_output.data.tolist()}")
    
    print("\n4. Parameter management:")
    all_params = network.parameters()
    print(f"   Total parameters: {len(all_params)}")
    
    total_param_count = 0
    for i, param in enumerate(all_params):
        total_param_count += param.data.size
        print(f"   Parameter {i}: shape {param.shape}, size {param.data.size}")
    
    print(f"   Total parameter count: {total_param_count:,}")
    
    print("\n5. Batch processing:")
    batch_input = Tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]])
    batch_output = network(batch_input)
    
    print(f"   Batch input shape: {batch_input.shape}")
    print(f"   Batch output shape: {batch_output.shape}")
    print("   ✅ Batch processing works automatically!")
    
    print("\n🎉 Layer composition enables building complex neural networks!")
    print("   • Individual layers are building blocks")
    print("   • Module class enables clean composition")
    print("   • Parameter management happens automatically")
    print("   • Batch processing scales to multiple samples")

demonstrate_layer_composition()

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Layers - Complete Neural Network Foundation

## 🎯 What You've Accomplished

You've successfully implemented the complete foundation for neural networks - all the essential components working together:

### ✅ **Complete Core System**
- **Module Base Class**: Parameter management and composition patterns for all neural network components
- **Matrix Multiplication**: The computational primitive underlying all neural network operations
- **Linear (Dense) Layers**: Complete implementation with proper parameter initialization and forward propagation
- **Sequential Networks**: Clean composition system for building complete neural network architectures
- **Flatten Operations**: Tensor reshaping to connect different layer types (essential for CNN→MLP transitions)

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

## 🔗 **Connection to Production ML Systems**

Your unified implementation mirrors the complete component systems used in:
- **PyTorch's nn.Module system**: Same parameter management and composition patterns
- **PyTorch's nn.Sequential**: Identical architecture composition approach
- **All major frameworks**: The same modular design principles that power TensorFlow, JAX, and others
- **Production ML systems**: Clean abstractions that enable complex models while maintaining manageable code

## 🚀 **What's Next**

With your complete layer foundation, you're ready to:
- **Add nonlinear activations** to enable complex function approximation
- **Implement loss functions** to define learning objectives
- **Build training algorithms** to optimize networks on data
- **Create specialized layers** like convolutions for computer vision

## 💡 **Key Systems Insights**

1. **Modular composition is the key to scalable ML systems** - clean interfaces enable complex behaviors
2. **Parameter management must be automatic** - manual parameter tracking doesn't scale to deep networks
3. **Tensor operations like flattening are architectural requirements** - different layer types need different tensor shapes
4. **Clean abstractions enable innovation** - good foundational design supports unlimited architectural experimentation

You now understand how to build complete, production-ready neural network foundations that can scale to any architecture!
"""

# %% nbgrader={"grade": false, "grade_id": "final-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    print("🔥 TinyTorch Layers Module - Complete Foundation Demo")
    print("=" * 60)
    
    # Test all core components
    print("\n🧪 Testing All Core Components:")
    test_matmul()
    test_dense_layer()
    test_dense_parameter_management()
    test_sequential_network()
    test_flatten_operations()
    
    print("\n" + "="*60)
    demonstrate_complete_networks()
    
    print("\n" + "="*60)
    demonstrate_layer_composition()
    
    print("\n🎉 Complete neural network foundation ready!")
    print("   ✅ Module system for parameter management")
    print("   ✅ Linear layers for transformations")
    print("   ✅ Sequential networks for composition")
    print("   ✅ Flatten operations for tensor reshaping")
    print("   ✅ All components tested and integrated!")
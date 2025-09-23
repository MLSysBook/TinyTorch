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

Welcome to the Layers module! You'll build the fundamental components that stack together to form any neural network architecture, from simple perceptrons to transformers.

## Learning Goals
- Systems understanding: How layer composition creates complex function approximators and why stacking enables deep learning
- Core implementation skill: Build matrix multiplication and Dense layers with proper parameter management
- Pattern recognition: Understand how different layer types solve different computational problems
- Framework connection: See how your layer implementations mirror PyTorch's nn.Module design patterns
- Performance insight: Learn why layer computation order and memory layout determine training speed

## Build → Use → Reflect
1. **Build**: Matrix multiplication primitives and Dense layers with parameter initialization strategies
2. **Use**: Compose layers into multi-layer networks and observe how data transforms through the stack
3. **Reflect**: Why does layer depth enable more complex functions, and when does it hurt performance?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how matrix operations enable neural networks to learn arbitrary functions
- Practical capability to build and compose layers into complex architectures
- Systems insight into why layer composition is the fundamental pattern for scalable ML systems
- Performance consideration of how layer size and depth affect memory usage and computational cost
- Connection to production ML systems and how frameworks optimize layer execution for different hardware

## Systems Reality Check
💡 **Production Context**: PyTorch's nn.Linear uses optimized BLAS operations and can automatically select GPU vs CPU execution based on data size
⚡ **Performance Note**: Large matrix multiplications can be memory-bound rather than compute-bound - understanding this shapes how production systems optimize layer execution
"""

# %% nbgrader={"grade": false, "grade_id": "layers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.layers

#| export
import numpy as np
import sys
import os

# Import our building blocks - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor, Parameter
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor, Parameter

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

**Learning Side:** You work in modules/source/04_layers/layers_dev.py  
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Dense, matmul  # All layer types together!
from tinytorch.core.tensor import Tensor  # The foundation
from tinytorch.core.activations import ReLU, Sigmoid  # Nonlinearity
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.Linear
- **Consistency:** All layer types live together in core.layers
- **Integration:** Works seamlessly with tensors and activations
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
    Matrix multiplication for tensors.
    
    Args:
        a: Left tensor (shape: ..., m, k)
        b: Right tensor (shape: ..., k, n)
    
    Returns:
        Result tensor (shape: ..., m, n)
    
    TODO: Implement matrix multiplication using numpy's @ operator.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy arrays from both tensors using .data
    2. Perform matrix multiplication: result_data = a_data @ b_data
    3. Wrap result in a new Tensor and return
    
    LEARNING CONNECTIONS:
    - This is the core operation in Dense layers: output = input @ weights
    - PyTorch uses optimized BLAS libraries for this operation
    - GPU implementations parallelize this across thousands of cores
    - Understanding this operation is key to neural network performance
    
    EXAMPLE:
    ```python
    a = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
    b = Tensor([[5, 6], [7, 8]])  # shape (2, 2)
    result = matmul(a, b)
    # result.data = [[19, 22], [43, 50]]
    ```
    
    IMPLEMENTATION HINTS:
    - Use the @ operator for clean matrix multiplication
    - Ensure you return a Tensor, not a numpy array
    - The operation should work for any compatible matrix shapes
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays from tensors
    a_data = a.data
    b_data = b.data
    
    # Perform matrix multiplication
    result_data = a_data @ b_data
    
    # Return new Tensor with result
    return Tensor(result_data)
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
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Linear layer.
        
        Args:
            x: Input tensor (shape: ..., input_size)
        
        Returns:
            Output tensor (shape: ..., output_size)
        
        TODO: Implement forward pass: output = input @ weights + bias
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Perform matrix multiplication: output = matmul(x, self.weights)
        2. If bias exists, add it: output = output + self.bias
        3. Return result as Tensor
        
        LEARNING CONNECTIONS:
        - This is the core neural network transformation
        - Matrix multiplication scales input features to output features  
        - Bias provides offset (like y-intercept in linear equations)
        - Broadcasting handles different batch sizes automatically
        
        IMPLEMENTATION HINTS:
        - Use the matmul function you implemented above
        - Handle bias addition with simple + operator
        - Check if self.bias is not None before adding
        """
        ### BEGIN SOLUTION
        # Matrix multiplication: input @ weights
        output = matmul(x, self.weights)
        
        # Add bias if it exists
        if self.bias is not None:
            output = output + self.bias
        
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
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented the core components, let's think about their implications for ML systems:
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
## 🎯 MODULE SUMMARY: Layers

## 🎯 What You've Accomplished

You've successfully implemented the fundamental building blocks of neural networks:

### ✅ **Core Implementations**
- **Matrix Multiplication**: The computational primitive underlying all neural network operations
- **Dense Layer**: Complete implementation with proper parameter initialization and forward propagation
- **Module System**: Clean composition patterns for building complex neural networks
- **Composition Patterns**: How layers stack together to form complex function approximators

### ✅ **Systems Understanding**
- **Memory Analysis**: How layer size affects memory usage and why this matters for deployment
- **Performance Characteristics**: Understanding computational complexity and scaling behavior
- **Production Context**: Connection to real-world ML systems and optimization techniques

### ✅ **ML Engineering Skills**
- **Parameter Management**: How neural networks store and update learnable parameters
- **Batch Processing**: Efficient handling of multiple data samples simultaneously
- **Architecture Design**: Trade-offs between network width, depth, and resource requirements

## 🔗 **Connection to Production ML Systems**

Your implementations mirror the core concepts used in:
- **PyTorch's nn.Linear**: Same mathematical operations with production optimizations
- **TensorFlow's Dense layers**: Identical parameter structure and forward pass logic
- **Transformer architectures**: Dense layers form the foundation of modern language models
- **Computer vision models**: ConvNets use similar principles with spatial structure

## 🚀 **What's Next**

With solid layer implementations, you're ready to:
- **Compose** these layers into complete neural networks
- **Add** nonlinear activations to enable complex function approximation
- **Implement** training algorithms to learn from data
- **Scale** to larger, more sophisticated architectures

## 💡 **Key Systems Insights**

1. **Matrix multiplication is the computational bottleneck** in neural networks
2. **Memory layout and access patterns** often matter more than raw compute power
3. **Layer composition** is the fundamental abstraction for building complex ML systems
4. **Parameter initialization and management** directly affects training success

You now understand the mathematical and computational foundations that enable neural networks to learn complex patterns from data!
"""

# %% nbgrader={"grade": false, "grade_id": "final-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    print("🔥 TinyTorch Layers Module - Final Demo")
    print("=" * 50)
    
    # Create a simple neural network architecture
    print("\n🏗️ Building a 3-layer neural network:")
    layer1 = Dense(784, 128)  # Input layer (like MNIST images)
    layer2 = Dense(128, 64)   # Hidden layer
    layer3 = Dense(64, 10)    # Output layer (10 classes)
    
    print(f"  Layer 1: {layer1.input_size} → {layer1.output_size} ({layer1.weights.data.size:,} parameters)")
    print(f"  Layer 2: {layer2.input_size} → {layer2.output_size} ({layer2.weights.data.size:,} parameters)")
    print(f"  Layer 3: {layer3.input_size} → {layer3.output_size} ({layer3.weights.data.size:,} parameters)")
    
    # Simulate forward pass
    print("\n🚀 Forward pass through network:")
    batch_size = 32
    input_data = Tensor(np.random.randn(batch_size, 784))
    
    print(f"  Input shape: {input_data.shape}")
    hidden1 = layer1(input_data)
    print(f"  After layer 1: {hidden1.shape}")
    hidden2 = layer2(hidden1)
    print(f"  After layer 2: {hidden2.shape}")
    output = layer3(hidden2)
    print(f"  Final output: {output.shape}")
    
    # Calculate total parameters
    total_params = (layer1.weights.data.size + layer1.bias.data.size + 
                   layer2.weights.data.size + layer2.bias.data.size +
                   layer3.weights.data.size + layer3.bias.data.size)
    
    print(f"\n📊 Network Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Memory usage: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"  Forward pass: {batch_size} samples processed simultaneously")
    
    print("\n✅ Neural network construction complete!")
    print("Ready for activation functions and training algorithms!")
    
    # Run layer composition demo
    print("\n" + "="*60)
    demonstrate_layer_composition()
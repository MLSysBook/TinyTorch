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
# Module 7: Autograd - Automatic Differentiation Engine

Welcome to the Autograd module! This is where TinyTorch becomes truly powerful. You'll implement the automatic differentiation engine that makes neural network training possible.

## Learning Goals
- Understand how automatic differentiation works through computational graphs
- Implement the Variable class that tracks gradients and operations
- Build backward propagation for gradient computation
- Create the foundation for neural network training
- Master the mathematical concepts behind backpropagation

## Build → Use → Analyze
1. **Build**: Create the Variable class and gradient computation system
2. **Use**: Perform automatic differentiation on complex expressions
3. **Analyze**: Understand how gradients flow through computational graphs
"""

# %% nbgrader={"grade": false, "grade_id": "autograd-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.autograd

#| export
import numpy as np
import sys
from typing import Union, List, Tuple, Optional, Any, Callable
from collections import defaultdict

# Import our existing components
from tinytorch.core.tensor import Tensor

# %% nbgrader={"grade": false, "grade_id": "autograd-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Autograd Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build automatic differentiation!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/07_autograd/autograd_dev.py`  
**Building Side:** Code exports to `tinytorch.core.autograd`

```python
# Final package structure:
from tinytorch.core.autograd import Variable, backward  # The gradient engine!
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
```

**Why this matters:**
- **Learning:** Focused module for understanding gradients
- **Production:** Proper organization like PyTorch's `torch.autograd`
- **Consistency:** All gradient operations live together in `core.autograd`
- **Foundation:** Enables training for all neural networks
"""

# %% [markdown]
"""
## Step 1: What is Automatic Differentiation?

### Definition
**Automatic differentiation (autograd)** is a technique that automatically computes derivatives of functions represented as computational graphs. It's the magic that makes neural network training possible.

### The Fundamental Challenge: Computing Gradients at Scale

#### **The Problem**
Neural networks have millions or billions of parameters. To train them, we need to compute the gradient of the loss function with respect to every single parameter:

```python
# For a neural network with parameters θ = [w1, w2, ..., wn, b1, b2, ..., bm]
# We need to compute: ∇θ L = [∂L/∂w1, ∂L/∂w2, ..., ∂L/∂wn, ∂L/∂b1, ∂L/∂b2, ..., ∂L/∂bm]
```

#### **Why Manual Differentiation Fails**
- **Complexity**: Neural networks are compositions of thousands of operations
- **Error-prone**: Manual computation is extremely difficult and error-prone
- **Inflexible**: Every architecture change requires re-deriving gradients
- **Inefficient**: Manual computation doesn't exploit computational structure

#### **Why Numerical Differentiation is Inadequate**
```python
# Numerical differentiation: f'(x) ≈ (f(x + h) - f(x)) / h
def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h
```

Problems:
- **Slow**: Requires 2 function evaluations per parameter
- **Imprecise**: Numerical errors accumulate
- **Unstable**: Sensitive to choice of h
- **Expensive**: O(n) cost for n parameters

### The Solution: Computational Graphs

#### **Key Insight: Every Computation is a Graph**
Any mathematical expression can be represented as a directed acyclic graph (DAG):

```python
# Expression: f(x, y) = (x + y) * (x - y)
# Graph representation:
#     x ──┐     ┌── add ──┐
#         │     │         │
#         ├─────┤         ├── multiply ── output
#         │     │         │
#     y ──┘     └── sub ──┘
```

#### **Forward Pass: Computing Values**
Traverse the graph from inputs to outputs, computing values at each node:

```python
# Forward pass for f(x, y) = (x + y) * (x - y)
x = 3, y = 2
add_result = x + y = 5
sub_result = x - y = 1
output = add_result * sub_result = 5
```

#### **Backward Pass: Computing Gradients**
Traverse the graph from outputs to inputs, computing gradients using the chain rule:

```python
# Backward pass for f(x, y) = (x + y) * (x - y)
# Starting from output gradient = 1
∂output/∂multiply = 1
∂output/∂add = ∂output/∂multiply * ∂multiply/∂add = 1 * sub_result = 1
∂output/∂sub = ∂output/∂multiply * ∂multiply/∂sub = 1 * add_result = 5
∂output/∂x = ∂output/∂add * ∂add/∂x + ∂output/∂sub * ∂sub/∂x = 1 * 1 + 5 * 1 = 6
∂output/∂y = ∂output/∂add * ∂add/∂y + ∂output/∂sub * ∂sub/∂y = 1 * 1 + 5 * (-1) = -4
```

### Mathematical Foundation: The Chain Rule

#### **Single Variable Chain Rule**
For composite functions: If z = f(g(x)), then:
```
dz/dx = (dz/df) * (df/dx)
```

#### **Multivariable Chain Rule**
For functions of multiple variables: If z = f(x, y) where x = g(t) and y = h(t), then:
```
dz/dt = (∂z/∂x) * (dx/dt) + (∂z/∂y) * (dy/dt)
```

#### **Chain Rule in Computational Graphs**
For any path from input to output through intermediate nodes:
```
∂output/∂input = ∏(∂node_{i+1}/∂node_i) for all nodes in the path
```

### Automatic Differentiation Modes

#### **Forward Mode (Forward Accumulation)**
- **Process**: Compute derivatives alongside forward pass
- **Efficiency**: Efficient when #inputs << #outputs
- **Use case**: Jacobian-vector products, sensitivity analysis

#### **Reverse Mode (Backpropagation)**
- **Process**: Compute derivatives in reverse pass after forward pass
- **Efficiency**: Efficient when #outputs << #inputs
- **Use case**: Neural network training (many parameters, few outputs)

#### **Why Reverse Mode Dominates ML**
Neural networks typically have:
- **Many inputs**: Millions of parameters
- **Few outputs**: Single loss value or small output vector
- **Reverse mode**: O(1) cost per parameter vs O(n) for forward mode

### The Computational Graph Abstraction

#### **Nodes: Operations and Variables**
- **Variable nodes**: Store values and gradients
- **Operation nodes**: Define how to compute forward and backward passes

#### **Edges: Data Dependencies**
- **Forward edges**: Data flow from inputs to outputs
- **Backward edges**: Gradient flow from outputs to inputs

#### **Dynamic vs Static Graphs**
- **Static graphs**: Define once, execute many times (TensorFlow 1.x)
- **Dynamic graphs**: Build graph during execution (PyTorch, TensorFlow 2.x)

### Real-World Impact: What Autograd Enables

#### **Deep Learning Revolution**
```python
# Before autograd: Manual gradient computation
def manual_gradient(x, y, w1, w2, b1, b2):
    # Forward pass
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2)
    loss = (a2 - y) ** 2
    
    # Backward pass (manual)
    dloss_da2 = 2 * (a2 - y)
    da2_dz2 = sigmoid_derivative(z2)
    dz2_dw2 = a1
    dz2_db2 = 1
    dz2_da1 = w2
    da1_dz1 = sigmoid_derivative(z1)
    dz1_dw1 = x
    dz1_db1 = 1
    
    # Chain rule application
    dloss_dw2 = dloss_da2 * da2_dz2 * dz2_dw2
    dloss_db2 = dloss_da2 * da2_dz2 * dz2_db2
    dloss_dw1 = dloss_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
    dloss_db1 = dloss_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_db1
    
    return dloss_dw1, dloss_db1, dloss_dw2, dloss_db2

# With autograd: Automatic gradient computation
def autograd_gradient(x, y, w1, w2, b1, b2):
    # Forward pass with gradient tracking
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2)
    loss = (a2 - y) ** 2
    
    # Backward pass (automatic)
    loss.backward()
    
    return w1.grad, b1.grad, w2.grad, b2.grad
```

#### **Scientific Computing**
- **Optimization**: Gradient-based optimization algorithms
- **Inverse problems**: Parameter estimation from observations
- **Sensitivity analysis**: How outputs change with input perturbations

#### **Modern AI Applications**
- **Neural architecture search**: Differentiable architecture optimization
- **Meta-learning**: Learning to learn with gradient-based meta-algorithms
- **Differentiable programming**: Entire programs as differentiable functions

### Performance Considerations

#### **Memory Management**
- **Intermediate storage**: Must store forward pass results for backward pass
- **Memory optimization**: Checkpointing, gradient accumulation
- **Trade-offs**: Memory vs computation time

#### **Computational Efficiency**
- **Graph optimization**: Fuse operations, eliminate redundancy
- **Parallelization**: Compute independent gradients simultaneously
- **Hardware acceleration**: Specialized gradient computation on GPUs/TPUs

#### **Numerical Stability**
- **Gradient clipping**: Prevent exploding gradients
- **Numerical precision**: Balance between float16 and float32
- **Accumulation order**: Minimize numerical errors

### Connection to Neural Network Training

#### **The Training Loop**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch.inputs)
        loss = criterion(predictions, batch.targets)
        
        # Backward pass (autograd)
        loss.backward()
        
        # Parameter update
        optimizer.step()
        optimizer.zero_grad()
```

#### **Gradient-Based Optimization**
- **Stochastic Gradient Descent**: Use gradients to update parameters
- **Adaptive methods**: Adam, RMSprop use gradient statistics
- **Second-order methods**: Use gradient and Hessian information

### Why Autograd is Revolutionary

#### **Democratization of Deep Learning**
- **Research acceleration**: Focus on architecture, not gradient computation
- **Experimentation**: Easy to try new ideas and architectures
- **Accessibility**: Researchers don't need to be differentiation experts

#### **Scalability**
- **Large models**: Handle millions/billions of parameters automatically
- **Complex architectures**: Support arbitrary computational graphs
- **Distributed training**: Coordinate gradients across multiple devices

Let's implement the Variable class that makes this magic possible!
"""

# %% [markdown]
"""
## Step 2: The Variable Class

### Core Concept
A **Variable** wraps a Tensor and tracks:
- **Data**: The actual values (forward pass)
- **Gradient**: The computed gradients (backward pass)
- **Computation history**: How this Variable was created
- **Backward function**: How to compute gradients

### Design Principles
- **Transparency**: Works seamlessly with existing Tensor operations
- **Efficiency**: Minimal overhead for forward pass
- **Flexibility**: Supports any differentiable operation
- **Correctness**: Implements the chain rule precisely
"""

# %% nbgrader={"grade": false, "grade_id": "variable-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Variable:
    """
    Variable: Tensor wrapper with automatic differentiation capabilities.
    
    The fundamental class for gradient computation in TinyTorch.
    Wraps Tensor objects and tracks computational history for backpropagation.
    """
    
    def __init__(self, data: Union[Tensor, np.ndarray, list, float, int], 
                 requires_grad: bool = True, grad_fn: Optional[Callable] = None):
        """
        Create a Variable with gradient tracking.
        
        Args:
            data: The data to wrap (will be converted to Tensor)
            requires_grad: Whether to compute gradients for this Variable
            grad_fn: Function to compute gradients (None for leaf nodes)
            
        TODO: Implement Variable initialization with gradient tracking.
        
        APPROACH:
        1. Convert data to Tensor if it's not already
        2. Store the tensor data
        3. Set gradient tracking flag
        4. Initialize gradient to None (will be computed later)
        5. Store the gradient function for backward pass
        6. Track if this is a leaf node (no grad_fn)
        
        EXAMPLE:
        Variable(5.0) → Variable wrapping Tensor(5.0)
        Variable([1, 2, 3]) → Variable wrapping Tensor([1, 2, 3])
        
        HINTS:
        - Use isinstance() to check if data is already a Tensor
        - Store requires_grad, grad_fn, and is_leaf flags
        - Initialize self.grad to None
        - A leaf node has grad_fn=None
        """
        ### BEGIN SOLUTION
        # Convert data to Tensor if needed
        if isinstance(data, Tensor):
            self.data = data
        else:
            self.data = Tensor(data)
        
        # Set gradient tracking
        self.requires_grad = requires_grad
        self.grad = None  # Will be initialized when needed
        self.grad_fn = grad_fn
        self.is_leaf = grad_fn is None
        
        # For computational graph
        self._backward_hooks = []
        ### END SOLUTION
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the underlying tensor."""
        return self.data.shape
    
    @property
    def size(self) -> int:
        """Get the total number of elements."""
        return self.data.size
    
    def __repr__(self) -> str:
        """String representation of the Variable."""
        grad_str = f", grad_fn={self.grad_fn.__name__}" if self.grad_fn else ""
        return f"Variable({self.data.data.tolist()}, requires_grad={self.requires_grad}{grad_str})"
    
    def backward(self, gradient: Optional['Variable'] = None) -> None:
        """
        Compute gradients using backpropagation.
        
        Args:
            gradient: The gradient to backpropagate (defaults to ones)
            
        TODO: Implement backward propagation.
        
        APPROACH:
        1. If gradient is None, create a gradient of ones with same shape
        2. If this Variable doesn't require gradients, return early
        3. If this is a leaf node, accumulate the gradient
        4. If this has a grad_fn, call it to propagate gradients
        
        EXAMPLE:
        x = Variable(5.0)
        y = x * 2
        y.backward()  # Computes x.grad = 2.0
        
        HINTS:
        - Use np.ones_like() to create default gradient
        - Accumulate gradients with += for leaf nodes
        - Call self.grad_fn(gradient) for non-leaf nodes
        """
        ### BEGIN SOLUTION
        # Default gradient is ones
        if gradient is None:
            gradient = Variable(np.ones_like(self.data.data))
        
        # Skip if gradients not required
        if not self.requires_grad:
            return
        
        # Accumulate gradient for leaf nodes
        if self.is_leaf:
            if self.grad is None:
                self.grad = Variable(np.zeros_like(self.data.data))
            self.grad.data._data += gradient.data.data
        else:
            # Propagate gradients through grad_fn
            if self.grad_fn is not None:
                self.grad_fn(gradient)
        ### END SOLUTION
    
    def zero_grad(self) -> None:
        """Zero out the gradient."""
        if self.grad is not None:
            self.grad.data._data.fill(0)
    
    # Arithmetic operations with gradient tracking
    def __add__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Addition with gradient tracking."""
        return add(self, other)
    
    def __mul__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Multiplication with gradient tracking."""
        return multiply(self, other)
    
    def __sub__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Subtraction with gradient tracking."""
        return subtract(self, other)
    
    def __truediv__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Division with gradient tracking."""
        return divide(self, other) 

# %% [markdown]
"""
## Step 3: Basic Operations with Gradients

### The Pattern
Every differentiable operation follows the same pattern:
1. **Forward pass**: Compute the result
2. **Create grad_fn**: Function that knows how to compute gradients
3. **Return Variable**: With the result and grad_fn

### Mathematical Rules
- **Addition**: `d(x + y)/dx = 1, d(x + y)/dy = 1`
- **Multiplication**: `d(x * y)/dx = y, d(x * y)/dy = x`
- **Subtraction**: `d(x - y)/dx = 1, d(x - y)/dy = -1`
- **Division**: `d(x / y)/dx = 1/y, d(x / y)/dy = -x/y²`

### Implementation Strategy
Each operation creates a closure that captures the input variables and implements the gradient computation rule.
"""

# %% nbgrader={"grade": false, "grade_id": "add-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def add(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Addition operation with gradient tracking.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Variable with sum and gradient function
        
    TODO: Implement addition with gradient computation.
    
    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass: result = a + b
    3. Create gradient function that distributes gradients
    4. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = x + y, then dz/dx = 1, dz/dy = 1
    
    EXAMPLE:
    x = Variable(2.0), y = Variable(3.0)
    z = add(x, y)  # z.data = 5.0
    z.backward()   # x.grad = 1.0, y.grad = 1.0
    
    HINTS:
    - Use isinstance() to check if inputs are Variables
    - Create a closure that captures a and b
    - In grad_fn, call a.backward() and b.backward() with appropriate gradients
    """
    ### BEGIN SOLUTION
    # Convert to Variables if needed
    if not isinstance(a, Variable):
        a = Variable(a, requires_grad=False)
    if not isinstance(b, Variable):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data + b.data
    
    # Create gradient function
    def grad_fn(grad_output):
        # Addition distributes gradients equally
        if a.requires_grad:
            a.backward(grad_output)
        if b.requires_grad:
            b.backward(grad_output)
    
    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad
    
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "multiply-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def multiply(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Multiplication operation with gradient tracking.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Variable with product and gradient function
        
    TODO: Implement multiplication with gradient computation.
    
    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass: result = a * b
    3. Create gradient function using product rule
    4. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = x * y, then dz/dx = y, dz/dy = x
    
    EXAMPLE:
    x = Variable(2.0), y = Variable(3.0)
    z = multiply(x, y)  # z.data = 6.0
    z.backward()        # x.grad = 3.0, y.grad = 2.0
    
    HINTS:
    - Store a.data and b.data for gradient computation
    - In grad_fn, multiply incoming gradient by the other operand
    - Handle broadcasting if shapes are different
    """
    ### BEGIN SOLUTION
    # Convert to Variables if needed
    if not isinstance(a, Variable):
        a = Variable(a, requires_grad=False)
    if not isinstance(b, Variable):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data * b.data
    
    # Create gradient function
    def grad_fn(grad_output):
        # Product rule: d(xy)/dx = y, d(xy)/dy = x
        if a.requires_grad:
            a_grad = Variable(grad_output.data * b.data)
            a.backward(a_grad)
        if b.requires_grad:
            b_grad = Variable(grad_output.data * a.data)
            b.backward(b_grad)
    
    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad
    
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "subtract-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def subtract(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Subtraction operation with gradient tracking.
    
    Args:
        a: First operand (minuend)
        b: Second operand (subtrahend)
        
    Returns:
        Variable with difference and gradient function
        
    TODO: Implement subtraction with gradient computation.
    
    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass: result = a - b
    3. Create gradient function with correct signs
    4. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = x - y, then dz/dx = 1, dz/dy = -1
    
    EXAMPLE:
    x = Variable(5.0), y = Variable(3.0)
    z = subtract(x, y)  # z.data = 2.0
    z.backward()        # x.grad = 1.0, y.grad = -1.0
    
    HINTS:
    - Forward pass is straightforward: a - b
    - Gradient for a is positive, for b is negative
    - Remember to negate the gradient for b
    """
    ### BEGIN SOLUTION
    # Convert to Variables if needed
    if not isinstance(a, Variable):
        a = Variable(a, requires_grad=False)
    if not isinstance(b, Variable):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data - b.data
    
    # Create gradient function
    def grad_fn(grad_output):
        # Subtraction rule: d(x-y)/dx = 1, d(x-y)/dy = -1
        if a.requires_grad:
            a.backward(grad_output)
        if b.requires_grad:
            b_grad = Variable(-grad_output.data.data)
            b.backward(b_grad)
    
    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad
    
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "divide-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def divide(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Division operation with gradient tracking.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Variable with quotient and gradient function
        
    TODO: Implement division with gradient computation.
    
    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass: result = a / b
    3. Create gradient function using quotient rule
    4. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = x / y, then dz/dx = 1/y, dz/dy = -x/y²
    
    EXAMPLE:
    x = Variable(6.0), y = Variable(2.0)
    z = divide(x, y)  # z.data = 3.0
    z.backward()      # x.grad = 0.5, y.grad = -1.5
    
    HINTS:
    - Forward pass: a.data / b.data
    - Gradient for a: grad_output / b.data
    - Gradient for b: -grad_output * a.data / (b.data ** 2)
    - Be careful with numerical stability
    """
    ### BEGIN SOLUTION
    # Convert to Variables if needed
    if not isinstance(a, Variable):
        a = Variable(a, requires_grad=False)
    if not isinstance(b, Variable):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data / b.data
    
    # Create gradient function
    def grad_fn(grad_output):
        # Quotient rule: d(x/y)/dx = 1/y, d(x/y)/dy = -x/y²
        if a.requires_grad:
            a_grad = Variable(grad_output.data.data / b.data.data)
            a.backward(a_grad)
        if b.requires_grad:
            b_grad = Variable(-grad_output.data.data * a.data.data / (b.data.data ** 2))
            b.backward(b_grad)
    
    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad
    
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% [markdown]
"""
## Step 4: Testing Basic Operations

Let's test our basic operations to ensure they compute gradients correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-basic-operations", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_basic_operations():
    """Test basic operations with gradient computation."""
    print("🔬 Testing basic operations...")
    
    # Test addition
    print("📊 Testing addition...")
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = add(x, y)
    
    assert abs(z.data.data.item() - 5.0) < 1e-6, f"Addition failed: expected 5.0, got {z.data.data.item()}"
    
    z.backward()
    assert abs(x.grad.data.data.item() - 1.0) < 1e-6, f"Addition gradient for x failed: expected 1.0, got {x.grad.data.data.item()}"
    assert abs(y.grad.data.data.item() - 1.0) < 1e-6, f"Addition gradient for y failed: expected 1.0, got {y.grad.data.data.item()}"
    print("✅ Addition test passed!")
    
    # Test multiplication
    print("📊 Testing multiplication...")
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = multiply(x, y)
    
    assert abs(z.data.data.item() - 6.0) < 1e-6, f"Multiplication failed: expected 6.0, got {z.data.data.item()}"
    
    z.backward()
    assert abs(x.grad.data.data.item() - 3.0) < 1e-6, f"Multiplication gradient for x failed: expected 3.0, got {x.grad.data.data.item()}"
    assert abs(y.grad.data.data.item() - 2.0) < 1e-6, f"Multiplication gradient for y failed: expected 2.0, got {y.grad.data.data.item()}"
    print("✅ Multiplication test passed!")
    
    # Test subtraction
    print("📊 Testing subtraction...")
    x = Variable(5.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = subtract(x, y)
    
    assert abs(z.data.data.item() - 2.0) < 1e-6, f"Subtraction failed: expected 2.0, got {z.data.data.item()}"
    
    z.backward()
    assert abs(x.grad.data.data.item() - 1.0) < 1e-6, f"Subtraction gradient for x failed: expected 1.0, got {x.grad.data.data.item()}"
    assert abs(y.grad.data.data.item() - (-1.0)) < 1e-6, f"Subtraction gradient for y failed: expected -1.0, got {y.grad.data.data.item()}"
    print("✅ Subtraction test passed!")
    
    # Test division
    print("📊 Testing division...")
    x = Variable(6.0, requires_grad=True)
    y = Variable(2.0, requires_grad=True)
    z = divide(x, y)
    
    assert abs(z.data.data.item() - 3.0) < 1e-6, f"Division failed: expected 3.0, got {z.data.data.item()}"
    
    z.backward()
    assert abs(x.grad.data.data.item() - 0.5) < 1e-6, f"Division gradient for x failed: expected 0.5, got {x.grad.data.data.item()}"
    assert abs(y.grad.data.data.item() - (-1.5)) < 1e-6, f"Division gradient for y failed: expected -1.5, got {y.grad.data.data.item()}"
    print("✅ Division test passed!")
    
    print("🎉 All basic operation tests passed!")
    return True

# Run the test
success = test_basic_operations()

# %% [markdown]
"""
## Step 5: Chain Rule Testing

Let's test more complex expressions to ensure the chain rule works correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-chain-rule", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_chain_rule():
    """Test chain rule with complex expressions."""
    print("🔬 Testing chain rule...")
    
    # Test: f(x, y) = (x + y) * (x - y) = x² - y²
    print("📊 Testing f(x, y) = (x + y) * (x - y)...")
    x = Variable(3.0, requires_grad=True)
    y = Variable(2.0, requires_grad=True)
    
    # Forward pass
    sum_xy = add(x, y)      # x + y = 5
    diff_xy = subtract(x, y) # x - y = 1
    result = multiply(sum_xy, diff_xy)  # (x + y) * (x - y) = 5
    
    assert abs(result.data.data.item() - 5.0) < 1e-6, f"Chain rule forward failed: expected 5.0, got {result.data.data.item()}"
    
    # Backward pass
    result.backward()
    
    # Analytical gradients: df/dx = 2x = 6, df/dy = -2y = -4
    expected_x_grad = 2 * 3.0  # 6.0
    expected_y_grad = -2 * 2.0  # -4.0
    
    assert abs(x.grad.data.data.item() - expected_x_grad) < 1e-6, f"Chain rule x gradient failed: expected {expected_x_grad}, got {x.grad.data.data.item()}"
    assert abs(y.grad.data.data.item() - expected_y_grad) < 1e-6, f"Chain rule y gradient failed: expected {expected_y_grad}, got {y.grad.data.data.item()}"
    print("✅ Chain rule test passed!")
    
    # Test: f(x) = x * x * x (x³)
    print("📊 Testing f(x) = x³...")
    x = Variable(2.0, requires_grad=True)
    
    # Forward pass
    x_squared = multiply(x, x)      # x²
    x_cubed = multiply(x_squared, x)  # x³
    
    assert abs(x_cubed.data.data.item() - 8.0) < 1e-6, f"x³ forward failed: expected 8.0, got {x_cubed.data.data.item()}"
    
    # Backward pass
    x_cubed.backward()
    
    # Analytical gradient: df/dx = 3x² = 12
    expected_grad = 3 * (2.0 ** 2)  # 12.0
    
    assert abs(x.grad.data.data.item() - expected_grad) < 1e-6, f"x³ gradient failed: expected {expected_grad}, got {x.grad.data.data.item()}"
    print("✅ x³ test passed!")
    
    print("🎉 All chain rule tests passed!")
    return True

# Run the test
success = test_chain_rule()

# %% [markdown]
"""
## Step 6: Activation Function Gradients

Now let's implement gradients for activation functions to integrate with our existing modules.
"""

# %% nbgrader={"grade": false, "grade_id": "relu-gradient", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def relu_with_grad(x: Variable) -> Variable:
    """
    ReLU activation with gradient tracking.
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with ReLU applied and gradient function
        
    TODO: Implement ReLU with gradient computation.
    
    APPROACH:
    1. Compute forward pass: max(0, x)
    2. Create gradient function using ReLU derivative
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    f(x) = max(0, x)
    f'(x) = 1 if x > 0, else 0
    
    EXAMPLE:
    x = Variable([-1.0, 0.0, 1.0])
    y = relu_with_grad(x)  # y.data = [0.0, 0.0, 1.0]
    y.backward()           # x.grad = [0.0, 0.0, 1.0]
    
    HINTS:
    - Use np.maximum(0, x.data.data) for forward pass
    - Use (x.data.data > 0) for gradient mask
    - Only propagate gradients where input was positive
    """
    ### BEGIN SOLUTION
    # Forward pass
    result_data = Tensor(np.maximum(0, x.data.data))
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # ReLU derivative: 1 if x > 0, else 0
            mask = (x.data.data > 0).astype(np.float32)
            x_grad = Variable(grad_output.data.data * mask)
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "sigmoid-gradient", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def sigmoid_with_grad(x: Variable) -> Variable:
    """
    Sigmoid activation with gradient tracking.
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with sigmoid applied and gradient function
        
    TODO: Implement sigmoid with gradient computation.
    
    APPROACH:
    1. Compute forward pass: 1 / (1 + exp(-x))
    2. Create gradient function using sigmoid derivative
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    
    EXAMPLE:
    x = Variable(0.0)
    y = sigmoid_with_grad(x)  # y.data = 0.5
    y.backward()              # x.grad = 0.25
    
    HINTS:
    - Use np.clip for numerical stability
    - Store sigmoid output for gradient computation
    - Gradient is sigmoid * (1 - sigmoid)
    """
    ### BEGIN SOLUTION
    # Forward pass with numerical stability
    clipped = np.clip(x.data.data, -500, 500)
    sigmoid_output = 1.0 / (1.0 + np.exp(-clipped))
    result_data = Tensor(sigmoid_output)
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # Sigmoid derivative: sigmoid * (1 - sigmoid)
            sigmoid_grad = sigmoid_output * (1.0 - sigmoid_output)
            x_grad = Variable(grad_output.data.data * sigmoid_grad)
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% [markdown]
"""
## Step 7: Integration Testing

Let's test our autograd system with a simple neural network scenario.
"""

# %% nbgrader={"grade": true, "grade_id": "test-integration", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_integration():
    """Test autograd integration with neural network scenario."""
    print("🔬 Testing autograd integration...")
    
    # Simple neural network: input -> linear -> ReLU -> output
    print("📊 Testing simple neural network...")
    
    # Input
    x = Variable(2.0, requires_grad=True)
    
    # Weights and bias
    w1 = Variable(0.5, requires_grad=True)
    b1 = Variable(0.1, requires_grad=True)
    w2 = Variable(1.5, requires_grad=True)
    
    # Forward pass
    linear1 = add(multiply(x, w1), b1)  # x * w1 + b1 = 2*0.5 + 0.1 = 1.1
    activation1 = relu_with_grad(linear1)  # ReLU(1.1) = 1.1
    output = multiply(activation1, w2)     # 1.1 * 1.5 = 1.65
    
    # Check forward pass
    expected_output = 1.65
    assert abs(output.data.data.item() - expected_output) < 1e-6, f"Integration forward failed: expected {expected_output}, got {output.data.data.item()}"
    
    # Backward pass
    output.backward()
    
    # Check gradients
    # dL/dx = dL/doutput * doutput/dactivation1 * dactivation1/dlinear1 * dlinear1/dx
    #       = 1 * w2 * 1 * w1 = 1.5 * 0.5 = 0.75
    expected_x_grad = 0.75
    assert abs(x.grad.data.data.item() - expected_x_grad) < 1e-6, f"Integration x gradient failed: expected {expected_x_grad}, got {x.grad.data.data.item()}"
    
    # dL/dw1 = dL/doutput * doutput/dactivation1 * dactivation1/dlinear1 * dlinear1/dw1
    #        = 1 * w2 * 1 * x = 1.5 * 2.0 = 3.0
    expected_w1_grad = 3.0
    assert abs(w1.grad.data.data.item() - expected_w1_grad) < 1e-6, f"Integration w1 gradient failed: expected {expected_w1_grad}, got {w1.grad.data.data.item()}"
    
    # dL/db1 = dL/doutput * doutput/dactivation1 * dactivation1/dlinear1 * dlinear1/db1
    #        = 1 * w2 * 1 * 1 = 1.5
    expected_b1_grad = 1.5
    assert abs(b1.grad.data.data.item() - expected_b1_grad) < 1e-6, f"Integration b1 gradient failed: expected {expected_b1_grad}, got {b1.grad.data.data.item()}"
    
    # dL/dw2 = dL/doutput * doutput/dw2 = 1 * activation1 = 1.1
    expected_w2_grad = 1.1
    assert abs(w2.grad.data.data.item() - expected_w2_grad) < 1e-6, f"Integration w2 gradient failed: expected {expected_w2_grad}, got {w2.grad.data.data.item()}"
    
    print("✅ Integration test passed!")
    print("🎉 All autograd tests passed!")
    return True

# Run the test
success = test_integration()

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented automatic differentiation for TinyTorch:

### What You've Accomplished
✅ **Variable Class**: Tensor wrapper with gradient tracking and computational graph  
✅ **Basic Operations**: Addition, multiplication, subtraction, division with gradients  
✅ **Chain Rule**: Automatic gradient computation through complex expressions  
✅ **Activation Functions**: ReLU and Sigmoid with proper gradient computation  
✅ **Integration**: Works seamlessly with neural network scenarios  

### Key Concepts You've Learned
- **Computational graphs** represent mathematical expressions as directed graphs
- **Forward pass** computes function values following the graph
- **Backward pass** computes gradients using the chain rule in reverse
- **Gradient functions** capture how to compute gradients for each operation
- **Variable tracking** enables automatic differentiation of any expression

### Mathematical Foundations
- **Chain rule**: The fundamental principle behind backpropagation
- **Partial derivatives**: How gradients flow through operations
- **Computational efficiency**: Reusing forward pass results in backward pass
- **Numerical stability**: Handling edge cases in gradient computation

### Real-World Applications
- **Neural network training**: Backpropagation through layers
- **Optimization**: Gradient descent and advanced optimizers
- **Scientific computing**: Sensitivity analysis and inverse problems
- **Machine learning**: Any gradient-based learning algorithm

### Next Steps
1. **Export your code**: `tito package nbdev --export 07_autograd`
2. **Test your implementation**: `tito module test 07_autograd`
3. **Use your autograd**: 
   ```python
   from tinytorch.core.autograd import Variable
   
   x = Variable(2.0, requires_grad=True)
   y = x**2 + 3*x + 1
   y.backward()
   print(x.grad)  # Your gradients in action!
   ```
4. **Move to Module 8**: Start building training loops and optimizers!

**Ready for the next challenge?** Let's use your autograd system to build complete training pipelines!
""" 

# %% [markdown]
"""
## Step 8: Performance Optimizations and Advanced Features

### Memory Management
- **Gradient Accumulation**: Efficient in-place gradient updates
- **Computational Graph Cleanup**: Release intermediate values when possible
- **Lazy Evaluation**: Compute gradients only when needed

### Numerical Stability
- **Gradient Clipping**: Prevent exploding gradients
- **Numerical Precision**: Handle edge cases gracefully
- **Overflow Protection**: Clip extreme values

### Advanced Features
- **Higher-Order Gradients**: Gradients of gradients
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **Custom Operations**: Framework for user-defined differentiable functions
"""

# %% nbgrader={"grade": false, "grade_id": "advanced-features", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def power(base: Variable, exponent: Union[float, int]) -> Variable:
    """
    Power operation with gradient tracking: base^exponent.
    
    Args:
        base: Base Variable
        exponent: Exponent (scalar)
        
    Returns:
        Variable with power applied and gradient function
        
    TODO: Implement power operation with gradient computation.
    
    APPROACH:
    1. Compute forward pass: base^exponent
    2. Create gradient function using power rule
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = x^n, then dz/dx = n * x^(n-1)
    
    EXAMPLE:
    x = Variable(2.0)
    y = power(x, 3)  # y.data = 8.0
    y.backward()     # x.grad = 3 * 2^2 = 12.0
    
    HINTS:
    - Use np.power() for forward pass
    - Power rule: gradient = exponent * base^(exponent-1)
    - Handle edge cases like exponent=0 or base=0
    """
    ### BEGIN SOLUTION
    # Forward pass
    result_data = Tensor(np.power(base.data.data, exponent))
    
    # Create gradient function
    def grad_fn(grad_output):
        if base.requires_grad:
            # Power rule: d(x^n)/dx = n * x^(n-1)
            if exponent == 0:
                # Special case: derivative of constant is 0
                base_grad = Variable(np.zeros_like(base.data.data))
            else:
                base_grad_data = exponent * np.power(base.data.data, exponent - 1)
                base_grad = Variable(grad_output.data.data * base_grad_data)
            base.backward(base_grad)
    
    return Variable(result_data, requires_grad=base.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "exp-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def exp(x: Variable) -> Variable:
    """
    Exponential operation with gradient tracking: e^x.
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with exponential applied and gradient function
        
    TODO: Implement exponential operation with gradient computation.
    
    APPROACH:
    1. Compute forward pass: e^x
    2. Create gradient function using exponential derivative
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = e^x, then dz/dx = e^x
    
    EXAMPLE:
    x = Variable(1.0)
    y = exp(x)  # y.data = e^1 ≈ 2.718
    y.backward()  # x.grad = e^1 ≈ 2.718
    
    HINTS:
    - Use np.exp() for forward pass
    - Exponential derivative is itself: d(e^x)/dx = e^x
    - Store result for gradient computation
    """
    ### BEGIN SOLUTION
    # Forward pass
    exp_result = np.exp(x.data.data)
    result_data = Tensor(exp_result)
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # Exponential derivative: d(e^x)/dx = e^x
            x_grad = Variable(grad_output.data.data * exp_result)
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "log-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def log(x: Variable) -> Variable:
    """
    Natural logarithm operation with gradient tracking: ln(x).
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with logarithm applied and gradient function
        
    TODO: Implement logarithm operation with gradient computation.
    
    APPROACH:
    1. Compute forward pass: ln(x)
    2. Create gradient function using logarithm derivative
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = ln(x), then dz/dx = 1/x
    
    EXAMPLE:
    x = Variable(2.0)
    y = log(x)  # y.data = ln(2) ≈ 0.693
    y.backward()  # x.grad = 1/2 = 0.5
    
    HINTS:
    - Use np.log() for forward pass
    - Logarithm derivative: d(ln(x))/dx = 1/x
    - Handle numerical stability for small x
    """
    ### BEGIN SOLUTION
    # Forward pass with numerical stability
    clipped_x = np.clip(x.data.data, 1e-8, np.inf)  # Avoid log(0)
    result_data = Tensor(np.log(clipped_x))
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # Logarithm derivative: d(ln(x))/dx = 1/x
            x_grad = Variable(grad_output.data.data / clipped_x)
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "sum-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def sum_all(x: Variable) -> Variable:
    """
    Sum all elements operation with gradient tracking.
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with sum and gradient function
        
    TODO: Implement sum operation with gradient computation.
    
    APPROACH:
    1. Compute forward pass: sum of all elements
    2. Create gradient function that broadcasts gradient back
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = sum(x), then dz/dx_i = 1 for all i
    
    EXAMPLE:
    x = Variable([[1, 2], [3, 4]])
    y = sum_all(x)  # y.data = 10
    y.backward()    # x.grad = [[1, 1], [1, 1]]
    
    HINTS:
    - Use np.sum() for forward pass
    - Gradient is ones with same shape as input
    - This is used for loss computation
    """
    ### BEGIN SOLUTION
    # Forward pass
    result_data = Tensor(np.sum(x.data.data))
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # Sum gradient: broadcasts to all elements
            x_grad = Variable(grad_output.data.data * np.ones_like(x.data.data))
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "mean-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def mean(x: Variable) -> Variable:
    """
    Mean operation with gradient tracking.
    
    Args:
        x: Input Variable
        
    Returns:
        Variable with mean and gradient function
        
    TODO: Implement mean operation with gradient computation.
    
    APPROACH:
    1. Compute forward pass: mean of all elements
    2. Create gradient function that distributes gradient evenly
    3. Return Variable with result and grad_fn
    
    MATHEMATICAL RULE:
    If z = mean(x), then dz/dx_i = 1/n for all i (where n is number of elements)
    
    EXAMPLE:
    x = Variable([[1, 2], [3, 4]])
    y = mean(x)  # y.data = 2.5
    y.backward()  # x.grad = [[0.25, 0.25], [0.25, 0.25]]
    
    HINTS:
    - Use np.mean() for forward pass
    - Gradient is 1/n for each element
    - This is commonly used for loss computation
    """
    ### BEGIN SOLUTION
    # Forward pass
    result_data = Tensor(np.mean(x.data.data))
    
    # Create gradient function
    def grad_fn(grad_output):
        if x.requires_grad:
            # Mean gradient: 1/n for each element
            n = x.data.size
            x_grad = Variable(grad_output.data.data * np.ones_like(x.data.data) / n)
            x.backward(x_grad)
    
    return Variable(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% [markdown]
"""
## Step 9: Gradient Utilities and Helper Functions

### Gradient Management
- **Gradient Clipping**: Prevent exploding gradients
- **Gradient Checking**: Verify gradient correctness
- **Parameter Collection**: Gather all parameters for optimization

### Debugging Tools
- **Gradient Visualization**: Inspect gradient flow
- **Computational Graph**: Visualize the computation graph
- **Gradient Statistics**: Monitor gradient magnitudes
"""

# %% nbgrader={"grade": false, "grade_id": "gradient-utilities", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def clip_gradients(variables: List[Variable], max_norm: float = 1.0) -> None:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        variables: List of Variables to clip gradients for
        max_norm: Maximum gradient norm allowed
        
    TODO: Implement gradient clipping.
    
    APPROACH:
    1. Compute total gradient norm across all variables
    2. If norm exceeds max_norm, scale all gradients down
    3. Modify gradients in-place
    
    MATHEMATICAL RULE:
    If ||g|| > max_norm, then g := g * (max_norm / ||g||)
    
    EXAMPLE:
    variables = [w1, w2, b1, b2]
    clip_gradients(variables, max_norm=1.0)
    
    HINTS:
    - Compute L2 norm of all gradients combined
    - Scale factor = max_norm / total_norm
    - Only clip if total_norm > max_norm
    """
    ### BEGIN SOLUTION
    # Compute total gradient norm
    total_norm = 0.0
    for var in variables:
        if var.grad is not None:
            total_norm += np.sum(var.grad.data.data ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        scale_factor = max_norm / total_norm
        for var in variables:
            if var.grad is not None:
                var.grad.data._data *= scale_factor
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "collect-parameters", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def collect_parameters(*modules) -> List[Variable]:
    """
    Collect all parameters from modules for optimization.
    
    Args:
        *modules: Variable number of modules/objects with parameters
        
    Returns:
        List of all Variables that require gradients
        
    TODO: Implement parameter collection.
    
    APPROACH:
    1. Iterate through all provided modules
    2. Find all Variable attributes that require gradients
    3. Return list of all such Variables
    
    EXAMPLE:
    layer1 = SomeLayer()
    layer2 = SomeLayer()
    params = collect_parameters(layer1, layer2)
    
    HINTS:
    - Use hasattr() and getattr() to find Variable attributes
    - Check if attribute is Variable and requires_grad
    - Handle different module types gracefully
    """
    ### BEGIN SOLUTION
    parameters = []
    for module in modules:
        if hasattr(module, '__dict__'):
            for attr_name, attr_value in module.__dict__.items():
                if isinstance(attr_value, Variable) and attr_value.requires_grad:
                    parameters.append(attr_value)
    return parameters
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "zero-gradients", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def zero_gradients(variables: List[Variable]) -> None:
    """
    Zero out gradients for all variables.
    
    Args:
        variables: List of Variables to zero gradients for
        
    TODO: Implement gradient zeroing.
    
    APPROACH:
    1. Iterate through all variables
    2. Call zero_grad() on each variable
    3. Handle None gradients gracefully
    
    EXAMPLE:
    parameters = [w1, w2, b1, b2]
    zero_gradients(parameters)
    
    HINTS:
    - Use the zero_grad() method on each Variable
    - Check if variable has gradients before zeroing
    - This is typically called before each training step
    """
    ### BEGIN SOLUTION
    for var in variables:
        if var.grad is not None:
            var.zero_grad()
    ### END SOLUTION

# %% [markdown]
"""
## Step 10: Advanced Testing

Let's test our advanced features and optimizations.
"""

# %% nbgrader={"grade": true, "grade_id": "test-advanced-operations", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_advanced_operations():
    """Test advanced mathematical operations."""
    print("🔬 Testing advanced operations...")
    
    # Test power operation
    print("📊 Testing power operation...")
    x = Variable(2.0, requires_grad=True)
    y = power(x, 3)  # x^3
    
    assert abs(y.data.data.item() - 8.0) < 1e-6, f"Power forward failed: expected 8.0, got {y.data.data.item()}"
    
    y.backward()
    # Gradient: d(x^3)/dx = 3x^2 = 3 * 4 = 12
    assert abs(x.grad.data.data.item() - 12.0) < 1e-6, f"Power gradient failed: expected 12.0, got {x.grad.data.data.item()}"
    print("✅ Power operation test passed!")
    
    # Test exponential operation
    print("📊 Testing exponential operation...")
    x = Variable(1.0, requires_grad=True)
    y = exp(x)  # e^x
    
    expected_exp = np.exp(1.0)
    assert abs(y.data.data.item() - expected_exp) < 1e-6, f"Exp forward failed: expected {expected_exp}, got {y.data.data.item()}"
    
    y.backward()
    # Gradient: d(e^x)/dx = e^x
    assert abs(x.grad.data.data.item() - expected_exp) < 1e-6, f"Exp gradient failed: expected {expected_exp}, got {x.grad.data.data.item()}"
    print("✅ Exponential operation test passed!")
    
    # Test logarithm operation
    print("📊 Testing logarithm operation...")
    x = Variable(2.0, requires_grad=True)
    y = log(x)  # ln(x)
    
    expected_log = np.log(2.0)
    assert abs(y.data.data.item() - expected_log) < 1e-6, f"Log forward failed: expected {expected_log}, got {y.data.data.item()}"
    
    y.backward()
    # Gradient: d(ln(x))/dx = 1/x = 1/2 = 0.5
    assert abs(x.grad.data.data.item() - 0.5) < 1e-6, f"Log gradient failed: expected 0.5, got {x.grad.data.data.item()}"
    print("✅ Logarithm operation test passed!")
    
    # Test sum operation
    print("📊 Testing sum operation...")
    x = Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = sum_all(x)  # sum of all elements
    
    assert abs(y.data.data.item() - 10.0) < 1e-6, f"Sum forward failed: expected 10.0, got {y.data.data.item()}"
    
    y.backward()
    # Gradient: all elements should be 1
    expected_grad = np.ones((2, 2))
    np.testing.assert_array_almost_equal(x.grad.data.data, expected_grad)
    print("✅ Sum operation test passed!")
    
    # Test mean operation
    print("📊 Testing mean operation...")
    x = Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = mean(x)  # mean of all elements
    
    assert abs(y.data.data.item() - 2.5) < 1e-6, f"Mean forward failed: expected 2.5, got {y.data.data.item()}"
    
    y.backward()
    # Gradient: all elements should be 1/4 = 0.25
    expected_grad = np.ones((2, 2)) * 0.25
    np.testing.assert_array_almost_equal(x.grad.data.data, expected_grad)
    print("✅ Mean operation test passed!")
    
    print("🎉 All advanced operation tests passed!")
    return True

# Run the test
success = test_advanced_operations()

# %% nbgrader={"grade": true, "grade_id": "test-gradient-utilities", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_gradient_utilities():
    """Test gradient utility functions."""
    print("🔬 Testing gradient utilities...")
    
    # Test gradient clipping
    print("📊 Testing gradient clipping...")
    x = Variable(1.0, requires_grad=True)
    y = Variable(1.0, requires_grad=True)
    
    # Create large gradients
    z = multiply(x, 10.0)  # Large gradient for x
    w = multiply(y, 10.0)  # Large gradient for y
    loss = add(z, w)
    loss.backward()
    
    # Check gradients are large before clipping
    assert abs(x.grad.data.data.item() - 10.0) < 1e-6
    assert abs(y.grad.data.data.item() - 10.0) < 1e-6
    
    # Clip gradients
    clip_gradients([x, y], max_norm=1.0)
    
    # Check gradients are clipped
    total_norm = np.sqrt(x.grad.data.data.item()**2 + y.grad.data.data.item()**2)
    assert abs(total_norm - 1.0) < 1e-6, f"Gradient clipping failed: total norm {total_norm}, expected 1.0"
    print("✅ Gradient clipping test passed!")
    
    # Test zero gradients
    print("📊 Testing zero gradients...")
    # Gradients should be non-zero before zeroing
    assert abs(x.grad.data.data.item()) > 1e-6
    assert abs(y.grad.data.data.item()) > 1e-6
    
    # Zero gradients
    zero_gradients([x, y])
    
    # Check gradients are zero
    assert abs(x.grad.data.data.item()) < 1e-6
    assert abs(y.grad.data.data.item()) < 1e-6
    print("✅ Zero gradients test passed!")
    
    print("🎉 All gradient utility tests passed!")
    return True

# Run the test
success = test_gradient_utilities()

# %% [markdown]
"""
## Step 11: Complete ML Pipeline Example

Let's demonstrate a complete machine learning pipeline using our autograd system.
"""

# %% nbgrader={"grade": true, "grade_id": "test-complete-pipeline", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_complete_ml_pipeline():
    """Test complete ML pipeline with autograd."""
    print("🔬 Testing complete ML pipeline...")
    
    # Create a simple regression problem: y = 2x + 1 + noise
    print("📊 Setting up regression problem...")
    
    # Training data
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_data = [3.1, 4.9, 7.2, 9.1, 10.8]  # Approximately 2x + 1 with noise
    
    # Model parameters
    w = Variable(0.1, requires_grad=True)  # Weight
    b = Variable(0.0, requires_grad=True)  # Bias
    
    # Training loop
    learning_rate = 0.01
    num_epochs = 100
    
    print("📊 Training model...")
    for epoch in range(num_epochs):
        total_loss = Variable(0.0, requires_grad=False)
        
        # Forward pass for all data points
        for x_val, y_val in zip(x_data, y_data):
            x = Variable(x_val, requires_grad=False)
            y_target = Variable(y_val, requires_grad=False)
            
            # Prediction: y_pred = w * x + b
            y_pred = add(multiply(w, x), b)
            
            # Loss: MSE = (y_pred - y_target)^2
            diff = subtract(y_pred, y_target)
            loss = multiply(diff, diff)
            
            # Accumulate loss
            total_loss = add(total_loss, loss)
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        w.data._data -= learning_rate * w.grad.data.data
        b.data._data -= learning_rate * b.grad.data.data
        
        # Zero gradients for next iteration
        zero_gradients([w, b])
        
        # Print progress
        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: Loss = {total_loss.data.data.item():.4f}, w = {w.data.data.item():.4f}, b = {b.data.data.item():.4f}")
    
    # Check final parameters
    print("📊 Checking final parameters...")
    final_w = w.data.data.item()
    final_b = b.data.data.item()
    
    # Should be close to true values: w=2, b=1
    assert abs(final_w - 2.0) < 0.5, f"Weight not learned correctly: expected ~2.0, got {final_w}"
    assert abs(final_b - 1.0) < 0.5, f"Bias not learned correctly: expected ~1.0, got {final_b}"
    
    print(f"✅ Model learned: w = {final_w:.3f}, b = {final_b:.3f}")
    print("✅ Complete ML pipeline test passed!")
    
    # Test prediction on new data
    print("📊 Testing prediction on new data...")
    x_test = Variable(6.0, requires_grad=False)
    y_pred = add(multiply(w, x_test), b)
    expected_pred = 2.0 * 6.0 + 1.0  # True function value
    
    print(f"   Prediction for x=6: {y_pred.data.data.item():.3f} (expected ~{expected_pred})")
    assert abs(y_pred.data.data.item() - expected_pred) < 1.0, "Prediction accuracy insufficient"
    
    print("🎉 Complete ML pipeline test passed!")
    return True

# Run the test
success = test_complete_ml_pipeline() 
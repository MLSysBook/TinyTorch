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
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

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
## What is Automatic Differentiation?

### The Problem: Computing Gradients at Scale
Neural networks have millions of parameters. To train them, we need gradients of the loss function with respect to every parameter:

```
∇θ L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ, ∂L/∂b₁, ∂L/∂b₂, ..., ∂L/∂bₘ]
```

**Manual differentiation fails** because:
- Networks have thousands of composed functions
- Manual computation is extremely error-prone
- Every architecture change requires re-deriving all gradients

### The Solution: Automatic Differentiation
**Autograd** automatically computes derivatives of functions represented as computational graphs:

```python
# Instead of manually computing: ∂(x² + 2xy + y²)/∂x = 2x + 2y
# Autograd does it automatically:
x = Variable(3.0, requires_grad=True)
y = Variable(4.0, requires_grad=True)
z = x**2 + 2*x*y + y**2
z.backward()
print(x.grad)  # 2*3 + 2*4 = 14 (computed automatically!)
```

### Why This is Revolutionary
- **Efficiency**: O(1) overhead per operation
- **Flexibility**: Works with any differentiable function
- **Correctness**: Implements chain rule precisely
- **Scale**: Handles millions of parameters automatically

### Real-World Impact
- **PyTorch**: `torch.autograd` enables all neural network training
- **TensorFlow**: `tf.GradientTape` provides similar functionality
- **JAX**: `jax.grad` for high-performance computing
- **Deep Learning**: Made training complex models practical

Let's build the engine that powers modern AI!
"""

# %% [markdown]
"""
## Step 1: The Variable Class - Gradient Tracking

### What is a Variable?
A **Variable** wraps a Tensor and tracks:
- **Data**: The actual values (forward pass)
- **Gradient**: The computed gradients (backward pass)
- **Computation history**: How this Variable was created
- **Backward function**: How to compute gradients

### The Computational Graph
Variables automatically build a computational graph:

```python
x = Variable(2.0)  # Leaf node
y = Variable(3.0)  # Leaf node
z = x * y          # Intermediate node: z = x * y
w = z + 1          # Output node: w = z + 1

# Graph: x ──→ * ──→ + ──→ w
#        y ──→   ──→   ──→
```

### Design Principles
- **Transparency**: Works seamlessly with existing operations
- **Efficiency**: Minimal overhead for forward pass
- **Flexibility**: Supports any differentiable operation
- **Correctness**: Implements chain rule precisely

### Real-World Context
This is like:
- **PyTorch**: `torch.autograd.Variable` (now integrated into tensors)
- **TensorFlow**: `tf.Variable` with gradient tracking
- **JAX**: Variables with `jax.grad` transformation
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
            
        TODO: Implement Variable initialization with gradient tracking.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert data to Tensor if it's not already a Tensor
        2. Store the tensor data in self.data
        3. Set gradient tracking flag (requires_grad)
        4. Initialize gradient to None (will be computed during backward pass)
        5. Store the gradient function for backward pass
        6. Track if this is a leaf node (no grad_fn means it's a leaf)
        
        EXAMPLE USAGE:
        ```python
        # Create leaf variables (input data)
        x = Variable(5.0, requires_grad=True)
        y = Variable([1, 2, 3], requires_grad=True)
        
        # Create intermediate variables (results of operations)
        z = x + y  # Has grad_fn for addition
        ```
        
        IMPLEMENTATION HINTS:
        - Use isinstance(data, Tensor) to check type
        - Convert with Tensor(data) if needed
        - Store requires_grad, grad_fn flags
        - Initialize self.grad = None
        - Leaf nodes have grad_fn = None
        - Set self.is_leaf = (grad_fn is None)
        
        LEARNING CONNECTIONS:
        - This is like torch.Tensor with requires_grad=True
        - Forms the basis for all neural network training
        - Each Variable is a node in the computational graph
        - Enables automatic gradient computation
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
        
        TODO: Implement backward pass for gradient computation.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. If gradient is None, create gradient of ones (for scalar outputs)
        2. If this Variable requires gradients, accumulate the gradient
        3. If this Variable has a grad_fn, call it to propagate gradients
        4. The grad_fn will recursively call backward on input Variables
        
        EXAMPLE USAGE:
        ```python
        x = Variable(2.0, requires_grad=True)
        y = Variable(3.0, requires_grad=True)
        z = add(x, y)  # z = 5.0
        z.backward()
        print(x.grad)  # 1.0 (∂z/∂x = 1)
        print(y.grad)  # 1.0 (∂z/∂y = 1)
        ```
        
        IMPLEMENTATION HINTS:
        - If gradient is None: gradient = Variable(np.ones_like(self.data.data))
        - If self.requires_grad: accumulate gradient into self.grad
        - If self.grad_fn: call self.grad_fn(gradient)
        - Handle gradient accumulation (add to existing gradient)
        
        LEARNING CONNECTIONS:
        - This implements the chain rule of calculus
        - Gradients flow backward through the computational graph
        - Each operation contributes its local gradient
        - Enables training of any differentiable function
        """
        ### BEGIN SOLUTION
        if gradient is None:
            gradient = Variable(np.ones_like(self.data.data))
        
        if self.requires_grad:
            if self.grad is None:
                self.grad = gradient
            else:
                # Accumulate gradients
                self.grad = Variable(self.grad.data.data + gradient.data.data)
        
            if self.grad_fn is not None:
                self.grad_fn(gradient)
        ### END SOLUTION
    
    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        self.grad = None
    
    def __add__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Addition operator: self + other"""
        return add(self, other)
    
    def __mul__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Multiplication operator: self * other"""
        return multiply(self, other)
    
    def __sub__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Subtraction operator: self - other"""
        return subtract(self, other)
    
    def __truediv__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Division operator: self / other"""
        return divide(self, other) 

# %% [markdown]
"""
### 🧪 Test Your Variable Class

Once you implement the Variable class above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-variable-class", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_variable_class():
    """Test Variable class implementation"""
    print("🔬 Unit Test: Variable Class...")
    
    # Test Variable creation
    x = Variable(5.0, requires_grad=True)
    assert x.requires_grad == True, "Variable should require gradients"
    assert x.is_leaf == True, "Variable should be a leaf node"
    assert x.grad is None, "Gradient should be None initially"
    
    # Test data access
    assert x.data.data.item() == 5.0, "Data should be accessible"
    assert x.shape == (), "Scalar should have empty shape"
    assert x.size == 1, "Scalar should have size 1"
    
    # Test with list input
    y = Variable([1, 2, 3], requires_grad=True)
    assert y.shape == (3,), "List should create 1D tensor"
    assert y.size == 3, "Size should be 3"
    
    # Test with requires_grad=False
    z = Variable(10.0, requires_grad=False)
    assert z.requires_grad == False, "Should not require gradients"
    
    # Test zero_grad
    x.grad = Variable(1.0)
    x.zero_grad()
    assert x.grad is None, "zero_grad should reset gradient to None"
    
    print("✅ Variable class tests passed!")
    print(f"✅ Variable creation and initialization working")
    print(f"✅ Data access and properties working")
    print(f"✅ Gradient management working")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_variable_class()

# %% [markdown]
"""
## Step 2: Basic Operations with Gradients

### The Chain Rule in Action
Every operation must implement:
1. **Forward pass**: Compute the result
2. **Backward pass**: Compute gradients for inputs

### Example: Addition
For z = x + y:
- **Forward**: z.data = x.data + y.data
- **Backward**: ∂z/∂x = 1, ∂z/∂y = 1

### Mathematical Foundation
The chain rule states:
```
∂f/∂x = ∂f/∂z · ∂z/∂x
```

For complex expressions like f(g(h(x))):
```
∂f/∂x = ∂f/∂g · ∂g/∂h · ∂h/∂x
```

### Implementation Pattern
Each operation returns a new Variable with:
- **Forward result**: Computed value
- **Backward function**: Gradient computation
"""

# %% nbgrader={"grade": false, "grade_id": "add-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def add(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Addition operation with gradient tracking: a + b
    
    TODO: Implement addition with automatic differentiation.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Convert inputs to Variables if they're scalars
    2. Compute forward pass: result = a.data + b.data
    3. Create gradient function that implements: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    4. Return new Variable with result and gradient function
    
    MATHEMATICAL FOUNDATION:
    - Forward: z = x + y
    - Backward: ∂z/∂x = 1, ∂z/∂y = 1
    - Chain rule: ∂L/∂x = ∂L/∂z · ∂z/∂x = ∂L/∂z · 1 = ∂L/∂z
    
    EXAMPLE USAGE:
    ```python
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = add(x, y)  # z = 5.0
    z.backward()
    print(x.grad)  # 1.0 (∂z/∂x = 1)
    print(y.grad)  # 1.0 (∂z/∂y = 1)
    ```
    
    IMPLEMENTATION HINTS:
    - Convert scalars: if isinstance(a, (int, float)): a = Variable(a, requires_grad=False)
    - Forward pass: result_data = a.data + b.data
    - Backward function: def grad_fn(grad_output): if a.requires_grad: a.backward(grad_output)
    - Return: Variable(result_data, grad_fn=grad_fn)
    - Only propagate gradients to Variables that require them
    
    LEARNING CONNECTIONS:
    - This is like torch.add() with autograd
    - Addition distributes gradients equally to both inputs
    - Forms the basis for bias addition in neural networks
    - Chain rule propagates gradients through the graph
    """
    ### BEGIN SOLUTION
    # Convert scalars to Variables
    if isinstance(a, (int, float)):
        a = Variable(a, requires_grad=False)
    if isinstance(b, (int, float)):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data + b.data
    
    # Backward function
    def grad_fn(grad_output):
        # Addition distributes gradients equally
        if a.requires_grad:
            a.backward(grad_output)
        if b.requires_grad:
            b.backward(grad_output)
    
    # Return new Variable with gradient function
    requires_grad = a.requires_grad or b.requires_grad
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Addition Operation

Once you implement the add function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-add-operation", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_add_operation():
    """Test addition operation with gradients"""
    print("🔬 Unit Test: Addition Operation...")
    
    # Test basic addition
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = add(x, y)
    
    assert z.data.data.item() == 5.0, "Addition result should be 5.0"
    assert z.requires_grad == True, "Result should require gradients"
    assert z.is_leaf == False, "Result should not be a leaf node"
    
    # Test backward pass
    z.backward()
    
    assert x.grad is not None, "x should have gradient"
    assert y.grad is not None, "y should have gradient"
    assert x.grad.data.data.item() == 1.0, "∂z/∂x should be 1.0"
    assert y.grad.data.data.item() == 1.0, "∂z/∂y should be 1.0"
    
    # Test with scalar
    a = Variable(5.0, requires_grad=True)
    b = add(a, 3.0)  # Add scalar
    
    assert b.data.data.item() == 8.0, "Addition with scalar should work"
    
    b.backward()
    assert a.grad.data.data.item() == 1.0, "Gradient through scalar addition should be 1.0"
    
    print("✅ Addition operation tests passed!")
    print(f"✅ Forward pass computing correct results")
    print(f"✅ Backward pass computing correct gradients")
    print(f"✅ Scalar addition working correctly")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_add_operation()

# %% [markdown]
"""
## Step 3: Multiplication Operation

### The Product Rule
For z = x * y:
- **Forward**: z = x * y
- **Backward**: ∂z/∂x = y, ∂z/∂y = x

### Why This Matters
Multiplication is everywhere in neural networks:
- **Weight scaling**: w * x in dense layers
- **Attention mechanisms**: attention_weights * values
- **Gating**: gate_signal * hidden_state

### Chain Rule Application
When gradients flow back through multiplication:
```
∂L/∂x = ∂L/∂z · ∂z/∂x = ∂L/∂z · y
∂L/∂y = ∂L/∂z · ∂z/∂y = ∂L/∂z · x
```
"""

# %% nbgrader={"grade": false, "grade_id": "multiply-operation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def multiply(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Multiplication operation with gradient tracking: a * b
    
    TODO: Implement multiplication with automatic differentiation.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Convert inputs to Variables if they're scalars
    2. Compute forward pass: result = a.data * b.data
    3. Create gradient function implementing product rule: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    4. Return new Variable with result and gradient function
    
    MATHEMATICAL FOUNDATION:
    - Forward: z = x * y
    - Backward: ∂z/∂x = y, ∂z/∂y = x
    - Chain rule: ∂L/∂x = ∂L/∂z · y, ∂L/∂y = ∂L/∂z · x
    
    EXAMPLE USAGE:
    ```python
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = multiply(x, y)  # z = 6.0
    z.backward()
    print(x.grad)  # 3.0 (∂z/∂x = y)
    print(y.grad)  # 2.0 (∂z/∂y = x)
    ```
    
    IMPLEMENTATION HINTS:
    - Convert scalars to Variables (same as addition)
    - Forward pass: result_data = a.data * b.data
    - Backward function: multiply incoming gradient by the other variable
    - For a: a.backward(grad_output * b.data)
    - For b: b.backward(grad_output * a.data)
    
    LEARNING CONNECTIONS:
    - This is like torch.mul() with autograd
    - Product rule is fundamental to backpropagation
    - Used in weight updates and attention mechanisms
    - Each input's gradient depends on the other input's value
    """
    ### BEGIN SOLUTION
    # Convert scalars to Variables
    if isinstance(a, (int, float)):
        a = Variable(a, requires_grad=False)
    if isinstance(b, (int, float)):
        b = Variable(b, requires_grad=False)
    
    # Forward pass
    result_data = a.data * b.data
    
    # Backward function
    def grad_fn(grad_output):
        # Product rule: d(xy)/dx = y, d(xy)/dy = x
        if a.requires_grad:
            a.backward(Variable(grad_output.data.data * b.data.data))
        if b.requires_grad:
            b.backward(Variable(grad_output.data.data * a.data.data))
    
    # Return new Variable with gradient function
    requires_grad = a.requires_grad or b.requires_grad
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Multiplication Operation

Once you implement the multiply function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-multiply-operation", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_multiply_operation():
    """Test multiplication operation with gradients"""
    print("🔬 Unit Test: Multiplication Operation...")
    
    # Test basic multiplication
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = multiply(x, y)
    
    assert z.data.data.item() == 6.0, "Multiplication result should be 6.0"
    assert z.requires_grad == True, "Result should require gradients"
    
    # Test backward pass
    z.backward()
    
    assert x.grad is not None, "x should have gradient"
    assert y.grad is not None, "y should have gradient"
    assert x.grad.data.data.item() == 3.0, "∂z/∂x should be y = 3.0"
    assert y.grad.data.data.item() == 2.0, "∂z/∂y should be x = 2.0"
    
    # Test with scalar
    a = Variable(4.0, requires_grad=True)
    b = multiply(a, 2.0)  # Multiply by scalar
    
    assert b.data.data.item() == 8.0, "Multiplication with scalar should work"
    
    b.backward()
    assert a.grad.data.data.item() == 2.0, "Gradient through scalar multiplication should be the scalar"
    
    print("✅ Multiplication operation tests passed!")
    print(f"✅ Forward pass computing correct results")
    print(f"✅ Backward pass implementing product rule correctly")
    print(f"✅ Scalar multiplication working correctly")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_multiply_operation()

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

# %% nbgrader={"grade": false, "grade_id": "test-subtract-operation", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_subtract_operation():
    """Test subtraction operation with gradients"""
    print("🔬 Unit Test: Subtraction Operation...")
    
    # Test basic subtraction
    x = Variable(5.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = subtract(x, y)
    
    assert z.data.data.item() == 2.0, "Subtraction result should be 2.0"
    assert z.requires_grad == True, "Result should require gradients"
    
    # Test backward pass
    z.backward()
    
    assert x.grad is not None, "x should have gradient"
    assert y.grad is not None, "y should have gradient"
    assert x.grad.data.data.item() == 1.0, "∂z/∂x should be 1.0"
    assert y.grad.data.data.item() == -1.0, "∂z/∂y should be -1.0"
    
    # Test with scalar
    a = Variable(4.0, requires_grad=True)
    b = subtract(a, 2.0)  # Subtract scalar
    
    assert b.data.data.item() == 2.0, "Subtraction with scalar should work"
    
    b.backward()
    assert a.grad.data.data.item() == 1.0, "Gradient through scalar subtraction should be 1.0"
    
    print("✅ Subtraction operation tests passed!")
    print(f"✅ Forward pass computing correct results")
    print(f"✅ Backward pass implementing subtraction rule correctly")
    print(f"✅ Scalar subtraction working correctly")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_subtract_operation()

# %% [markdown]
"""
## Step 4: Chain Rule in Complex Expressions

### Building Complex Computations
Now let's test how multiple operations work together through the chain rule:

### Example: f(x, y) = (x + y) * (x - y)
This creates a computational graph:
```
x ──→ + ──→ * ──→ result
y ──→   ──→   ──→
│            ↑
└──→ - ──────┘
```

### Chain Rule Application
- **Forward**: Compute each operation in sequence
- **Backward**: Gradients flow back through each operation
- **Automatic**: No manual gradient computation needed!

### Real-World Significance
Complex neural networks are just larger versions of this:
- **Millions of operations**: Each tracked automatically
- **Complex architectures**: ResNet, Transformer, etc.
- **Efficient computation**: O(1) overhead per operation
"""

# %% nbgrader={"grade": true, "grade_id": "test-chain-rule", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_chain_rule():
    """Test chain rule with complex expressions"""
    print("🔬 Unit Test: Chain Rule with Complex Expressions...")
    
    # Test: f(x, y) = (x + y) * (x - y) = x² - y²
    x = Variable(3.0, requires_grad=True)
    y = Variable(2.0, requires_grad=True)
    
    # Build expression step by step
    sum_xy = add(x, y)      # x + y = 5.0
    diff_xy = subtract(x, y) # x - y = 1.0
    result = multiply(sum_xy, diff_xy)  # (x + y) * (x - y) = 5.0
    
    # Check forward pass
    assert result.data.data.item() == 5.0, "Forward pass should compute 5.0"
    
    # Compute gradients
    result.backward()
    
    # Check gradients: ∂(x²-y²)/∂x = 2x, ∂(x²-y²)/∂y = -2y
    expected_x_grad = 2 * x.data.data.item()  # 2 * 3 = 6
    expected_y_grad = -2 * y.data.data.item()  # -2 * 2 = -4
    
    assert abs(x.grad.data.data.item() - expected_x_grad) < 1e-6, f"x gradient should be {expected_x_grad}"
    assert abs(y.grad.data.data.item() - expected_y_grad) < 1e-6, f"y gradient should be {expected_y_grad}"
    
    # Test more complex expression: f(x) = (x + 1) * (x + 2) * (x + 3)
    x2 = Variable(1.0, requires_grad=True)
    
    term1 = add(x2, 1.0)    # x + 1 = 2.0
    term2 = add(x2, 2.0)    # x + 2 = 3.0
    term3 = add(x2, 3.0)    # x + 3 = 4.0
    
    product1 = multiply(term1, term2)  # (x + 1) * (x + 2) = 6.0
    result2 = multiply(product1, term3)  # * (x + 3) = 24.0
    
    assert result2.data.data.item() == 24.0, "Complex expression should compute 24.0"
    
    result2.backward()
    
    # For f(x) = (x+1)(x+2)(x+3), f'(x) = 3x² + 12x + 11
    # At x=1: f'(1) = 3 + 12 + 11 = 26
    expected_grad = 3 * (1.0**2) + 12 * 1.0 + 11  # 26
    
    assert abs(x2.grad.data.data.item() - expected_grad) < 1e-6, f"Complex gradient should be {expected_grad}"
    
    print("✅ Chain rule tests passed!")
    print(f"✅ Simple expression: (x+y)*(x-y) = x²-y²")
    print(f"✅ Complex expression: (x+1)*(x+2)*(x+3)")
    print(f"✅ Automatic gradient computation working correctly")
    print(f"✅ Chain rule implemented correctly")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_chain_rule()

# %% [markdown]
"""
## Step 5: Integration with Neural Network Training

### The Complete Training Loop
Let's see how autograd enables neural network training:

1. **Forward pass**: Compute predictions
2. **Loss computation**: Compare with targets
3. **Backward pass**: Compute gradients automatically
4. **Parameter update**: Update weights using gradients

### Example: Simple Linear Regression
   ```python
# Model: y = wx + b
w = Variable(0.5, requires_grad=True)
b = Variable(0.1, requires_grad=True)

    # Forward pass
prediction = w * x + b

# Loss: mean squared error
loss = (prediction - target)**2

# Backward pass (automatic!)
loss.backward()

# Update parameters
w.data = w.data - learning_rate * w.grad.data
b.data = b.data - learning_rate * b.grad.data
```

### Why This is Powerful
- **Automatic**: No manual gradient computation
- **Flexible**: Works with any differentiable function
- **Efficient**: Minimal computational overhead
- **Scalable**: Handles millions of parameters
"""

# %% nbgrader={"grade": true, "grade_id": "test-neural-network-training", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_neural_network_training():
    """Test autograd in neural network training scenario"""
    print("🔬 Unit Test: Neural Network Training Integration...")
    
    # Simple linear regression: y = wx + b
    # Training data: y = 2x + 1 + noise
    
    # Initialize parameters
    w = Variable(0.1, requires_grad=True)  # Start with small random value
    b = Variable(0.0, requires_grad=True)  # Start with zero bias
    
    # Training data
    x_data = [1.0, 2.0, 3.0, 4.0]
    y_data = [3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
    
    learning_rate = 0.01
    
    # Training loop
    for epoch in range(100):
        total_loss = Variable(0.0)
        
        for x_val, y_val in zip(x_data, y_data):
            # Create input variable
            x = Variable(x_val, requires_grad=False)
            target = Variable(y_val, requires_grad=False)
            
    # Forward pass
            prediction = add(multiply(w, x), b)  # wx + b
            
            # Loss: squared error
            error = subtract(prediction, target)
            loss = multiply(error, error)  # (pred - target)²
            
            # Accumulate loss
            total_loss = add(total_loss, loss)
        
        # Backward pass
        w.zero_grad()
        b.zero_grad()
        total_loss.backward()
        
        # Update parameters
        if w.grad is not None:
            w.data = Tensor(w.data.data - learning_rate * w.grad.data.data)
        if b.grad is not None:
            b.data = Tensor(b.data.data - learning_rate * b.grad.data.data)
    
    # Check that parameters converged to correct values
    final_w = w.data.data.item()
    final_b = b.data.data.item()
    
    print(f"Final weights: w = {final_w:.3f}, b = {final_b:.3f}")
    print(f"Target weights: w = 2.000, b = 1.000")
    
    # Should be close to w=2, b=1
    assert abs(final_w - 2.0) < 0.1, f"Weight should be close to 2.0, got {final_w}"
    assert abs(final_b - 1.0) < 0.1, f"Bias should be close to 1.0, got {final_b}"
    
    # Test prediction with learned parameters
    test_x = Variable(5.0, requires_grad=False)
    test_prediction = add(multiply(w, test_x), b)
    expected_output = 2.0 * 5.0 + 1.0  # 11.0
    
    prediction_error = abs(test_prediction.data.data.item() - expected_output)
    assert prediction_error < 0.5, f"Prediction error should be small, got {prediction_error}"
    
    print("✅ Neural network training integration tests passed!")
    print(f"✅ Parameters converged to correct values")
    print(f"✅ Model makes accurate predictions")
    print(f"✅ Autograd enables automatic training")
    print(f"✅ Ready for complex neural network architectures!")

# Run inline tests when module is executed directly
if __name__ == "__main__":
    test_neural_network_training()

# %% [markdown]
"""
## 🎯 Module Summary: Automatic Differentiation Mastery!

Congratulations! You've successfully implemented the automatic differentiation engine that powers all modern deep learning:

### ✅ What You've Built
- **Variable Class**: Tensor wrapper with gradient tracking and computational graph construction
- **Automatic Differentiation**: Forward and backward pass implementation
- **Basic Operations**: Addition and multiplication with proper gradient computation
- **Chain Rule**: Automatic gradient flow through complex expressions
- **Training Integration**: Complete neural network training with automatic gradients

### ✅ Key Learning Outcomes
- **Understanding**: How automatic differentiation works through computational graphs
- **Implementation**: Built the gradient engine from scratch
- **Mathematical mastery**: Chain rule, product rule, and gradient computation
- **Real-world application**: Saw how autograd enables neural network training
- **Systems thinking**: Understanding the foundation of modern AI systems

### ✅ Mathematical Foundations Mastered
- **Chain Rule**: ∂f/∂x = ∂f/∂z · ∂z/∂x for composite functions
- **Product Rule**: ∂(xy)/∂x = y, ∂(xy)/∂y = x for multiplication
- **Gradient Accumulation**: Handling multiple paths to the same variable
- **Computational Graphs**: Forward pass builds graph, backward pass computes gradients

### ✅ Professional Skills Developed
- **Systems architecture**: Designed a scalable gradient computation system
- **Memory management**: Efficient gradient storage and computation
- **API design**: Clean interfaces for automatic differentiation
- **Testing methodology**: Comprehensive validation of gradient computation

### ✅ Ready for Advanced Applications
Your autograd engine now enables:
- **Deep Neural Networks**: Automatic gradient computation for any architecture
- **Optimization**: Gradient-based parameter updates
- **Complex Models**: Transformers, ResNets, any differentiable model
- **Research**: Foundation for experimenting with new architectures

### 🔗 Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.autograd` provides identical functionality
- **TensorFlow**: `tf.GradientTape` implements similar concepts
- **JAX**: `jax.grad` for high-performance automatic differentiation
- **Industry Standard**: Every major ML framework uses these exact principles

### 🎯 The Power of Automatic Differentiation
You've unlocked the key technology that made modern AI possible:
- **Scalability**: Handles millions of parameters automatically
- **Flexibility**: Works with any differentiable function
- **Efficiency**: Minimal computational overhead
- **Universality**: Enables training of any neural network architecture

### 🧠 Deep Learning Revolution
You now understand the technology that revolutionized AI:
- **Before autograd**: Manual gradient computation limited model complexity
- **After autograd**: Automatic gradients enabled deep learning revolution
- **Modern AI**: GPT, BERT, ResNet all rely on automatic differentiation
- **Future**: Your understanding enables you to build next-generation AI systems

### 🚀 What's Next
Your autograd engine is the foundation for:
- **Optimizers**: SGD, Adam, and other gradient-based optimizers
- **Training Loops**: Complete neural network training systems
- **Advanced Architectures**: Transformers, GANs, and more complex models
- **Research**: Experimenting with new differentiable algorithms

**Next Module**: Advanced training systems, optimizers, and complete neural network architectures!

You've built the engine that powers modern AI. Now let's use it to train intelligent systems that can learn to solve complex problems!
""" 
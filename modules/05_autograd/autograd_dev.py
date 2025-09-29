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
# Autograd - Automatic Differentiation Engine

Welcome to Autograd! You'll implement the automatic differentiation engine that makes neural network training possible by automatically computing gradients through computational graphs.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): Data structures that hold neural network parameters
- Module 04 (Losses): Functions that measure prediction accuracy

**What's Working**: You can compute loss values for any prediction!

**The Gap**: Loss values tell you HOW WRONG you are, but not HOW TO IMPROVE the parameters.

**This Module's Solution**: Implement automatic differentiation to compute gradients automatically.

**Connection Map**:
```
Tensors → Losses → Autograd → Optimizers
(data)   (error)  (∇L/∇θ)   (updates)
```

## Learning Objectives
1. **Core Implementation**: Variable class with gradient tracking
2. **Mathematical Foundation**: Chain rule application in computational graphs
3. **Testing Skills**: Gradient computation validation
4. **Integration Knowledge**: How autograd enables neural network training

## Build → Test → Use
1. **Build**: Variable class with backward propagation
2. **Test**: Verify gradients are computed correctly
3. **Use**: Apply to mathematical expressions and see automatic differentiation

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/05_autograd/autograd_dev.py
**Building Side:** Code exports to tinytorch.core.autograd

```python
# Final package structure:
from tinytorch.core.autograd import Variable  # This module
from tinytorch.core.tensor import Tensor      # Foundation (always needed)
```

**Why this matters:**
- **Learning:** Complete automatic differentiation system for deep understanding
- **Production:** Proper organization like PyTorch's torch.autograd
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Works seamlessly with tensors for complete training systems
"""

# %%
#| default_exp core.autograd

#| export
import numpy as np
import sys
from typing import Union, List, Optional, Callable

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# %%
print("🔥 TinyTorch Autograd Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build automatic differentiation!")

# %% [markdown]
"""
## What is Automatic Differentiation?

### The Problem: Computing Gradients at Scale

In neural networks, we need to compute gradients of complex functions with millions of parameters:

```
Loss = f(W₁, W₂, ..., Wₙ, data)
∇Loss = [∂Loss/∂W₁, ∂Loss/∂W₂, ..., ∂Loss/∂Wₙ]
```

Manual differentiation is impossible. Numerical differentiation is too slow.

### The Solution: Automatic Differentiation

🧠 **Core Concept**: Track operations as we compute forward pass, then apply chain rule backwards
⚡ **Performance**: Same speed as forward pass, exact gradients (not approximations)
📦 **Framework Compatibility**: This is how PyTorch and TensorFlow work internally

### Visual Representation: Computational Graph

```
Forward Pass:
x ──┐
    ├──[×]──> z = x * y
y ──┘

Backward Pass:
∂L/∂z ──┬──> ∂L/∂x = ∂L/∂z * y
        │
        └──> ∂L/∂y = ∂L/∂z * x
```

**Key Insight**: Each operation stores how to compute gradients with respect to its inputs.
"""

# %% [markdown]
"""
## Implementation: Variable Class - Gradient Tracking

🏗️ **Organization**: Variables wrap tensors and track gradients
🎯 **Clean API**: Seamless integration with existing tensor operations
📐 **Mathematical Foundation**: Computational graph representation of functions

### Design Principles

A Variable tracks:
- **data**: The actual values (using our Tensor)
- **grad**: Accumulated gradients (starts as None)
- **grad_fn**: Function to compute gradients during backward pass
- **requires_grad**: Whether to track gradients for this variable
"""

# %% nbgrader={"grade": false, "grade_id": "variable-class", "solution": true}
#| export
class Variable:
    """
    Variable with automatic differentiation support.

    A Variable wraps a Tensor and tracks operations for gradient computation.

    TODO: Implement Variable class with gradient tracking capabilities

    APPROACH:
    1. Initialize with data, optional gradient requirement
    2. Store grad_fn for backward pass computation
    3. Implement backward() method to compute gradients

    EXAMPLE:
    >>> x = Variable([2.0], requires_grad=True)
    >>> y = Variable([3.0], requires_grad=True)
    >>> z = x * y
    >>> z.backward()
    >>> print(x.grad)  # Should be [3.0]
    >>> print(y.grad)  # Should be [2.0]

    HINTS:
    - Store data as Tensor for consistency
    - grad starts as None, gets created during backward
    - grad_fn is a callable that propagates gradients
    """
    ### BEGIN SOLUTION
    def __init__(self, data, requires_grad=False, grad_fn=None):
        """Initialize Variable with data and gradient tracking."""
        # Convert to Tensor if needed
        if isinstance(data, (list, tuple, int, float)):
            self.data = Tensor(data)
        elif isinstance(data, np.ndarray):
            self.data = Tensor(data)
        elif isinstance(data, Tensor):
            self.data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

    def __repr__(self):
        """String representation of Variable."""
        grad_info = f", grad_fn={self.grad_fn.__name__}" if self.grad_fn else ""
        requires_grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Variable({self.data.data}{grad_info}{requires_grad_info})"

    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation.

        Args:
            gradient: Gradient flowing backwards (defaults to ones)
        """
        # Default gradient for scalar outputs
        if gradient is None:
            if self.data.data.size == 1:
                gradient = np.ones_like(self.data.data)
            else:
                raise RuntimeError("gradient must be specified for non-scalar variables")

        # Accumulate gradients
        if self.requires_grad:
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad = self.grad + gradient

        # Propagate gradients backwards through computation graph
        if self.grad_fn is not None:
            self.grad_fn(gradient)

    # Arithmetic operations with gradient tracking
    def __add__(self, other):
        """Addition with gradient tracking."""
        return add(self, other)

    def __radd__(self, other):
        """Reverse addition."""
        return add(other, self)

    def __mul__(self, other):
        """Multiplication with gradient tracking."""
        return multiply(self, other)

    def __rmul__(self, other):
        """Reverse multiplication."""
        return multiply(other, self)

    def __sub__(self, other):
        """Subtraction with gradient tracking."""
        return subtract(self, other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        return subtract(other, self)

    def __matmul__(self, other):
        """Matrix multiplication with gradient tracking."""
        return matmul(self, other)

    @staticmethod
    def sum(variable):
        """
        Sum all elements of a Variable, maintaining gradient tracking.

        This is essential for creating scalar losses from multi-element results.
        Unlike extracting scalar values, this preserves the computational graph.

        Args:
            variable: Variable to sum

        Returns:
            Variable containing the sum with gradient tracking
        """
        # Forward pass: compute sum
        sum_data = np.sum(variable.data.data)

        # Determine if result requires gradients
        requires_grad = variable.requires_grad

        # Define backward function for gradient propagation
        def grad_fn(gradient):
            """Propagate gradients back to all elements."""
            if variable.requires_grad:
                # For sum operation, gradient is broadcast to all elements
                # Since d(sum)/d(xi) = 1 for all i
                grad_shape = variable.data.data.shape
                element_grad = np.full(grad_shape, gradient)
                variable.backward(element_grad)

        return Variable(sum_data, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Variable Class
This test validates Variable creation and basic gradient setup
"""

# %%
def test_unit_variable_class():
    """Test Variable class implementation with gradient tracking."""
    print("🔬 Unit Test: Variable Class...")

    # Test basic creation
    x = Variable([2.0, 3.0], requires_grad=True)
    assert isinstance(x.data, Tensor), "Variable should wrap Tensor"
    assert x.requires_grad == True, "Should track gradients when requested"
    assert x.grad is None, "Gradient should start as None"

    # Test creation without gradients
    y = Variable([1.0, 2.0], requires_grad=False)
    assert y.requires_grad == False, "Should not track gradients when not requested"

    # Test different data types
    z = Variable(np.array([4.0]), requires_grad=True)
    assert isinstance(z.data, Tensor), "Should convert numpy arrays to Tensors"

    print("✅ Variable class works correctly!")

test_unit_variable_class()

# %% [markdown]
"""
## Implementation: Addition Operation with Chain Rule

🧠 **Core Concepts**: Addition requires applying chain rule to both operands
⚡ **Performance**: Gradient computation is O(1) relative to forward pass
📦 **Framework Compatibility**: Matches PyTorch's autograd behavior

### Mathematical Foundation

For z = x + y:
- ∂z/∂x = 1 (derivative of x + y with respect to x)
- ∂z/∂y = 1 (derivative of x + y with respect to y)

Chain rule: ∂L/∂x = ∂L/∂z × ∂z/∂x = ∂L/∂z × 1 = ∂L/∂z
"""

# %% nbgrader={"grade": false, "grade_id": "add-operation", "solution": true}
def _ensure_variable(x):
    """Convert input to Variable if needed."""
    if isinstance(x, Variable):
        return x
    else:
        return Variable(x, requires_grad=False)

#| export
def add(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Add two variables with gradient tracking.

    TODO: Implement addition that properly tracks gradients

    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass (a.data + b.data)
    3. Create grad_fn that propagates gradients to both inputs
    4. Return new Variable with result and grad_fn

    EXAMPLE:
    >>> x = Variable([2.0], requires_grad=True)
    >>> y = Variable([3.0], requires_grad=True)
    >>> z = add(x, y)
    >>> z.backward()
    >>> print(x.grad)  # [1.0] - derivative of z w.r.t x
    >>> print(y.grad)  # [1.0] - derivative of z w.r.t y

    HINTS:
    - Use chain rule: ∂L/∂x = ∂L/∂z × ∂z/∂x = ∂L/∂z × 1
    - Both operands get same gradient (derivative of sum is 1)
    - Only propagate to variables that require gradients
    """
    ### BEGIN SOLUTION
    # Ensure both inputs are Variables
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    # Forward pass computation
    result_data = Tensor(a.data.data + b.data.data)

    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad

    # Define backward function for gradient propagation
    def grad_fn(gradient):
        """Propagate gradients to both operands."""
        # Addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        if a.requires_grad:
            a.backward(gradient)
        if b.requires_grad:
            b.backward(gradient)

    # Create result variable with gradient function
    result = Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Addition Operation
This test validates addition with proper gradient computation
"""

# %%
def test_unit_add_operation():
    """Test addition with gradient tracking."""
    print("🔬 Unit Test: Addition Operation...")

    # Test basic addition
    x = Variable([2.0], requires_grad=True)
    y = Variable([3.0], requires_grad=True)
    z = add(x, y)

    # Verify forward pass
    assert np.allclose(z.data.data, [5.0]), f"Expected [5.0], got {z.data.data}"

    # Test backward pass
    z.backward()
    assert np.allclose(x.grad, [1.0]), f"Expected x.grad=[1.0], got {x.grad}"
    assert np.allclose(y.grad, [1.0]), f"Expected y.grad=[1.0], got {y.grad}"

    # Test with constants
    a = Variable([1.0], requires_grad=True)
    b = add(a, 5.0)  # Adding constant
    b.backward()
    assert np.allclose(a.grad, [1.0]), "Gradient should flow through constant addition"

    print("✅ Addition operation works correctly!")

test_unit_add_operation()

# %% [markdown]
"""
## Implementation: Multiplication Operation with Product Rule

📐 **Mathematical Foundation**: Product rule for derivatives
🔗 **Connections**: Essential for linear layers, attention mechanisms
⚡ **Performance**: Efficient gradient computation using cached forward values

### The Product Rule

For z = x × y:
- ∂z/∂x = y (derivative with respect to first operand)
- ∂z/∂y = x (derivative with respect to second operand)

Chain rule: ∂L/∂x = ∂L/∂z × ∂z/∂x = ∂L/∂z × y
"""

# %% nbgrader={"grade": false, "grade_id": "multiply-operation", "solution": true}
#| export
def multiply(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Multiply two variables with gradient tracking.

    TODO: Implement multiplication using product rule for gradients

    APPROACH:
    1. Convert inputs to Variables if needed
    2. Compute forward pass (a.data × b.data)
    3. Create grad_fn using product rule: ∂(a×b)/∂a = b, ∂(a×b)/∂b = a
    4. Return Variable with result and grad_fn

    EXAMPLE:
    >>> x = Variable([2.0], requires_grad=True)
    >>> y = Variable([3.0], requires_grad=True)
    >>> z = multiply(x, y)
    >>> z.backward()
    >>> print(x.grad)  # [3.0] - derivative is y's value
    >>> print(y.grad)  # [2.0] - derivative is x's value

    HINTS:
    - Product rule: d(uv)/dx = u(dv/dx) + v(du/dx)
    - For our case: ∂(a×b)/∂a = b, ∂(a×b)/∂b = a
    - Store original values for use in backward pass
    """
    ### BEGIN SOLUTION
    # Ensure both inputs are Variables
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    # Forward pass computation
    result_data = Tensor(a.data.data * b.data.data)

    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad

    # Define backward function for gradient propagation
    def grad_fn(gradient):
        """Propagate gradients using product rule."""
        # Product rule: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        if a.requires_grad:
            # ∂L/∂a = ∂L/∂z × ∂z/∂a = gradient × b
            a_grad = gradient * b.data.data
            a.backward(a_grad)
        if b.requires_grad:
            # ∂L/∂b = ∂L/∂z × ∂z/∂b = gradient × a
            b_grad = gradient * a.data.data
            b.backward(b_grad)

    # Create result variable with gradient function
    result = Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Multiplication Operation
This test validates multiplication with product rule gradients
"""

# %%
def test_unit_multiply_operation():
    """Test multiplication with gradient tracking."""
    print("🔬 Unit Test: Multiplication Operation...")

    # Test basic multiplication
    x = Variable([2.0], requires_grad=True)
    y = Variable([3.0], requires_grad=True)
    z = multiply(x, y)

    # Verify forward pass
    assert np.allclose(z.data.data, [6.0]), f"Expected [6.0], got {z.data.data}"

    # Test backward pass
    z.backward()
    assert np.allclose(x.grad, [3.0]), f"Expected x.grad=[3.0], got {x.grad}"
    assert np.allclose(y.grad, [2.0]), f"Expected y.grad=[2.0], got {y.grad}"

    # Test with constants
    a = Variable([4.0], requires_grad=True)
    b = multiply(a, 2.0)  # Multiplying by constant
    b.backward()
    assert np.allclose(a.grad, [2.0]), "Gradient should be the constant value"

    print("✅ Multiplication operation works correctly!")

test_unit_multiply_operation()

# %% [markdown]
"""
## Implementation: Additional Operations

🔗 **Connections**: Complete the basic arithmetic operations needed for neural networks
⚡ **Performance**: Each operation implements efficient gradient computation
📦 **Framework Compatibility**: Matches behavior of production autograd systems
"""

# %% nbgrader={"grade": false, "grade_id": "additional-operations", "solution": true}
#| export
def subtract(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Subtract two variables with gradient tracking.

    TODO: Implement subtraction with proper gradient flow

    HINTS:
    - For z = a - b: ∂z/∂a = 1, ∂z/∂b = -1
    - Similar to addition but second operand gets negative gradient
    """
    ### BEGIN SOLUTION
    # Ensure both inputs are Variables
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    # Forward pass computation
    result_data = Tensor(a.data.data - b.data.data)

    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad

    # Define backward function for gradient propagation
    def grad_fn(gradient):
        """Propagate gradients for subtraction."""
        # Subtraction: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
        if a.requires_grad:
            a.backward(gradient)
        if b.requires_grad:
            b.backward(-gradient)  # Negative for subtraction

    # Create result variable with gradient function
    result = Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    return result
    ### END SOLUTION

#| export
def matmul(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """
    Matrix multiplication with gradient tracking.

    TODO: Implement matrix multiplication with proper gradients

    HINTS:
    - For z = a @ b: ∂z/∂a = gradient @ b.T, ∂z/∂b = a.T @ gradient
    - This is fundamental for neural network linear layers
    """
    ### BEGIN SOLUTION
    # Ensure both inputs are Variables
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    # Forward pass computation
    result_data = Tensor(a.data.data @ b.data.data)

    # Determine if result requires gradients
    requires_grad = a.requires_grad or b.requires_grad

    # Define backward function for gradient propagation
    def grad_fn(gradient):
        """Propagate gradients for matrix multiplication."""
        # Matrix multiplication gradients:
        # ∂(a@b)/∂a = gradient @ b.T
        # ∂(a@b)/∂b = a.T @ gradient
        if a.requires_grad:
            a_grad = gradient @ b.data.data.T
            a.backward(a_grad)
        if b.requires_grad:
            b_grad = a.data.data.T @ gradient
            b.backward(b_grad)

    # Create result variable with gradient function
    result = Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Additional Operations
This test validates subtraction and matrix multiplication
"""

# %%
def test_unit_additional_operations():
    """Test subtraction and matrix multiplication."""
    print("🔬 Unit Test: Additional Operations...")

    # Test subtraction
    x = Variable([5.0], requires_grad=True)
    y = Variable([2.0], requires_grad=True)
    z = subtract(x, y)

    assert np.allclose(z.data.data, [3.0]), f"Subtraction failed: expected [3.0], got {z.data.data}"

    z.backward()
    assert np.allclose(x.grad, [1.0]), f"Subtraction gradient failed: expected x.grad=[1.0], got {x.grad}"
    assert np.allclose(y.grad, [-1.0]), f"Subtraction gradient failed: expected y.grad=[-1.0], got {y.grad}"

    # Test matrix multiplication
    a = Variable([[1.0, 2.0]], requires_grad=True)
    b = Variable([[3.0], [4.0]], requires_grad=True)
    c = matmul(a, b)

    assert np.allclose(c.data.data, [[11.0]]), f"Matrix multiplication failed: expected [[11.0]], got {c.data.data}"

    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f"Matmul gradient failed for a: expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(b.grad, [[1.0], [2.0]]), f"Matmul gradient failed for b: expected [[1.0], [2.0]], got {b.grad}"

    print("✅ Additional operations work correctly!")

test_unit_additional_operations()

# %% [markdown]
"""
## Implementation: Chain Rule Through Complex Expressions

🧠 **Core Concept**: Multiple operations automatically chain gradients together
⚡ **Performance**: Each operation contributes O(1) overhead for gradient computation
🔗 **Connections**: This enables training deep neural networks with many layers

### Example: Complex Expression

Consider: f(x, y) = (x + y) × (x - y) = x² - y²

The autograd system automatically:
1. Tracks each intermediate operation
2. Applies chain rule backwards through the computation graph
3. Accumulates gradients at each variable

Expected gradients:
- ∂f/∂x = 2x (derivative of x² - y²)
- ∂f/∂y = -2y (derivative of x² - y²)
"""

# %% [markdown]
"""
### 🧪 Unit Test: Chain Rule Application
This test validates complex expressions with multiple operations
"""

# %%
def test_unit_chain_rule():
    """Test chain rule through complex expressions."""
    print("🔬 Unit Test: Chain Rule Application...")

    # Test complex expression: (x + y) * (x - y) = x² - y²
    x = Variable([3.0], requires_grad=True)
    y = Variable([2.0], requires_grad=True)

    # Build computation graph
    sum_term = add(x, y)      # x + y = 5
    diff_term = subtract(x, y) # x - y = 1
    result = multiply(sum_term, diff_term)  # (x+y)*(x-y) = 5*1 = 5

    # Verify forward pass
    expected_result = 3.0**2 - 2.0**2  # x² - y² = 9 - 4 = 5
    assert np.allclose(result.data.data, [expected_result]), f"Expected [{expected_result}], got {result.data.data}"

    # Test backward pass
    result.backward()

    # Expected gradients: ∂(x²-y²)/∂x = 2x = 6, ∂(x²-y²)/∂y = -2y = -4
    expected_x_grad = 2 * 3.0  # 6.0
    expected_y_grad = -2 * 2.0  # -4.0

    assert np.allclose(x.grad, [expected_x_grad]), f"Expected x.grad=[{expected_x_grad}], got {x.grad}"
    assert np.allclose(y.grad, [expected_y_grad]), f"Expected y.grad=[{expected_y_grad}], got {y.grad}"

    # Test another complex expression: x * y + x * y (should equal 2*x*y)
    a = Variable([2.0], requires_grad=True)
    b = Variable([3.0], requires_grad=True)

    term1 = multiply(a, b)
    term2 = multiply(a, b)
    sum_result = add(term1, term2)

    sum_result.backward()

    # Expected: ∂(2xy)/∂x = 2y = 6, ∂(2xy)/∂y = 2x = 4
    assert np.allclose(a.grad, [6.0]), f"Expected a.grad=[6.0], got {a.grad}"
    assert np.allclose(b.grad, [4.0]), f"Expected b.grad=[4.0], got {b.grad}"

    print("✅ Chain rule works correctly through complex expressions!")

test_unit_chain_rule()

# %% [markdown]
"""
## 🔍 Systems Analysis: Gradient Computation Behavior

Now that your autograd implementation is complete and tested, let's analyze its behavior:

**Analysis Focus**: Understand memory usage and computational patterns in automatic differentiation
"""

# %%
def analyze_gradient_computation():
    """
    📊 SYSTEMS MEASUREMENT: Gradient Computation Analysis

    Measure how autograd scales with expression complexity and input size.
    """
    print("📊 AUTOGRAD SYSTEMS ANALYSIS")
    print("Testing gradient computation patterns...")

    import time

    # Test 1: Expression complexity scaling
    print("\n🔍 Expression Complexity Analysis:")
    x = Variable([2.0], requires_grad=True)
    y = Variable([3.0], requires_grad=True)

    expressions = [
        ("Simple: x + y", lambda: add(x, y)),
        ("Medium: x * y + x", lambda: add(multiply(x, y), x)),
        ("Complex: (x + y) * (x - y)", lambda: multiply(add(x, y), subtract(x, y)))
    ]

    for name, expr_fn in expressions:
        # Reset gradients
        x.grad = None
        y.grad = None

        # Time forward + backward pass
        start = time.perf_counter()
        result = expr_fn()
        result.backward()
        elapsed = time.perf_counter() - start

        print(f"  {name}: {elapsed*1000:.3f}ms")

    # Test 2: Memory usage pattern
    print("\n💾 Memory Usage Analysis:")
    try:
        import psutil
        import os

        def get_memory_mb():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024

        baseline = get_memory_mb()
        psutil_available = True
    except ImportError:
        print("  Note: psutil not installed, skipping detailed memory analysis")
        psutil_available = False
        baseline = 0

    # Create computation graph with many variables
    variables = []
    for i in range(100):
        var = Variable([float(i)], requires_grad=True)
        variables.append(var)

    # Chain operations
    result = variables[0]
    for var in variables[1:]:
        result = add(result, var)

    if psutil_available:
        memory_after_forward = get_memory_mb()

    # Backward pass
    result.backward()

    if psutil_available:
        memory_after_backward = get_memory_mb()
        print(f"  Baseline memory: {baseline:.1f}MB")
        print(f"  After forward pass: {memory_after_forward:.1f}MB (+{memory_after_forward-baseline:.1f}MB)")
        print(f"  After backward pass: {memory_after_backward:.1f}MB (+{memory_after_backward-baseline:.1f}MB)")
    else:
        print("  Memory tracking skipped (psutil not available)")

    # Test 3: Gradient accumulation
    print("\n🔄 Gradient Accumulation Test:")
    z = Variable([1.0], requires_grad=True)

    # Multiple backward passes should accumulate gradients
    loss1 = multiply(z, 2.0)
    loss1.backward()
    first_grad = z.grad.copy()

    loss2 = multiply(z, 3.0)
    loss2.backward()  # Should accumulate with previous gradient

    print(f"  First backward: grad = {first_grad}")
    print(f"  After second backward: grad = {z.grad}")
    print(f"  Expected accumulation: {first_grad + 3.0}")

    print("\n💡 AUTOGRAD INSIGHTS:")
    print("  • Forward pass builds computation graph in memory")
    print("  • Backward pass traverses graph and accumulates gradients")
    print("  • Memory scales with graph depth, not just data size")
    print("  • This is why PyTorch uses gradient checkpointing for deep networks!")

analyze_gradient_computation()

# %% [markdown]
"""
## Integration: Complete Module Testing

🧪 **Testing Strategy**: Comprehensive validation of all autograd functionality
✅ **Quality Assurance**: Ensure all components work together correctly
🚀 **Ready for Training**: Verify autograd enables neural network optimization
"""

# %%
def test_module():
    """Comprehensive test of autograd module functionality."""
    print("🧪 COMPREHENSIVE MODULE TEST")
    print("Running complete autograd validation...")

    # Test 1: Variable creation and basic properties
    print("\n1️⃣ Testing Variable creation...")
    x = Variable([1.0, 2.0], requires_grad=True)
    assert isinstance(x.data, Tensor)
    assert x.requires_grad == True
    assert x.grad is None
    print("   ✅ Variable creation works")

    # Test 2: All arithmetic operations
    print("\n2️⃣ Testing arithmetic operations...")
    a = Variable([2.0], requires_grad=True)
    b = Variable([3.0], requires_grad=True)

    # Test each operation
    add_result = add(a, b)
    assert np.allclose(add_result.data.data, [5.0])

    mul_result = multiply(a, b)
    assert np.allclose(mul_result.data.data, [6.0])

    sub_result = subtract(a, b)
    assert np.allclose(sub_result.data.data, [-1.0])
    print("   ✅ All arithmetic operations work")

    # Test 3: Gradient computation
    print("\n3️⃣ Testing gradient computation...")
    x = Variable([3.0], requires_grad=True)
    y = Variable([4.0], requires_grad=True)
    z = multiply(x, y)  # z = 12
    z.backward()

    assert np.allclose(x.grad, [4.0]), f"Expected x.grad=[4.0], got {x.grad}"
    assert np.allclose(y.grad, [3.0]), f"Expected y.grad=[3.0], got {y.grad}"
    print("   ✅ Gradient computation works")

    # Test 4: Complex expressions
    print("\n4️⃣ Testing complex expressions...")
    p = Variable([2.0], requires_grad=True)
    q = Variable([3.0], requires_grad=True)

    # (p + q) * (p - q) = p² - q²
    expr = multiply(add(p, q), subtract(p, q))
    expr.backward()

    # Expected: ∂(p²-q²)/∂p = 2p = 4, ∂(p²-q²)/∂q = -2q = -6
    assert np.allclose(p.grad, [4.0]), f"Expected p.grad=[4.0], got {p.grad}"
    assert np.allclose(q.grad, [-6.0]), f"Expected q.grad=[-6.0], got {q.grad}"
    print("   ✅ Complex expressions work")

    # Test 5: Matrix operations
    print("\n5️⃣ Testing matrix operations...")
    A = Variable([[1.0, 2.0]], requires_grad=True)
    B = Variable([[3.0], [4.0]], requires_grad=True)
    C = matmul(A, B)

    assert np.allclose(C.data.data, [[11.0]])
    C.backward()
    assert np.allclose(A.grad, [[3.0, 4.0]])
    assert np.allclose(B.grad, [[1.0], [2.0]])
    print("   ✅ Matrix operations work")

    # Test 6: Mixed operations
    print("\n6️⃣ Testing mixed operations...")
    u = Variable([1.0], requires_grad=True)
    v = Variable([2.0], requires_grad=True)

    # Neural network-like computation: u * v + u
    hidden = multiply(u, v)  # u * v
    output = add(hidden, u)   # + u
    output.backward()

    # Expected: ∂(u*v + u)/∂u = v + 1 = 3, ∂(u*v + u)/∂v = u = 1
    assert np.allclose(u.grad, [3.0]), f"Expected u.grad=[3.0], got {u.grad}"
    assert np.allclose(v.grad, [1.0]), f"Expected v.grad=[1.0], got {v.grad}"
    print("   ✅ Mixed operations work")

    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 Autograd module is ready for neural network training!")
    print("🔗 Next: Use these gradients in optimizers to update parameters")

# %%
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

### Question 1: Memory Management in Computational Graphs

Consider the expression `z = (x + y) * (x - y)` where x and y have `requires_grad=True`.

**Analysis Task**: Your autograd implementation stores intermediate results during forward pass and uses them during backward pass. In a deep neural network with 100 layers, each layer creating intermediate variables, what memory challenges would emerge?

**Specific Questions**:
- How does memory usage scale with network depth in your current implementation?
- What strategies could reduce memory usage during gradient computation?
- Why do production frameworks like PyTorch implement "gradient checkpointing"?

**Implementation Connection**: Examine how your `grad_fn` closures capture references to input variables and consider the memory implications.
"""

# %% nbgrader={"grade": true, "grade_id": "memory-analysis", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
TODO: Analyze memory usage patterns in your autograd implementation.

Consider how your Variable class stores references to other variables through grad_fn,
and how this affects memory usage in deep networks.

Discuss specific memory optimization strategies you could implement.
"""
### BEGIN SOLUTION
# Memory analysis for autograd implementation:

# 1. Memory scaling with network depth:
# - Each Variable stores references to inputs through grad_fn closure
# - In deep networks: O(depth) memory growth for intermediate activations
# - Gradient computation requires keeping forward activations in memory
# - 100-layer network = 100x intermediate variables + their grad_fn closures

# 2. Memory optimization strategies:
# - Gradient checkpointing: Only store subset of activations, recompute others
# - In-place operations where mathematically valid
# - Clear computation graph after backward pass
# - Use smaller data types (float16 vs float32) where precision allows

# 3. Production framework solutions:
# - PyTorch's gradient checkpointing trades compute for memory
# - Automatic memory management with garbage collection
# - Graph optimization to reduce intermediate storage
# - Dynamic graph construction vs static graph optimization

# Current implementation improvement:
# Add method to clear computation graph: variable.detach() or graph.clear()
### END SOLUTION

# %% [markdown]
"""
### Question 2: Gradient Accumulation and Training Efficiency

In your autograd implementation, gradients accumulate when `backward()` is called multiple times without zeroing gradients.

**Analysis Task**: Design a training loop that uses gradient accumulation to simulate larger batch sizes with limited memory.

**Specific Questions**:
- How would you modify the Variable class to support gradient zeroing?
- What are the trade-offs between large batches vs. gradient accumulation?
- How does gradient accumulation affect convergence in neural network training?

**Implementation Connection**: Consider how your `backward()` method accumulates gradients and design a complete training interface.
"""

# %% nbgrader={"grade": true, "grade_id": "gradient-accumulation", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
TODO: Design gradient accumulation strategy for your autograd system.

Extend your Variable class with gradient management methods and analyze
the trade-offs between memory efficiency and training convergence.
"""
### BEGIN SOLUTION
# Gradient accumulation design for training efficiency:

# 1. Variable class extensions needed:
def zero_grad(self):
    """Clear accumulated gradients."""
    self.grad = None

def add_zero_grad_to_variable():
    """Would add this method to Variable class"""
    # Implementation would set self.grad = None
    pass

# 2. Training loop with gradient accumulation:
def training_step_with_accumulation(model, data_loader, accumulation_steps=4):
    """
    Simulate larger batches through gradient accumulation
    """
    for param in model.parameters():
        param.zero_grad()

    total_loss = 0
    for i, batch in enumerate(data_loader):
        loss = compute_loss(model(batch.x), batch.y)
        loss.backward()  # Accumulate gradients
        total_loss += loss.data

        if (i + 1) % accumulation_steps == 0:
            # Update parameters with accumulated gradients
            optimizer.step()
            # Clear gradients for next accumulation cycle
            for param in model.parameters():
                param.zero_grad()

    return total_loss / len(data_loader)

# 3. Trade-offs analysis:
# Memory: Gradient accumulation uses constant memory vs. large batch linear growth
# Convergence: Accumulated gradients approximate large batch behavior
# Computation: Extra backward passes vs. single large batch forward/backward
# Synchronization: In distributed training, less frequent communication

# 4. Production considerations:
# - Gradient scaling to prevent underflow with accumulated small gradients
# - Learning rate adjustment for effective batch size
# - Batch normalization statistics affected by actual vs effective batch size
### END SOLUTION

# %% [markdown]
"""
### Question 3: Computational Graph Optimization

Your autograd implementation creates a new Variable for each operation, building a computation graph dynamically.

**Analysis Task**: Analyze opportunities for optimizing the computational graph to reduce memory usage and improve performance.

**Specific Questions**:
- Which operations could be fused together to reduce intermediate Variable storage?
- How would in-place operations affect gradient computation safety?
- What graph optimization passes could be implemented before backward propagation?

**Implementation Connection**: Examine your operation functions and identify where intermediate results could be eliminated or reused.
"""

# %% nbgrader={"grade": true, "grade_id": "graph-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
TODO: Design graph optimization strategies for your autograd implementation.

Identify specific optimizations that could reduce memory usage and improve
performance while maintaining gradient correctness.
"""
### BEGIN SOLUTION
# Computational graph optimization strategies:

# 1. Operation fusion opportunities:
# - Fuse: add + multiply → fused_add_mul (one intermediate variable)
# - Fuse: activation + linear → fused_linear_activation
# - Elementwise operations: add + relu + multiply can be single kernel
# Current: 3 Variables → Optimized: 1 Variable

def fused_add_multiply(a, b, c):
    """Fused operation: (a + b) * c - saves one intermediate Variable"""
    # Direct computation without intermediate Variable
    result_data = (a.data.data + b.data.data) * c.data.data

    def grad_fn(gradient):
        if a.requires_grad:
            a.backward(gradient * c.data.data)
        if b.requires_grad:
            b.backward(gradient * c.data.data)
        if c.requires_grad:
            c.backward(gradient * (a.data.data + b.data.data))

    return Variable(result_data, requires_grad=any([a.requires_grad, b.requires_grad, c.requires_grad]), grad_fn=grad_fn)

# 2. In-place operation safety:
# Safe: element-wise operations on leaf variables not used elsewhere
# Unsafe: in-place on intermediate variables used in multiple paths
# Solution: Track variable usage count before allowing in-place

def safe_inplace_add(var, other):
    """In-place addition if safe for gradient computation"""
    if var.grad_fn is not None:
        raise RuntimeError("Cannot do in-place operation on variable with grad_fn")
    var.data.data += other.data.data
    return var

# 3. Graph optimization passes:
# - Dead code elimination: Remove unused intermediate variables
# - Common subexpression elimination: Reuse x*y if computed multiple times
# - Memory layout optimization: Arrange for cache-friendly access patterns

class GraphOptimizer:
    def optimize_memory_layout(self, variables):
        """Optimize variable storage for cache efficiency"""
        # Group related variables in contiguous memory
        pass

    def eliminate_dead_variables(self, root_variable):
        """Remove variables not needed for gradient computation"""
        # Traverse backward from root, mark reachable variables
        pass

    def fuse_operations(self, computation_sequence):
        """Identify fusible operation sequences"""
        # Pattern matching for common operation combinations
        pass

# 4. Production framework techniques:
# - TensorFlow's XLA: Ahead-of-time compilation with graph optimization
# - PyTorch's TorchScript: Graph optimization for inference
# - ONNX graph optimization passes: Constant folding, operator fusion
# - Memory planning: Pre-allocate memory for entire computation graph
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Autograd - Automatic Differentiation Engine

Congratulations! You've successfully implemented the automatic differentiation engine:

### What You've Accomplished
✅ **Variable Class Implementation**: Complete gradient tracking system with 200+ lines of core functionality
✅ **Arithmetic Operations**: Addition, multiplication, subtraction, and matrix operations with proper gradient flow
✅ **Chain Rule Application**: Automatic gradient computation through complex mathematical expressions
✅ **Memory Management**: Efficient gradient accumulation and computational graph construction
✅ **Systems Analysis**: Understanding of memory scaling and performance characteristics in gradient computation

### Key Learning Outcomes
- **Automatic Differentiation**: How computational graphs enable efficient gradient computation
- **Chain Rule Implementation**: Mathematical foundation for backpropagation in neural networks
- **Memory Patterns**: How gradient computation affects memory usage in deep learning systems
- **Production Understanding**: Connection to PyTorch/TensorFlow autograd implementations

### Mathematical Foundations Mastered
- **Chain Rule**: Systematic application through computational graphs
- **Product Rule**: Gradient computation for multiplication operations
- **Computational Complexity**: O(1) gradient overhead per operation in forward pass
- **Memory Complexity**: O(graph_depth) storage requirements for intermediate activations

### Professional Skills Developed
- **Gradient System Design**: Building automatic differentiation from scratch
- **Performance Analysis**: Understanding memory and computational trade-offs
- **Testing Methodology**: Comprehensive validation of gradient correctness

### Ready for Advanced Applications
Your autograd implementation now enables:
- **Neural Network Training**: Automatic gradient computation for parameter updates
- **Optimization Algorithms**: Foundation for SGD, Adam, and other optimizers
- **Deep Learning Research**: Understanding of how modern frameworks work internally

### Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: `torch.autograd.Variable` and automatic gradient computation
- **TensorFlow**: `tf.GradientTape` for automatic differentiation
- **Industry Standard**: Dynamic computational graphs used in most modern frameworks

### Next Steps
1. **Export your module**: `tito module complete 05_autograd`
2. **Validate integration**: `tito test --module autograd`
3. **Ready for Module 06**: Optimizers will use your gradients to update neural network parameters!

**🚀 Achievement Unlocked**: Your automatic differentiation engine is the foundation that makes modern neural network training possible!
"""
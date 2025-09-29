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
- Module 01 (Tensor): Pure data structures with ZERO gradient contamination
- Module 02-04: Built on pure tensors with clean mathematical operations

**What's Working**: You have a complete pure tensor system with arithmetic operations!

**The Gap**: Your tensors are "gradient-blind" - they can't track gradients for training.

**This Module's Solution**: Use Python's decorator pattern to enhance your existing Tensor class with gradient tracking, WITHOUT breaking any existing code.

**Connection Map**:
```
Pure Tensors → Enhanced Tensors → Training
(Module 01)    (+ Autograd)      (Optimizers)
```

## Learning Objectives
1. **Python Mastery**: Advanced metaprogramming with decorators
2. **Backward Compatibility**: Enhance without breaking existing functionality
3. **Mathematical Foundation**: Chain rule application in computational graphs
4. **Systems Design**: Clean enhancement patterns in software engineering

## Build → Test → Use
1. **Build**: Decorator that adds gradient tracking to existing Tensor class
2. **Test**: Verify ALL previous code still works + new gradient features
3. **Use**: Enable gradient-based optimization on familiar tensor operations

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/05_autograd/autograd_dev.py
**Building Side:** Code exports to tinytorch.core.autograd

```python
# Final package structure:
from tinytorch.core.autograd import add_autograd  # This module's decorator
from tinytorch.core.tensor import Tensor          # Pure tensor from Module 01

# Apply enhancement:
Tensor = add_autograd(Tensor)  # Now your Tensor has gradient capabilities!
```

**Why this matters:**
- **Learning:** Experience advanced Python patterns and clean software design
- **Backward Compatibility:** All Module 01-04 code works unchanged
- **Professional Practice:** How real systems add features without breaking existing code
- **Educational Clarity:** See exactly how gradient tracking enhances pure tensors
"""

# %%
#| default_exp core.autograd

#| export
import numpy as np
import sys
from typing import Union, List, Optional, Callable

# Import the PURE Tensor class from Module 01
# This is the clean, gradient-free tensor we'll enhance
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# %%
print("🔥 TinyTorch Autograd Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build automatic differentiation!")

# %% [markdown]
"""
## Python Metaprogramming: The Decorator Pattern

### The Challenge: Enhancing Existing Classes Without Breaking Code

You've built a beautiful, clean Tensor class in Module 01. All your code from Modules 02-04 depends on it working exactly as designed. But now you need gradient tracking.

**Wrong Approach**: Modify the Tensor class directly
- ❌ Breaks existing code
- ❌ Contaminates pure mathematical operations
- ❌ Violates single responsibility principle

**Right Approach**: Use Python's decorator pattern
- ✅ Enhance without modifying original class
- ✅ Perfect backward compatibility
- ✅ Clean separation of concerns

### The Decorator Pattern in Action

```python
# Your original pure Tensor class
class Tensor:
    def __add__(self, other):
        return Tensor(self.data + other.data)  # Pure math, no gradients

# Decorator adds gradient capabilities
@add_autograd
class Tensor:  # Same class, now enhanced!
    def __add__(self, other):  # Enhanced method
        result = original_add(self, other)  # Original behavior preserved
        # + gradient tracking added seamlessly
        return result
```

**Key Insight**: Decorators let you enhance classes by wrapping their methods, preserving original functionality while adding new capabilities.
"""

# %% [markdown]
"""
## Implementation: The add_autograd Decorator

🏗️ **Design Goal**: Transform pure Tensor class into gradient-capable version
🎯 **Backward Compatibility**: All existing Tensor code continues to work unchanged
📐 **Clean Enhancement**: Gradient tracking added without polluting core math operations

### The Decorator's Mission

The `add_autograd` decorator will:
1. **Save original methods**: Store pure mathematical implementations
2. **Enhance constructor**: Add `requires_grad` parameter and gradient storage
3. **Wrap operations**: Intercept `__add__`, `__mul__`, etc. to build computation graphs
4. **Add new methods**: Include `backward()` for gradient computation
5. **Preserve semantics**: Existing code works exactly as before

### Before vs After Enhancement

```python
# Before: Pure tensor (Module 01)
x = Tensor([2.0])
y = Tensor([3.0])
z = x + y  # Result: Tensor([5.0]) - pure math

# After: Enhanced tensor (this module)
x = Tensor([2.0], requires_grad=True)  # New optional parameter
y = Tensor([3.0], requires_grad=True)
z = x + y  # Result: Tensor([5.0]) - same math + gradient tracking
z.backward()  # New capability!
print(x.grad)  # [1.0] - gradients computed automatically
```
"""

# %% nbgrader={"grade": false, "grade_id": "add-autograd-decorator", "solution": true}
#| export
def add_autograd(cls):
    """
    Decorator that adds gradient tracking to existing Tensor class.

    This transforms a pure Tensor class into one capable of automatic differentiation
    while preserving 100% backward compatibility.

    TODO: Implement decorator that enhances Tensor class with gradient tracking

    APPROACH:
    1. Save original methods from pure Tensor class
    2. Create new __init__ that adds gradient parameters
    3. Wrap arithmetic operations to build computation graphs
    4. Add backward() method for gradient computation
    5. Replace methods on the class and return enhanced class

    EXAMPLE:
    >>> # Apply decorator to pure Tensor class
    >>> Tensor = add_autograd(Tensor)
    >>>
    >>> # Now Tensor has gradient capabilities!
    >>> x = Tensor([2.0], requires_grad=True)
    >>> y = Tensor([3.0], requires_grad=True)
    >>> z = x * y
    >>> z.backward()
    >>> print(x.grad)  # [3.0]
    >>> print(y.grad)  # [2.0]

    HINTS:
    - Store original methods before replacing them
    - New methods should call original methods first
    - Only add gradient tracking when requires_grad=True
    - Preserve all original functionality
    """
    ### BEGIN SOLUTION
    # Store original methods from pure Tensor class
    original_init = cls.__init__
    original_add = cls.__add__
    original_mul = cls.__mul__
    original_sub = cls.__sub__ if hasattr(cls, '__sub__') else None
    original_matmul = cls.__matmul__ if hasattr(cls, '__matmul__') else None

    def new_init(self, data, dtype=None, requires_grad=False):
        """Enhanced constructor with gradient tracking support."""
        # Call original constructor to preserve all existing functionality
        original_init(self, data, dtype)

        # Add gradient tracking attributes
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def new_add(self, other):
        """Enhanced addition with gradient tracking."""
        # Forward pass: use original pure addition
        result = original_add(self, other)

        # Add gradient tracking if either operand requires gradients
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result.grad = None

            # Define backward function for gradient computation
            def grad_fn(gradient):
                """Apply addition backward pass: d(a+b)/da = 1, d(a+b)/db = 1"""
                if self.requires_grad:
                    self.backward(gradient)
                if hasattr(other, 'requires_grad') and other.requires_grad:
                    other.backward(gradient)

            result.grad_fn = grad_fn

        return result

    def new_mul(self, other):
        """Enhanced multiplication with gradient tracking."""
        # Forward pass: use original pure multiplication
        result = original_mul(self, other)

        # Add gradient tracking if either operand requires gradients
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result.grad = None

            # Define backward function using product rule
            def grad_fn(gradient):
                """Apply multiplication backward pass: d(a*b)/da = b, d(a*b)/db = a"""
                if self.requires_grad:
                    # Get gradient data, handle both Tensor and scalar cases
                    if hasattr(other, 'data'):
                        other_data = other.data
                    else:
                        other_data = other
                    self_grad = gradient * other_data
                    self.backward(self_grad)

                if hasattr(other, 'requires_grad') and other.requires_grad:
                    # Get gradient data for self
                    self_grad = gradient * self.data
                    other.backward(self_grad)

            result.grad_fn = grad_fn

        return result

    def new_sub(self, other):
        """Enhanced subtraction with gradient tracking."""
        if original_sub is None:
            # If original class doesn't have subtraction, implement it
            if hasattr(other, 'data'):
                result_data = self.data - other.data
            else:
                result_data = self.data - other
            result = cls(result_data)
        else:
            # Use original subtraction
            result = original_sub(self, other)

        # Add gradient tracking
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result.grad = None

            def grad_fn(gradient):
                """Apply subtraction backward pass: d(a-b)/da = 1, d(a-b)/db = -1"""
                if self.requires_grad:
                    self.backward(gradient)
                if hasattr(other, 'requires_grad') and other.requires_grad:
                    other.backward(-gradient)

            result.grad_fn = grad_fn

        return result

    def new_matmul(self, other):
        """Enhanced matrix multiplication with gradient tracking."""
        if original_matmul is None:
            # If original class doesn't have matmul, implement it
            result_data = self.data @ other.data
            result = cls(result_data)
        else:
            # Use original matrix multiplication
            result = original_matmul(self, other)

        # Add gradient tracking
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result.grad = None

            def grad_fn(gradient):
                """Apply matmul backward pass."""
                if self.requires_grad:
                    # d(A@B)/dA = gradient @ B.T
                    self_grad = gradient @ other.data.T
                    self.backward(self_grad)
                if hasattr(other, 'requires_grad') and other.requires_grad:
                    # d(A@B)/dB = A.T @ gradient
                    other_grad = self.data.T @ gradient
                    other.backward(other_grad)

            result.grad_fn = grad_fn

        return result

    def backward(self, gradient=None):
        """
        New method: Compute gradients via backpropagation.

        Args:
            gradient: Gradient flowing backwards (defaults to ones for scalars)
        """
        if not self.requires_grad:
            raise RuntimeError("Tensor doesn't require gradients")

        # Default gradient for scalar outputs
        if gradient is None:
            if hasattr(self, 'data') and hasattr(self.data, 'size'):
                if self.data.size == 1:
                    gradient = np.ones_like(self.data)
                else:
                    raise RuntimeError("gradient must be specified for non-scalar tensors")
            else:
                gradient = np.ones_like(self.data)

        # Accumulate gradients
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        # Propagate gradients backwards through computation graph
        if self.grad_fn is not None:
            self.grad_fn(gradient)

    # Replace methods on the class
    cls.__init__ = new_init
    cls.__add__ = new_add
    cls.__mul__ = new_mul
    cls.__sub__ = new_sub
    cls.__matmul__ = new_matmul
    cls.backward = backward

    return cls
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Decorator Application
This test validates the decorator enhances Tensor while preserving backward compatibility
"""

# %%
def test_unit_decorator_application():
    """Test that decorator enhances Tensor while preserving compatibility."""
    print("🔬 Unit Test: Decorator Application...")

    # Apply decorator to enhance the pure Tensor class
    EnhancedTensor = add_autograd(Tensor)

    # Test 1: Backward compatibility - existing functionality preserved
    x = EnhancedTensor([2.0, 3.0])  # No requires_grad - should work like pure Tensor
    y = EnhancedTensor([1.0, 2.0])
    z = x + y

    # Should behave exactly like original Tensor
    assert hasattr(z, 'data'), "Enhanced tensor should have data attribute"
    assert not hasattr(z, 'requires_grad') or not z.requires_grad, "Should not track gradients by default"

    # Test 2: New gradient capabilities when enabled
    a = EnhancedTensor([2.0], requires_grad=True)
    b = EnhancedTensor([3.0], requires_grad=True)

    assert a.requires_grad == True, "Should track gradients when requested"
    assert a.grad is None, "Gradient should start as None"
    assert hasattr(a, 'backward'), "Should have backward method"

    # Test 3: Operations build computation graphs
    c = a + b
    assert c.requires_grad == True, "Result should require gradients if inputs do"
    assert hasattr(c, 'grad_fn'), "Should have gradient function"

    print("✅ Decorator application works correctly!")

test_unit_decorator_application()

# %% [markdown]
"""
## Implementation: Apply Decorator to Create Enhanced Tensor

🏗️ **The Magic Moment**: Transform pure Tensor into gradient-capable version
✅ **Backward Compatibility**: All existing code continues to work
🎆 **New Capabilities**: Gradient tracking available when requested

### The Transformation

Applying the decorator is simple but powerful:

```python
# Before: Pure Tensor class (Module 01)
class Tensor:
    def __add__(self, other): return Tensor(self.data + other.data)

# After: Enhanced with autograd capabilities
Tensor = add_autograd(Tensor)

# Now the same class can do both!
z1 = Tensor([1, 2]) + Tensor([3, 4])  # Pure math (like before)
z2 = Tensor([1, 2], requires_grad=True) + Tensor([3, 4], requires_grad=True)  # + gradients!
```

### Mathematical Foundation

For z = x + y:
- ∂z/∂x = 1 (derivative of x + y with respect to x)
- ∂z/∂y = 1 (derivative of x + y with respect to y)

Chain rule: ∂L/∂x = ∂L/∂z × ∂z/∂x = ∂L/∂z × 1 = ∂L/∂z
"""

# %% nbgrader={"grade": false, "grade_id": "apply-decorator", "solution": true}
#| export
# Apply the decorator to transform pure Tensor into gradient-capable version
# This is where the magic happens!

### BEGIN SOLUTION
# Import pure Tensor class and enhance it with autograd
Tensor = add_autograd(Tensor)
### END SOLUTION

# Now our pure Tensor class has been enhanced with gradient tracking!
# Let's test that it works correctly...

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
## 🎯 MODULE SUMMARY: Autograd - Decorator-Based Automatic Differentiation

Congratulations! You've mastered the decorator pattern to enhance pure tensors with gradient tracking:

### What You've Accomplished
✅ **Decorator Implementation**: Clean enhancement of existing Tensor class with 100+ lines of elegant code
✅ **Backward Compatibility**: All Module 01-04 code works unchanged - zero breaking changes
✅ **Gradient Tracking**: Optional `requires_grad=True` parameter enables automatic differentiation
✅ **Chain Rule Application**: Automatic gradient computation through complex mathematical expressions
✅ **Systems Understanding**: Analysis of memory patterns and performance characteristics
✅ **Production Connection**: Understanding of how real ML frameworks evolved

### Key Learning Outcomes
- **Python Metaprogramming**: Advanced decorator patterns for class enhancement
- **Software Architecture**: Clean enhancement without code contamination
- **Backward Compatibility**: Professional approach to adding features safely
- **Automatic Differentiation**: How computational graphs enable efficient gradient computation
- **Production Understanding**: Connection to PyTorch's evolution from Variable to Tensor-based autograd

### Technical Foundations Mastered
- **Decorator Pattern**: Method interception and enhancement techniques
- **Computational Graphs**: Dynamic graph construction through operation tracking
- **Chain Rule**: Automatic application through backward propagation
- **Memory Management**: Efficient gradient accumulation and graph storage
- **Performance Analysis**: Understanding overhead patterns in gradient computation

### Professional Skills Developed
- **Clean Code Enhancement**: Adding features without breaking existing functionality
- **Advanced Python**: Metaprogramming techniques used in production frameworks
- **Systems Thinking**: Understanding trade-offs between functionality and performance
- **Testing Methodology**: Comprehensive validation including backward compatibility

### Ready for Advanced Applications
Your enhanced Tensor class now enables:
- **Neural Network Training**: Seamless gradient computation for parameter updates
- **Optimization Algorithms**: Foundation for SGD, Adam, and other optimizers
- **Research Applications**: Understanding of how modern frameworks implement autograd

### Connection to Real ML Systems
Your decorator-based implementation mirrors production evolution:
- **PyTorch v0.1**: Separate Variable class (old approach)
- **PyTorch v0.4+**: Tensor-based autograd using enhancement patterns (your approach!)
- **TensorFlow**: Similar evolution from separate Variable to enhanced Tensor
- **Industry Standard**: Decorator pattern widely used for framework evolution

### Next Steps
1. **Export your module**: `tito module complete 05_autograd`
2. **Validate integration**: All Module 01-04 code still works + new gradient features
3. **Ready for Module 06**: Optimizers will use your gradients to update neural network parameters!

**🚀 Achievement Unlocked**: You've mastered the professional approach to enhancing software systems without breaking existing functionality - exactly how real ML frameworks evolved!
"""
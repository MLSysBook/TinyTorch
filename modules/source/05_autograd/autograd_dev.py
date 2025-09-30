# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 05: Autograd - Awakening the Gradient Engine

Welcome to Module 05! Today you'll bring gradients to life and unlock automatic differentiation.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor operations, activations, layers, and loss functions
**You'll Build**: The autograd system that computes gradients automatically
**You'll Enable**: Learning! Training! The ability to optimize neural networks!

**Connection Map**:
```
Modules 01-04 → Autograd → Training (Module 06-07)
(forward pass) (backward pass) (learning loops)
```

## Learning Objectives
By the end of this module, you will:
1. Implement the backward() method for Tensor to enable gradient computation
2. Create a Function base class for operation tracking
3. Build computation graphs for automatic differentiation
4. Test gradient correctness and chain rule implementation

**CRITICAL**: This module enhances the existing Tensor class by implementing its dormant gradient features!

Let's awaken the gradient engine!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/05_autograd/autograd_dev.py
**Building Side:** Code exports to tinytorch.core.autograd

```python
# Final package structure:
from tinytorch.core.autograd import Function  # This module - gradient computation
from tinytorch.core.tensor import Tensor  # Enhanced with gradients from this module
```

**Why this matters:**
- **Learning:** Complete autograd system enabling automatic differentiation
- **Production:** PyTorch-style computational graph and backward pass
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Enhances existing Tensor without breaking anything
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.autograd
#| export

import numpy as np
from typing import List, Optional, Callable
import sys
import os

# Import the modern Tensor class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

# %% [markdown]
"""
## 1. Introduction: What is Automatic Differentiation?

Automatic differentiation (autograd) is the magic that makes neural networks learn. Instead of manually computing gradients for every parameter, autograd tracks operations and automatically computes gradients via the chain rule.

### The Challenge
In Module 04, you implemented a loss function. To train a model, you need:
```
Loss = f(W₃, f(W₂, f(W₁, x)))
∂Loss/∂W₁ = ?  ∂Loss/∂W₂ = ?  ∂Loss/∂W₃ = ?
```

Manual gradient computation becomes impossible for complex models with millions of parameters.

### The Solution: Computational Graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
```

**Complete Autograd Process Visualization:**
```
┌─ FORWARD PASS ──────────────────────────────────────────────┐
│                                                             │
│ x ──┬── W₁ ──┐                                              │
│     │        ├──[Linear₁]──→ z₁ ──[ReLU]──→ a₁ ──┬── W₂ ──┐ │
│     └── b₁ ──┘                               │        ├─→ Loss
│                                              └── b₂ ──┘ │
│                                                             │
└─ COMPUTATION GRAPH BUILT ──────────────────────────────────┘
                             │
                             ▼
┌─ BACKWARD PASS ─────────────────────────────────────────────┐
│                                                             │
│∇x ←┬← ∇W₁ ←┐                                               │
│    │       ├←[Linear₁]←─ ∇z₁ ←[ReLU]← ∇a₁ ←┬← ∇W₂ ←┐      │
│    └← ∇b₁ ←┘                             │       ├← ∇Loss  │
│                                          └← ∇b₂ ←┘      │
│                                                             │
└─ GRADIENTS COMPUTED ───────────────────────────────────────┘

Key Insight: Each [operation] stores how to compute its backward pass.
The chain rule automatically flows gradients through the entire graph.
```

Each operation records how to compute its backward pass. The chain rule connects them all.
"""

# %% [markdown]
"""
## 2. Foundations: The Chain Rule in Action

### Mathematical Foundation
For composite functions: f(g(x)), the derivative is:
```
df/dx = (df/dg) × (dg/dx)
```

### Computational Graph Example
```
Simple computation: L = (x * y + 5)²

Forward Pass:
  x=2 ──┐
        ├──[×]──→ z=6 ──[+5]──→ w=11 ──[²]──→ L=121
  y=3 ──┘

Backward Pass (Chain Rule in Action):
  ∂L/∂x = ∂L/∂w × ∂w/∂z × ∂z/∂x
        = 2w  ×  1  ×  y
        = 2(11) × 1 × 3 = 66

  ∂L/∂y = ∂L/∂w × ∂w/∂z × ∂z/∂y
        = 2w  ×  1  ×  x
        = 2(11) × 1 × 2 = 44

Gradient Flow Visualization:
  ∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←──[²]←── ∇L=1
  ∇y=44 ←──┘
```

### Memory Layout During Backpropagation
```
Computation Graph Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Forward Pass (stored for backward)                      │
├─────────────────────────────────────────────────────────┤
│ Node 1: x=2 (leaf, requires_grad=True) │ grad: None→66  │
│ Node 2: y=3 (leaf, requires_grad=True) │ grad: None→44  │
│ Node 3: z=x*y (MulFunction)            │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (AddFunction)            │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w² (PowFunction)             │ grad: 1        │
│         saved: (w=11)                  │ inputs: [w]    │
└─────────────────────────────────────────────────────────┘

Memory Cost: 2× parameters (data + gradients) + graph overhead
```
"""

# %% [markdown]
"""
## 3. Implementation: Building the Autograd Engine

Let's implement the autograd system step by step. We'll enhance the existing Tensor class and create supporting infrastructure.

### The Function Architecture

Every differentiable operation needs two things:
1. **Forward pass**: Compute the result
2. **Backward pass**: Compute gradients for inputs

```
Function Class Design:
┌─────────────────────────────────────┐
│ Function (Base Class)               │
├─────────────────────────────────────┤
│ • save_for_backward()  ← Store data │
│ • forward()           ← Compute     │
│ • backward()          ← Gradients   │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Function│ │Function│ │Function│ │Function│
└────────┘ └────────┘ └────────┘ └────────┘
```

Each operation inherits from Function and implements specific gradient rules.
"""

# %% [markdown]
"""
### Function Base Class - The Foundation of Autograd

The Function class is the foundation that makes autograd possible. Every differentiable operation (addition, multiplication, etc.) inherits from this class.

**Why Functions Matter:**
- They remember inputs needed for backward pass
- They implement forward computation
- They implement gradient computation via backward()
- They connect to form computation graphs

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.backward() → grad_inputs
```

This pattern enables the chain rule to flow gradients through complex computations.
"""

# %% nbgrader={"grade": false, "grade_id": "function-base", "solution": true}
class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class.
    """

    def __init__(self):
        """Initialize function with empty input tracking."""
        self.inputs = []
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        """
        Save tensors needed for backward pass.

        TODO: Store tensors that backward() will need

        EXAMPLE:
        In multiplication: y = a * b
        We need to save 'a' and 'b' because:
        ∂y/∂a = b and ∂y/∂b = a
        """
        ### BEGIN SOLUTION
        self.saved_tensors = tensors
        ### END SOLUTION

    def forward(self, *inputs):
        """
        Compute forward pass.

        TODO: Implement in subclasses
        This should be overridden by each specific operation.
        """
        raise NotImplementedError("Forward pass must be implemented by subclasses")

    def backward(self, grad_output):
        """
        Compute backward pass.

        TODO: Implement in subclasses

        APPROACH:
        1. Take gradient flowing backward (grad_output)
        2. Apply chain rule with local gradients
        3. Return gradients for inputs
        """
        raise NotImplementedError("Backward pass must be implemented by subclasses")

# %% [markdown]
"""
### 🔬 Unit Test: Function Base Class
This test validates our Function base class works correctly.
**What we're testing**: Function initialization and interface
**Why it matters**: Foundation for all differentiable operations
**Expected**: Proper initialization and save_for_backward functionality
"""

# %% nbgrader={"grade": true, "grade_id": "test-function-base", "locked": true, "points": 10}
def test_unit_function_base():
    """🔬 Test Function base class."""
    print("🔬 Unit Test: Function Base Class...")

    # Test initialization
    func = Function()
    assert func.inputs == []
    assert func.saved_tensors == []

    # Test save_for_backward
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    func.save_for_backward(tensor1, tensor2)
    assert len(func.saved_tensors) == 2
    assert func.saved_tensors[0] is tensor1
    assert func.saved_tensors[1] is tensor2

    print("✅ Function base class works correctly!")

if __name__ == "__main__":
    test_unit_function_base()

# %% [markdown]
"""
### Operation Functions - Implementing Gradient Rules

Now we'll implement specific operations that compute gradients correctly. Each operation has mathematical rules for how gradients flow backward.

**Gradient Flow Visualization:**
```
Addition (z = a + b):
    ∂z/∂a = 1    ∂z/∂b = 1

    a ──┐           grad_a ←──┐
        ├─[+]─→ z          ├─[+]←── grad_z
    b ──┘           grad_b ←──┘

Multiplication (z = a * b):
    ∂z/∂a = b    ∂z/∂b = a

    a ──┐           grad_a = grad_z * b
        ├─[×]─→ z
    b ──┘           grad_b = grad_z * a

Matrix Multiplication (Z = A @ B):
    ∂Z/∂A = grad_Z @ B.T
    ∂Z/∂B = A.T @ grad_Z

    A ──┐           grad_A = grad_Z @ B.T
        ├─[@]─→ Z
    B ──┘           grad_B = A.T @ grad_Z
```

Each operation stores the inputs it needs for computing gradients.
"""

# %% [markdown]
"""
### AddFunction - Gradient Rules for Addition

Addition is the simplest gradient operation: gradients flow unchanged to both inputs.

**Mathematical Principle:**
```
If z = a + b, then:
∂z/∂a = 1  (gradient of z w.r.t. a)
∂z/∂b = 1  (gradient of z w.r.t. b)

By chain rule:
∂Loss/∂a = ∂Loss/∂z × ∂z/∂a = grad_output × 1 = grad_output
∂Loss/∂b = ∂Loss/∂z × ∂z/∂b = grad_output × 1 = grad_output
```

**Broadcasting Challenge:**
When tensors have different shapes, NumPy broadcasts automatically in forward pass,
but we must "unbroadcast" gradients in backward pass to match original shapes.
"""

# %% nbgrader={"grade": false, "grade_id": "operation-functions", "solution": true}
class AddFunction(Function):
    """Gradient computation for tensor addition."""

    def forward(self, a, b):
        """
        Forward pass: compute a + b

        TODO: Implement addition forward pass
        """
        ### BEGIN SOLUTION
        # Save inputs for backward pass (shapes might be needed)
        self.save_for_backward(a, b)

        # Compute addition
        if isinstance(b, Tensor):
            result = a.data + b.data
        else:
            result = a.data + b

        return result
        ### END SOLUTION

    def backward(self, grad_output):
        """
        Backward pass: compute gradients for addition

        TODO: Implement addition backward pass

        MATH: If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1
        So: ∂loss/∂a = ∂loss/∂z × 1 = grad_output
            ∂loss/∂b = ∂loss/∂z × 1 = grad_output

        BROADCASTING CHALLENGE:
        If shapes differ, we need to sum gradients appropriately
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors

        # Gradient for 'a' - same shape as grad_output initially
        grad_a = grad_output

        # Gradient for 'b' - same as grad_output initially
        grad_b = grad_output

        # Handle broadcasting: if original shapes differed, sum gradients
        # For tensor + scalar case
        if not isinstance(b, Tensor):
            grad_b = np.sum(grad_output)
        else:
            # Handle shape differences due to broadcasting
            if a.shape != grad_output.shape:
                # Sum out added dimensions and squeeze
                grad_a = _handle_broadcasting_backward(grad_a, a.shape)

            if b.shape != grad_output.shape:
                grad_b = _handle_broadcasting_backward(grad_b, b.shape)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### MulFunction - Gradient Rules for Element-wise Multiplication

Element-wise multiplication follows the product rule of calculus.

**Mathematical Principle:**
```
If z = a * b (element-wise), then:
∂z/∂a = b  (gradient w.r.t. a equals the other input)
∂z/∂b = a  (gradient w.r.t. b equals the other input)

By chain rule:
∂Loss/∂a = grad_output * b
∂Loss/∂b = grad_output * a
```

**Visual Example:**
```
Forward:  a=[2,3] * b=[4,5] = z=[8,15]
Backward: grad_z=[1,1]
          grad_a = grad_z * b = [1,1] * [4,5] = [4,5]
          grad_b = grad_z * a = [1,1] * [2,3] = [2,3]
```
"""

class MulFunction(Function):
    """Gradient computation for tensor multiplication."""

    def forward(self, a, b):
        """
        Forward pass: compute a * b (element-wise)

        TODO: Implement multiplication forward pass
        """
        ### BEGIN SOLUTION
        self.save_for_backward(a, b)

        if isinstance(b, Tensor):
            result = a.data * b.data
        else:
            result = a.data * b

        return result
        ### END SOLUTION

    def backward(self, grad_output):
        """
        Backward pass: compute gradients for multiplication

        TODO: Implement multiplication backward pass

        MATH: If z = a * b, then:
        ∂z/∂a = b and ∂z/∂b = a
        So: ∂loss/∂a = grad_output * b
            ∂loss/∂b = grad_output * a
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors

        if isinstance(b, Tensor):
            grad_a = grad_output * b.data
            grad_b = grad_output * a.data

            # Handle broadcasting
            if a.shape != grad_output.shape:
                grad_a = _handle_broadcasting_backward(grad_a, a.shape)
            if b.shape != grad_output.shape:
                grad_b = _handle_broadcasting_backward(grad_b, b.shape)
        else:
            # b is a scalar
            grad_a = grad_output * b
            grad_b = np.sum(grad_output * a.data)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### MatmulFunction - Gradient Rules for Matrix Multiplication

Matrix multiplication has more complex gradient rules based on matrix calculus.

**Mathematical Principle:**
```
If Z = A @ B (matrix multiplication), then:
∂Z/∂A = grad_Z @ B.T
∂Z/∂B = A.T @ grad_Z
```

**Why These Rules Work:**
```
For element Z[i,j] = Σ_k A[i,k] * B[k,j]
∂Z[i,j]/∂A[i,k] = B[k,j]  ← This gives us grad_Z @ B.T
∂Z[i,j]/∂B[k,j] = A[i,k]  ← This gives us A.T @ grad_Z
```

**Dimension Analysis:**
```
Forward:  A(m×k) @ B(k×n) = Z(m×n)
Backward: grad_Z(m×n) @ B.T(n×k) = grad_A(m×k) ✓
          A.T(k×m) @ grad_Z(m×n) = grad_B(k×n) ✓
```
"""

class MatmulFunction(Function):
    """Gradient computation for matrix multiplication."""

    def forward(self, a, b):
        """
        Forward pass: compute a @ b (matrix multiplication)

        TODO: Implement matmul forward pass
        """
        ### BEGIN SOLUTION
        self.save_for_backward(a, b)
        result = np.dot(a.data, b.data)
        return result
        ### END SOLUTION

    def backward(self, grad_output):
        """
        Backward pass: compute gradients for matrix multiplication

        TODO: Implement matmul backward pass

        MATH: If Z = A @ B, then:
        ∂Z/∂A = grad_output @ B.T
        ∂Z/∂B = A.T @ grad_output
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors

        # Gradient w.r.t. a: grad_output @ b.T
        grad_a = np.dot(grad_output, b.data.T)

        # Gradient w.r.t. b: a.T @ grad_output
        grad_b = np.dot(a.data.T, grad_output)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### SumFunction - Gradient Rules for Reduction Operations

Sum operations reduce tensor dimensions, so gradients must be broadcast back.

**Mathematical Principle:**
```
If z = sum(a), then ∂z/∂a[i] = 1 for all i
Gradient is broadcasted from scalar result back to input shape.
```

**Gradient Broadcasting Examples:**
```
Case 1: Full sum
  Forward:  a=[1,2,3] → sum() → z=6 (scalar)
  Backward: grad_z=1 → broadcast → grad_a=[1,1,1]

Case 2: Axis sum
  Forward:  a=[[1,2],[3,4]] → sum(axis=0) → z=[4,6]
  Backward: grad_z=[1,1] → broadcast → grad_a=[[1,1],[1,1]]

Case 3: Keepdims
  Forward:  a=[[1,2],[3,4]] → sum(axis=0,keepdims=True) → z=[[4,6]]
  Backward: grad_z=[[1,1]] → broadcast → grad_a=[[1,1],[1,1]]
```
"""

class SumFunction(Function):
    """Gradient computation for tensor sum."""

    def forward(self, a, axis=None, keepdims=False):
        """
        Forward pass: compute tensor sum

        TODO: Implement sum forward pass
        """
        ### BEGIN SOLUTION
        self.save_for_backward(a)
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = a.shape

        result = np.sum(a.data, axis=axis, keepdims=keepdims)
        return result
        ### END SOLUTION

    def backward(self, grad_output):
        """
        Backward pass: compute gradients for sum

        TODO: Implement sum backward pass

        MATH: If z = sum(a), then ∂z/∂a[i] = 1 for all i
        So gradient is broadcast back to original shape
        """
        ### BEGIN SOLUTION
        # Sum distributes gradient to all input elements
        # Need to broadcast grad_output back to input shape

        if self.axis is None:
            # Summed all elements - broadcast scalar back to input shape
            grad_a = np.full(self.input_shape, grad_output)
        else:
            # Summed along specific axis - need to broadcast properly
            grad_a = grad_output

            # If keepdims=False, we need to expand the summed dimensions
            if not self.keepdims:
                if isinstance(self.axis, int):
                    grad_a = np.expand_dims(grad_a, self.axis)
                else:
                    for ax in sorted(self.axis):
                        grad_a = np.expand_dims(grad_a, ax)

            # Broadcast to input shape
            grad_a = np.broadcast_to(grad_a, self.input_shape)

        return grad_a
        ### END SOLUTION

def _handle_broadcasting_backward(grad, target_shape):
    """
    Helper function to handle gradient broadcasting.

    When forward pass used broadcasting, we need to sum gradients
    back to the original tensor's shape.
    """
    ### BEGIN SOLUTION
    # Start with the gradient
    result = grad

    # Sum out dimensions that were broadcasted (added dimensions)
    # If target has fewer dimensions, sum out the leading dimensions
    while len(result.shape) > len(target_shape):
        result = np.sum(result, axis=0)

    # For dimensions that were size 1 in target but expanded in grad
    for i, (grad_dim, target_dim) in enumerate(zip(result.shape, target_shape)):
        if target_dim == 1 and grad_dim > 1:
            result = np.sum(result, axis=i, keepdims=True)

    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: Operation Functions
This test validates our operation functions compute gradients correctly.
**What we're testing**: Forward and backward passes for each operation
**Why it matters**: These are the building blocks of autograd
**Expected**: Correct gradients that satisfy mathematical definitions
"""

# %% nbgrader={"grade": true, "grade_id": "test-operation-functions", "locked": true, "points": 15}
def test_unit_operation_functions():
    """🔬 Test operation functions."""
    print("🔬 Unit Test: Operation Functions...")

    # Test AddFunction
    add_func = AddFunction()
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = add_func.forward(a, b)
    expected = np.array([5, 7, 9])
    assert np.allclose(result, expected)

    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.backward(grad_output)
    assert np.allclose(grad_a, grad_output)
    assert np.allclose(grad_b, grad_output)

    # Test MulFunction
    mul_func = MulFunction()
    result = mul_func.forward(a, b)
    expected = np.array([4, 10, 18])
    assert np.allclose(result, expected)

    grad_a, grad_b = mul_func.backward(grad_output)
    assert np.allclose(grad_a, b.data)  # grad w.r.t a = b
    assert np.allclose(grad_b, a.data)  # grad w.r.t b = a

    # Test MatmulFunction
    matmul_func = MatmulFunction()
    a_mat = Tensor([[1, 2], [3, 4]])
    b_mat = Tensor([[5, 6], [7, 8]])
    result = matmul_func.forward(a_mat, b_mat)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(result, expected)

    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.backward(grad_output)
    assert grad_a.shape == a_mat.shape
    assert grad_b.shape == b_mat.shape

    print("✅ Operation functions work correctly!")

if __name__ == "__main__":
    test_unit_operation_functions()

# %% [markdown]
"""
### Enhancing Tensor with Autograd Capabilities

Now we'll enhance the existing Tensor class to use these gradient functions and build computation graphs automatically.

**Computation Graph Formation:**
```
Before Autograd:             After Autograd:
  x → operation → y           x → [Function] → y
                                     ↓
                               Stores operation
                               for backward pass
```

**The Enhancement Strategy:**
1. **Add backward() method** - Triggers gradient computation
2. **Enhance operations** - Replace simple ops with gradient-tracking versions
3. **Track computation graphs** - Each tensor remembers how it was created
4. **Maintain compatibility** - All existing code continues to work

**Critical Design Decision:**
We enhance the EXISTING Tensor class rather than creating a new one.
This means:
- ✅ All previous modules continue working unchanged
- ✅ No import changes needed
- ✅ Gradients are "opt-in" via requires_grad=True
- ✅ No confusion between Tensor types
"""

# %% [markdown]
"""
### The Backward Pass Algorithm

The backward() method implements reverse-mode automatic differentiation.

**Algorithm Visualization:**
```
Computation Graph (Forward):
  x₁ ──┐
       ├─[op₁]── z₁ ──┐
  x₂ ──┘              ├─[op₂]── y
  x₃ ──────[op₃]── z₂ ──┘

Gradient Flow (Backward):
  ∇x₁ ←──┐
         ├─[op₁.backward()]← ∇z₁ ←──┐
  ∇x₂ ←──┘                      ├─[op₂.backward()]← ∇y
  ∇x₃ ←────[op₃.backward()]← ∇z₂ ←──┘
```

**Backward Pass Steps:**
1. Start from output tensor (∇y = 1)
2. For each operation in reverse order:
   - Apply chain rule: ∇inputs = operation.backward(∇output)
   - Accumulate gradients (handle shared variables)
   - Continue to parent tensors
3. Gradients accumulate in tensor.grad attributes
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-enhancements", "solution": true}
def implement_tensor_backward_method():
    """
    Implement the backward method for the Tensor class.

    CRITICAL: We modify the Tensor class in place to activate gradient features.
    The dormant features are now brought to life!
    """

    def backward_implementation(self, gradient=None):
        """
        Compute gradients for this tensor and all tensors in its computation graph.

        TODO: Implement the backward pass

        APPROACH:
        1. Check if this tensor requires gradients
        2. Initialize gradient if starting point
        3. Traverse computation graph backwards
        4. Apply chain rule at each step

        EXAMPLE:
        >>> x = Tensor([2.0], requires_grad=True)
        >>> y = x * 3
        >>> y.backward()
        >>> print(x.grad)  # Should be [3.0]
        """
        ### BEGIN SOLUTION
        if not self.requires_grad:
            return

        # Initialize gradient if this is the starting point
        if gradient is None:
            if self.data.shape == ():
                # Scalar tensor
                gradient = np.array(1.0)
            else:
                # Non-scalar: gradient should be ones of same shape
                gradient = np.ones_like(self.data)

        # Accumulate gradient
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        # If this tensor has a gradient function, propagate backwards
        if hasattr(self, 'grad_fn') and self.grad_fn is not None:
            grads = self.grad_fn.backward(gradient)

            # grads could be a single gradient or tuple of gradients
            if not isinstance(grads, tuple):
                grads = (grads,)

            # Propagate to input tensors
            if hasattr(self.grad_fn, 'inputs'):
                for tensor, grad in zip(self.grad_fn.inputs, grads):
                    if isinstance(tensor, Tensor) and tensor.requires_grad:
                        tensor.backward(grad)
        ### END SOLUTION

    # Replace the placeholder backward method with the real implementation
    Tensor.backward = backward_implementation
    print("🚀 Tensor backward method activated!")

# Activate the backward method
implement_tensor_backward_method()

def create_gradient_tracking_tensor(data, requires_grad, grad_fn=None, inputs=None):
    """
    Helper function to create tensors with gradient tracking.

    This function helps operations create result tensors that properly
    track gradients and maintain the computation graph.
    """
    result = Tensor(data, requires_grad=requires_grad)

    if requires_grad and grad_fn is not None:
        result.grad_fn = grad_fn
        if inputs is not None:
            grad_fn.inputs = inputs

    return result

def enhance_tensor_operations():
    """
    Enhance existing Tensor operations to support gradient tracking.

    This modifies the existing methods to use gradient-tracking functions
    when requires_grad=True.
    """

    # Store original methods
    original_add = Tensor.__add__
    original_mul = Tensor.__mul__
    original_matmul = Tensor.matmul
    original_sum = Tensor.sum

    def gradient_aware_add(self, other):
        """
        Addition that tracks gradients when needed.
        """
        # Check if gradient tracking is needed
        requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)

        if requires_grad:
            # Use gradient-tracking version
            add_func = AddFunction()
            result_data = add_func.forward(self, other)
            inputs = [self, other] if isinstance(other, Tensor) else [self]
            return create_gradient_tracking_tensor(result_data, requires_grad, add_func, inputs)
        else:
            # Use original method (no gradient tracking)
            return original_add(self, other)

    def gradient_aware_mul(self, other):
        """
        Multiplication that tracks gradients when needed.
        """
        requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)

        if requires_grad:
            mul_func = MulFunction()
            result_data = mul_func.forward(self, other)
            inputs = [self, other] if isinstance(other, Tensor) else [self]
            return create_gradient_tracking_tensor(result_data, requires_grad, mul_func, inputs)
        else:
            return original_mul(self, other)

    def gradient_aware_matmul(self, other):
        """
        Matrix multiplication that tracks gradients when needed.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got {type(other)}")

        requires_grad = self.requires_grad or other.requires_grad

        if requires_grad:
            matmul_func = MatmulFunction()
            result_data = matmul_func.forward(self, other)
            inputs = [self, other]
            return create_gradient_tracking_tensor(result_data, requires_grad, matmul_func, inputs)
        else:
            return original_matmul(self, other)

    def gradient_aware_sum(self, axis=None, keepdims=False):
        """
        Sum that tracks gradients when needed.
        """
        if self.requires_grad:
            sum_func = SumFunction()
            result_data = sum_func.forward(self, axis, keepdims)
            inputs = [self]
            return create_gradient_tracking_tensor(result_data, self.requires_grad, sum_func, inputs)
        else:
            return original_sum(self, axis, keepdims)

    # Replace methods with gradient-aware versions
    Tensor.__add__ = gradient_aware_add
    Tensor.__mul__ = gradient_aware_mul
    Tensor.matmul = gradient_aware_matmul
    Tensor.sum = gradient_aware_sum

    print("🚀 Tensor operations enhanced with gradient tracking!")

# Enhance the operations
enhance_tensor_operations()

# %% [markdown]
"""
### 🔬 Unit Test: Tensor Autograd Enhancement
This test validates our enhanced Tensor class computes gradients correctly.
**What we're testing**: Gradient computation and chain rule implementation
**Why it matters**: This is the core of automatic differentiation
**Expected**: Correct gradients for various operations and computation graphs
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-autograd", "locked": true, "points": 20}
def test_unit_tensor_autograd():
    """🔬 Test Tensor autograd enhancement."""
    print("🔬 Unit Test: Tensor Autograd Enhancement...")

    # Test simple gradient computation
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1  # z = 3x + 1, so dz/dx = 3

    z.backward()
    assert np.allclose(x.grad, [3.0]), f"Expected [3.0], got {x.grad}"

    # Test matrix multiplication gradients
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # 1x2
    b = Tensor([[3.0], [4.0]], requires_grad=True)  # 2x1
    c = a.matmul(b)  # 1x1, result = [[11.0]]

    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f"Expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(b.grad, [[1.0], [2.0]]), f"Expected [[1.0], [2.0]], got {b.grad}"

    # Test computation graph with multiple operations
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2      # y = [2, 4]
    z = y.sum()    # z = 6

    z.backward()
    assert np.allclose(x.grad, [2.0, 2.0]), f"Expected [2.0, 2.0], got {x.grad}"

    print("✅ Tensor autograd enhancement works correctly!")

if __name__ == "__main__":
    test_unit_tensor_autograd()

# %% [markdown]
"""
## 4. Integration: Building Complex Computation Graphs

Let's test how our autograd system handles complex neural network computations.

### Complex Computation Graph Example

Neural networks create complex computation graphs with shared parameters and multiple paths.

**Detailed Neural Network Computation Graph:**
```
Forward Pass with Function Tracking:
                    x (input)
                    │ requires_grad=True
           ┌────────▼────────┐
           │ MatmulFunction  │ stores: (x, W₁)
           │   h₁ = x @ W₁   │
           └────────┬────────┘
                    │ grad_fn=MatmulFunction
           ┌────────▼────────┐
           │  AddFunction    │ stores: (h₁, b₁)
           │  z₁ = h₁ + b₁   │
           └────────┬────────┘
                    │ grad_fn=AddFunction
           ┌────────▼────────┐
           │  ReLU (manual)  │ Note: We'll implement
           │ a₁ = max(0,z₁)  │ ReLUFunction later
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ MatmulFunction  │ stores: (a₁, W₂)
           │   h₂ = a₁ @ W₂  │
           └────────┬────────┘
                    │ grad_fn=MatmulFunction
           ┌────────▼────────┐
           │  AddFunction    │ stores: (h₂, b₂)
           │   y = h₂ + b₂   │ (final output)
           └─────────────────┘

Backward Pass Chain Rule Application:
                   ∇x ←─────────────────────────────┐
                                                     │
    ┌─────────────────────────────────────────────────────────┐
    │ MatmulFunction.backward(∇h₁):                           │
    │   ∇x = ∇h₁ @ W₁.T                                      │
    │   ∇W₁ = x.T @ ∇h₁                                      │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │ AddFunction.backward(∇z₁):                              │
    │   ∇h₁ = ∇z₁  (gradient passes through unchanged)       │
    │   ∇b₁ = ∇z₁                                            │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │ Manual ReLU backward:                                   │
    │   ∇z₁ = ∇a₁ * (z₁ > 0)  (zero out negative gradients) │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │ MatmulFunction.backward(∇h₂):                           │
    │   ∇a₁ = ∇h₂ @ W₂.T                                     │
    │   ∇W₂ = a₁.T @ ∇h₂                                     │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │ AddFunction.backward(∇y):                               │
    │   ∇h₂ = ∇y  (gradient passes through unchanged)        │
    │   ∇b₂ = ∇y                                             │
    └─────────────────────────────────────────────────────────┘
```

**Key Autograd Concepts:**
1. **Function Chaining**: Each operation creates a Function that stores inputs
2. **Gradient Accumulation**: Multiple paths to a parameter accumulate gradients
3. **Automatic Traversal**: backward() walks the graph in reverse topological order
4. **Chain Rule**: Local gradients multiply according to calculus rules
"""


# %% [markdown]
"""
## 5. Systems Analysis: Memory and Performance of Autograd

Understanding the computational and memory costs of automatic differentiation.

### Autograd Memory Architecture

**Memory Layout Comparison:**
```
Forward-Only Mode:
┌─────────────┐
│ Parameters  │ 4N bytes (float32)
└─────────────┘

Autograd Mode:
┌─────────────┐
│ Parameters  │ 4N bytes
├─────────────┤
│ Gradients   │ 4N bytes (additional)
├─────────────┤
│ Graph Nodes │ Variable overhead
├─────────────┤
│ Activations │ Depends on graph depth
└─────────────┘
Total: ~2-3× forward memory
```

**Computation Graph Memory Growth:**
```
Shallow Network (3 layers):
  Graph: x → W₁ → ReLU → W₂ → ReLU → W₃ → loss
  Memory: Base + 3 × (weights + activations)

Deep Network (50 layers):
  Graph: x → [W₁...W₅₀] → loss
  Memory: Base + 50 × (weights + activations)

Gradient Checkpointing (optimization):
  Store only every K layers, recompute others
  Memory: Base + K × (weights + activations)
  Time: +20% compute, -80% memory
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-autograd-memory", "solution": true}
def analyze_autograd_memory():
    """📊 Analyze memory usage of autograd vs no-grad computation."""
    print("📊 Analyzing Autograd Memory Usage...")

    # Test different tensor sizes
    sizes = [100, 500, 1000]

    for size in sizes:
        # Forward-only computation
        x_no_grad = Tensor(np.random.randn(size, size), requires_grad=False)
        y_no_grad = Tensor(np.random.randn(size, size), requires_grad=False)
        z_no_grad = x_no_grad.matmul(y_no_grad)

        # Forward + backward computation
        x_grad = Tensor(np.random.randn(size, size), requires_grad=True)
        y_grad = Tensor(np.random.randn(size, size), requires_grad=True)
        z_grad = x_grad.matmul(y_grad)

        # Memory analysis
        no_grad_elements = x_no_grad.size + y_no_grad.size + z_no_grad.size
        grad_elements = x_grad.size + y_grad.size + z_grad.size
        grad_storage = x_grad.size + y_grad.size  # For gradients

        print(f"Size {size}×{size}:")
        print(f"  No grad: {no_grad_elements:,} elements")
        print(f"  With grad: {grad_elements + grad_storage:,} elements")
        print(f"  Memory overhead: {grad_storage / no_grad_elements:.1%}")

    print("\n💡 Autograd Memory Pattern:")
    print("- Each parameter tensor needs gradient storage (2× memory)")
    print("- Computation graph nodes add overhead")
    print("- Trade-off: 2× memory for automatic gradients")

# Function defined above, will be called in main block

# %% nbgrader={"grade": false, "grade_id": "analyze-gradient-computation", "solution": true}
def analyze_gradient_computation():
    """📊 Analyze computational cost of gradient computation."""
    print("📊 Analyzing Gradient Computation Cost...")

    import time

    # Test computation times
    size = 500
    x = Tensor(np.random.randn(size, size), requires_grad=True)
    y = Tensor(np.random.randn(size, size), requires_grad=True)

    # Time forward pass
    start_time = time.time()
    z = x.matmul(y)
    forward_time = time.time() - start_time

    # Time backward pass
    start_time = time.time()
    z.backward()
    backward_time = time.time() - start_time

    print(f"Matrix size: {size}×{size}")
    print(f"Forward pass: {forward_time:.4f}s")
    print(f"Backward pass: {backward_time:.4f}s")
    print(f"Backward/Forward ratio: {backward_time/forward_time:.1f}×")

    print(f"\n💡 Gradient Computation Analysis:")
    print(f"- Forward: O(n³) matrix multiplication")
    print(f"- Backward: 2× O(n³) operations (gradients for both inputs)")
    print(f"- Total training cost: ~3× forward-only computation")

# Function defined above, will be called in main block

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 25}
def test_module():
    """
    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Autograd works for complex computation graphs
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_function_base()
    test_unit_operation_functions()
    test_unit_tensor_autograd()

    print("\nRunning integration scenarios...")

    # Test 1: Multi-layer computation graph
    print("🔬 Integration Test: Multi-layer Neural Network...")

    # Create a 3-layer computation: x -> Linear -> Linear -> Linear -> loss
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    W1 = Tensor([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]], requires_grad=True)
    b1 = Tensor([[0.1, 0.2, 0.3]], requires_grad=True)

    # First layer
    h1 = x.matmul(W1) + b1
    assert h1.shape == (1, 3)
    assert h1.requires_grad == True

    # Second layer
    W2 = Tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    h2 = h1.matmul(W2)
    assert h2.shape == (1, 1)

    # Compute simple loss (just square the output for testing)
    loss = h2 * h2

    # Backward pass
    loss.backward()

    # Verify all parameters have gradients
    assert x.grad is not None
    assert W1.grad is not None
    assert b1.grad is not None
    assert W2.grad is not None
    assert x.grad.shape == x.shape
    assert W1.grad.shape == W1.shape

    print("✅ Multi-layer neural network gradients work!")

    # Test 2: Gradient accumulation
    print("🔬 Integration Test: Gradient Accumulation...")

    x = Tensor([2.0], requires_grad=True)

    # First computation
    y1 = x * 3
    y1.backward()
    first_grad = x.grad.copy()

    # Second computation (should accumulate)
    y2 = x * 5
    y2.backward()

    assert np.allclose(x.grad, first_grad + 5.0), "Gradients should accumulate"
    print("✅ Gradient accumulation works!")

    # Test 3: Complex mathematical operations
    print("🔬 Integration Test: Complex Operations...")

    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

    # Complex computation: ((a @ b) + a) * b
    temp1 = a.matmul(b)  # Matrix multiplication
    temp2 = temp1 + a    # Addition
    result = temp2 * b   # Element-wise multiplication
    final = result.sum() # Sum reduction

    final.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    print("✅ Complex mathematical operations work!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 05_autograd")

# Test function defined above, will be called in main block

# %%
# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Autograd Engine

Congratulations! You've built the gradient engine that makes neural networks learn!

### Key Accomplishments
- Implemented Function base class for tracking differentiable operations
- Enhanced existing Tensor class with backward() method (no new classes!)
- Built computation graph tracking for automatic differentiation
- Created operation functions (Add, Mul, Matmul, Sum) with correct gradients
- Tested complex multi-layer computation graphs with gradient propagation
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your autograd implementation enables optimization! The dormant gradient features from Module 01 are now fully active. Every tensor can track gradients, every operation builds computation graphs, and backward() computes gradients automatically.

Export with: `tito module complete 05_autograd`

**Next**: Module 06 will add optimizers (SGD, Adam) that use these gradients to actually train neural networks!
"""
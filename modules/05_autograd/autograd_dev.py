# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 05: Autograd ⚡ - The Gradient Engine

Welcome to Module 05! Today you'll awaken the gradient engine and unlock automatic differentiation.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor operations, activations, layers, and loss functions  
**You'll Build**: The autograd system that computes gradients automatically  
**You'll Enable**: Learning! Training! The ability to optimize neural networks!

**Connection Map**:
```
Modules 01-04 → Autograd → Training (Module 06-07)
(forward pass) (backward pass) (learning loops)
```

## Learning Objectives ⭐⭐
By the end of this module, you will:
1. **Enhance Tensor** with automatic differentiation capabilities
2. **Build computation graphs** that track operations for gradient flow
3. **Implement backward()** method for reverse-mode differentiation
4. **Create Function classes** for operation-specific gradient rules
5. **Test gradient correctness** with mathematical validation

**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/05_autograd/autograd_dev.py`  
**Building Side:** Code exports to `tinytorch.core.autograd`

```python
# How to use this module:
from tinytorch.core.autograd import Function, enable_autograd
```

**Why this matters:**
- **Learning:** Complete autograd system enabling automatic differentiation
- **Production:** PyTorch-style computational graph and backward pass
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Enhances existing Tensor without breaking anything

Let's build the gradient engine that makes neural networks learn! 🚀
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.autograd
#| export

import numpy as np
from typing import Optional, List, Tuple

from tinytorch.core.tensor import Tensor

# %% [markdown]
"""
## 1. Introduction: What is Automatic Differentiation?

Automatic differentiation (autograd) is the magic that makes neural networks learn. Instead of manually computing gradients for every parameter, autograd tracks operations and automatically computes gradients via the chain rule.

### The Challenge
In previous modules, you implemented layers and loss functions. To train a model, you need:
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
│ • saved_tensors    ← Store data     │
│ • apply()          ← Compute grads  │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Backward│ │Backward│ │Backward│ │Backward│
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
- They implement gradient computation via apply()
- They connect to form computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.apply() → grad_inputs
```

This pattern enables the chain rule to flow gradients through complex computations.
"""

# %% nbgrader={"grade": false, "grade_id": "function-base", "solution": true}
#| export
class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.
    
    **Key Concepts:**
    - **saved_tensors**: Store inputs needed for backward pass
    - **apply()**: Compute gradients using chain rule
    - **next_functions**: Track computation graph connections
    
    **Example Usage:**
    ```python
    class AddBackward(Function):
        def apply(self, grad_output):
            # Addition distributes gradients equally
            return grad_output, grad_output
    ```
    """

    def __init__(self, *tensors):
        """
        Initialize function with input tensors.
        
        Args:
            *tensors: Input tensors that will be saved for backward pass
        """
        self.saved_tensors = tensors
        self.next_functions = []

        # Build computation graph connections
        for t in tensors:
            if isinstance(t, Tensor) and t.requires_grad:
                if hasattr(t, '_grad_fn'):
                    self.next_functions.append(t._grad_fn)

    def apply(self, grad_output):
        """
        Compute gradients for inputs.
        
        Args:
            grad_output: Gradient flowing backward from the output
            
        Returns:
            Tuple of gradients for each input tensor
            
        **Must be implemented by subclasses**
        """
        raise NotImplementedError("Each Function must implement apply() method")

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
### AddBackward - Gradient Rules for Addition

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

# %% nbgrader={"grade": false, "grade_id": "add-backward", "solution": true}
#| export
class AddBackward(Function):
    """
    Gradient computation for tensor addition.
    
    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1
    
    **Key Insight:** Addition distributes gradients equally to both inputs.
    The gradient flowing backward is passed unchanged to each input.
    
    **Broadcasting Handling:** When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.
    """

    def apply(self, grad_output):
        """
        Compute gradients for addition.
        
        Args:
            grad_output: Gradient flowing backward from output
            
        Returns:
            Tuple of (grad_a, grad_b) for the two inputs
            
        **Mathematical Foundation:**
        - ∂(a+b)/∂a = 1 → grad_a = grad_output
        - ∂(a+b)/∂b = 1 → grad_b = grad_output
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        # Gradient for second input  
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output

        return grad_a, grad_b

# %% [markdown]
"""
### MulBackward - Gradient Rules for Element-wise Multiplication

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

# %% nbgrader={"grade": false, "grade_id": "mul-backward", "solution": true}
#| export
class MulBackward(Function):
    """
    Gradient computation for tensor multiplication.
    
    **Mathematical Rule:** If z = a * b, then ∂z/∂a = b and ∂z/∂b = a
    
    **Key Insight:** Each input's gradient equals the gradient output 
    multiplied by the OTHER input's value (product rule).
    
    **Applications:** Used in weight scaling, attention mechanisms,
    and anywhere element-wise multiplication occurs.
    """

    def apply(self, grad_output):
        """
        Compute gradients for multiplication.
        
        Args:
            grad_output: Gradient flowing backward from output
            
        Returns:
            Tuple of (grad_a, grad_b) for the two inputs
            
        **Mathematical Foundation:**
        - ∂(a*b)/∂a = b → grad_a = grad_output * b
        - ∂(a*b)/∂b = a → grad_b = grad_output * a
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output * b
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b

        # Gradient for second input: grad_output * a
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data

        return grad_a, grad_b

# %%



# %% [markdown]
"""
### MatmulBackward - Gradient Rules for Matrix Multiplication

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

# %% nbgrader={"grade": false, "grade_id": "matmul-backward", "solution": true}
#| export
class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.
    
    **Mathematical Rule:** If Z = A @ B, then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z
    
    **Key Insight:** Matrix multiplication gradients involve transposing
    one input and multiplying with the gradient output.
    
    **Applications:** Core operation in neural networks for weight updates
    in linear layers, attention mechanisms, and transformers.
    """

    def apply(self, grad_output):
        """
        Compute gradients for matrix multiplication.
        
        Args:
            grad_output: Gradient flowing backward from output
            
        Returns:
            Tuple of (grad_a, grad_b) for the two matrix inputs
            
        **Mathematical Foundation:**
        - ∂(A@B)/∂A = grad_output @ B.T
        - ∂(A@B)/∂B = A.T @ grad_output
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output @ b.T
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = np.dot(grad_output, b.data.T)

        # Gradient for second input: a.T @ grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = np.dot(a.data.T, grad_output)

        return grad_a, grad_b

# %% [markdown]
"""
### SumBackward - Gradient Rules for Reduction Operations

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
```
"""

# %% nbgrader={"grade": false, "grade_id": "sum-backward", "solution": true}
#| export
class SumBackward(Function):
    """
    Gradient computation for tensor sum.
    
    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i
    
    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.
    
    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.
    """

    def apply(self, grad_output):
        """
        Compute gradients for sum operation.
        
        Args:
            grad_output: Gradient flowing backward from output
            
        Returns:
            Tuple containing gradient for the input tensor
            
        **Mathematical Foundation:**
        - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Gradient is 1 for all elements, scaled by grad_output
            return np.ones_like(tensor.data) * grad_output,
        return None,

# %%



# %%



# %% [markdown]
"""
### 🔬 Unit Test: Function Classes
This test validates our Function classes compute gradients correctly.
**What we're testing**: Forward and backward passes for each operation
**Why it matters**: These are the building blocks of autograd
**Expected**: Correct gradients that satisfy mathematical definitions
"""

# %% nbgrader={"grade": true, "grade_id": "test-function-classes", "locked": true, "points": 15}
def test_unit_function_classes():
    """🔬 Test Function classes."""
    print("🔬 Unit Test: Function Classes...")

    # Test AddBackward
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    add_func = AddBackward(a, b)
    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.apply(grad_output)
    assert np.allclose(grad_a, grad_output), f"AddBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, grad_output), f"AddBackward grad_b failed: {grad_b}"

    # Test MulBackward
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(grad_output)
    assert np.allclose(grad_a, b.data), f"MulBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, a.data), f"MulBackward grad_b failed: {grad_b}"

    # Test MatmulBackward
    a_mat = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b_mat = Tensor([[5, 6], [7, 8]], requires_grad=True)
    matmul_func = MatmulBackward(a_mat, b_mat)
    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape, f"MatmulBackward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape, f"MatmulBackward grad_b shape: {grad_b.shape}"

    print("✅ Function classes work correctly!")

if __name__ == "__main__":
    test_unit_function_classes()

# %% [markdown]
"""
## 4. Enhancing Tensor with Autograd Capabilities

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
### The enable_autograd() Function

This function is the magic that brings gradients to life! It enhances the existing Tensor class with autograd capabilities by:

1. **Monkey-patching operations** - Replaces `__add__`, `__mul__`, etc. with gradient-aware versions
2. **Adding backward() method** - Implements reverse-mode automatic differentiation
3. **Maintaining compatibility** - All existing code continues to work unchanged

**The Pattern:**
```
Original: x + y → simple addition
Enhanced: x + y → addition + gradient tracking (if requires_grad=True)
```

This approach follows PyTorch 2.0 style - clean, modern, and educational.
"""

# %% nbgrader={"grade": false, "grade_id": "relu-backward", "solution": true}
#| export
class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation.
    
    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """
    
    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)
    
    def apply(self, grad_output):
        """Compute gradient for ReLU."""
        tensor, = self.saved_tensors
        
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad,
        return None,

# %%



# %% nbgrader={"grade": false, "grade_id": "sigmoid-backward", "solution": true}
#| export
class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation.
    
    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    """
    
    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.
        
        Args:
            input_tensor: Original input to sigmoid
            output_tensor: Output of sigmoid (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
    
    def apply(self, grad_output):
        """Compute gradient for sigmoid."""
        tensor, = self.saved_tensors
        
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad,
        return None,


# %% nbgrader={"grade": false, "grade_id": "mse-backward", "solution": true}
#| export
class MSEBackward(Function):
    """
    Gradient computation for Mean Squared Error Loss.
    
    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
    """
    
    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)
    
    def apply(self, grad_output):
        """Compute gradient for MSE loss."""
        predictions, = self.saved_tensors
        
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples
            
            return grad * grad_output,
        return None,


# %% nbgrader={"grade": false, "grade_id": "bce-backward", "solution": true}
#| export
class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.
    
    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
    """
    
    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)
    
    def apply(self, grad_output):
        """Compute gradient for BCE loss."""
        predictions, = self.saved_tensors
        
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = 1e-7
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data
            
            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)
            
            return grad * grad_output,
        return None,


# %% nbgrader={"grade": false, "grade_id": "ce-backward", "solution": true}
#| export
class CrossEntropyBackward(Function):
    """
    Gradient computation for Cross-Entropy Loss.
    
    CrossEntropy: L = -mean(log_softmax(logits)[targets])
    
    The gradient with respect to logits is remarkably elegant:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N
    
    This is one of the most beautiful results in machine learning:
    - The gradient is simply the difference between predictions and targets
    - It naturally scales with how wrong we are
    - It's numerically stable when computed via softmax
    """
    
    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]
    
    def apply(self, grad_output):
        """Compute gradient for cross-entropy loss."""
        logits, = self.saved_tensors
        
        if isinstance(logits, Tensor) and logits.requires_grad:
            # Compute softmax probabilities
            # Using stable softmax: subtract max for numerical stability
            logits_data = logits.data
            max_logits = np.max(logits_data, axis=1, keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Create one-hot encoding of targets
            one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.targets_data] = 1.0
            
            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size
            
            return grad * grad_output,
        return None,


# %% nbgrader={"grade": false, "grade_id": "enable-autograd", "solution": true}
#| export
def enable_autograd():
    """
    Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.

    **What it does:**
    - Replaces Tensor operations with gradient-tracking versions
    - Adds backward() method for reverse-mode differentiation
    - Enables computation graph building
    - Maintains full backward compatibility

    **After calling this:**
    - Tensor operations will track computation graphs
    - backward() method becomes available
    - Gradients will flow through operations
    - requires_grad=True enables tracking per tensor

    **Example:**
    ```python
    enable_autograd()  # Call once
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    y.backward()
    print(x.grad)  # [3.0]
    ```
    """

    # Check if already enabled
    if hasattr(Tensor, '_autograd_enabled'):
        print("⚠️ Autograd already enabled")
        return

    # Store original operations
    _original_add = Tensor.__add__
    _original_mul = Tensor.__mul__
    _original_matmul = Tensor.matmul if hasattr(Tensor, 'matmul') else None

    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """
        Addition with gradient tracking.
        
        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Call original operation
        result = _original_add(self, other)

        # Track gradient if needed
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def tracked_mul(self, other):
        """
        Multiplication with gradient tracking.
        
        Enhances the original __mul__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Convert scalar to Tensor if needed for consistency
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other

        # Call original operation
        result = _original_mul(self, other)

        # Track gradient if needed
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def tracked_matmul(self, other):
        """
        Matrix multiplication with gradient tracking.
        
        Enhances the original matmul method to build computation graphs
        when requires_grad=True for any input.
        """
        if _original_matmul:
            result = _original_matmul(self, other)
        else:
            # Fallback if matmul doesn't exist
            result = Tensor(np.dot(self.data, other.data))

        # Track gradient if needed
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """
        Sum operation with gradient tracking.
        
        Creates a new sum method that builds computation graphs
        when requires_grad=True.
        """
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = SumBackward(self)

        return result

    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.
        
        **Algorithm:**
        1. Initialize gradient if not provided (for scalar outputs)
        2. Accumulate gradient in self.grad
        3. If this tensor has a _grad_fn, call it to propagate gradients
        4. Recursively call backward() on parent tensors
        
        **Example:**
        ```python
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        y.backward()  # Computes gradients for x
        print(x.grad)  # [3.0]
        ```
        """
        # Only compute gradients if required
        if not self.requires_grad:
            return

        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError("backward() requires gradient for non-scalar outputs")

        # Initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        # Handle broadcasting: sum gradient to match self.data shape
        # This happens when operations broadcast tensors (e.g., adding bias to batch)
        if gradient.shape != self.grad.shape:
            # Step 1: Remove extra leading dimensions added during forward pass
            # Example: gradient (batch_size, features) → self.grad (features,)
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)
            
            # Step 2: Sum over dimensions that were size-1 in original tensor
            # Example: bias with shape (1,) broadcast to (batch_size,) during forward
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)
        
        self.grad += gradient

        # Propagate gradients through computation graph
        if hasattr(self, '_grad_fn') and self._grad_fn:
            grads = self._grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(self._grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad)

    def zero_grad(self):
        """
        Reset gradients to zero.
        
        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        """
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__mul__ = tracked_mul
    Tensor.matmul = tracked_matmul
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from tinytorch.core.activations import Sigmoid, ReLU
        from tinytorch.core.losses import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss
        
        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward
        
        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)
            
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)
            
            return result
        
        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)
            
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)
            
            return result
        
        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Compute BCE loss
            eps = 1e-7
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)
            bce_loss = np.mean(bce_per_sample)
            
            result = Tensor(bce_loss)
            
            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)
            
            return result
        
        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Compute MSE loss
            diff = predictions.data - targets.data
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)
            
            result = Tensor(mse)
            
            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)
            
            return result
        
        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            from tinytorch.core.losses import log_softmax
            
            # Compute log-softmax for numerical stability
            log_probs = log_softmax(logits, dim=-1)
            
            # Select log-probabilities for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
            
            # Return negative mean
            ce_loss = -np.mean(selected_log_probs)
            
            result = Tensor(ce_loss)
            
            if logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)
            
            return result
        
        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward
        
    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    print("✅ Autograd enabled! Tensors now track gradients.")
    print("   - Operations build computation graphs")
    print("   - backward() computes gradients")
    print("   - requires_grad=True enables tracking")

# Auto-enable when module is imported
enable_autograd()

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
    test_unit_function_classes()
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

### Key Accomplishments ⭐⭐
- **Enhanced Tensor class** with backward() method (no new wrapper classes!)
- **Built computation graph tracking** for automatic differentiation
- **Implemented Function classes** (Add, Mul, Matmul, Sum) with correct gradients
- **Created enable_autograd()** function that activates gradients globally
- **Tested complex multi-layer** computation graphs with gradient propagation
- **All tests pass** ✅ (validated by `test_module()`)

### Ready for Next Steps 🚀
Your autograd implementation enables optimization! The dormant gradient features from Module 01 are now fully active. Every tensor can track gradients, every operation builds computation graphs, and backward() computes gradients automatically.

**What you can do now:**
```python
# Create tensors with gradient tracking
x = Tensor([2.0], requires_grad=True)
W = Tensor([[0.5, 0.3]], requires_grad=True)

# Build computation graphs automatically
y = x.matmul(W.T)  # Forward pass
loss = (y - 1.0) ** 2  # Simple loss

# Compute gradients automatically
loss.backward()  # Magic happens here!

# Access gradients
print(f"x.grad: {x.grad}")  # Gradient w.r.t. x
print(f"W.grad: {W.grad}")  # Gradient w.r.t. W
```

Export with: `tito module complete 05_autograd`

**Next**: Module 06 will add optimizers (SGD, Adam) that use these gradients to actually train neural networks! 🎯

### 📈 Progress: Autograd ✓
```
✅ Module 01: Tensor (Foundation)
✅ Module 02: Activations (Non-linearities) 
✅ Module 03: Layers (Building blocks)
✅ Module 04: Losses (Training objectives)
✅ Module 05: Autograd (Gradient engine) ← YOU ARE HERE
🔄 Module 06: Optimizers (Learning algorithms)
🔄 Module 07: Training (Complete training loops)
```
"""

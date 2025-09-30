"""
Module 05: Autograd - Progressive Enhancement Pattern

This module enhances the existing Tensor class with automatic differentiation.
No Variable class - just pure Tensor with gradient tracking!

Following PyTorch 2.0 style - modern, clean, educational.
"""

#| default_exp core.autograd

import numpy as np
from typing import Optional, List, Tuple
import sys
import os

# Import the Tensor from Module 01
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

print("🔥 TinyTorch Autograd Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to enable automatic differentiation!")

#| export
class Function:
    """Base class for all differentiable operations.

    Each operation that needs gradients will have a corresponding Function
    that knows how to compute its gradient.
    """

    def __init__(self, *tensors):
        """Store input tensors for backward pass."""
        self.saved_tensors = tensors
        self.next_functions = []

        # Build computation graph connections
        for t in tensors:
            if isinstance(t, Tensor) and t.requires_grad:
                if hasattr(t, '_grad_fn'):
                    self.next_functions.append(t._grad_fn)

    def apply(self, *args):
        """Compute gradients. Must be implemented by subclasses."""
        raise NotImplementedError

#| export
class AddBackward(Function):
    """Backward function for addition."""

    def apply(self, grad_output):
        """Addition gradient: both inputs get the same gradient."""
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        # Gradient for second input
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output

        return grad_a, grad_b

#| export
class MulBackward(Function):
    """Backward function for multiplication."""

    def apply(self, grad_output):
        """Multiplication gradient: each input gets grad * other input."""
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

#| export
class MatmulBackward(Function):
    """Backward function for matrix multiplication."""

    def apply(self, grad_output):
        """Matrix multiplication gradient using chain rule."""
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output @ b.T
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = np.dot(grad_output, b.data.T)

        # Gradient for second input: a.T @ grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = np.dot(a.data.T, grad_output)

        return grad_a, grad_b

#| export
class SumBackward(Function):
    """Backward function for sum operation."""

    def apply(self, grad_output):
        """Sum gradient: distribute gradient to all elements."""
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Gradient is 1 for all elements, scaled by grad_output
            return np.ones_like(tensor.data) * grad_output,
        return None,

#| export
def enable_autograd():
    """Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.

    After calling this:
    - Tensor operations will track computation graphs
    - backward() method becomes available
    - Gradients will flow through operations
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
        """Addition with gradient tracking."""
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
        """Multiplication with gradient tracking."""
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
        """Matrix multiplication with gradient tracking."""
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
        """Sum operation with gradient tracking."""
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = SumBackward(self)

        return result

    def backward(self, gradient=None):
        """Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.
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
        self.grad += gradient

        # Propagate gradients through computation graph
        if hasattr(self, '_grad_fn') and self._grad_fn:
            grads = self._grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(self._grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad)

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__mul__ = tracked_mul
    Tensor.matmul = tracked_matmul
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Mark as enabled
    Tensor._autograd_enabled = True

    print("✅ Autograd enabled! Tensors now track gradients.")
    print("   - Operations build computation graphs")
    print("   - backward() computes gradients")
    print("   - requires_grad=True enables tracking")

# Auto-enable when module is imported
enable_autograd()

# Test the system
def test_autograd():
    """Test that autograd is working correctly."""
    print("\n🧪 Testing Autograd System...")

    # Test 1: Simple multiplication
    x = Tensor([[2.0]], requires_grad=True)
    y = x * 3
    y.backward()
    assert np.allclose(x.grad, [[3.0]]), f"Expected grad [[3.0]], got {x.grad}"
    print("✅ Test 1: Simple multiplication - PASSED")

    # Test 2: Addition
    x = Tensor([[1.0]], requires_grad=True)
    y = Tensor([[2.0]], requires_grad=True)
    z = x + y
    z.backward()
    assert np.allclose(x.grad, [[1.0]]), f"Expected x.grad [[1.0]], got {x.grad}"
    assert np.allclose(y.grad, [[1.0]]), f"Expected y.grad [[1.0]], got {y.grad}"
    print("✅ Test 2: Addition - PASSED")

    # Test 3: Chain of operations
    x = Tensor([[2.0]], requires_grad=True)
    y = x * 3  # y = 3x
    z = y + 1  # z = 3x + 1
    z.backward()
    assert np.allclose(x.grad, [[3.0]]), f"Expected grad [[3.0]], got {x.grad}"
    print("✅ Test 3: Chain of operations - PASSED")

    # Test 4: Matrix multiplication
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    W = Tensor([[3.0], [4.0]], requires_grad=True)
    y = x.matmul(W)  # y = 1*3 + 2*4 = 11
    y.backward()
    assert np.allclose(x.grad, [[3.0, 4.0]]), f"Expected x.grad [[3, 4]], got {x.grad}"
    assert np.allclose(W.grad, [[1.0], [2.0]]), f"Expected W.grad [[1], [2]], got {W.grad}"
    print("✅ Test 4: Matrix multiplication - PASSED")

    print("\n🎉 All autograd tests passed! The gradient engine is working!")

if __name__ == "__main__":
    test_autograd()
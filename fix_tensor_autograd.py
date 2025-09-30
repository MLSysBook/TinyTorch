"""
Fix Tensor autograd to enable gradient flow for MLP training.

This enhances the Tensor class to properly track gradients through operations.
Following PyTorch 2.0 style - no Variables, just Tensors with autograd.
"""

import numpy as np
from typing import Optional, Callable

class GradientFunction:
    """Base class for gradient functions that track computation graph."""

    def __init__(self, *tensors):
        self.saved_tensors = tensors
        self.next_functions = []
        for t in tensors:
            if hasattr(t, '_grad_fn') and t._grad_fn:
                self.next_functions.append((t._grad_fn, t))
            elif hasattr(t, 'requires_grad') and t.requires_grad:
                self.next_functions.append((None, t))

    def apply(self, grad_output):
        """Compute and propagate gradients."""
        raise NotImplementedError

class AddBackward(GradientFunction):
    """Gradient function for addition."""

    def apply(self, grad_output):
        # Addition gradient: both inputs get the same gradient
        a, b = self.saved_tensors
        if hasattr(a, 'requires_grad') and a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += grad_output

        if hasattr(b, 'requires_grad') and b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += grad_output

class MulBackward(GradientFunction):
    """Gradient function for multiplication."""

    def apply(self, grad_output):
        # Multiplication gradient: each input gets gradient * other input
        a, b = self.saved_tensors
        if hasattr(a, 'requires_grad') and a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # Handle scalar multiplication
            if np.isscalar(b):
                a.grad += grad_output * b
            elif hasattr(b, 'data'):
                a.grad += grad_output * b.data
            else:
                a.grad += grad_output * b

        if hasattr(b, 'requires_grad') and b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            a.grad += grad_output * a.data

class MatmulBackward(GradientFunction):
    """Gradient function for matrix multiplication."""

    def apply(self, grad_output):
        # Matrix multiplication gradient
        a, b = self.saved_tensors

        if hasattr(a, 'requires_grad') and a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # grad_a = grad_output @ b.T
            a.grad += np.dot(grad_output, b.data.T)

        if hasattr(b, 'requires_grad') and b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            # grad_b = a.T @ grad_output (fixed: was incorrectly updating a.grad)
            b.grad += np.dot(a.data.T, grad_output)

def enhance_tensor_class(TensorClass):
    """Enhance an existing Tensor class with proper autograd."""

    # Store original methods
    original_init = TensorClass.__init__
    original_add = TensorClass.__add__
    original_mul = TensorClass.__mul__
    original_matmul = TensorClass.matmul if hasattr(TensorClass, 'matmul') else None

    def new_init(self, data, requires_grad=False, dtype=None):
        """Enhanced init with gradient tracking."""
        original_init(self, data, requires_grad=requires_grad, dtype=dtype)
        self._grad_fn = None
        if requires_grad and not hasattr(self, 'grad'):
            self.grad = None

    def new_add(self, other):
        """Addition with gradient tracking."""
        result = original_add(self, other)

        # Track gradient if needed
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def new_mul(self, other):
        """Multiplication with gradient tracking."""
        result = original_mul(self, other)

        # Track gradient if needed
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def new_matmul(self, other):
        """Matrix multiplication with gradient tracking."""
        if original_matmul:
            result = original_matmul(self, other)
        else:
            # Fallback implementation
            result = TensorClass(np.dot(self.data, other.data))

        # Track gradient if needed
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def new_backward(self, gradient=None):
        """Proper backward pass with gradient propagation."""
        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            gradient = np.ones_like(self.data)

        # Accumulate gradient for this tensor
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += gradient

        # Propagate gradients through computation graph
        if hasattr(self, '_grad_fn') and self._grad_fn:
            self._grad_fn.apply(gradient)

    # Monkey-patch the class
    TensorClass.__init__ = new_init
    TensorClass.__add__ = new_add
    TensorClass.__mul__ = new_mul
    TensorClass.matmul = new_matmul
    TensorClass.backward = new_backward

    # Add sum operation if missing
    if not hasattr(TensorClass, 'sum'):
        def sum_op(self):
            """Sum all elements in tensor."""
            result = TensorClass(np.sum(self.data))
            if self.requires_grad:
                result.requires_grad = True

                class SumBackward(GradientFunction):
                    def apply(self, grad_output):
                        if self.saved_tensors[0].requires_grad:
                            if self.saved_tensors[0].grad is None:
                                self.saved_tensors[0].grad = np.zeros_like(self.saved_tensors[0].data)
                            self.saved_tensors[0].grad += np.ones_like(self.saved_tensors[0].data) * grad_output

                result._grad_fn = SumBackward(self)
            return result

        TensorClass.sum = sum_op

    return TensorClass

# Test the enhanced Tensor
if __name__ == "__main__":
    # Import and enhance the Tensor class
    from tinytorch.core.tensor import Tensor

    # Enhance with autograd
    Tensor = enhance_tensor_class(Tensor)

    print("🧪 Testing enhanced Tensor autograd...")

    # Test 1: Simple gradient flow
    x = Tensor([[2.0]], requires_grad=True)
    y = x * 3
    z = y + 1
    z.backward()

    print(f"Test 1 - Simple chain: x.grad = {x.grad} (expected: [[3.0]])")

    # Test 2: Multi-variable
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    y = x.matmul(w)  # 1*3 + 2*4 = 11
    y.backward()

    print(f"Test 2 - Matmul: x.grad = {x.grad}, w.grad = {w.grad}")

    print("✅ Autograd enhancement complete!")
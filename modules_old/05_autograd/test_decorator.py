#!/usr/bin/env python3
"""
Simple test of the decorator-based autograd implementation
"""
import sys
import os
import numpy as np

# Import the pure Tensor class from Module 01
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

def add_autograd(cls):
    """
    Decorator that adds gradient tracking to existing Tensor class.
    """
    # Store original methods from pure Tensor class
    original_init = cls.__init__
    original_add = cls.__add__
    original_mul = cls.__mul__
    original_sub = cls.__sub__ if hasattr(cls, '__sub__') else None

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

    def backward(self, gradient=None):
        """
        New method: Compute gradients via backpropagation.
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
    cls.backward = backward

    return cls

def test_decorator():
    """Test the decorator-based autograd implementation"""
    print("🧪 Testing Decorator-Based Autograd")
    print("=" * 40)

    # Apply decorator to enhance the pure Tensor class
    EnhancedTensor = add_autograd(Tensor)

    # Test 1: Backward compatibility (no gradients)
    print("Test 1: Backward Compatibility")
    x = EnhancedTensor([1.0, 2.0])
    y = EnhancedTensor([3.0, 4.0])
    z = x + y
    expected = np.array([4.0, 6.0])
    actual = z.data if hasattr(z, 'data') else z._data
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"
    print("✅ Pure tensor behavior preserved")

    # Test 2: Gradient tracking
    print("\nTest 2: Gradient Tracking")
    a = EnhancedTensor([2.0], requires_grad=True)
    b = EnhancedTensor([3.0], requires_grad=True)
    c = a * b  # c = 6.0

    # Backward pass
    c.backward()

    # Check gradients: dc/da = b = 3, dc/db = a = 2
    assert np.allclose(a.grad, [3.0]), f"Expected a.grad=[3.0], got {a.grad}"
    assert np.allclose(b.grad, [2.0]), f"Expected b.grad=[2.0], got {b.grad}"
    print("✅ Gradient computation works")

    # Test 3: Complex expression
    print("\nTest 3: Complex Expression")
    p = EnhancedTensor([4.0], requires_grad=True)
    q = EnhancedTensor([2.0], requires_grad=True)

    # f(p,q) = (p + q) * p = p² + pq
    sum_term = p + q  # p + q = 6
    result = sum_term * p  # (p + q) * p = 6 * 4 = 24

    result.backward()

    # Expected gradients: df/dp = 2p + q = 8 + 2 = 10, df/dq = p = 4
    expected_p_grad = 2 * 4.0 + 2.0  # 10.0
    expected_q_grad = 4.0            # 4.0

    assert np.allclose(p.grad, [expected_p_grad]), f"Expected p.grad=[{expected_p_grad}], got {p.grad}"
    assert np.allclose(q.grad, [expected_q_grad]), f"Expected q.grad=[{expected_q_grad}], got {q.grad}"
    print("✅ Complex expression gradients work")

    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 Decorator-based autograd implementation successful!")

if __name__ == "__main__":
    test_decorator()
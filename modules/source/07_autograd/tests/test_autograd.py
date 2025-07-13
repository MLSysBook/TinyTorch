"""
Test suite for the autograd module.
This tests the autograd implementations using mock classes to avoid cross-module dependencies.
"""

import pytest
import numpy as np
import sys
import os

# Add the module path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the autograd module directly
from autograd_dev import Variable, add, multiply, subtract, divide, relu_with_grad, sigmoid_with_grad


class MockTensor:
    """Mock Tensor class for testing autograd without dependencies."""
    
    def __init__(self, data):
        if isinstance(data, (int, float)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)
    
    @property
    def data(self):
        return self._data
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def size(self):
        return self._data.size
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        else:
            return MockTensor(self._data + other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data * other._data)
        else:
            return MockTensor(self._data * other)
    
    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data - other._data)
        else:
            return MockTensor(self._data - other)
    
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data / other._data)
        else:
            return MockTensor(self._data / other)
    
    def item(self):
        return self._data.item()


class TestVariableCreation:
    """Test Variable creation and basic properties."""
    
    def test_variable_from_scalar(self):
        """Test creating Variable from scalar values."""
        # Float scalar
        v1 = Variable(5.0)
        assert v1.shape == ()
        assert v1.size == 1
        assert v1.requires_grad == True
        assert v1.is_leaf == True
        assert v1.grad is None
        
        # Integer scalar
        v2 = Variable(42)
        assert v2.shape == ()
        assert v2.size == 1
        assert abs(v2.data.data.item() - 42.0) < 1e-6
    
    def test_variable_from_list(self):
        """Test creating Variable from list."""
        v = Variable([1.0, 2.0, 3.0])
        assert v.shape == (3,)
        assert v.size == 3
        assert v.requires_grad == True
        assert v.is_leaf == True
        np.testing.assert_array_almost_equal(v.data.data, [1.0, 2.0, 3.0])
    
    def test_variable_from_numpy(self):
        """Test creating Variable from numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = Variable(arr)
        assert v.shape == (2, 2)
        assert v.size == 4
        np.testing.assert_array_almost_equal(v.data.data, arr)
    
    def test_variable_requires_grad_flag(self):
        """Test requires_grad flag functionality."""
        v1 = Variable(5.0, requires_grad=True)
        assert v1.requires_grad == True
        
        v2 = Variable(5.0, requires_grad=False)
        assert v2.requires_grad == False
    
    def test_variable_with_grad_fn(self):
        """Test Variable with gradient function (non-leaf)."""
        def dummy_grad_fn(grad):
            pass
        
        v = Variable(5.0, requires_grad=True, grad_fn=dummy_grad_fn)
        assert v.requires_grad == True
        assert v.is_leaf == False
        assert v.grad_fn == dummy_grad_fn
    
    def test_variable_repr(self):
        """Test string representation of Variable."""
        v = Variable(5.0)
        repr_str = repr(v)
        assert 'Variable' in repr_str
        assert 'requires_grad' in repr_str


class TestBasicOperations:
    """Test basic arithmetic operations with gradient tracking."""
    
    def test_addition_operation(self):
        """Test addition operation and gradients."""
        x = Variable(2.0, requires_grad=True)
        y = Variable(3.0, requires_grad=True)
        z = add(x, y)
        
        # Test forward pass
        assert abs(z.data.data.item() - 5.0) < 1e-6
        assert z.requires_grad == True
        assert z.is_leaf == False
        
        # Test backward pass
        z.backward()
        assert abs(x.grad.data.data.item() - 1.0) < 1e-6
        assert abs(y.grad.data.data.item() - 1.0) < 1e-6
    
    def test_multiplication_operation(self):
        """Test multiplication operation and gradients."""
        x = Variable(2.0, requires_grad=True)
        y = Variable(3.0, requires_grad=True)
        z = multiply(x, y)
        
        # Test forward pass
        assert abs(z.data.data.item() - 6.0) < 1e-6
        assert z.requires_grad == True
        assert z.is_leaf == False
        
        # Test backward pass
        z.backward()
        assert abs(x.grad.data.data.item() - 3.0) < 1e-6  # dy/dx = y = 3
        assert abs(y.grad.data.data.item() - 2.0) < 1e-6  # dy/dy = x = 2
    
    def test_subtraction_operation(self):
        """Test subtraction operation and gradients."""
        x = Variable(5.0, requires_grad=True)
        y = Variable(3.0, requires_grad=True)
        z = subtract(x, y)
        
        # Test forward pass
        assert abs(z.data.data.item() - 2.0) < 1e-6
        assert z.requires_grad == True
        assert z.is_leaf == False
        
        # Test backward pass
        z.backward()
        assert abs(x.grad.data.data.item() - 1.0) < 1e-6   # dz/dx = 1
        assert abs(y.grad.data.data.item() - (-1.0)) < 1e-6  # dz/dy = -1
    
    def test_division_operation(self):
        """Test division operation and gradients."""
        x = Variable(6.0, requires_grad=True)
        y = Variable(2.0, requires_grad=True)
        z = divide(x, y)
        
        # Test forward pass
        assert abs(z.data.data.item() - 3.0) < 1e-6
        assert z.requires_grad == True
        assert z.is_leaf == False
        
        # Test backward pass
        z.backward()
        assert abs(x.grad.data.data.item() - 0.5) < 1e-6    # dz/dx = 1/y = 1/2
        assert abs(y.grad.data.data.item() - (-1.5)) < 1e-6  # dz/dy = -x/y² = -6/4
    
    def test_operations_with_constants(self):
        """Test operations with constant values."""
        x = Variable(2.0, requires_grad=True)
        
        # Addition with constant
        z1 = add(x, 3.0)
        assert abs(z1.data.data.item() - 5.0) < 1e-6
        z1.backward()
        assert abs(x.grad.data.data.item() - 1.0) < 1e-6
        
        # Reset gradient
        x.zero_grad()
        
        # Multiplication with constant
        z2 = multiply(x, 4.0)
        assert abs(z2.data.data.item() - 8.0) < 1e-6
        z2.backward()
        assert abs(x.grad.data.data.item() - 4.0) < 1e-6
    
    def test_no_grad_propagation(self):
        """Test that gradients don't propagate when requires_grad=False."""
        x = Variable(2.0, requires_grad=False)
        y = Variable(3.0, requires_grad=True)
        z = add(x, y)
        
        z.backward()
        assert x.grad is None  # No gradient for x
        assert abs(y.grad.data.data.item() - 1.0) < 1e-6


class TestChainRule:
    """Test chain rule implementation with complex expressions."""
    
    def test_simple_chain_rule(self):
        """Test f(x, y) = (x + y) * (x - y) = x² - y²."""
        x = Variable(3.0, requires_grad=True)
        y = Variable(2.0, requires_grad=True)
        
        # Forward pass
        sum_xy = add(x, y)
        diff_xy = subtract(x, y)
        result = multiply(sum_xy, diff_xy)
        
        # Check forward pass
        assert abs(result.data.data.item() - 5.0) < 1e-6  # (3+2)*(3-2) = 5
        
        # Backward pass
        result.backward()
        
        # Check gradients: df/dx = 2x = 6, df/dy = -2y = -4
        assert abs(x.grad.data.data.item() - 6.0) < 1e-6
        assert abs(y.grad.data.data.item() - (-4.0)) < 1e-6
    
    def test_cubic_function(self):
        """Test f(x) = x³ using x * x * x."""
        x = Variable(2.0, requires_grad=True)
        
        # Forward pass
        x_squared = multiply(x, x)
        x_cubed = multiply(x_squared, x)
        
        # Check forward pass
        assert abs(x_cubed.data.data.item() - 8.0) < 1e-6  # 2³ = 8
        
        # Backward pass
        x_cubed.backward()
        
        # Check gradient: df/dx = 3x² = 12
        assert abs(x.grad.data.data.item() - 12.0) < 1e-6
    
    def test_complex_expression(self):
        """Test f(x, y) = (x * y) + (x / y)."""
        x = Variable(4.0, requires_grad=True)
        y = Variable(2.0, requires_grad=True)
        
        # Forward pass
        product = multiply(x, y)
        quotient = divide(x, y)
        result = add(product, quotient)
        
        # Check forward pass: (4*2) + (4/2) = 8 + 2 = 10
        assert abs(result.data.data.item() - 10.0) < 1e-6
        
        # Backward pass
        result.backward()
        
        # Check gradients: df/dx = y + 1/y = 2 + 0.5 = 2.5
        #                  df/dy = x - x/y² = 4 - 4/4 = 3
        assert abs(x.grad.data.data.item() - 2.5) < 1e-6
        assert abs(y.grad.data.data.item() - 3.0) < 1e-6
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        x = Variable(2.0, requires_grad=True)
        
        # First computation
        y1 = multiply(x, 3.0)
        y1.backward()
        first_grad = x.grad.data.data.item()
        
        # Second computation (should accumulate)
        y2 = multiply(x, 4.0)
        y2.backward()
        second_grad = x.grad.data.data.item()
        
        # Gradient should accumulate: 3 + 4 = 7
        assert abs(second_grad - 7.0) < 1e-6
    
    def test_zero_grad_functionality(self):
        """Test zero_grad functionality."""
        x = Variable(2.0, requires_grad=True)
        y = multiply(x, 3.0)
        y.backward()
        
        # Check gradient exists
        assert x.grad is not None
        assert abs(x.grad.data.data.item() - 3.0) < 1e-6
        
        # Zero the gradient
        x.zero_grad()
        assert abs(x.grad.data.data.item() - 0.0) < 1e-6


class TestActivationGradients:
    """Test activation functions with gradient computation."""
    
    def test_relu_activation(self):
        """Test ReLU activation and its gradient."""
        # Test positive input
        x1 = Variable(2.0, requires_grad=True)
        y1 = relu_with_grad(x1)
        
        assert abs(y1.data.data.item() - 2.0) < 1e-6  # ReLU(2) = 2
        y1.backward()
        assert abs(x1.grad.data.data.item() - 1.0) < 1e-6  # gradient = 1 for x > 0
        
        # Test negative input
        x2 = Variable(-1.0, requires_grad=True)
        y2 = relu_with_grad(x2)
        
        assert abs(y2.data.data.item() - 0.0) < 1e-6  # ReLU(-1) = 0
        y2.backward()
        assert abs(x2.grad.data.data.item() - 0.0) < 1e-6  # gradient = 0 for x < 0
        
        # Test zero input
        x3 = Variable(0.0, requires_grad=True)
        y3 = relu_with_grad(x3)
        
        assert abs(y3.data.data.item() - 0.0) < 1e-6  # ReLU(0) = 0
        y3.backward()
        assert abs(x3.grad.data.data.item() - 0.0) < 1e-6  # gradient = 0 for x = 0
    
    def test_sigmoid_activation(self):
        """Test Sigmoid activation and its gradient."""
        # Test zero input
        x1 = Variable(0.0, requires_grad=True)
        y1 = sigmoid_with_grad(x1)
        
        assert abs(y1.data.data.item() - 0.5) < 1e-6  # sigmoid(0) = 0.5
        y1.backward()
        assert abs(x1.grad.data.data.item() - 0.25) < 1e-6  # gradient = 0.5 * 0.5 = 0.25
        
        # Test positive input
        x2 = Variable(2.0, requires_grad=True)
        y2 = sigmoid_with_grad(x2)
        
        expected_sigmoid = 1.0 / (1.0 + np.exp(-2.0))
        assert abs(y2.data.data.item() - expected_sigmoid) < 1e-6
        
        y2.backward()
        expected_grad = expected_sigmoid * (1.0 - expected_sigmoid)
        assert abs(x2.grad.data.data.item() - expected_grad) < 1e-6
        
        # Test negative input
        x3 = Variable(-1.0, requires_grad=True)
        y3 = sigmoid_with_grad(x3)
        
        expected_sigmoid = 1.0 / (1.0 + np.exp(1.0))
        assert abs(y3.data.data.item() - expected_sigmoid) < 1e-6
        
        y3.backward()
        expected_grad = expected_sigmoid * (1.0 - expected_sigmoid)
        assert abs(x3.grad.data.data.item() - expected_grad) < 1e-6
    
    def test_activation_chaining(self):
        """Test chaining activation functions."""
        x = Variable(1.0, requires_grad=True)
        
        # Chain: x -> ReLU -> Sigmoid
        relu_out = relu_with_grad(x)
        sigmoid_out = sigmoid_with_grad(relu_out)
        
        # Forward pass
        expected_relu = 1.0  # ReLU(1) = 1
        expected_sigmoid = 1.0 / (1.0 + np.exp(-1.0))  # sigmoid(1)
        
        assert abs(relu_out.data.data.item() - expected_relu) < 1e-6
        assert abs(sigmoid_out.data.data.item() - expected_sigmoid) < 1e-6
        
        # Backward pass
        sigmoid_out.backward()
        
        # Check that gradient flows through both activations
        assert x.grad is not None
        assert abs(x.grad.data.data.item()) > 1e-6  # Should have non-zero gradient


class TestNeuralNetworkScenarios:
    """Test autograd in realistic neural network scenarios."""
    
    def test_simple_linear_layer(self):
        """Test simple linear transformation: y = Wx + b."""
        # Input
        x = Variable(2.0, requires_grad=True)
        
        # Parameters
        w = Variable(0.5, requires_grad=True)
        b = Variable(0.1, requires_grad=True)
        
        # Forward pass
        linear_out = add(multiply(x, w), b)  # y = x*w + b = 2*0.5 + 0.1 = 1.1
        
        assert abs(linear_out.data.data.item() - 1.1) < 1e-6
        
        # Backward pass
        linear_out.backward()
        
        # Check gradients
        assert abs(x.grad.data.data.item() - 0.5) < 1e-6  # dy/dx = w = 0.5
        assert abs(w.grad.data.data.item() - 2.0) < 1e-6  # dy/dw = x = 2.0
        assert abs(b.grad.data.data.item() - 1.0) < 1e-6  # dy/db = 1 = 1.0
    
    def test_two_layer_network(self):
        """Test two-layer neural network."""
        # Input
        x = Variable(1.0, requires_grad=True)
        
        # Layer 1 parameters
        w1 = Variable(2.0, requires_grad=True)
        b1 = Variable(0.5, requires_grad=True)
        
        # Layer 2 parameters
        w2 = Variable(1.5, requires_grad=True)
        b2 = Variable(0.2, requires_grad=True)
        
        # Forward pass
        # Layer 1: h = x*w1 + b1 = 1*2 + 0.5 = 2.5
        h = add(multiply(x, w1), b1)
        # ReLU activation
        h_relu = relu_with_grad(h)  # ReLU(2.5) = 2.5
        # Layer 2: y = h*w2 + b2 = 2.5*1.5 + 0.2 = 3.95
        y = add(multiply(h_relu, w2), b2)
        
        assert abs(y.data.data.item() - 3.95) < 1e-6
        
        # Backward pass
        y.backward()
        
        # Check that all parameters have gradients
        assert x.grad is not None
        assert w1.grad is not None
        assert b1.grad is not None
        assert w2.grad is not None
        assert b2.grad is not None
        
        # Check specific gradient values
        assert abs(b2.grad.data.data.item() - 1.0) < 1e-6  # dy/db2 = 1
        assert abs(w2.grad.data.data.item() - 2.5) < 1e-6  # dy/dw2 = h_relu = 2.5
        assert abs(b1.grad.data.data.item() - 1.5) < 1e-6  # dy/db1 = w2 = 1.5
        assert abs(w1.grad.data.data.item() - 1.5) < 1e-6  # dy/dw1 = x * w2 = 1 * 1.5
        assert abs(x.grad.data.data.item() - 3.0) < 1e-6   # dy/dx = w1 * w2 = 2 * 1.5
    
    def test_loss_computation(self):
        """Test loss computation with gradients."""
        # Prediction and target
        pred = Variable(3.0, requires_grad=True)
        target = Variable(2.0, requires_grad=False)
        
        # Mean squared error: loss = (pred - target)²
        diff = subtract(pred, target)  # 3 - 2 = 1
        loss = multiply(diff, diff)    # 1² = 1
        
        assert abs(loss.data.data.item() - 1.0) < 1e-6
        
        # Backward pass
        loss.backward()
        
        # Check gradient: d_loss/d_pred = 2 * (pred - target) = 2 * 1 = 2
        assert abs(pred.grad.data.data.item() - 2.0) < 1e-6
        assert target.grad is None  # No gradient for target
    
    def test_batch_processing_simulation(self):
        """Test simulation of batch processing."""
        # Simulate batch of 3 samples
        x1 = Variable(1.0, requires_grad=True)
        x2 = Variable(2.0, requires_grad=True)
        x3 = Variable(3.0, requires_grad=True)
        
        # Shared parameters
        w = Variable(0.5, requires_grad=True)
        b = Variable(0.1, requires_grad=True)
        
        # Forward pass for each sample
        y1 = add(multiply(x1, w), b)  # 1*0.5 + 0.1 = 0.6
        y2 = add(multiply(x2, w), b)  # 2*0.5 + 0.1 = 1.1
        y3 = add(multiply(x3, w), b)  # 3*0.5 + 0.1 = 1.6
        
        # Compute batch loss (sum of individual losses)
        loss1 = multiply(y1, y1)  # 0.6² = 0.36
        loss2 = multiply(y2, y2)  # 1.1² = 1.21
        loss3 = multiply(y3, y3)  # 1.6² = 2.56
        
        batch_loss = add(add(loss1, loss2), loss3)  # 0.36 + 1.21 + 2.56 = 4.13
        
        assert abs(batch_loss.data.data.item() - 4.13) < 1e-6
        
        # Backward pass
        batch_loss.backward()
        
        # Check that gradients accumulated for shared parameters
        assert w.grad is not None
        assert b.grad is not None
        
        # w gradient should be sum of individual contributions
        # dL/dw = 2*y1*x1 + 2*y2*x2 + 2*y3*x3 = 2*(0.6*1 + 1.1*2 + 1.6*3) = 2*7.6 = 15.2
        expected_w_grad = 2 * (0.6*1 + 1.1*2 + 1.6*3)
        assert abs(w.grad.data.data.item() - expected_w_grad) < 1e-6
        
        # b gradient should be sum of individual contributions
        # dL/db = 2*y1 + 2*y2 + 2*y3 = 2*(0.6 + 1.1 + 1.6) = 2*3.3 = 6.6
        expected_b_grad = 2 * (0.6 + 1.1 + 1.6)
        assert abs(b.grad.data.data.item() - expected_b_grad) < 1e-6


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_division_handling(self):
        """Test division by zero handling."""
        x = Variable(1.0, requires_grad=True)
        y = Variable(0.0, requires_grad=True)
        
        # This should not crash but may produce inf/nan
        z = divide(x, y)
        
        # Check that the operation completes
        assert z.data.data.item() == np.inf or np.isnan(z.data.data.item())
    
    def test_large_gradient_values(self):
        """Test handling of large gradient values."""
        x = Variable(100.0, requires_grad=True)
        y = Variable(100.0, requires_grad=True)
        
        # Large multiplication
        z = multiply(x, y)  # 100 * 100 = 10000
        z.backward()
        
        # Gradients should be large but finite
        assert np.isfinite(x.grad.data.data.item())
        assert np.isfinite(y.grad.data.data.item())
        assert abs(x.grad.data.data.item() - 100.0) < 1e-6
        assert abs(y.grad.data.data.item() - 100.0) < 1e-6
    
    def test_very_small_values(self):
        """Test handling of very small values."""
        x = Variable(1e-10, requires_grad=True)
        y = Variable(2e-10, requires_grad=True)
        
        z = add(x, y)
        z.backward()
        
        # Gradients should still be computed correctly
        assert abs(x.grad.data.data.item() - 1.0) < 1e-6
        assert abs(y.grad.data.data.item() - 1.0) < 1e-6
    
    def test_mixed_requires_grad(self):
        """Test operations with mixed requires_grad settings."""
        x = Variable(2.0, requires_grad=True)
        y = Variable(3.0, requires_grad=False)
        
        z = multiply(x, y)
        
        # Result should require gradients
        assert z.requires_grad == True
        
        z.backward()
        
        # Only x should have gradients
        assert x.grad is not None
        assert y.grad is None
        assert abs(x.grad.data.data.item() - 3.0) < 1e-6


# Integration tests that combine multiple concepts
class TestIntegration:
    """Integration tests combining multiple autograd concepts."""
    
    def test_complete_training_step(self):
        """Test a complete training step simulation."""
        # Model parameters
        w1 = Variable(0.1, requires_grad=True)
        b1 = Variable(0.0, requires_grad=True)
        w2 = Variable(0.2, requires_grad=True)
        b2 = Variable(0.0, requires_grad=True)
        
        # Training data
        x = Variable(1.5, requires_grad=False)
        target = Variable(2.0, requires_grad=False)
        
        # Forward pass
        h1 = add(multiply(x, w1), b1)  # Linear layer 1
        h1_relu = relu_with_grad(h1)   # ReLU activation
        output = add(multiply(h1_relu, w2), b2)  # Linear layer 2
        
        # Loss computation (MSE)
        diff = subtract(output, target)
        loss = multiply(diff, diff)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        assert w1.grad is not None
        assert b1.grad is not None
        assert w2.grad is not None
        assert b2.grad is not None
        
        # Simulate parameter update (gradient descent)
        learning_rate = 0.01
        
        # Save old parameter values
        old_w1 = w1.data.data.item()
        old_b1 = b1.data.data.item()
        old_w2 = w2.data.data.item()
        old_b2 = b2.data.data.item()
        
        # Update parameters: param = param - lr * grad
        w1.data._data -= learning_rate * w1.grad.data.data
        b1.data._data -= learning_rate * b1.grad.data.data
        w2.data._data -= learning_rate * w2.grad.data.data
        b2.data._data -= learning_rate * b2.grad.data.data
        
        # Check that parameters actually changed
        assert abs(w1.data.data.item() - old_w1) > 1e-6
        assert abs(b1.data.data.item() - old_b1) > 1e-6
        assert abs(w2.data.data.item() - old_w2) > 1e-6
        assert abs(b2.data.data.item() - old_b2) > 1e-6
    
    def test_multi_output_gradients(self):
        """Test gradients when multiple outputs depend on same input."""
        x = Variable(2.0, requires_grad=True)
        
        # Create multiple outputs from same input
        y1 = multiply(x, 3.0)  # y1 = 3x
        y2 = multiply(x, x)    # y2 = x²
        
        # Combine outputs
        combined = add(y1, y2)  # combined = 3x + x²
        
        combined.backward()
        
        # Gradient should be sum of individual contributions
        # d(combined)/dx = d(3x)/dx + d(x²)/dx = 3 + 2x = 3 + 2*2 = 7
        assert abs(x.grad.data.data.item() - 7.0) < 1e-6
    
    def test_gradient_flow_through_complex_network(self):
        """Test gradient flow through a more complex network."""
        # Input
        x = Variable(1.0, requires_grad=True)
        
        # Create a diamond-shaped computation graph
        #     x
        #   /   \
        #  a     b
        #   \   /
        #     c
        
        a = multiply(x, 2.0)  # a = 2x
        b = add(x, 1.0)       # b = x + 1
        c = multiply(a, b)    # c = a * b = 2x * (x + 1) = 2x² + 2x
        
        # Expected: c = 2x² + 2x, so dc/dx = 4x + 2 = 4*1 + 2 = 6
        c.backward()
        
        assert abs(x.grad.data.data.item() - 6.0) < 1e-6
    
    def test_nested_function_composition(self):
        """Test deeply nested function composition."""
        x = Variable(2.0, requires_grad=True)
        
        # Create nested composition: f(g(h(x)))
        h = multiply(x, 2.0)        # h(x) = 2x
        g = add(h, 1.0)             # g(h(x)) = 2x + 1
        f = multiply(g, g)          # f(g(h(x))) = (2x + 1)²
        
        # Expected: f = (2x + 1)², so df/dx = 2(2x + 1) * 2 = 4(2x + 1) = 4(2*2 + 1) = 20
        f.backward()
        
        assert abs(x.grad.data.data.item() - 20.0) < 1e-6 
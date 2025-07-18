"""
Integration Tests - Tensor and Autograd

Tests real integration between Tensor and Autograd modules.
Uses actual TinyTorch components to verify they work together correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable, add, multiply


class TestTensorAutogradIntegration:
    """Test integration between Tensor and Autograd components."""
    
    def test_variable_wraps_real_tensors(self):
        """Test Variable properly wraps real Tensor objects."""
        # Create real tensor
        tensor_data = Tensor([1.0, 2.0, 3.0])
        
        # Wrap in Variable
        var = Variable(tensor_data, requires_grad=True)
        
        # Verify Variable properties
        assert isinstance(var.data, Tensor), "Variable should wrap a Tensor"
        assert var.requires_grad is True, "Variable should track gradients"
        assert var.grad is None, "Initial gradient should be None"
        
        # Verify tensor data is preserved
        np.testing.assert_array_equal(var.data.data, tensor_data.data)
        assert var.data.shape == tensor_data.shape
        assert var.data.dtype == tensor_data.dtype
    
    def test_add_operation_with_real_tensors(self):
        """Test addition operation with real tensor data."""
        # Create real tensor inputs
        a_tensor = Tensor([1.0, 2.0])
        b_tensor = Tensor([3.0, 4.0])
        
        # Create Variables
        a = Variable(a_tensor, requires_grad=True)
        b = Variable(b_tensor, requires_grad=True)
        
        # Test addition
        c = add(a, b)
        
        # Verify result
        assert isinstance(c, Variable), "Result should be a Variable"
        assert isinstance(c.data, Tensor), "Result data should be a Tensor"
        
        expected_data = np.array([4.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(c.data.data, expected_data, decimal=5)
        
        # Verify gradient tracking
        assert c.requires_grad is True, "Result should track gradients"
        assert c.grad_fn is not None, "Result should have gradient function"
    
    def test_multiply_operation_with_real_tensors(self):
        """Test multiplication operation with real tensor data."""
        # Create real tensor inputs
        a_tensor = Tensor([2.0, 3.0])
        b_tensor = Tensor([4.0, 5.0])
        
        # Create Variables
        a = Variable(a_tensor, requires_grad=True)
        b = Variable(b_tensor, requires_grad=True)
        
        # Test multiplication
        c = multiply(a, b)
        
        # Verify result
        assert isinstance(c, Variable), "Result should be a Variable"
        assert isinstance(c.data, Tensor), "Result data should be a Tensor"
        
        expected_data = np.array([8.0, 15.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(c.data.data, expected_data, decimal=5)
        
        # Verify gradient tracking
        assert c.requires_grad is True, "Result should track gradients"
        assert c.grad_fn is not None, "Result should have gradient function"
    
    def test_relu_with_real_tensors(self):
        """Test ReLU operation with real tensor data."""
        # Create real tensor with negative and positive values
        tensor_data = Tensor([-1.0, 0.0, 1.0, 2.0])
        var = Variable(tensor_data, requires_grad=True)
        
        # Apply ReLU
        output = relu_with_grad(var)
        
        # Verify result
        assert isinstance(output, Variable), "Result should be a Variable"
        assert isinstance(output.data, Tensor), "Result data should be a Tensor"
        
        expected_data = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.data.data, expected_data, decimal=5)
        
        # Verify gradient tracking
        assert output.requires_grad is True, "Result should track gradients"
        assert output.grad_fn is not None, "Result should have gradient function"
    
    def test_sigmoid_with_real_tensors(self):
        """Test Sigmoid operation with real tensor data."""
        # Create real tensor data
        tensor_data = Tensor([0.0, 1.0, -1.0])
        var = Variable(tensor_data, requires_grad=True)
        
        # Apply Sigmoid
        output = sigmoid_with_grad(var)
        
        # Verify result
        assert isinstance(output, Variable), "Result should be a Variable"
        assert isinstance(output.data, Tensor), "Result data should be a Tensor"
        
        # Verify sigmoid values (approximately)
        expected_data = np.array([0.5, 0.731, 0.269], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.data.data, expected_data, decimal=2)
        
        # Verify gradient tracking
        assert output.requires_grad is True, "Result should track gradients"
        assert output.grad_fn is not None, "Result should have gradient function"


class TestTensorAutogradBackwardPass:
    """Test backward pass integration with real tensors."""
    
    def test_simple_addition_backward(self):
        """Test backward pass through addition with real tensors."""
        # Create real tensor inputs
        a_tensor = Tensor([1.0, 2.0])
        b_tensor = Tensor([3.0, 4.0])
        
        # Create Variables
        a = Variable(a_tensor, requires_grad=True)
        b = Variable(b_tensor, requires_grad=True)
        
        # Forward pass
        c = add(a, b)
        
        # Create gradient tensor for backward pass
        grad_output = Variable(Tensor([1.0, 1.0]), requires_grad=False)
        
        # Backward pass
        c.backward(grad_output)
        
        # Verify gradients
        assert a.grad is not None, "Input 'a' should have gradient"
        assert b.grad is not None, "Input 'b' should have gradient"
        
        # For addition, gradients should be passed through unchanged
        expected_grad = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(a.grad.data.data, expected_grad, decimal=5)
        np.testing.assert_array_almost_equal(b.grad.data.data, expected_grad, decimal=5)
    
    def test_multiplication_backward(self):
        """Test backward pass through multiplication with real tensors."""
        # Create real tensor inputs
        a_tensor = Tensor([2.0, 3.0])
        b_tensor = Tensor([4.0, 5.0])
        
        # Create Variables
        a = Variable(a_tensor, requires_grad=True)
        b = Variable(b_tensor, requires_grad=True)
        
        # Forward pass
        c = multiply(a, b)
        
        # Create gradient tensor for backward pass
        grad_output = Variable(Tensor([1.0, 1.0]), requires_grad=False)
        
        # Backward pass
        c.backward(grad_output)
        
        # Verify gradients
        assert a.grad is not None, "Input 'a' should have gradient"
        assert b.grad is not None, "Input 'b' should have gradient"
        
        # For multiplication: grad_a = grad_output * b, grad_b = grad_output * a
        expected_grad_a = np.array([4.0, 5.0], dtype=np.float32)  # b values
        expected_grad_b = np.array([2.0, 3.0], dtype=np.float32)  # a values
        
        np.testing.assert_array_almost_equal(a.grad.data.data, expected_grad_a, decimal=5)
        np.testing.assert_array_almost_equal(b.grad.data.data, expected_grad_b, decimal=5)
    
    def test_relu_backward(self):
        """Test backward pass through ReLU with real tensors."""
        # Create real tensor with negative and positive values
        tensor_data = Tensor([-1.0, 0.0, 1.0, 2.0])
        var = Variable(tensor_data, requires_grad=True)
        
        # Forward pass
        output = relu_with_grad(var)
        
        # Create gradient tensor for backward pass
        grad_output = Variable(Tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=False)
        
        # Backward pass
        output.backward(grad_output)
        
        # Verify gradients
        assert var.grad is not None, "Input should have gradient"
        
        # For ReLU: gradient is 0 for negative inputs, 1 for positive inputs
        expected_grad = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(var.grad.data.data, expected_grad, decimal=5)


class TestTensorAutogradComputationGraph:
    """Test computation graph construction with real tensors."""
    
    def test_chain_operations_with_real_tensors(self):
        """Test chaining operations with real tensor data."""
        # Create real tensor input
        x_tensor = Tensor([1.0, 2.0])
        x = Variable(x_tensor, requires_grad=True)
        
        # Chain operations: y = (x + 1) * 2
        temp = add(x, Variable(Tensor([1.0, 1.0]), requires_grad=False))
        y = multiply(temp, Variable(Tensor([2.0, 2.0]), requires_grad=False))
        
        # Verify intermediate result
        assert isinstance(temp, Variable), "Intermediate result should be Variable"
        assert isinstance(y, Variable), "Final result should be Variable"
        
        # Verify final result
        expected_data = np.array([4.0, 6.0], dtype=np.float32)  # (1+1)*2, (2+1)*2
        np.testing.assert_array_almost_equal(y.data.data, expected_data, decimal=5)
        
        # Verify gradient tracking
        assert y.requires_grad is True, "Final result should track gradients"
        assert y.grad_fn is not None, "Final result should have gradient function"
    
    def test_complex_computation_graph(self):
        """Test complex computation graph with real tensors."""
        # Create real tensor inputs
        a_tensor = Tensor([2.0])
        b_tensor = Tensor([3.0])
        
        a = Variable(a_tensor, requires_grad=True)
        b = Variable(b_tensor, requires_grad=True)
        
        # Build computation graph: z = (a + b) * (a - b)
        sum_ab = add(a, b)
        # Note: We don't have subtract function, so we'll use add with negative
        neg_b = multiply(b, Variable(Tensor([-1.0]), requires_grad=False))
        diff_ab = add(a, neg_b)
        z = multiply(sum_ab, diff_ab)
        
        # Verify result
        expected_data = np.array([5.0 * (-1.0)], dtype=np.float32)  # (2+3) * (2-3) = 5 * (-1)
        np.testing.assert_array_almost_equal(z.data.data, expected_data, decimal=5)
        
        # Verify gradient tracking
        assert z.requires_grad is True, "Result should track gradients"
        assert z.grad_fn is not None, "Result should have gradient function"


class TestTensorAutogradDataTypes:
    """Test autograd operations with different tensor data types."""
    
    def test_float32_tensor_integration(self):
        """Test autograd with float32 tensors."""
        # Create float32 tensor
        tensor_data = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        var = Variable(tensor_data, requires_grad=True)
        
        # Apply operation
        result = relu_with_grad(var)
        
        # Verify data type preservation
        assert var.data.dtype == np.float32, "Input should be float32"
        assert result.data.dtype == np.float32, "Result should be float32"
    
    def test_different_tensor_shapes(self):
        """Test autograd with different tensor shapes."""
        test_cases = [
            Tensor([1.0]),  # 1D single element
            Tensor([1.0, 2.0]),  # 1D multiple elements
            Tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
        ]
        
        for tensor_data in test_cases:
            var = Variable(tensor_data, requires_grad=True)
            result = relu_with_grad(var)
            
            # Verify shape preservation
            assert result.data.shape == tensor_data.shape, f"Shape should be preserved: {tensor_data.shape}"
            assert isinstance(result.data, Tensor), "Result should be a Tensor"


class TestTensorAutogradRealisticScenarios:
    """Test autograd operations with realistic tensor scenarios."""
    
    def test_neural_network_like_computation(self):
        """Test autograd with neural network-like computation."""
        # Create input tensor (batch_size=1, features=2)
        x_tensor = Tensor([[1.0, 2.0]])
        x = Variable(x_tensor, requires_grad=True)
        
        # Create weight tensor
        w_tensor = Tensor([[0.5, 0.3], [0.2, 0.8]])
        w = Variable(w_tensor, requires_grad=True)
        
        # Note: We would need matrix multiplication for full neural network
        # For now, test element-wise operations
        
        # Apply activation to input
        activated = relu_with_grad(x)
        
        # Verify realistic computation
        expected_data = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(activated.data.data, expected_data, decimal=5)
        
        assert activated.requires_grad is True, "Should track gradients"
        assert isinstance(activated.data, Tensor), "Should produce Tensor"
    
    def test_gradient_accumulation_scenario(self):
        """Test gradient accumulation with real tensors."""
        # Create parameter tensor
        param_tensor = Tensor([1.0, 2.0])
        param = Variable(param_tensor, requires_grad=True)
        
        # Simulate multiple forward passes
        for i in range(3):
            # Forward pass
            output = multiply(param, Variable(Tensor([float(i+1), float(i+1)]), requires_grad=False))
            
            # Backward pass
            grad_output = Variable(Tensor([1.0, 1.0]), requires_grad=False)
            output.backward(grad_output)
            
            # Verify gradient exists
            assert param.grad is not None, f"Gradient should exist after pass {i+1}"
            
            # Note: In a real system, we'd accumulate gradients
            # For now, just verify the gradient computation works
            expected_grad = np.array([float(i+1), float(i+1)], dtype=np.float32)
            np.testing.assert_array_almost_equal(param.grad.data.data, expected_grad, decimal=5)
            
            # Reset gradient for next iteration (simulating optimizer step)
            param.grad = None 
"""
Test suite for the activations module.
This tests the student implementations to ensure they work correctly.
"""

import pytest
import numpy as np
import sys
import os

# Import from the main package (rock solid foundation)
from tinytorch.core.tensor import Tensor

# Import our implementations from the local module for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from activations_dev import ReLU, Sigmoid, Tanh, Softmax


class TestReLU:
    """Test the ReLU activation function."""
    
    def test_relu_basic_functionality(self):
        """Test basic ReLU behavior: max(0, x)"""
        relu = ReLU()
        
        # Test mixed positive/negative values
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = relu(x)
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
        
        assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    
    def test_relu_all_positive(self):
        """Test ReLU with all positive values (should be unchanged)"""
        relu = ReLU()
        
        x = Tensor([[1.0, 2.5, 3.7, 10.0]])
        y = relu(x)
        
        assert np.allclose(y.data, x.data), "ReLU should preserve positive values"
    
    def test_relu_all_negative(self):
        """Test ReLU with all negative values (should be zeros)"""
        relu = ReLU()
        
        x = Tensor([[-1.0, -2.5, -3.7, -10.0]])
        y = relu(x)
        expected = np.zeros_like(x.data)
        
        assert np.allclose(y.data, expected), "ReLU should zero out negative values"
    
    def test_relu_zero_input(self):
        """Test ReLU with zero input"""
        relu = ReLU()
        
        x = Tensor([[0.0]])
        y = relu(x)
        
        assert y.data[0, 0] == 0.0, "ReLU(0) should be 0"
    
    def test_relu_shape_preservation(self):
        """Test that ReLU preserves tensor shape"""
        relu = ReLU()
        
        # Test different shapes
        shapes = [(1, 5), (2, 3), (4, 1), (3, 3)]
        for shape in shapes:
            x = Tensor(np.random.randn(*shape))
            y = relu(x)
            assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
    
    def test_relu_callable(self):
        """Test that ReLU can be called directly"""
        relu = ReLU()
        x = Tensor([[1.0, -1.0]])
        
        y1 = relu(x)
        y2 = relu.forward(x)
        
        assert np.allclose(y1.data, y2.data), "Direct call should match forward method"


class TestSigmoid:
    """Test the Sigmoid activation function."""
    
    def test_sigmoid_basic_functionality(self):
        """Test basic Sigmoid behavior"""
        sigmoid = Sigmoid()
        
        # Test known values
        x = Tensor([[0.0]])
        y = sigmoid(x)
        assert abs(y.data[0, 0] - 0.5) < 1e-6, "Sigmoid(0) should be 0.5"
    
    def test_sigmoid_range(self):
        """Test that Sigmoid outputs are in (0, 1)"""
        sigmoid = Sigmoid()
        
        # Test wide range of inputs
        x = Tensor([[-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]])
        y = sigmoid(x)
        
        assert np.all(y.data > 0), "Sigmoid outputs should be > 0"
        assert np.all(y.data < 1), "Sigmoid outputs should be < 1"
    
    def test_sigmoid_numerical_stability(self):
        """Test Sigmoid with extreme values (numerical stability)"""
        sigmoid = Sigmoid()
        
        # Test extreme values that could cause overflow
        x = Tensor([[-100.0, -50.0, 50.0, 100.0]])
        y = sigmoid(x)
        
        # Should not contain NaN or inf
        assert not np.any(np.isnan(y.data)), "Sigmoid should not produce NaN"
        assert not np.any(np.isinf(y.data)), "Sigmoid should not produce inf"
        
        # Should be close to 0 for very negative, close to 1 for very positive
        assert y.data[0, 0] < 1e-10, "Sigmoid(-100) should be very close to 0"
        assert y.data[0, 1] < 1e-10, "Sigmoid(-50) should be very close to 0"
        assert y.data[0, 2] > 1 - 1e-10, "Sigmoid(50) should be very close to 1"
        assert y.data[0, 3] > 1 - 1e-10, "Sigmoid(100) should be very close to 1"
    
    def test_sigmoid_monotonicity(self):
        """Test that Sigmoid is monotonically increasing"""
        sigmoid = Sigmoid()
        
        x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
        y = sigmoid(x)
        
        # Check that outputs are increasing
        for i in range(len(y.data[0]) - 1):
            assert y.data[0, i] < y.data[0, i + 1], "Sigmoid should be monotonically increasing"
    
    def test_sigmoid_shape_preservation(self):
        """Test that Sigmoid preserves tensor shape"""
        sigmoid = Sigmoid()
        
        shapes = [(1, 5), (2, 3), (4, 1)]
        for shape in shapes:
            x = Tensor(np.random.randn(*shape))
            y = sigmoid(x)
            assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
    
    def test_sigmoid_callable(self):
        """Test that Sigmoid can be called directly"""
        sigmoid = Sigmoid()
        x = Tensor([[1.0, -1.0]])
        
        y1 = sigmoid(x)
        y2 = sigmoid.forward(x)
        
        assert np.allclose(y1.data, y2.data), "Direct call should match forward method"


class TestTanh:
    """Test the Tanh activation function."""
    
    def test_tanh_basic_functionality(self):
        """Test basic Tanh behavior"""
        tanh = Tanh()
        
        # Test known values
        x = Tensor([[0.0]])
        y = tanh(x)
        assert abs(y.data[0, 0] - 0.0) < 1e-6, "Tanh(0) should be 0"
    
    def test_tanh_range(self):
        """Test that Tanh outputs are in [-1, 1]"""
        tanh = Tanh()
        
        # Test wide range of inputs
        x = Tensor([[-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]])
        y = tanh(x)
        
        assert np.all(y.data >= -1), "Tanh outputs should be >= -1"
        assert np.all(y.data <= 1), "Tanh outputs should be <= 1"
    
    def test_tanh_symmetry(self):
        """Test that Tanh is symmetric: tanh(-x) = -tanh(x)"""
        tanh = Tanh()
        
        x = Tensor([[1.0, 2.0, 3.0]])
        x_neg = Tensor([[-1.0, -2.0, -3.0]])
        
        y_pos = tanh(x)
        y_neg = tanh(x_neg)
        
        assert np.allclose(y_neg.data, -y_pos.data), "Tanh should be symmetric"
    
    def test_tanh_monotonicity(self):
        """Test that Tanh is monotonically increasing"""
        tanh = Tanh()
        
        x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
        y = tanh(x)
        
        # Check that outputs are increasing
        for i in range(len(y.data[0]) - 1):
            assert y.data[0, i] < y.data[0, i + 1], "Tanh should be monotonically increasing"
    
    def test_tanh_extreme_values(self):
        """Test Tanh with extreme values"""
        tanh = Tanh()
        
        x = Tensor([[-100.0, 100.0]])
        y = tanh(x)
        
        # Should be close to -1 and 1 respectively
        assert abs(y.data[0, 0] - (-1.0)) < 1e-10, "Tanh(-100) should be very close to -1"
        assert abs(y.data[0, 1] - 1.0) < 1e-10, "Tanh(100) should be very close to 1"
    
    def test_tanh_shape_preservation(self):
        """Test that Tanh preserves tensor shape"""
        tanh = Tanh()
        
        shapes = [(1, 5), (2, 3), (4, 1)]
        for shape in shapes:
            x = Tensor(np.random.randn(*shape))
            y = tanh(x)
            assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
    
    def test_tanh_callable(self):
        """Test that Tanh can be called directly"""
        tanh = Tanh()
        x = Tensor([[1.0, -1.0]])
        
        y1 = tanh(x)
        y2 = tanh.forward(x)
        
        assert np.allclose(y1.data, y2.data), "Direct call should match forward method"


class TestActivationComparison:
    """Test interactions and comparisons between activation functions."""
    
    def test_activation_consistency(self):
        """Test that all activations work with the same input"""
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        
        # All should process without error
        y_relu = relu(x)
        y_sigmoid = sigmoid(x)
        y_tanh = tanh(x)
        
        # All should preserve shape
        assert y_relu.shape == x.shape
        assert y_sigmoid.shape == x.shape
        assert y_tanh.shape == x.shape
    
    def test_activation_ranges(self):
        """Test that activations have expected output ranges"""
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        x = Tensor([[-5.0, -2.0, 0.0, 2.0, 5.0]])
        
        y_relu = relu(x)
        y_sigmoid = sigmoid(x)
        y_tanh = tanh(x)
        
        # ReLU: [0, inf)
        assert np.all(y_relu.data >= 0), "ReLU should be non-negative"
        
        # Sigmoid: (0, 1)
        assert np.all(y_sigmoid.data > 0), "Sigmoid should be positive"
        assert np.all(y_sigmoid.data < 1), "Sigmoid should be less than 1"
        
        # Tanh: (-1, 1)
        assert np.all(y_tanh.data > -1), "Tanh should be greater than -1"
        assert np.all(y_tanh.data < 1), "Tanh should be less than 1"


# Integration tests with edge cases
class TestActivationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_tensor(self):
        """Test all activations with zero tensor"""
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        x = Tensor([[0.0, 0.0, 0.0]])
        
        y_relu = relu(x)
        y_sigmoid = sigmoid(x)
        y_tanh = tanh(x)
        
        assert np.allclose(y_relu.data, [0.0, 0.0, 0.0]), "ReLU(0) should be 0"
        assert np.allclose(y_sigmoid.data, [0.5, 0.5, 0.5]), "Sigmoid(0) should be 0.5"
        assert np.allclose(y_tanh.data, [0.0, 0.0, 0.0]), "Tanh(0) should be 0"
    
    def test_single_element_tensor(self):
        """Test all activations with single element tensor"""
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        x = Tensor([[1.0]])
        
        y_relu = relu(x)
        y_sigmoid = sigmoid(x)
        y_tanh = tanh(x)
        
        assert y_relu.shape == (1, 1)
        assert y_sigmoid.shape == (1, 1)
        assert y_tanh.shape == (1, 1)
    
    def test_large_tensor(self):
        """Test activations with larger tensors"""
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        # Create a 10x10 tensor
        x = Tensor(np.random.randn(10, 10))
        
        y_relu = relu(x)
        y_sigmoid = sigmoid(x)
        y_tanh = tanh(x)
        
        assert y_relu.shape == (10, 10)
        assert y_sigmoid.shape == (10, 10)
        assert y_tanh.shape == (10, 10)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 
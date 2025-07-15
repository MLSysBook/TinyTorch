"""
Integration Tests - Tensor and Activations

Tests real integration between Tensor and Activation modules.
Uses actual TinyTorch components to verify they work together correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax


class TestTensorActivationIntegration:
    """Test real integration between Tensor and Activation modules."""
    
    def test_relu_with_real_tensors(self):
        """Test ReLU activation with real Tensor objects."""
        relu = ReLU()
        
        # Test with negative, zero, and positive values
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        result = relu(x)
        
        # Verify it returns a real Tensor
        assert isinstance(result, Tensor)
        assert result.shape == x.shape
        
        # Verify ReLU behavior: max(0, x)
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_allclose(result.data, expected)
    
    def test_sigmoid_with_real_tensors(self):
        """Test Sigmoid activation with real Tensor objects."""
        sigmoid = Sigmoid()
        
        # Test with various inputs
        x = Tensor([[-5.0, -1.0, 0.0, 1.0, 5.0]])
        result = sigmoid(x)
        
        # Verify it returns a real Tensor
        assert isinstance(result, Tensor)
        assert result.shape == x.shape
        
        # Verify sigmoid properties
        assert np.all(result.data > 0.0)  # All positive
        assert np.all(result.data < 1.0)  # All less than 1
        assert np.isclose(result.data[0, 2], 0.5, atol=1e-6)  # sigmoid(0) = 0.5
    
    def test_tanh_with_real_tensors(self):
        """Test Tanh activation with real Tensor objects."""
        tanh = Tanh()
        
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        result = tanh(x)
        
        # Verify it returns a real Tensor
        assert isinstance(result, Tensor)
        assert result.shape == x.shape
        
        # Verify tanh properties
        assert np.all(result.data > -1.0)  # All greater than -1
        assert np.all(result.data < 1.0)   # All less than 1
        assert np.isclose(result.data[0, 2], 0.0, atol=1e-6)  # tanh(0) = 0
    
    def test_softmax_with_real_tensors(self):
        """Test Softmax activation with real Tensor objects."""
        softmax = Softmax()
        
        # Test with logits
        x = Tensor([[1.0, 2.0, 3.0]])
        result = softmax(x)
        
        # Verify it returns a real Tensor
        assert isinstance(result, Tensor)
        assert result.shape == x.shape
        
        # Verify softmax properties
        assert np.all(result.data > 0.0)  # All positive
        assert np.all(result.data < 1.0)  # All less than 1
        assert np.isclose(np.sum(result.data), 1.0, atol=1e-6)  # Sums to 1
    
    def test_activation_chaining_with_real_tensors(self):
        """Test chaining activations with real Tensors."""
        relu = ReLU()
        sigmoid = Sigmoid()
        
        # Start with mixed positive/negative values
        x = Tensor([[-1.0, 0.0, 1.0, 2.0]])
        
        # Apply ReLU first: negative values become 0
        relu_result = relu(x)
        expected_after_relu = np.array([[0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_allclose(relu_result.data, expected_after_relu)
        
        # Apply Sigmoid to ReLU output
        final_result = sigmoid(relu_result)
        
        # Verify final result properties
        assert isinstance(final_result, Tensor)
        assert np.all(final_result.data > 0.0)
        assert np.all(final_result.data < 1.0)
        
        # First two should be sigmoid(0) = 0.5
        assert np.isclose(final_result.data[0, 0], 0.5, atol=1e-6)
        assert np.isclose(final_result.data[0, 1], 0.5, atol=1e-6)
    
    def test_batch_processing_integration(self):
        """Test activation functions work with batched tensors."""
        activations = [ReLU(), Sigmoid(), Tanh()]
        
        # Create batch of samples
        batch_x = Tensor([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [0.5, 1.5, -0.5, -1.5, 0.0],
            [3.0, -3.0, 1.0, -1.0, 0.0]
        ])
        
        for activation in activations:
            result = activation(batch_x)
            
            # Verify batch processing preserves shape
            assert isinstance(result, Tensor)
            assert result.shape == batch_x.shape
            assert not np.any(np.isnan(result.data))
            assert not np.any(np.isinf(result.data))
    
    def test_softmax_batch_integration(self):
        """Test Softmax works correctly with batched tensors."""
        softmax = Softmax()
        
        # Create batch of logits
        batch_x = Tensor([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0]
        ])
        
        result = softmax(batch_x)
        
        # Verify batch processing
        assert isinstance(result, Tensor)
        assert result.shape == batch_x.shape
        
        # Each row should sum to 1
        for i in range(batch_x.shape[0]):
            row_sum = np.sum(result.data[i])
            assert np.isclose(row_sum, 1.0, atol=1e-6)
    
    def test_tensor_type_preservation(self):
        """Test that activations preserve tensor type and properties."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        # Test with different tensor shapes
        test_tensors = [
            Tensor([5.0]),                    # Scalar
            Tensor([1.0, 2.0, 3.0]),          # 1D
            Tensor([[1.0, 2.0], [3.0, 4.0]]), # 2D
        ]
        
        for tensor in test_tensors:
            for activation in activations:
                result = activation(tensor)
                
                # Verify type preservation
                assert isinstance(result, Tensor)
                assert result.shape == tensor.shape
                assert hasattr(result, 'data')
                assert hasattr(result, 'shape')
    
    def test_numerical_stability_integration(self):
        """Test numerical stability when using real tensors with activations."""
        # Test with extreme values
        extreme_tensor = Tensor([[-1000.0, 1000.0, 0.0]])
        
        # ReLU should handle extreme values
        relu = ReLU()
        relu_result = relu(extreme_tensor)
        assert np.all(np.isfinite(relu_result.data))
        
        # Sigmoid should handle extreme values
        sigmoid = Sigmoid()
        sigmoid_result = sigmoid(extreme_tensor)
        assert np.all(np.isfinite(sigmoid_result.data))
        assert np.all(sigmoid_result.data >= 0.0)
        assert np.all(sigmoid_result.data <= 1.0)
        
        # Tanh should handle extreme values  
        tanh = Tanh()
        tanh_result = tanh(extreme_tensor)
        assert np.all(np.isfinite(tanh_result.data))
        assert np.all(tanh_result.data >= -1.0)
        assert np.all(tanh_result.data <= 1.0)


class TestActivationPolymorphism:
    """Test that activations work with different tensor-like objects."""
    
    def test_activation_type_preservation(self):
        """Test that activations preserve input type."""
        relu = ReLU()
        
        # Test with real Tensor
        tensor_input = Tensor([[1.0, -1.0, 2.0]])
        tensor_result = relu(tensor_input)
        
        # Should return same type as input
        assert type(tensor_result) == type(tensor_input)
        assert isinstance(tensor_result, Tensor)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"]) 
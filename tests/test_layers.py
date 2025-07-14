"""
Comprehensive Layers Module Tests

Tests Dense layer functionality using simple mock objects.
Used for instructor grading and comprehensive validation.

This file demonstrates the mock-based testing approach where we use
simple, visible mocks instead of depending on other TinyTorch modules.
"""

import numpy as np
import pytest


# Simple Mock Objects - Visible and Educational
class MockTensor:
    """
    Simple mock tensor for testing layers.
    
    Shows exactly what interface the Dense layer expects:
    - .data (numpy array): The actual numerical data
    - .shape (tuple): Dimensions of the data
    """
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"MockTensor(shape={self.shape})"


# Import the student's implementation
# Note: In a real setup, this would import from the student's module
try:
    from modules.source.layers.layers_dev import Dense, matmul_naive
except ImportError:
    # Fallback for different import paths
    try:
        from tinytorch.core.layers import Dense, matmul_naive
    except ImportError:
        # Skip tests if module not found
        pytest.skip("Layers module not found", allow_module_level=True)


class TestMatrixMultiplication:
    """Comprehensive tests for matrix multiplication implementation."""
    
    def test_basic_2x2_multiplication(self):
        """Test basic 2x2 matrix multiplication."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        result = matmul_naive(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == (2, 2)
    
    def test_different_shapes(self):
        """Test matrix multiplication with different shapes."""
        # Test 1x3 × 3x1 = 1x1
        A = np.array([[1, 2, 3]], dtype=np.float32)
        B = np.array([[4], [5], [6]], dtype=np.float32)
        
        result = matmul_naive(A, B)
        expected = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == (1, 1)
        
        # Test 3x2 × 2x4 = 3x4
        A2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        B2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        
        result2 = matmul_naive(A2, B2)
        expected2 = A2 @ B2  # Use NumPy for verification
        
        np.testing.assert_array_almost_equal(result2, expected2)
        assert result2.shape == (3, 4)
    
    def test_edge_cases(self):
        """Test matrix multiplication edge cases."""
        # Test with zeros
        A_zero = np.zeros((2, 3), dtype=np.float32)
        B_zero = np.zeros((3, 2), dtype=np.float32)
        result_zero = matmul_naive(A_zero, B_zero)
        expected_zero = np.zeros((2, 2), dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result_zero, expected_zero)
        
        # Test with identity
        A_id = np.array([[1, 2]], dtype=np.float32)
        B_id = np.array([[1, 0], [0, 1]], dtype=np.float32)
        result_id = matmul_naive(A_id, B_id)
        expected_id = np.array([[1, 2]], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result_id, expected_id)
    
    def test_comparison_with_numpy(self):
        """Test that our implementation matches NumPy."""
        # Random test cases
        np.random.seed(42)
        
        for _ in range(5):
            m, n, k = np.random.randint(1, 10, 3)
            A = np.random.randn(m, n).astype(np.float32)
            B = np.random.randn(n, k).astype(np.float32)
            
            our_result = matmul_naive(A, B)
            numpy_result = A @ B
            
            np.testing.assert_array_almost_equal(our_result, numpy_result, decimal=5)


class TestDenseLayerInitialization:
    """Test Dense layer initialization and parameter setup."""
    
    def test_initialization_with_bias(self):
        """Test Dense layer initialization with bias."""
        layer = Dense(input_size=3, output_size=2, use_bias=True)
        
        # Check shapes
        assert layer.weights.shape == (3, 2)
        assert layer.bias is not None
        assert layer.bias.shape == (2,)
        
        # Check initialization
        assert not np.allclose(layer.weights, 0), "Weights should not be all zeros"
        assert np.allclose(layer.bias, 0), "Bias should be initialized to zeros"
    
    def test_initialization_without_bias(self):
        """Test Dense layer initialization without bias."""
        layer = Dense(input_size=4, output_size=3, use_bias=False)
        
        # Check shapes
        assert layer.weights.shape == (4, 3)
        assert layer.bias is None
        
        # Check weight initialization
        assert not np.allclose(layer.weights, 0), "Weights should not be all zeros"
    
    def test_different_sizes(self):
        """Test Dense layer with different input/output sizes."""
        test_configs = [
            (1, 1),
            (10, 5),
            (100, 50),
            (784, 128)  # MNIST-like
        ]
        
        for input_size, output_size in test_configs:
            layer = Dense(input_size=input_size, output_size=output_size)
            
            assert layer.weights.shape == (input_size, output_size)
            if layer.bias is not None:
                assert layer.bias.shape == (output_size,)


class TestDenseLayerForward:
    """Test Dense layer forward pass computation."""
    
    def test_single_sample_forward(self):
        """Test forward pass with single sample."""
        layer = Dense(input_size=3, output_size=2, use_bias=True)
        
        # Use mock tensor
        x = MockTensor([[1, 2, 3]])
        y = layer(x)
        
        # Check output shape
        assert y.shape == (1, 2)
        
        # Verify computation manually
        expected = np.dot(x.data, layer.weights) + layer.bias
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_batch_forward(self):
        """Test forward pass with batch of samples."""
        layer = Dense(input_size=3, output_size=2, use_bias=True)
        
        # Use mock tensor with batch
        x = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = layer(x)
        
        # Check output shape
        assert y.shape == (3, 2)
        
        # Verify computation manually
        expected = np.dot(x.data, layer.weights) + layer.bias
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_forward_without_bias(self):
        """Test forward pass without bias."""
        layer = Dense(input_size=2, output_size=3, use_bias=False)
        
        x = MockTensor([[1, 2]])
        y = layer(x)
        
        # Check output shape
        assert y.shape == (1, 3)
        
        # Verify computation (should be just matrix multiplication)
        expected = np.dot(x.data, layer.weights)
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_naive_vs_optimized_matmul(self):
        """Test that naive and optimized matrix multiplication give same results."""
        layer_naive = Dense(input_size=2, output_size=2, use_naive_matmul=True)
        layer_optimized = Dense(input_size=2, output_size=2, use_naive_matmul=False)
        
        # Set same weights for comparison
        layer_optimized.weights = layer_naive.weights.copy()
        if layer_naive.bias is not None:
            layer_optimized.bias = layer_naive.bias.copy()
        
        x = MockTensor([[1, 2]])
        y_naive = layer_naive(x)
        y_optimized = layer_optimized(x)
        
        # Both should give same results
        np.testing.assert_array_almost_equal(y_naive.data, y_optimized.data)


class TestDenseLayerEdgeCases:
    """Test Dense layer edge cases and robustness."""
    
    def test_zero_input(self):
        """Test layer with zero input."""
        layer = Dense(input_size=3, output_size=2, use_bias=True)
        
        x = MockTensor([[0, 0, 0]])
        y = layer(x)
        
        # Output should be just the bias
        expected = layer.bias
        np.testing.assert_array_almost_equal(y.data.flatten(), expected)
    
    def test_large_values(self):
        """Test layer with large input values."""
        layer = Dense(input_size=2, output_size=2)
        
        x = MockTensor([[1000, -1000]])
        y = layer(x)
        
        # Should not produce NaN or Inf
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isinf(y.data))
        assert y.shape == (1, 2)
    
    def test_negative_values(self):
        """Test layer with negative input values."""
        layer = Dense(input_size=3, output_size=2)
        
        x = MockTensor([[-1, -2, -3]])
        y = layer(x)
        
        # Should handle negative values correctly
        assert y.shape == (1, 2)
        assert not np.any(np.isnan(y.data))
    
    def test_single_neuron(self):
        """Test layer with single input and output neuron."""
        layer = Dense(input_size=1, output_size=1)
        
        x = MockTensor([[5]])
        y = layer(x)
        
        assert y.shape == (1, 1)
        
        # Manual verification
        expected = x.data * layer.weights + (layer.bias if layer.bias is not None else 0)
        np.testing.assert_array_almost_equal(y.data, expected)


class TestDenseLayerIntegration:
    """Test Dense layer integration scenarios."""
    
    def test_layer_chaining(self):
        """Test chaining multiple Dense layers."""
        layer1 = Dense(input_size=4, output_size=3)
        layer2 = Dense(input_size=3, output_size=2)
        layer3 = Dense(input_size=2, output_size=1)
        
        x = MockTensor([[1, 2, 3, 4]])
        
        # Chain layers
        h1 = layer1(x)
        h2 = layer2(h1)
        h3 = layer3(h2)
        
        # Check shapes
        assert h1.shape == (1, 3)
        assert h2.shape == (1, 2)
        assert h3.shape == (1, 1)
    
    def test_parameter_counting(self):
        """Test parameter counting for different layer configurations."""
        # Layer with bias
        layer_bias = Dense(input_size=10, output_size=5, use_bias=True)
        expected_params_bias = 10 * 5 + 5  # weights + bias
        actual_params_bias = layer_bias.weights.size + (layer_bias.bias.size if layer_bias.bias is not None else 0)
        assert actual_params_bias == expected_params_bias
        
        # Layer without bias
        layer_no_bias = Dense(input_size=10, output_size=5, use_bias=False)
        expected_params_no_bias = 10 * 5  # only weights
        actual_params_no_bias = layer_no_bias.weights.size
        assert actual_params_no_bias == expected_params_no_bias
    
    def test_batch_consistency(self):
        """Test that batch processing is consistent with single sample processing."""
        layer = Dense(input_size=3, output_size=2)
        
        # Single samples
        x1 = MockTensor([[1, 2, 3]])
        x2 = MockTensor([[4, 5, 6]])
        
        y1 = layer(x1)
        y2 = layer(x2)
        
        # Batch processing
        x_batch = MockTensor([[1, 2, 3], [4, 5, 6]])
        y_batch = layer(x_batch)
        
        # Results should be consistent
        np.testing.assert_array_almost_equal(y_batch.data[0], y1.data[0])
        np.testing.assert_array_almost_equal(y_batch.data[1], y2.data[0])


class TestDenseLayerPerformance:
    """Test Dense layer performance characteristics."""
    
    def test_batch_processing(self):
        """Test layer with reasonable batch sizes."""
        layer = Dense(input_size=10, output_size=5)
        
        # Realistic batch size for educational context
        batch_size = 32
        x = MockTensor(np.random.randn(batch_size, 10))
        y = layer(x)
        
        assert y.shape == (batch_size, 5)
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isinf(y.data))
    
    def test_memory_efficiency(self):
        """Test that layer doesn't create unnecessary copies."""
        layer = Dense(input_size=5, output_size=3)
        
        x = MockTensor([[1, 2, 3, 4, 5]])
        original_weights = layer.weights.copy()
        original_bias = layer.bias.copy() if layer.bias is not None else None
        
        # Forward pass shouldn't modify weights
        y = layer(x)
        
        np.testing.assert_array_equal(layer.weights, original_weights)
        if original_bias is not None:
            np.testing.assert_array_equal(layer.bias, original_bias)


# Test runner for command line execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
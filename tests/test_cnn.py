"""
Mock-based module tests for CNN module.

This test file uses simple mocks to avoid cross-module dependencies while thoroughly
testing the CNN module functionality. The MockTensor class provides a minimal
interface that matches expected behavior without requiring actual implementations.

Test Philosophy:
- Use simple, visible mocks instead of complex mocking frameworks
- Test interface contracts and behavior, not implementation details
- Avoid dependency cascade where CNN tests fail due to tensor bugs
- Focus on convolution operations, Conv2D layers, and flatten functionality
- Ensure educational value with clear test structure
"""

import pytest
import numpy as np
import sys
import os

# Add the module source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'source', '05_cnn'))

from cnn_dev import Conv2D, conv2d_naive, flatten


class MockTensor:
    """
    Simple mock tensor for testing CNN operations without tensor dependencies.
    
    This mock provides just enough functionality to test CNN operations
    without requiring the full Tensor implementation.
    """
    
    def __init__(self, data):
        """Initialize with numpy array data."""
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array([data], dtype=np.float32)
    
    @property
    def shape(self):
        """Return shape of the underlying data."""
        return self.data.shape
    
    def __repr__(self):
        return f"MockTensor({self.data})"
    
    def __eq__(self, other):
        """Check equality with another MockTensor."""
        if isinstance(other, MockTensor):
            return np.allclose(self.data, other.data)
        return False


class TestConv2DNaive:
    """Test conv2d_naive function with numpy arrays."""
    
    def test_conv2d_naive_basic(self):
        """Test basic convolution operation."""
        # Simple 3x3 input with 2x2 kernel
        input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        kernel_array = np.array([[1, 0], [0, 1]], dtype=np.float32)
        
        result = conv2d_naive(input_array, kernel_array)
        expected = np.array([[6, 8], [12, 14]], dtype=np.float32)
        
        assert np.allclose(result, expected)
        assert result.shape == (2, 2)
    
    def test_conv2d_naive_edge_detection(self):
        """Test convolution with edge detection kernel."""
        # Create a simple edge pattern
        input_array = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        edge_kernel = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
        
        result = conv2d_naive(input_array, edge_kernel)
        
        # Should detect vertical edge
        assert result.shape == (2, 2)
        assert result[0, 0] == 0  # No edge at left
        assert result[0, 1] > 0   # Edge detected at right
    
    def test_conv2d_naive_different_sizes(self):
        """Test convolution with different kernel sizes."""
        # Test with 5x5 input and 3x3 kernel
        input_5x5 = np.random.randn(5, 5).astype(np.float32)
        kernel_3x3 = np.random.randn(3, 3).astype(np.float32)
        
        result = conv2d_naive(input_5x5, kernel_3x3)
        expected_shape = (3, 3)  # 5-3+1 = 3
        
        assert result.shape == expected_shape
        assert result.dtype == np.float32
    
    def test_conv2d_naive_identity_kernel(self):
        """Test convolution with identity-like kernel."""
        input_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        identity_kernel = np.array([[1]], dtype=np.float32)
        
        result = conv2d_naive(input_array, identity_kernel)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        assert np.allclose(result, expected)
    
    def test_conv2d_naive_zero_kernel(self):
        """Test convolution with zero kernel."""
        input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        zero_kernel = np.array([[0, 0], [0, 0]], dtype=np.float32)
        
        result = conv2d_naive(input_array, zero_kernel)
        expected = np.zeros((1, 2), dtype=np.float32)
        
        assert np.allclose(result, expected)
    
    def test_conv2d_naive_large_kernel(self):
        """Test convolution with large kernel."""
        input_array = np.ones((10, 10), dtype=np.float32)
        large_kernel = np.ones((5, 5), dtype=np.float32)
        
        result = conv2d_naive(input_array, large_kernel)
        expected_shape = (6, 6)  # 10-5+1 = 6
        
        assert result.shape == expected_shape
        # All ones input with all ones kernel should give 25 (5*5) everywhere
        assert np.allclose(result, 25.0)


class TestConv2DLayer:
    """Test Conv2D layer class with mock tensors."""
    
    def test_conv2d_layer_initialization(self):
        """Test Conv2D layer initialization."""
        layer = Conv2D(kernel_size=(3, 3))
        
        assert layer.kernel_size == (3, 3)
        assert layer.kernel.shape == (3, 3)
        assert layer.kernel.dtype == np.float32
        assert not np.allclose(layer.kernel, 0)  # Should be randomly initialized
    
    def test_conv2d_layer_different_sizes(self):
        """Test Conv2D layer with different kernel sizes."""
        sizes = [(2, 2), (3, 3), (5, 5), (1, 1)]
        
        for size in sizes:
            layer = Conv2D(kernel_size=size)
            assert layer.kernel_size == size
            assert layer.kernel.shape == size
    
    def test_conv2d_layer_forward_pass(self):
        """Test Conv2D layer forward pass."""
        layer = Conv2D(kernel_size=(2, 2))
        
        # Test with 3x3 input
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output = layer(input_tensor)
        
        assert isinstance(output, MockTensor)
        assert output.shape == (2, 2)  # 3-2+1 = 2
    
    def test_conv2d_layer_batch_processing(self):
        """Test Conv2D layer with batch input."""
        layer = Conv2D(kernel_size=(2, 2))
        
        # Note: Current implementation might not support batches
        # This test checks if it handles single images correctly
        input_tensor = MockTensor(np.random.randn(5, 5))
        output = layer(input_tensor)
        
        assert isinstance(output, MockTensor)
        assert output.shape == (4, 4)  # 5-2+1 = 4
    
    def test_conv2d_layer_kernel_consistency(self):
        """Test that Conv2D layer uses consistent kernel."""
        layer = Conv2D(kernel_size=(2, 2))
        
        # Store original kernel
        original_kernel = layer.kernel.copy()
        
        # Forward pass shouldn't change kernel
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output = layer(input_tensor)
        
        assert np.allclose(layer.kernel, original_kernel)
    
    def test_conv2d_layer_different_inputs(self):
        """Test Conv2D layer with different input sizes."""
        layer = Conv2D(kernel_size=(3, 3))
        
        input_sizes = [(5, 5), (8, 8), (10, 10)]
        
        for h, w in input_sizes:
            input_tensor = MockTensor(np.random.randn(h, w))
            output = layer(input_tensor)
            
            expected_h, expected_w = h - 3 + 1, w - 3 + 1
            assert output.shape == (expected_h, expected_w)
    
    def test_conv2d_layer_callable(self):
        """Test that Conv2D layer is callable."""
        layer = Conv2D(kernel_size=(2, 2))
        
        # Should be callable both ways
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        output1 = layer(input_tensor)
        output2 = layer.forward(input_tensor)
        
        # Both should work and produce same output
        assert isinstance(output1, MockTensor)
        assert isinstance(output2, MockTensor)
        assert output1.shape == output2.shape


class TestFlattenFunction:
    """Test flatten function with mock tensors."""
    
    def test_flatten_2d_tensor(self):
        """Test flattening 2D tensor."""
        input_tensor = MockTensor([[1, 2], [3, 4]])
        flattened = flatten(input_tensor)
        
        assert isinstance(flattened, MockTensor)
        assert flattened.shape == (1, 4)
        expected = np.array([[1, 2, 3, 4]], dtype=np.float32)
        assert np.allclose(flattened.data, expected)
    
    def test_flatten_3x3_tensor(self):
        """Test flattening 3x3 tensor."""
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        flattened = flatten(input_tensor)
        
        assert flattened.shape == (1, 9)
        expected = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.float32)
        assert np.allclose(flattened.data, expected)
    
    def test_flatten_different_shapes(self):
        """Test flatten with different tensor shapes."""
        test_shapes = [
            (2, 3),    # 2x3 -> (1, 6)
            (4, 4),    # 4x4 -> (1, 16)
            (1, 5),    # 1x5 -> (1, 5)
            (6, 1),    # 6x1 -> (1, 6)
        ]
        
        for h, w in test_shapes:
            input_data = np.random.randn(h, w)
            input_tensor = MockTensor(input_data)
            flattened = flatten(input_tensor)
            
            assert flattened.shape == (1, h * w)
            assert np.allclose(flattened.data.flatten(), input_data.flatten())
    
    def test_flatten_preserves_order(self):
        """Test that flatten preserves row-major order."""
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6]])
        flattened = flatten(input_tensor)
        
        expected = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        assert np.allclose(flattened.data, expected)
    
    def test_flatten_single_element(self):
        """Test flatten with single element tensor."""
        input_tensor = MockTensor([[5]])
        flattened = flatten(input_tensor)
        
        assert flattened.shape == (1, 1)
        assert np.allclose(flattened.data, [[5]])
    
    def test_flatten_batch_dimension(self):
        """Test that flatten adds batch dimension correctly."""
        input_tensor = MockTensor([[1, 2], [3, 4]])
        flattened = flatten(input_tensor)
        
        # Should have batch dimension of 1
        assert flattened.shape[0] == 1
        assert len(flattened.shape) == 2


class TestCNNIntegration:
    """Test integration between CNN components."""
    
    def test_conv2d_to_flatten_pipeline(self):
        """Test pipeline from Conv2D to flatten."""
        # Create Conv2D layer
        conv_layer = Conv2D(kernel_size=(2, 2))
        
        # Apply convolution
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        conv_output = conv_layer(input_tensor)
        
        # Flatten the result
        flattened = flatten(conv_output)
        
        # Should have correct shapes
        assert conv_output.shape == (2, 2)
        assert flattened.shape == (1, 4)
    
    def test_multiple_conv2d_layers(self):
        """Test multiple Conv2D layers in sequence."""
        # Create two Conv2D layers
        conv1 = Conv2D(kernel_size=(2, 2))
        conv2 = Conv2D(kernel_size=(2, 2))
        
        # Apply them in sequence
        input_tensor = MockTensor(np.random.randn(5, 5))
        
        # First convolution: 5x5 -> 4x4
        h1 = conv1(input_tensor)
        assert h1.shape == (4, 4)
        
        # Second convolution: 4x4 -> 3x3
        h2 = conv2(h1)
        assert h2.shape == (3, 3)
    
    def test_conv2d_with_different_kernels(self):
        """Test Conv2D layers with different kernel sizes."""
        input_tensor = MockTensor(np.random.randn(10, 10))
        
        # Test different kernel sizes
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        
        for kernel_size in kernel_sizes:
            conv_layer = Conv2D(kernel_size=kernel_size)
            output = conv_layer(input_tensor)
            
            expected_h = 10 - kernel_size[0] + 1
            expected_w = 10 - kernel_size[1] + 1
            assert output.shape == (expected_h, expected_w)
    
    def test_cnn_feature_extraction_pipeline(self):
        """Test complete CNN feature extraction pipeline."""
        # Simulate image classification pipeline
        input_image = MockTensor(np.random.randn(8, 8))
        
        # Feature extraction
        conv1 = Conv2D(kernel_size=(3, 3))  # 8x8 -> 6x6
        features1 = conv1(input_image)
        
        conv2 = Conv2D(kernel_size=(2, 2))  # 6x6 -> 5x5
        features2 = conv2(features1)
        
        # Flatten for dense layer
        flattened = flatten(features2)  # 5x5 -> (1, 25)
        
        # Verify pipeline
        assert features1.shape == (6, 6)
        assert features2.shape == (5, 5)
        assert flattened.shape == (1, 25)


class TestCNNEdgeCases:
    """Test edge cases and error conditions for CNN operations."""
    
    def test_conv2d_minimal_input(self):
        """Test Conv2D with minimal input size."""
        # Test with input same size as kernel
        layer = Conv2D(kernel_size=(2, 2))
        input_tensor = MockTensor([[1, 2], [3, 4]])
        
        output = layer(input_tensor)
        assert output.shape == (1, 1)  # 2-2+1 = 1
    
    def test_conv2d_large_kernel(self):
        """Test Conv2D with large kernel relative to input."""
        layer = Conv2D(kernel_size=(3, 3))
        input_tensor = MockTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        output = layer(input_tensor)
        assert output.shape == (1, 1)  # 3-3+1 = 1
    
    def test_flatten_edge_shapes(self):
        """Test flatten with edge case shapes."""
        # Very wide tensor
        wide_tensor = MockTensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        flattened = flatten(wide_tensor)
        assert flattened.shape == (1, 8)
        
        # Very tall tensor
        tall_tensor = MockTensor([[1], [2], [3], [4]])
        flattened = flatten(tall_tensor)
        assert flattened.shape == (1, 4)
    
    def test_conv2d_numerical_stability(self):
        """Test Conv2D with extreme values."""
        layer = Conv2D(kernel_size=(2, 2))
        
        # Test with very large values
        large_input = MockTensor([[1000, 2000], [3000, 4000]])
        output = layer(large_input)
        assert np.all(np.isfinite(output.data))
        
        # Test with very small values
        small_input = MockTensor([[1e-6, 2e-6], [3e-6, 4e-6]])
        output = layer(small_input)
        assert np.all(np.isfinite(output.data))
    
    def test_conv2d_zero_input(self):
        """Test Conv2D with zero input."""
        layer = Conv2D(kernel_size=(2, 2))
        zero_input = MockTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
        output = layer(zero_input)
        assert output.shape == (2, 2)
        # Output should be finite (kernel is random, not zero)
        assert np.all(np.isfinite(output.data))


class TestCNNPerformance:
    """Test CNN performance characteristics."""
    
    def test_conv2d_consistency(self):
        """Test Conv2D produces consistent results."""
        layer = Conv2D(kernel_size=(3, 3))
        input_tensor = MockTensor(np.random.randn(5, 5))
        
        # Multiple forward passes should be consistent
        output1 = layer(input_tensor)
        output2 = layer(input_tensor)
        
        # Should be identical (deterministic)
        assert np.allclose(output1.data, output2.data)
    
    def test_conv2d_different_instances(self):
        """Test different Conv2D instances have different kernels."""
        layer1 = Conv2D(kernel_size=(3, 3))
        layer2 = Conv2D(kernel_size=(3, 3))
        
        # Should have different random kernels
        assert not np.allclose(layer1.kernel, layer2.kernel)
    
    def test_flatten_efficiency(self):
        """Test flatten operation efficiency."""
        # Test with different sizes
        sizes = [(5, 5), (10, 10), (20, 20)]
        
        for h, w in sizes:
            input_tensor = MockTensor(np.random.randn(h, w))
            flattened = flatten(input_tensor)
            
            # Should preserve all data
            assert flattened.shape == (1, h * w)
            assert np.allclose(flattened.data.flatten(), input_tensor.data.flatten())
    
    def test_conv2d_scalability(self):
        """Test Conv2D with different scales."""
        kernel_sizes = [(2, 2), (3, 3), (5, 5)]
        input_sizes = [(10, 10), (20, 20), (50, 50)]
        
        for kernel_size in kernel_sizes:
            for input_size in input_sizes:
                if input_size[0] >= kernel_size[0] and input_size[1] >= kernel_size[1]:
                    layer = Conv2D(kernel_size=kernel_size)
                    input_tensor = MockTensor(np.random.randn(*input_size))
                    
                    output = layer(input_tensor)
                    expected_h = input_size[0] - kernel_size[0] + 1
                    expected_w = input_size[1] - kernel_size[1] + 1
                    assert output.shape == (expected_h, expected_w)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 
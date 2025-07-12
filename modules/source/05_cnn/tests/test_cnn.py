"""
Test suite for the CNN module.
This tests the CNN implementations to ensure they work correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the CNN module to the path
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Import from the exported package
    from tinytorch.core.cnn import conv2d_naive, Conv2D, flatten
except ImportError:
    # Fallback for when module isn't exported yet
    from cnn_dev import conv2d_naive, Conv2D, flatten

from tinytorch.core.tensor import Tensor

def safe_numpy(tensor):
    """Get numpy array from tensor, using .data attribute"""
    return tensor.data


class TestConv2DNaive:
    """Test the naive convolution implementation."""
    
    def test_conv2d_naive_small(self):
        """Test basic convolution with small matrices."""
        input = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        kernel = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.float32)
        expected = np.array([
            [1*1+2*0+4*0+5*(-1), 2*1+3*0+5*0+6*(-1)],
            [4*1+5*0+7*0+8*(-1), 5*1+6*0+8*0+9*(-1)]
        ], dtype=np.float32)
        output = conv2d_naive(input, kernel)
        assert np.allclose(output, expected), f"conv2d_naive output incorrect!\nExpected:\n{expected}\nGot:\n{output}"
    
    def test_conv2d_naive_edge_detection(self):
        """Test convolution with edge detection kernel."""
        input = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Vertical edge detection kernel
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        
        output = conv2d_naive(input, kernel)
        assert output.shape == (3, 3), f"Expected shape (3, 3), got {output.shape}"
        
        # Should detect vertical edges
        assert np.abs(output[1, 0]) > 0, "Should detect left edge"
        assert np.abs(output[1, 2]) > 0, "Should detect right edge"
        assert np.abs(output[1, 1]) < 1, "Should be small in center"
    
    def test_conv2d_naive_identity_kernel(self):
        """Test convolution with identity kernel."""
        input = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        
        # Identity kernel
        kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        
        output = conv2d_naive(input, kernel)
        expected = np.array([[5]], dtype=np.float32)  # Only center value
        assert np.allclose(output, expected), f"Identity kernel failed: got {output}, expected {expected}"
    
    def test_conv2d_naive_different_sizes(self):
        """Test convolution with different input and kernel sizes."""
        # 4x4 input, 2x2 kernel
        input = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32)
        
        kernel = np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.float32)
        
        output = conv2d_naive(input, kernel)
        assert output.shape == (3, 3), f"Expected shape (3, 3), got {output.shape}"
        
        # Check first element: 1+2+5+6 = 14
        assert np.isclose(output[0, 0], 14), f"First element should be 14, got {output[0, 0]}"
    
    def test_conv2d_naive_single_pixel(self):
        """Test convolution with single pixel input."""
        input = np.array([[5]], dtype=np.float32)
        kernel = np.array([[2]], dtype=np.float32)
        
        output = conv2d_naive(input, kernel)
        expected = np.array([[10]], dtype=np.float32)
        assert np.allclose(output, expected), f"Single pixel convolution failed: got {output}, expected {expected}"


class TestConv2DLayer:
    """Test the Conv2D layer implementation."""
    
    def test_conv2d_layer_creation(self):
        """Test Conv2D layer creation."""
        conv = Conv2D((3, 3))
        assert conv.kernel_size == (3, 3), f"Kernel size should be (3, 3), got {conv.kernel_size}"
        assert conv.kernel.shape == (3, 3), f"Kernel shape should be (3, 3), got {conv.kernel.shape}"
    
    def test_conv2d_layer_forward_pass(self):
        """Test Conv2D layer forward pass."""
        conv = Conv2D((2, 2))
        x = Tensor(np.ones((4, 4), dtype=np.float32))
        
        output = conv(x)
        assert output.shape == (3, 3), f"Expected output shape (3, 3), got {output.shape}"
        assert hasattr(output, 'data'), "Output should be a Tensor with data attribute"
    
    def test_conv2d_layer_different_sizes(self):
        """Test Conv2D layer with different input sizes."""
        conv = Conv2D((2, 2))
        
        # Test with 3x3 input
        x1 = Tensor(np.ones((3, 3), dtype=np.float32))
        out1 = conv(x1)
        assert out1.shape == (2, 2), f"3x3 input should give (2, 2) output, got {out1.shape}"
        
        # Test with 5x5 input
        x2 = Tensor(np.ones((5, 5), dtype=np.float32))
        out2 = conv(x2)
        assert out2.shape == (4, 4), f"5x5 input should give (4, 4) output, got {out2.shape}"
    
    def test_conv2d_layer_kernel_initialization(self):
        """Test that Conv2D layer initializes kernel properly."""
        conv = Conv2D((3, 3))
        
        # Kernel should not be all zeros
        assert not np.allclose(conv.kernel, 0), "Kernel should not be all zeros"
        
        # Kernel should be reasonable size (not too large)
        assert np.abs(conv.kernel).max() < 10, "Kernel values should be reasonable"
    
    def test_conv2d_layer_reproducibility(self):
        """Test that Conv2D layer gives consistent results."""
        conv = Conv2D((2, 2))
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
        
        # Multiple forward passes should give same result
        out1 = conv(x)
        out2 = conv(x)
        
        assert np.allclose(safe_numpy(out1), safe_numpy(out2)), "Conv2D should be deterministic"


class TestFlattenFunction:
    """Test the flatten function implementation."""
    
    def test_flatten_2d_matrix(self):
        """Test flattening a 2D matrix."""
        x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        flattened = flatten(x)
        
        expected = np.array([[1, 2, 3, 4]], dtype=np.float32)
        assert np.array_equal(safe_numpy(flattened), expected), f"Flatten failed: got {safe_numpy(flattened)}, expected {expected}"
        assert flattened.shape == (1, 4), f"Expected shape (1, 4), got {flattened.shape}"
    
    def test_flatten_3d_tensor(self):
        """Test flattening a 3D tensor."""
        x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
        flattened = flatten(x)
        
        expected = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
        assert np.array_equal(safe_numpy(flattened), expected), f"3D flatten failed: got {safe_numpy(flattened)}, expected {expected}"
        assert flattened.shape == (1, 8), f"Expected shape (1, 8), got {flattened.shape}"
    
    def test_flatten_1d_tensor(self):
        """Test flattening a 1D tensor."""
        x = Tensor(np.array([1, 2, 3, 4], dtype=np.float32))
        flattened = flatten(x)
        
        expected = np.array([[1, 2, 3, 4]], dtype=np.float32)
        assert np.array_equal(safe_numpy(flattened), expected), f"1D flatten failed: got {safe_numpy(flattened)}, expected {expected}"
        assert flattened.shape == (1, 4), f"Expected shape (1, 4), got {flattened.shape}"
    
    def test_flatten_single_element(self):
        """Test flattening a single element tensor."""
        x = Tensor(np.array([[[[5]]]], dtype=np.float32))
        flattened = flatten(x)
        
        expected = np.array([[5]], dtype=np.float32)
        assert np.array_equal(safe_numpy(flattened), expected), f"Single element flatten failed: got {safe_numpy(flattened)}, expected {expected}"
        assert flattened.shape == (1, 1), f"Expected shape (1, 1), got {flattened.shape}"
    
    def test_flatten_preserves_data_type(self):
        """Test that flatten preserves data type."""
        x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        flattened = flatten(x)
        
        assert safe_numpy(flattened).dtype == np.float32, f"Data type should be preserved: got {safe_numpy(flattened).dtype}"


class TestCNNIntegration:
    """Test integration between CNN components."""
    
    def test_conv_then_flatten(self):
        """Test convolution followed by flatten (typical CNN pattern)."""
        # Create a simple input
        x = Tensor(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32))
        
        # Apply convolution
        conv = Conv2D((2, 2))
        conv_out = conv(x)
        assert conv_out.shape == (3, 3), f"Conv output should be (3, 3), got {conv_out.shape}"
        
        # Apply flatten
        flat_out = flatten(conv_out)
        assert flat_out.shape == (1, 9), f"Flatten output should be (1, 9), got {flat_out.shape}"
        
        # Check that data is preserved
        assert safe_numpy(flat_out).size == 9, "Should have 9 elements after flatten"
    
    def test_multiple_conv_layers(self):
        """Test multiple convolution layers (deeper CNN)."""
        x = Tensor(np.ones((5, 5), dtype=np.float32))
        
        # First conv layer
        conv1 = Conv2D((2, 2))
        out1 = conv1(x)
        assert out1.shape == (4, 4), f"First conv should give (4, 4), got {out1.shape}"
        
        # Second conv layer
        conv2 = Conv2D((2, 2))
        out2 = conv2(out1)
        assert out2.shape == (3, 3), f"Second conv should give (3, 3), got {out2.shape}"
        
        # Final flatten
        final = flatten(out2)
        assert final.shape == (1, 9), f"Final flatten should give (1, 9), got {final.shape}"
    
    def test_conv_output_range(self):
        """Test that convolution outputs are in reasonable range."""
        # Create input with known range
        x = Tensor(np.random.rand(4, 4).astype(np.float32))  # Values 0-1
        
        conv = Conv2D((2, 2))
        output = conv(x)
        
        # Output should be finite
        assert np.all(np.isfinite(safe_numpy(output))), "Conv output should be finite"
        
        # Output should not be extremely large
        assert np.abs(safe_numpy(output)).max() < 100, "Conv output should not be extremely large"


class TestCNNEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_conv2d_naive_minimum_size(self):
        """Test convolution with minimum possible sizes."""
        # 1x1 input, 1x1 kernel
        input = np.array([[1]], dtype=np.float32)
        kernel = np.array([[2]], dtype=np.float32)
        
        output = conv2d_naive(input, kernel)
        expected = np.array([[2]], dtype=np.float32)
        assert np.allclose(output, expected), f"Minimum size convolution failed: got {output}, expected {expected}"
    
    def test_conv2d_layer_minimum_size(self):
        """Test Conv2D layer with minimum input size."""
        conv = Conv2D((1, 1))
        x = Tensor(np.array([[5]], dtype=np.float32))
        
        output = conv(x)
        assert output.shape == (1, 1), f"Minimum size layer should give (1, 1), got {output.shape}"
    
    def test_flatten_empty_handling(self):
        """Test flatten with various edge cases."""
        # Very small tensor
        x = Tensor(np.array([1], dtype=np.float32))
        flattened = flatten(x)
        assert flattened.shape == (1, 1), f"Single element should give (1, 1), got {flattened.shape}"
    
    def test_conv_with_zeros(self):
        """Test convolution with zero inputs."""
        # All zeros input
        x = Tensor(np.zeros((3, 3), dtype=np.float32))
        conv = Conv2D((2, 2))
        output = conv(x)
        
        # Should not crash and should produce valid output
        assert output.shape == (2, 2), f"Zero input should give (2, 2), got {output.shape}"
        assert np.all(np.isfinite(safe_numpy(output))), "Zero input should produce finite output"
    
    def test_conv_with_negative_values(self):
        """Test convolution with negative inputs."""
        x = Tensor(np.array([[-1, -2], [-3, -4]], dtype=np.float32))
        conv = Conv2D((2, 2))
        output = conv(x)
        
        # Should handle negative values properly
        assert output.shape == (1, 1), f"Negative input should give (1, 1), got {output.shape}"
        assert np.all(np.isfinite(safe_numpy(output))), "Negative input should produce finite output"


class TestCNNPerformance:
    """Test performance characteristics of CNN operations."""
    
    def test_conv_reasonable_speed(self):
        """Test that convolution completes in reasonable time."""
        import time
        
        # Medium-sized input
        x = Tensor(np.random.rand(10, 10).astype(np.float32))
        conv = Conv2D((3, 3))
        
        start_time = time.time()
        output = conv(x)
        end_time = time.time()
        
        # Should complete quickly (less than 1 second)
        assert end_time - start_time < 1.0, "Convolution should complete quickly"
        assert output.shape == (8, 8), f"Expected (8, 8), got {output.shape}"
    
    def test_flatten_preserves_size(self):
        """Test that flatten preserves total number of elements."""
        shapes = [(2, 3), (4, 4), (1, 10), (5, 2, 3)]
        
        for shape in shapes:
            x = Tensor(np.random.rand(*shape).astype(np.float32))
            flattened = flatten(x)
            
            original_size = np.prod(shape)
            flattened_size = flattened.shape[1]  # Second dimension since flatten returns (1, N)
            
            assert original_size == flattened_size, f"Size mismatch for shape {shape}: {original_size} != {flattened_size}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 
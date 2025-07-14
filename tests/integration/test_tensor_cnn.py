"""
Integration Tests: Tensor ↔ CNN Operations

Tests the integration between core Tensor data structures and CNN operations:
- Conv2D operations with real tensors
- Flatten operations with real tensors
- CNN data flow with proper tensor shapes
- Error handling with real tensor inputs

These tests verify that CNN operations work correctly with real TinyTorch tensors,
not mocks or synthetic data.
"""

import pytest
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.cnn import Conv2D, conv2d_naive, flatten


class TestTensorCNNIntegration:
    """Test integration between Tensor and CNN components."""
    
    def test_conv2d_naive_with_real_tensors(self):
        """Test conv2d_naive function with real tensor data."""
        # Create real tensor data
        input_data = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]], dtype=np.float32)
        
        kernel_data = np.array([[1.0, 0.0],
                                [0.0, -1.0]], dtype=np.float32)
        
        # Test with real numpy arrays (function takes arrays, not tensors)
        result = conv2d_naive(input_data, kernel_data)
        
        # Verify correct shape
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        
        # Verify correct computation
        expected = np.array([[-4.0, -4.0],
                             [-4.0, -4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_conv2d_layer_with_real_tensors(self):
        """Test Conv2D layer with real tensor inputs."""
        # Create real tensor input
        input_tensor = Tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]])
        
        # Create Conv2D layer
        conv_layer = Conv2D(kernel_size=(2, 2))
        
        # Test forward pass
        output = conv_layer(input_tensor)
        
        # Verify output is a tensor
        assert isinstance(output, Tensor), "Conv2D output should be a Tensor"
        
        # Verify correct shape
        assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
        
        # Verify data type consistency
        assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
    
    def test_flatten_with_real_tensors(self):
        """Test flatten function with real tensor inputs."""
        # Create real 2D tensor
        input_tensor = Tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test flatten
        output = flatten(input_tensor)
        
        # Verify output is a tensor
        assert isinstance(output, Tensor), "Flatten output should be a Tensor"
        
        # Verify correct shape (batch dimension added)
        assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
        
        # Verify correct data
        expected_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.data, expected_data, decimal=5)
    
    def test_cnn_pipeline_with_real_tensors(self):
        """Test complete CNN pipeline with real tensor data flow."""
        # Create real input tensor (small image)
        input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0, 16.0]])
        
        # Create CNN components
        conv_layer = Conv2D(kernel_size=(2, 2))
        
        # Test complete pipeline
        conv_output = conv_layer(input_tensor)
        flattened_output = flatten(conv_output)
        
        # Verify shapes through pipeline
        assert conv_output.shape == (3, 3), f"Conv output shape should be (3, 3), got {conv_output.shape}"
        assert flattened_output.shape == (1, 9), f"Flattened shape should be (1, 9), got {flattened_output.shape}"
        
        # Verify all outputs are tensors
        assert isinstance(conv_output, Tensor), "Conv output should be a Tensor"
        assert isinstance(flattened_output, Tensor), "Flattened output should be a Tensor"
        
        # Verify data types
        assert conv_output.dtype == np.float32, "Conv output should be float32"
        assert flattened_output.dtype == np.float32, "Flattened output should be float32"


class TestTensorCNNShapeHandling:
    """Test CNN operations with various tensor shapes."""
    
    def test_conv2d_with_different_input_shapes(self):
        """Test Conv2D with different input tensor shapes."""
        # Test with different input sizes
        test_cases = [
            (Tensor([[1.0, 2.0], [3.0, 4.0]]), (2, 2), (1, 1)),  # Minimal size
            (Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), (2, 2), (2, 2)),  # Standard size
            (Tensor(np.random.rand(5, 5).astype(np.float32)), (3, 3), (3, 3)),  # Larger input
        ]
        
        for input_tensor, kernel_size, expected_shape in test_cases:
            conv_layer = Conv2D(kernel_size=kernel_size)
            output = conv_layer(input_tensor)
            
            assert output.shape == expected_shape, f"For input {input_tensor.shape} and kernel {kernel_size}, expected {expected_shape}, got {output.shape}"
            assert isinstance(output, Tensor), "Output should be a Tensor"
    
    def test_flatten_with_different_shapes(self):
        """Test flatten with different tensor shapes."""
        test_cases = [
            (Tensor([[1.0, 2.0]]), (1, 2)),  # 1x2 → 1x2
            (Tensor([[1.0, 2.0], [3.0, 4.0]]), (1, 4)),  # 2x2 → 1x4
            (Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), (1, 6)),  # 2x3 → 1x6
        ]
        
        for input_tensor, expected_shape in test_cases:
            output = flatten(input_tensor)
            
            assert output.shape == expected_shape, f"For input {input_tensor.shape}, expected {expected_shape}, got {output.shape}"
            assert isinstance(output, Tensor), "Output should be a Tensor"


class TestTensorCNNDataTypes:
    """Test CNN operations with different tensor data types."""
    
    def test_conv2d_preserves_data_types(self):
        """Test that Conv2D preserves appropriate data types."""
        # Create input with specific dtype
        input_data = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]], dtype=np.float32)
        input_tensor = Tensor(input_data)
        
        conv_layer = Conv2D(kernel_size=(2, 2))
        output = conv_layer(input_tensor)
        
        # Verify data type consistency
        assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
        assert input_tensor.dtype == np.float32, f"Input should be float32, got {input_tensor.dtype}"
    
    def test_flatten_preserves_data_types(self):
        """Test that flatten preserves tensor data types."""
        # Test with float32
        input_float = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        output_float = flatten(input_float)
        assert output_float.dtype == np.float32, f"Expected float32, got {output_float.dtype}"
        
        # Test with int32
        input_int = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
        output_int = flatten(input_int)
        assert output_int.dtype == np.int32, f"Expected int32, got {output_int.dtype}"


class TestTensorCNNErrorHandling:
    """Test error handling in CNN operations with real tensors."""
    
    def test_conv2d_with_minimal_valid_tensor(self):
        """Test Conv2D with minimal valid tensor input."""
        # Test with minimal valid input (2x2 with 2x2 kernel)
        minimal_input = Tensor([[1.0, 2.0], [3.0, 4.0]])
        conv_layer = Conv2D(kernel_size=(2, 2))
        
        # Should produce 1x1 output
        output = conv_layer(minimal_input)
        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
        assert isinstance(output, Tensor), "Output should be a Tensor"
    
    def test_conv2d_naive_with_edge_case_shapes(self):
        """Test conv2d_naive with edge case but valid shapes."""
        # Test with minimal valid case
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        kernel_data = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        
        # Should produce 1x1 output
        result = conv2d_naive(input_data, kernel_data)
        assert result.shape == (1, 1), f"Expected (1, 1), got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be numpy array"


class TestTensorCNNRealisticScenarios:
    """Test CNN operations with realistic tensor scenarios."""
    
    def test_image_processing_pipeline(self):
        """Test CNN operations with image-like tensor data."""
        # Create realistic image-like data (8x8 "image")
        image_data = np.random.rand(8, 8).astype(np.float32)
        image_tensor = Tensor(image_data)
        
        # Apply 3x3 convolution (common in CNNs)
        conv_layer = Conv2D(kernel_size=(3, 3))
        features = conv_layer(image_tensor)
        
        # Flatten for fully connected layer
        flattened = flatten(features)
        
        # Verify realistic shapes
        assert features.shape == (6, 6), f"Expected (6, 6) feature map, got {features.shape}"
        assert flattened.shape == (1, 36), f"Expected (1, 36) flattened, got {flattened.shape}"
        
        # Verify realistic data ranges
        assert np.all(np.isfinite(features.data)), "Features should be finite"
        assert np.all(np.isfinite(flattened.data)), "Flattened data should be finite"
    
    def test_multiple_convolutions(self):
        """Test multiple convolution operations in sequence."""
        # Start with larger input
        input_tensor = Tensor(np.random.rand(6, 6).astype(np.float32))
        
        # Apply first convolution
        conv1 = Conv2D(kernel_size=(3, 3))
        features1 = conv1(input_tensor)
        
        # Apply second convolution
        conv2 = Conv2D(kernel_size=(2, 2))
        features2 = conv2(features1)
        
        # Verify shape progression
        assert features1.shape == (4, 4), f"First conv should produce (4, 4), got {features1.shape}"
        assert features2.shape == (3, 3), f"Second conv should produce (3, 3), got {features2.shape}"
        
        # Verify all are tensors
        assert isinstance(features1, Tensor), "First features should be Tensor"
        assert isinstance(features2, Tensor), "Second features should be Tensor"
    
    def test_conv_to_dense_integration_preparation(self):
        """Test CNN output preparation for dense layer integration."""
        # Create input that will work with dense layers
        input_tensor = Tensor(np.random.rand(5, 5).astype(np.float32))
        
        # Apply convolution
        conv_layer = Conv2D(kernel_size=(2, 2))
        conv_output = conv_layer(input_tensor)
        
        # Flatten for dense layer
        flattened = flatten(conv_output)
        
        # Verify shape is suitable for dense layer (batch_size, features)
        assert len(flattened.shape) == 2, f"Flattened should be 2D, got {flattened.shape}"
        assert flattened.shape[0] == 1, f"Batch size should be 1, got {flattened.shape[0]}"
        
        # Verify data is ready for dense layer consumption
        assert flattened.dtype == np.float32, "Data should be float32 for dense layers"
        assert np.all(np.isfinite(flattened.data)), "Data should be finite for dense layers" 
"""
Integration Tests - CNN and Networks

Tests real integration between CNN and Network modules.
Uses actual TinyTorch components to verify they work together correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Softmax, Sigmoid, Tanh
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential
from tinytorch.core.cnn import Conv2D, flatten


class TestCNNNetworkIntegration:
    """Test real integration between CNN layers and Networks."""
    
    def test_conv2d_in_sequential_network(self):
        """Test Conv2D layer works within Sequential network."""
        # Create a simple CNN architecture: Conv2D -> ReLU -> Flatten -> Dense
        network = Sequential([
            Conv2D(kernel_size=(3, 3)),
            ReLU(),
            lambda x: flatten(x),  # Flatten function as lambda
            Dense(input_size=36, output_size=10)
        ])
        
        # Test with sample input
        input_image = Tensor(np.random.randn(8, 8))
        output = network(input_image)
        
        # Verify integration
        assert isinstance(output, Tensor), "Sequential with Conv2D should return Tensor"
        assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
        assert not np.any(np.isnan(output.data)), "CNN network should not produce NaN"
    
    def test_multiple_conv2d_layers_in_network(self):
        """Test multiple Conv2D layers in a Sequential network."""
        # Create deeper CNN: Conv2D -> ReLU -> Conv2D -> ReLU -> Flatten -> Dense
        network = Sequential([
            Conv2D(kernel_size=(3, 3)),  # 10x10 -> 8x8
            ReLU(),
            Conv2D(kernel_size=(3, 3)),  # 8x8 -> 6x6
            ReLU(),
            lambda x: flatten(x),        # 6x6 -> 36
            Dense(input_size=36, output_size=5)
        ])
        
        # Test with larger input
        input_image = Tensor(np.random.randn(10, 10))
        output = network(input_image)
        
        # Verify deep CNN integration
        assert isinstance(output, Tensor), "Deep CNN network should return Tensor"
        assert output.shape == (1, 5), f"Expected shape (1, 5), got {output.shape}"
        assert not np.any(np.isnan(output.data)), "Deep CNN should not produce NaN"
    
    def test_conv2d_with_different_activations(self):
        """Test Conv2D with different activation functions in networks."""
        activations = [ReLU(), Sigmoid(), Tanh()]
        
        for activation in activations:
            network = Sequential([
                Conv2D(kernel_size=(2, 2)),
                activation,
                lambda x: flatten(x),
                Dense(input_size=16, output_size=3)
            ])
            
            input_image = Tensor(np.random.randn(5, 5))
            output = network(input_image)
            
            assert isinstance(output, Tensor), f"Network with {activation.__class__.__name__} should return Tensor"
            assert output.shape == (1, 3), f"Expected shape (1, 3), got {output.shape}"
            assert not np.any(np.isnan(output.data)), f"Network with {activation.__class__.__name__} should not produce NaN"
    
    def test_conv2d_batch_processing_in_network(self):
        """Test Conv2D handles batch processing within networks."""
        # Create network
        network = Sequential([
            Conv2D(kernel_size=(2, 2)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=9, output_size=2)
        ])
        
        # Test with batch input (simulate multiple images)
        batch_images = []
        for _ in range(4):
            batch_images.append(Tensor(np.random.randn(4, 4)))
        
        # Process each image in the batch
        batch_outputs = []
        for image in batch_images:
            output = network(image)
            batch_outputs.append(output)
        
        # Verify batch processing
        assert len(batch_outputs) == 4, "Should process all images in batch"
        for i, output in enumerate(batch_outputs):
            assert isinstance(output, Tensor), f"Batch item {i} should return Tensor"
            assert output.shape == (1, 2), f"Batch item {i} should have shape (1, 2)"
            assert not np.any(np.isnan(output.data)), f"Batch item {i} should not produce NaN"
    
    def test_conv2d_different_kernel_sizes_in_network(self):
        """Test Conv2D with different kernel sizes in networks."""
        kernel_sizes = [(2, 2), (3, 3), (5, 5)]
        input_sizes = [6, 8, 10]  # Adjust input size for each kernel
        
        for kernel_size, input_size in zip(kernel_sizes, input_sizes):
            # Calculate expected output size after convolution
            conv_output_size = input_size - kernel_size[0] + 1
            flatten_size = conv_output_size * conv_output_size
            
            network = Sequential([
                Conv2D(kernel_size=kernel_size),
                ReLU(),
                lambda x: flatten(x),
                Dense(input_size=flatten_size, output_size=1)
            ])
            
            input_image = Tensor(np.random.randn(input_size, input_size))
            output = network(input_image)
            
            assert isinstance(output, Tensor), f"Network with kernel {kernel_size} should return Tensor"
            assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"
            assert not np.any(np.isnan(output.data)), f"Network with kernel {kernel_size} should not produce NaN"


class TestCNNNetworkComposition:
    """Test composition of CNN components with different network architectures."""
    
    def test_feature_extraction_pipeline(self):
        """Test CNN as feature extractor with dense classifier."""
        # Feature extractor: Conv2D -> ReLU -> Flatten
        feature_extractor = Sequential([
            Conv2D(kernel_size=(3, 3)),
            ReLU(),
            lambda x: flatten(x)
        ])
        
        # Classifier: Dense -> ReLU -> Dense
        classifier = Sequential([
            Dense(input_size=36, output_size=16),
            ReLU(),
            Dense(input_size=16, output_size=3)
        ])
        
        # Test feature extraction
        input_image = Tensor(np.random.randn(8, 8))
        features = feature_extractor(input_image)
        
        assert isinstance(features, Tensor), "Feature extractor should return Tensor"
        assert features.shape == (1, 36), f"Expected features shape (1, 36), got {features.shape}"
        
        # Test classification
        predictions = classifier(features)
        
        assert isinstance(predictions, Tensor), "Classifier should return Tensor"
        assert predictions.shape == (1, 3), f"Expected predictions shape (1, 3), got {predictions.shape}"
        assert not np.any(np.isnan(predictions.data)), "Complete pipeline should not produce NaN"
    
    def test_cnn_network_parameter_count(self):
        """Test that CNN networks have reasonable parameter counts."""
        # Create CNN network
        network = Sequential([
            Conv2D(kernel_size=(3, 3)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=36, output_size=10)
        ])
        
        # Test with input to ensure network is initialized
        input_image = Tensor(np.random.randn(8, 8))
        output = network(input_image)
        
        # CNN should have fewer parameters than equivalent fully connected
        # Conv2D(3x3) has 9 parameters
        # Dense(36->10) has 36*10 + 10 = 370 parameters
        # Total: ~379 parameters vs ~6400 for fully connected (64->10)
        
        assert isinstance(output, Tensor), "CNN network should work"
        assert output.shape == (1, 10), "CNN network should produce correct output shape"
        
        # Verify CNN efficiency (this is conceptual - actual parameter counting 
        # would require more sophisticated tracking)
        conv_layer = network.layers[0]
        dense_layer = network.layers[3]
        
        # Conv2D kernel should be 3x3
        assert conv_layer.kernel.shape == (3, 3), "Conv2D should have correct kernel shape"
        
        # Dense layer should connect flattened features to output
        assert dense_layer.weights.shape == (36, 10), "Dense layer should have correct weight shape"
    
    def test_cnn_vs_dense_comparison(self):
        """Test CNN vs pure dense network comparison."""
        # Create CNN network
        cnn_network = Sequential([
            Conv2D(kernel_size=(2, 2)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=9, output_size=5)
        ])
        
        # Create equivalent dense network (much larger)
        dense_network = Sequential([
            Dense(input_size=16, output_size=16),  # Simulate full connectivity
            ReLU(),
            Dense(input_size=16, output_size=5)
        ])
        
        # Test with same input
        input_image = Tensor(np.random.randn(4, 4))
        input_flat = flatten(input_image)  # For dense network
        
        cnn_output = cnn_network(input_image)
        dense_output = dense_network(input_flat)
        
        # Both should work but CNN is more parameter-efficient
        assert isinstance(cnn_output, Tensor), "CNN network should work"
        assert isinstance(dense_output, Tensor), "Dense network should work"
        assert cnn_output.shape == dense_output.shape, "Both networks should have same output shape"
        
        # CNN should have fewer parameters (conceptually)
        # Conv2D(2x2) + Dense(9->5) = 4 + 45 = 49 parameters
        # Dense(16->16) + Dense(16->5) = 256 + 80 = 336 parameters
        # CNN is ~7x more parameter efficient!


class TestCNNNetworkEdgeCases:
    """Test edge cases and error handling in CNN-Network integration."""
    
    def test_minimal_input_size(self):
        """Test CNN networks with minimal valid input sizes."""
        # Minimal case: 2x2 input with 2x2 kernel -> 1x1 output
        network = Sequential([
            Conv2D(kernel_size=(2, 2)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=1, output_size=1)
        ])
        
        input_image = Tensor(np.random.randn(2, 2))
        output = network(input_image)
        
        assert isinstance(output, Tensor), "Minimal CNN should work"
        assert output.shape == (1, 1), "Minimal CNN should produce scalar output"
    
    def test_shape_compatibility_validation(self):
        """Test that CNN networks properly validate shape compatibility."""
        # This tests the integration between Conv2D output and Dense input
        network = Sequential([
            Conv2D(kernel_size=(2, 2)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=9, output_size=3)  # Expects 3x3 = 9 from 4x4->2x2 conv
        ])
        
        # Correct input size
        correct_input = Tensor(np.random.randn(4, 4))
        output = network(correct_input)
        
        assert isinstance(output, Tensor), "Correct input should work"
        assert output.shape == (1, 3), "Correct input should produce expected output"
        
        # The network should handle the shape transformation correctly
        # Conv2D(2x2) on 4x4 input -> 3x3 output
        # Flatten 3x3 -> 9 features
        # Dense(9->3) -> 3 outputs
    
    def test_data_type_preservation(self):
        """Test that CNN networks preserve data types properly."""
        network = Sequential([
            Conv2D(kernel_size=(2, 2)),
            ReLU(),
            lambda x: flatten(x),
            Dense(input_size=4, output_size=2)
        ])
        
        # Test with float32 input
        input_image = Tensor(np.random.randn(3, 3).astype(np.float32))
        output = network(input_image)
        
        assert isinstance(output, Tensor), "Network should preserve tensor type"
        assert output.data.dtype == np.float32, "Network should preserve float32 dtype"
        assert output.shape == (1, 2), "Network should produce correct output shape"


def test_integration_summary():
    """Summary test demonstrating complete CNN-Network integration."""
    print("🎯 Integration Summary: CNN ↔ Networks")
    print("=" * 50)
    
    # Create a realistic CNN architecture
    print("🏗️  Building CNN architecture...")
    cnn_classifier = Sequential([
        Conv2D(kernel_size=(3, 3)),    # Feature extraction (8x8 -> 6x6)
        ReLU(),                         # Nonlinearity
        Conv2D(kernel_size=(2, 2)),    # Further feature extraction (6x6 -> 5x5)
        ReLU(),                         # Nonlinearity
        lambda x: flatten(x),          # Prepare for dense layers (5x5 -> 25)
        Dense(input_size=25, output_size=8),  # Feature compression (25 -> 8)
        ReLU(),                         # Nonlinearity
        Dense(input_size=8, output_size=3)    # Final classification (8 -> 3)
    ])
    
    # Test with realistic input
    print("📊 Testing with sample input...")
    input_image = Tensor(np.random.randn(8, 8))
    output = cnn_classifier(input_image)
    
    # Verify complete integration
    assert isinstance(output, Tensor), "Complete CNN should return Tensor"
    assert output.shape == (1, 3), "Complete CNN should produce classification output"
    assert not np.any(np.isnan(output.data)), "Complete CNN should not produce NaN"
    
    print("✅ CNN-Network integration successful!")
    print(f"   Input: {input_image.shape} -> Output: {output.shape}")
    print("   Architecture: Conv2D -> ReLU -> Conv2D -> ReLU -> Flatten -> Dense -> ReLU -> Dense")
    print("   Components: CNN layers, activations, dense layers, sequential composition")
    print("🎉 Ready for real computer vision applications!")


if __name__ == "__main__":
    test_integration_summary() 
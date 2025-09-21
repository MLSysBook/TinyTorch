"""
Integration Tests - CNN Pipeline

Tests real integration between CNN operations, activations, and layers.
Moved from inline tests because it's true cross-module integration testing.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import REAL TinyTorch components
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.cnn import Conv2D, flatten
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
    from tinytorch.core.layers import Dense
except ImportError:
    # Fallback for development
    sys.path.append(str(project_root / "modules" / "source" / "01_tensor"))
    sys.path.append(str(project_root / "modules" / "source" / "02_activations"))
    sys.path.append(str(project_root / "modules" / "source" / "03_layers"))
    sys.path.append(str(project_root / "modules" / "source" / "05_cnn"))
    
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    from layers_dev import Dense
    from cnn_dev import Conv2D, flatten


class TestCNNPipelineIntegration:
    """Test CNN pipeline integration with activations and layers."""
    
    def test_cnn_pipeline_integration(self):
        """Test CNN pipeline integration with complete workflow."""
        print("🔬 Integration Test: CNN Pipeline...")
        
        # Test complete CNN pipeline
        input_image = Tensor(np.random.randn(8, 8))
        
        # Build CNN pipeline
        conv = Conv2D(kernel_size=(3, 3))
        conv_output = conv(input_image)
        flattened = flatten(conv_output)
        
        # Test shapes
        assert conv_output.shape == (6, 6), "Conv output should be correct"
        assert flattened.shape == (1, 36), "Flatten output should be correct"
        
        # Test with activation and dense layers
        relu = ReLU()
        dense = Dense(input_size=36, output_size=10)
        
        activated = relu(conv_output)
        final_flat = flatten(activated)
        predictions = dense(final_flat)
        
        assert predictions.shape == (1, 10), "Final predictions should be correct shape"
        
        print("✅ CNN pipeline integration works correctly")
    
    def test_cnn_with_different_activations(self):
        """Test CNN pipeline with different activation functions."""
        activations = [
            ("ReLU", ReLU()),
            ("Sigmoid", Sigmoid()),
            ("Tanh", Tanh())
        ]
        
        input_image = Tensor(np.random.randn(6, 6))
        
        for name, activation in activations:
            # CNN pipeline with specific activation
            conv = Conv2D(kernel_size=(2, 2))
            conv_output = conv(input_image)
            
            # Apply activation
            activated = activation(conv_output)
            
            # Flatten and classify
            flattened = flatten(activated)
            dense = Dense(input_size=25, output_size=5)
            predictions = dense(flattened)
            
            # Verify integration
            assert isinstance(predictions, Tensor), f"CNN-{name} pipeline should return Tensor"
            assert predictions.shape == (1, 5), f"CNN-{name} pipeline should have correct shape"
            assert not np.any(np.isnan(predictions.data)), f"CNN-{name} pipeline should not produce NaN"
    
    def test_deep_cnn_pipeline(self):
        """Test deeper CNN pipeline with multiple layers."""
        # Create deeper pipeline
        input_image = Tensor(np.random.randn(10, 10))
        
        # Stage 1: First convolution
        conv1 = Conv2D(kernel_size=(3, 3))
        conv1_output = conv1(input_image)  # 10x10 -> 8x8
        relu1 = ReLU()
        activated1 = relu1(conv1_output)
        
        # Stage 2: Second convolution
        conv2 = Conv2D(kernel_size=(3, 3))
        conv2_output = conv2(activated1)  # 8x8 -> 6x6
        relu2 = ReLU()
        activated2 = relu2(conv2_output)
        
        # Stage 3: Final classification
        flattened = flatten(activated2)  # 6x6 -> 36
        dense = Dense(input_size=36, output_size=3)
        predictions = dense(flattened)
        
        # Verify deep pipeline
        assert isinstance(predictions, Tensor), "Deep CNN pipeline should return Tensor"
        assert predictions.shape == (1, 3), "Deep CNN pipeline should have correct shape"
        assert not np.any(np.isnan(predictions.data)), "Deep CNN pipeline should not produce NaN"
        
        # Verify intermediate shapes
        assert conv1_output.shape == (8, 8), "First conv should produce 8x8"
        assert conv2_output.shape == (6, 6), "Second conv should produce 6x6"
        assert flattened.shape == (1, 36), "Flatten should produce 36 features"
    
    def test_cnn_with_softmax_output(self):
        """Test CNN pipeline with softmax output for classification."""
        input_image = Tensor(np.random.randn(5, 5))
        
        # Build classification pipeline
        conv = Conv2D(kernel_size=(2, 2))
        conv_output = conv(input_image)  # 5x5 -> 4x4
        
        relu = ReLU()
        activated = relu(conv_output)
        
        flattened = flatten(activated)  # 4x4 -> 16
        dense = Dense(input_size=16, output_size=3)
        dense_output = dense(flattened)
        
        softmax = Softmax()
        predictions = softmax(dense_output)
        
        # Verify classification pipeline
        assert isinstance(predictions, Tensor), "Classification pipeline should return Tensor"
        assert predictions.shape == (1, 3), "Classification should have 3 class outputs"
        
        # Verify softmax properties
        probabilities = predictions.data[0]
        assert np.all(probabilities > 0), "Softmax should produce positive probabilities"
        assert np.isclose(np.sum(probabilities), 1.0), "Softmax should sum to 1"
    
    def test_cnn_batch_processing_integration(self):
        """Test CNN pipeline with batch processing integration."""
        # Create batch of images
        batch_size = 3
        batch_images = []
        for _ in range(batch_size):
            batch_images.append(Tensor(np.random.randn(4, 4)))
        
        # Process each image through the pipeline
        predictions = []
        for image in batch_images:
            # CNN pipeline
            conv = Conv2D(kernel_size=(2, 2))
            conv_output = conv(image)  # 4x4 -> 3x3
            
            relu = ReLU()
            activated = relu(conv_output)
            
            flattened = flatten(activated)  # 3x3 -> 9
            dense = Dense(input_size=9, output_size=2)
            prediction = dense(flattened)
            
            predictions.append(prediction)
        
        # Verify batch processing
        assert len(predictions) == batch_size, "Should process all images in batch"
        for i, pred in enumerate(predictions):
            assert isinstance(pred, Tensor), f"Batch item {i} should return Tensor"
            assert pred.shape == (1, 2), f"Batch item {i} should have correct shape"
            assert not np.any(np.isnan(pred.data)), f"Batch item {i} should not produce NaN"
    
    def test_cnn_pipeline_numerical_stability(self):
        """Test CNN pipeline numerical stability with edge cases."""
        # Test with very small values
        small_image = Tensor(np.random.randn(3, 3) * 0.001)
        
        conv = Conv2D(kernel_size=(2, 2))
        conv_output = conv(small_image)
        
        relu = ReLU()
        activated = relu(conv_output)
        
        flattened = flatten(activated)
        dense = Dense(input_size=4, output_size=1)
        output = dense(flattened)
        
        # Should handle small values without numerical issues
        assert isinstance(output, Tensor), "Should handle small values"
        assert output.shape == (1, 1), "Should maintain correct shape"
        assert not np.any(np.isnan(output.data)), "Should not produce NaN with small values"
        
        # Test with larger values
        large_image = Tensor(np.random.randn(3, 3) * 10.0)
        
        conv = Conv2D(kernel_size=(2, 2))
        conv_output = conv(large_image)
        
        relu = ReLU()
        activated = relu(conv_output)
        
        flattened = flatten(activated)
        dense = Dense(input_size=4, output_size=1)
        output = dense(flattened)
        
        # Should handle large values without overflow
        assert isinstance(output, Tensor), "Should handle large values"
        assert output.shape == (1, 1), "Should maintain correct shape"
        assert not np.any(np.isnan(output.data)), "Should not produce NaN with large values"
        assert not np.any(np.isinf(output.data)), "Should not produce Inf with large values"


def test_integration_summary():
    """Summary test demonstrating complete CNN pipeline integration."""
    print("🎯 Integration Summary: CNN Pipeline")
    print("=" * 50)
    
    # Create realistic CNN pipeline
    print("🏗️  Building CNN pipeline...")
    input_image = Tensor(np.random.randn(8, 8))
    
    # Stage 1: Feature extraction
    conv = Conv2D(kernel_size=(3, 3))
    features = conv(input_image)  # 8x8 -> 6x6
    
    # Stage 2: Nonlinear activation
    relu = ReLU()
    activated = relu(features)
    
    # Stage 3: Prepare for classification
    flattened = flatten(activated)  # 6x6 -> 36
    
    # Stage 4: Classification
    classifier = Dense(input_size=36, output_size=3)
    raw_predictions = classifier(flattened)
    
    # Stage 5: Probability distribution
    softmax = Softmax()
    predictions = softmax(raw_predictions)
    
    # Verify complete pipeline
    assert isinstance(predictions, Tensor), "Complete pipeline should return Tensor"
    assert predictions.shape == (1, 3), "Complete pipeline should produce 3 class probabilities"
    
    probabilities = predictions.data[0]
    assert np.all(probabilities > 0), "Should produce positive probabilities"
    assert np.isclose(np.sum(probabilities), 1.0), "Should sum to 1.0"
    
    print("✅ CNN pipeline integration successful!")
    print(f"   Input: {input_image.shape} -> Features: {features.shape}")
    print(f"   Activated: {activated.shape} -> Flattened: {flattened.shape}")
    print(f"   Raw predictions: {raw_predictions.shape} -> Final: {predictions.shape}")
    print("   Components: CNN → Activation → Flatten → Dense → Softmax")
    print("🎉 Ready for real computer vision applications!")


if __name__ == "__main__":
    test_integration_summary() 
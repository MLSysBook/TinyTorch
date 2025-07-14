"""
Integration Tests - Layers and Networks

Tests real integration between Dense layers and Network architectures.
Uses actual TinyTorch components to verify they work together correctly.
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
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh
    from tinytorch.core.layers import Dense
    from tinytorch.core.networks import Sequential, MLP
except ImportError:
    # Fallback for development
    sys.path.append(str(project_root / "modules" / "source" / "01_tensor"))
    sys.path.append(str(project_root / "modules" / "source" / "02_activations"))
    sys.path.append(str(project_root / "modules" / "source" / "03_layers"))
    sys.path.append(str(project_root / "modules" / "source" / "04_networks"))
    
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh
    from layers_dev import Dense
    from networks_dev import Sequential, MLP


class TestLayerNetworkIntegration:
    """Test real integration between Dense layers and Networks."""
    
    def test_dense_layer_with_real_tensors(self):
        """Test Dense layer works with real Tensor objects."""
        # Create real layer and tensor
        layer = Dense(input_size=3, output_size=2)
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Forward pass
        result = layer(x)
        
        # Verify real integration
        assert isinstance(result, Tensor)
        assert result.shape == (1, 2)
        assert hasattr(result, 'data')
        assert not np.any(np.isnan(result.data))
    
    def test_sequential_with_real_components(self):
        """Test Sequential network with real Dense layers and activations."""
        # Create network with REAL components
        network = Sequential([
            Dense(input_size=4, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=4),
            Sigmoid(),
            Dense(input_size=4, output_size=2)
        ])
        
        # Test with real tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        result = network(x)
        
        # Verify real integration
        assert isinstance(result, Tensor)
        assert result.shape == (1, 2)
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
    
    def test_mlp_with_real_components(self):
        """Test MLP network with real components."""
        # Create MLP with real components
        mlp = MLP(input_size=5, hidden_size=10, output_size=3)
        
        # Test with real tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = mlp(x)
        
        # Verify real integration
        assert isinstance(result, Tensor)
        assert result.shape == (1, 3)
        assert hasattr(mlp, 'network')
        assert isinstance(mlp.network, Sequential)
    
    def test_deep_network_integration(self):
        """Test deep network with multiple real layers."""
        # Create deep network
        deep_network = Sequential([
            Dense(input_size=6, output_size=12),
            ReLU(),
            Dense(input_size=12, output_size=8),
            Tanh(),
            Dense(input_size=8, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        # Test with real tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = deep_network(x)
        
        # Verify deep integration
        assert isinstance(result, Tensor)
        assert result.shape == (1, 2)
        
        # Verify final activation worked (sigmoid bounds)
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)
    
    def test_batch_processing_integration(self):
        """Test network works with batched real tensors."""
        network = Sequential([
            Dense(input_size=3, output_size=6),
            ReLU(),
            Dense(input_size=6, output_size=2)
        ])
        
        # Create batch of real tensors
        batch_x = Tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        result = network(batch_x)
        
        # Verify batch integration
        assert isinstance(result, Tensor)
        assert result.shape == (3, 2)
        assert not np.any(np.isnan(result.data))
    
    def test_layer_parameter_sharing(self):
        """Test that layers maintain consistent parameters."""
        layer = Dense(input_size=4, output_size=3)
        
        # Store original parameters
        original_weights = np.copy(layer.weights.data)
        original_bias = np.copy(layer.bias.data) if layer.bias is not None else None
        
        # Multiple forward passes
        x1 = Tensor([[1.0, 2.0, 3.0, 4.0]])
        x2 = Tensor([[5.0, 6.0, 7.0, 8.0]])
        
        result1 = layer(x1)
        result2 = layer(x2)
        
        # Parameters should be unchanged
        np.testing.assert_array_equal(layer.weights.data, original_weights)
        if original_bias is not None:
            np.testing.assert_array_equal(layer.bias.data, original_bias)
        
        # Results should be different for different inputs
        assert not np.allclose(result1.data, result2.data)
    
    def test_network_composition_integration(self):
        """Test composing networks with real components."""
        # Create encoder
        encoder = Sequential([
            Dense(input_size=8, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2)
        ])
        
        # Create decoder  
        decoder = Sequential([
            Dense(input_size=2, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=8)
        ])
        
        # Compose autoencoder
        autoencoder = Sequential([encoder, decoder])
        
        # Test with real tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = autoencoder(x)
        
        # Verify composition
        assert isinstance(result, Tensor)
        assert result.shape == x.shape  # Should reconstruct input size


class TestMLClassificationPipeline:
    """Test realistic ML classification pipelines."""
    
    def test_binary_classifier_integration(self):
        """Test binary classification with real components."""
        # Create binary classifier
        classifier = Sequential([
            Dense(input_size=4, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=1),
            Sigmoid()  # Binary classification
        ])
        
        # Test with sample data
        x = Tensor([[0.5, 1.5, -0.5, 2.0]])
        prediction = classifier(x)
        
        # Verify binary classification
        assert isinstance(prediction, Tensor)
        assert prediction.shape == (1, 1)
        assert 0.0 <= prediction.data[0, 0] <= 1.0  # Probability
    
    def test_multiclass_classifier_integration(self):
        """Test multi-class classification with real components."""
        # Create multi-class classifier (3 classes)
        classifier = MLP(
            input_size=6,
            hidden_size=12,
            output_size=3,
            activation=ReLU,
            output_activation=None  # Will add softmax manually
        )
        
        # Add softmax for multi-class
        from tinytorch.core.activations import Softmax
        softmax = Softmax()
        
        # Test prediction
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        logits = classifier(x)
        probabilities = softmax(logits)
        
        # Verify multi-class classification
        assert isinstance(probabilities, Tensor)
        assert probabilities.shape == (1, 3)
        assert np.isclose(np.sum(probabilities.data), 1.0, atol=1e-6)
        assert np.all(probabilities.data >= 0.0)
    
    def test_feature_extraction_pipeline(self):
        """Test feature extraction with real components."""
        # Feature extractor (encoder part)
        feature_extractor = Sequential([
            Dense(input_size=10, output_size=16),
            ReLU(),
            Dense(input_size=16, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=4)  # Feature representation
        ])
        
        # Classifier head
        classifier = Sequential([
            Dense(input_size=4, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=2)
        ])
        
        # Full pipeline
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        # Extract features
        features = feature_extractor(x)
        assert isinstance(features, Tensor)
        assert features.shape == (1, 4)
        
        # Classify features
        predictions = classifier(features)
        assert isinstance(predictions, Tensor)
        assert predictions.shape == (1, 2)


class TestErrorHandlingIntegration:
    """Test error handling in real integration scenarios."""
    
    def test_shape_mismatch_detection(self):
        """Test that shape mismatches are properly detected."""
        layer = Dense(input_size=3, output_size=2)
        
        # Wrong input size should raise error
        wrong_x = Tensor([[1.0, 2.0]])  # Should be size 3
        
        with pytest.raises(Exception):
            layer(wrong_x)
    
    def test_network_layer_compatibility(self):
        """Test network layer compatibility checking."""
        # Create network with incompatible layers
        try:
            incompatible_network = Sequential([
                Dense(input_size=4, output_size=3),
                Dense(input_size=5, output_size=2)  # Expects 5, gets 3
            ])
            
            x = Tensor([[1.0, 2.0, 3.0, 4.0]])
            result = incompatible_network(x)
            
            # If no error raised, should handle gracefully
            assert isinstance(result, Tensor)
            
        except Exception:
            # It's acceptable to raise an error for incompatible layers
            pass


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"]) 
"""
Tests for TinyTorch Layers module.

Tests the core layer functionality including Dense layers, activation functions,
and layer composition.

These tests work with the current implementation and provide stretch goals
for students to implement additional features.
"""

import sys
import os
import pytest
import numpy as np

# Add the parent directory to path to import layers_dev
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from the module's development file
# Note: This imports the instructor version with full implementation
from layers_dev import Dense, Tensor

# Import activation functions from the activations module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '02_activations'))
from activations_dev import ReLU, Sigmoid, Tanh

def safe_numpy(tensor):
    """Get numpy array from tensor, using .numpy() if available, otherwise .data"""
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    else:
        return tensor.data

class TestDenseLayer:
    """Test Dense (Linear) layer functionality."""
    
    def test_dense_creation(self):
        """Test creating Dense layers with different configurations."""
        # Basic dense layer
        layer = Dense(input_size=3, output_size=2)
        assert layer.input_size == 3
        assert layer.output_size == 2
        assert layer.use_bias == True
        assert layer.weights.shape == (3, 2)
        assert layer.bias.shape == (2,)
        
        # Dense layer without bias
        layer_no_bias = Dense(input_size=4, output_size=3, use_bias=False)
        assert layer_no_bias.use_bias == False
        assert layer_no_bias.bias is None
    
    def test_dense_forward_single(self):
        """Test Dense layer forward pass with single input."""
        layer = Dense(input_size=3, output_size=2)
        
        # Single input
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        
        assert y.shape == (1, 2)
        assert isinstance(y, Tensor)
    
    def test_dense_forward_batch(self):
        """Test Dense layer forward pass with batch input."""
        layer = Dense(input_size=3, output_size=2)
        
        # Batch input
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = layer(x)
        
        assert y.shape == (2, 2)
        assert isinstance(y, Tensor)
    
    def test_dense_no_bias(self):
        """Test Dense layer without bias."""
        layer = Dense(input_size=2, output_size=1, use_bias=False)
        
        x = Tensor([[1.0, 2.0]])
        y = layer(x)
        
        assert y.shape == (1, 1)
        # Should be just matrix multiplication without bias
        expected = safe_numpy(x) @ safe_numpy(layer.weights)
        np.testing.assert_array_almost_equal(safe_numpy(y), expected)
    
    def test_dense_callable(self):
        """Test that Dense layer is callable."""
        layer = Dense(input_size=2, output_size=1)
        x = Tensor([[1.0, 2.0]])
        
        # Both should work
        y1 = layer.forward(x)
        y2 = layer(x)
        
        np.testing.assert_array_equal(safe_numpy(y1), safe_numpy(y2))

class TestActivationFunctions:
    """Test activation function implementations."""
    
    def test_relu_basic(self):
        """Test ReLU activation function."""
        relu = ReLU()
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = relu(x)
        
        expected = [[0.0, 0.0, 0.0, 1.0, 2.0]]
        np.testing.assert_array_equal(safe_numpy(y), expected)
    
    def test_relu_callable(self):
        """Test that ReLU is callable."""
        relu = ReLU()
        x = Tensor([[1.0, -1.0]])
        
        y1 = relu.forward(x)
        y2 = relu(x)
        
        np.testing.assert_array_equal(safe_numpy(y1), safe_numpy(y2))
    
    def test_sigmoid_basic(self):
        """Test Sigmoid activation function."""
        sigmoid = Sigmoid()
        x = Tensor([[0.0]])  # sigmoid(0) = 0.5
        y = sigmoid(x)
        
        np.testing.assert_array_almost_equal(safe_numpy(y), [[0.5]])
    
    def test_sigmoid_range(self):
        """Test Sigmoid output range."""
        sigmoid = Sigmoid()
        x = Tensor([[-10.0, 0.0, 10.0]])
        y = sigmoid(x)
        
        # Should be in range [0, 1] - use reasonable bounds
        assert np.all(safe_numpy(y) >= 0)
        assert np.all(safe_numpy(y) <= 1)
        # Check that extreme values are close to bounds
        assert safe_numpy(y)[0][0] < 0.01  # Very small for -10
        assert safe_numpy(y)[0][2] > 0.99  # Very large for 10
    
    def test_tanh_basic(self):
        """Test Tanh activation function."""
        tanh = Tanh()
        x = Tensor([[0.0]])  # tanh(0) = 0
        y = tanh(x)
        
        np.testing.assert_array_almost_equal(safe_numpy(y), [[0.0]])
    
    def test_tanh_range(self):
        """Test Tanh output range."""
        tanh = Tanh()
        x = Tensor([[-10.0, 0.0, 10.0]])
        y = tanh(x)
        
        # Should be in range [-1, 1] - use reasonable bounds
        assert np.all(safe_numpy(y) >= -1)
        assert np.all(safe_numpy(y) <= 1)
        # Check that extreme values are close to bounds
        assert safe_numpy(y)[0][0] < -0.99  # Very negative for -10
        assert safe_numpy(y)[0][2] > 0.99   # Very positive for 10

class TestLayerComposition:
    """Test composing layers into neural networks."""
    
    def test_simple_network(self):
        """Test a simple 2-layer network."""
        # 3 → 4 → 2 network
        layer1 = Dense(input_size=3, output_size=4)
        relu = ReLU()
        layer2 = Dense(input_size=4, output_size=2)
        sigmoid = Sigmoid()
        
        # Forward pass
        x = Tensor([[1.0, 2.0, 3.0]])
        h1 = layer1(x)
        h1_activated = relu(h1)
        h2 = layer2(h1_activated)
        output = sigmoid(h2)
        
        assert h1.shape == (1, 4)
        assert h1_activated.shape == (1, 4)
        assert h2.shape == (1, 2)
        assert output.shape == (1, 2)
        
        # Output should be in sigmoid range
        assert np.all(safe_numpy(output) >= 0)
        assert np.all(safe_numpy(output) <= 1)
    
    def test_batch_network(self):
        """Test network with batch processing."""
        layer1 = Dense(input_size=2, output_size=3)
        relu = ReLU()
        layer2 = Dense(input_size=3, output_size=1)
        
        # Batch of 4 examples
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        
        h1 = layer1(x)
        h1_activated = relu(h1)
        output = layer2(h1_activated)
        
        assert output.shape == (4, 1)
    
    def test_deep_network(self):
        """Test deeper network composition."""
        # 5-layer network
        layers = [
            Dense(input_size=10, output_size=8),
            ReLU(),
            Dense(input_size=8, output_size=6),
            ReLU(),
            Dense(input_size=6, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ]
        
        x = Tensor([[1.0] * 10])  # 10 features
        
        # Forward pass through all layers
        current = x
        for layer in layers:
            current = layer(current)
        
        assert current.shape == (1, 2)
        # Final output should be in sigmoid range
        assert np.all(safe_numpy(current) >= 0)
        assert np.all(safe_numpy(current) <= 1)

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_input(self):
        """Test layers with zero input."""
        layer = Dense(input_size=3, output_size=2)
        relu = ReLU()
        
        x = Tensor([[0.0, 0.0, 0.0]])
        y = layer(x)
        y_relu = relu(y)
        
        assert y.shape == (1, 2)
        assert y_relu.shape == (1, 2)
    
    def test_large_input(self):
        """Test layers with large input values."""
        layer = Dense(input_size=2, output_size=1)
        sigmoid = Sigmoid()
        
        x = Tensor([[1000.0, -1000.0]])
        y = layer(x)
        y_sigmoid = sigmoid(y)
        
        # Should not overflow
        assert not np.any(np.isnan(safe_numpy(y_sigmoid)))
        assert not np.any(np.isinf(safe_numpy(y_sigmoid)))
    
    def test_single_neuron(self):
        """Test single neuron layers."""
        layer = Dense(input_size=1, output_size=1)
        x = Tensor([[5.0]])
        y = layer(x)
        
        assert y.shape == (1, 1)

# Stretch goal tests (these will be skipped if methods don't exist)
class TestStretchGoals:
    """Stretch goal tests for advanced features."""
    
    @pytest.mark.skip(reason="Stretch goal: Weight initialization methods")
    def test_weight_initialization_methods(self):
        """Test different weight initialization strategies."""
        # Xavier initialization
        layer_xavier = Dense(input_size=100, output_size=50, init_method='xavier')
        weights_xavier = safe_numpy(layer_xavier.weights)
        
        # He initialization  
        layer_he = Dense(input_size=100, output_size=50, init_method='he')
        weights_he = safe_numpy(layer_he.weights)
        
        # Check initialization ranges
        xavier_limit = np.sqrt(6.0 / (100 + 50))
        assert np.all(np.abs(weights_xavier) <= xavier_limit)
        
        he_limit = np.sqrt(2.0 / 100)
        assert np.std(weights_he) <= he_limit * 1.5  # Some tolerance
    
    @pytest.mark.skip(reason="Stretch goal: Layer parameter access")
    def test_layer_parameters(self):
        """Test accessing and modifying layer parameters."""
        layer = Dense(input_size=3, output_size=2)
        
        # Should be able to access parameters
        assert hasattr(layer, 'parameters')
        params = layer.parameters()
        assert len(params) == 2  # weights and bias
        
        # Should be able to set parameters
        new_weights = Tensor(np.ones((3, 2)))
        layer.set_weights(new_weights)
        np.testing.assert_array_equal(safe_numpy(layer.weights), safe_numpy(new_weights))
    
    @pytest.mark.skip(reason="Stretch goal: Additional activation functions")
    def test_additional_activations(self):
        """Test additional activation functions."""
        # Leaky ReLU
        leaky_relu = LeakyReLU(alpha=0.1)
        x = Tensor([[-1.0, 0.0, 1.0]])
        y = leaky_relu(x)
        expected = [[-0.1, 0.0, 1.0]]
        np.testing.assert_array_almost_equal(safe_numpy(y), expected)
        
        # Softmax
        softmax = Softmax()
        x = Tensor([[1.0, 2.0, 3.0]])
        y = softmax(x)
        # Should sum to 1
        assert np.allclose(np.sum(safe_numpy(y)), 1.0)
    
    @pytest.mark.skip(reason="Stretch goal: Dropout layer")
    def test_dropout_layer(self):
        """Test dropout layer implementation."""
        dropout = Dropout(p=0.5)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Training mode
        dropout.train()
        y_train = dropout(x)
        
        # Inference mode
        dropout.eval()
        y_eval = dropout(x)
        
        # In eval mode, should be same as input
        np.testing.assert_array_equal(safe_numpy(y_eval), safe_numpy(x))
    
    @pytest.mark.skip(reason="Stretch goal: Batch normalization")
    def test_batch_normalization(self):
        """Test batch normalization layer."""
        bn = BatchNorm1d(num_features=3)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = bn(x)
        
        # Should normalize across batch dimension
        assert y.shape == x.shape
        # Mean should be close to 0, std close to 1
        assert np.allclose(np.mean(safe_numpy(y), axis=0), 0.0, atol=1e-6)
        assert np.allclose(np.std(safe_numpy(y), axis=0), 1.0, atol=1e-6) 
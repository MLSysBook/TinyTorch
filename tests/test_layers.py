"""
Integration tests for TinyTorch Layers package.

Tests the exported layers functionality that students will use.
These tests ensure the student experience works correctly.
"""

import pytest
import numpy as np
from tinytorch.core.layers import Dense, ReLU, Sigmoid, Tanh
from tinytorch.core.tensor import Tensor


class TestDenseLayerIntegration:
    """Test Dense layer integration with exported package."""
    
    def test_dense_basic_functionality(self):
        """Test basic Dense layer functionality."""
        layer = Dense(input_size=3, output_size=2)
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        
        assert y.shape == (1, 2)
        assert isinstance(y, Tensor)
    
    def test_dense_batch_processing(self):
        """Test Dense layer with batch processing."""
        layer = Dense(input_size=2, output_size=3)
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = layer(x)
        
        assert y.shape == (3, 3)
        assert isinstance(y, Tensor)
    
    def test_dense_no_bias(self):
        """Test Dense layer without bias."""
        layer = Dense(input_size=2, output_size=1, use_bias=False)
        x = Tensor([[1.0, 2.0]])
        y = layer(x)
        
        assert y.shape == (1, 1)
        assert layer.bias is None


class TestActivationFunctionsIntegration:
    """Test activation functions integration."""
    
    def test_relu_integration(self):
        """Test ReLU activation function."""
        relu = ReLU()
        x = Tensor([[-1.0, 0.0, 1.0]])
        y = relu(x)
        
        expected = [[0.0, 0.0, 1.0]]
        np.testing.assert_array_equal(y.data, expected)
    
    def test_sigmoid_integration(self):
        """Test Sigmoid activation function."""
        sigmoid = Sigmoid()
        x = Tensor([[0.0]])
        y = sigmoid(x)
        
        np.testing.assert_array_almost_equal(y.data, [[0.5]])
    
    def test_tanh_integration(self):
        """Test Tanh activation function."""
        tanh = Tanh()
        x = Tensor([[0.0]])
        y = tanh(x)
        
        np.testing.assert_array_almost_equal(y.data, [[0.0]])


class TestNeuralNetworkIntegration:
    """Test complete neural network integration."""
    
    def test_simple_network_integration(self):
        """Test building a simple neural network."""
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
        
        assert output.shape == (1, 2)
        # Output should be in sigmoid range
        assert np.all(output.data >= 0)
        assert np.all(output.data <= 1)
    
    def test_batch_network_integration(self):
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
    
    def test_image_classification_network(self):
        """Test a realistic image classification network."""
        # Simulate MNIST: 784 → 128 → 64 → 10
        layer1 = Dense(input_size=784, output_size=128)
        relu1 = ReLU()
        layer2 = Dense(input_size=128, output_size=64)
        relu2 = ReLU()
        layer3 = Dense(input_size=64, output_size=10)
        sigmoid = Sigmoid()
        
        # Simulate a batch of 3 images
        batch_size = 3
        fake_images = Tensor(np.random.randn(batch_size, 784).astype(np.float32))
        
        # Forward pass
        h1 = relu1(layer1(fake_images))
        h2 = relu2(layer2(h1))
        predictions = sigmoid(layer3(h2))
        
        assert predictions.shape == (batch_size, 10)
        # All predictions should be in [0, 1] range
        assert np.all(predictions.data >= 0)
        assert np.all(predictions.data <= 1)


class TestLayerCompositionIntegration:
    """Test layer composition patterns."""
    
    def test_sequential_composition(self):
        """Test sequential layer composition."""
        layers = [
            Dense(input_size=5, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=3),
            ReLU(),
            Dense(input_size=3, output_size=2),
            Sigmoid()
        ]
        
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # Apply layers sequentially
        current = x
        for layer in layers:
            current = layer(current)
        
        assert current.shape == (1, 2)
        assert np.all(current.data >= 0)
        assert np.all(current.data <= 1)
    
    def test_different_activation_functions(self):
        """Test using different activation functions."""
        # Network with different activations
        layer1 = Dense(input_size=3, output_size=4)
        relu = ReLU()
        layer2 = Dense(input_size=4, output_size=4)
        tanh = Tanh()
        layer3 = Dense(input_size=4, output_size=2)
        sigmoid = Sigmoid()
        
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Forward pass
        h1 = relu(layer1(x))
        h2 = tanh(layer2(h1))
        output = sigmoid(layer3(h2))
        
        assert output.shape == (1, 2)
        # Final output should be in sigmoid range
        assert np.all(output.data >= 0)
        assert np.all(output.data <= 1)


class TestStudentExperience:
    """Test the typical student experience."""
    
    def test_first_neural_network(self):
        """Test the first neural network a student would build."""
        # Simple 2-layer network like in the tutorial
        layer1 = Dense(input_size=3, output_size=4)
        activation1 = ReLU()
        layer2 = Dense(input_size=4, output_size=2)
        activation2 = Sigmoid()
        
        # Sample data
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Forward pass
        h1 = layer1(x)
        h1_activated = activation1(h1)
        h2 = layer2(h1_activated)
        output = activation2(h2)
        
        # Should work without errors
        assert output.shape == (2, 2)
        assert isinstance(output, Tensor)
    
    def test_layer_inspection(self):
        """Test that students can inspect layer properties."""
        layer = Dense(input_size=3, output_size=2)
        
        # Students should be able to access these properties
        assert hasattr(layer, 'input_size')
        assert hasattr(layer, 'output_size')
        assert hasattr(layer, 'weights')
        assert hasattr(layer, 'bias')
        
        assert layer.input_size == 3
        assert layer.output_size == 2
        assert layer.weights.shape == (3, 2)
        assert layer.bias.shape == (2,)
    
    def test_activation_function_behavior(self):
        """Test activation function behavior that students will observe."""
        # ReLU clips negative values
        relu = ReLU()
        x = Tensor([[-1.0, 0.0, 1.0]])
        y = relu(x)
        assert np.array_equal(y.data, [[0.0, 0.0, 1.0]])
        
        # Sigmoid maps to (0, 1)
        sigmoid = Sigmoid()
        x = Tensor([[0.0]])
        y = sigmoid(x)
        assert np.isclose(y.data[0][0], 0.5)
        
        # Tanh maps to (-1, 1)
        tanh = Tanh()
        x = Tensor([[0.0]])
        y = tanh(x)
        assert np.isclose(y.data[0][0], 0.0) 
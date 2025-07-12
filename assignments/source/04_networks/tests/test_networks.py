"""
Tests for the Networks module.

Tests network composition, visualization, and practical applications.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the modules we're testing
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Sigmoid, Tanh

# Import the networks module
try:
    # Import from the exported package
    from tinytorch.core.networks import (
        Sequential, 
        create_mlp, 
        create_classification_network,
        create_regression_network,
        visualize_network_architecture,
        visualize_data_flow,
        compare_networks,
        analyze_network_behavior
    )
except ImportError:
    # Fallback for when module isn't exported yet
    sys.path.append(str(project_root / "modules" / "04_networks"))
    from networks_dev import (
        Sequential, 
        create_mlp, 
        create_classification_network,
        create_regression_network,
        visualize_network_architecture,
        visualize_data_flow,
        compare_networks,
        analyze_network_behavior
    )


class TestSequentialNetwork:
    """Test the Sequential network class."""
    
    def test_sequential_initialization(self):
        """Test Sequential network initialization."""
        layers = [Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()]
        network = Sequential(layers)
        
        assert len(network.layers) == 4
        assert isinstance(network.layers[0], Dense)
        assert isinstance(network.layers[1], ReLU)
        assert isinstance(network.layers[2], Dense)
        assert isinstance(network.layers[3], Sigmoid)
    
    def test_sequential_forward_pass(self):
        """Test Sequential network forward pass."""
        network = Sequential([
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
            Sigmoid()
        ])
        
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = network(x)
        
        assert output.shape == (2, 2)
        assert isinstance(output, Tensor)
        # Sigmoid output should be between 0 and 1
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
    
    def test_sequential_callable(self):
        """Test that Sequential network is callable."""
        network = Sequential([Dense(2, 3), ReLU()])
        x = Tensor([[1.0, 2.0]])
        
        # Test both forward() and __call__()
        output1 = network.forward(x)
        output2 = network(x)
        
        assert np.allclose(output1.data, output2.data)
    
    def test_empty_sequential(self):
        """Test Sequential network with no layers."""
        network = Sequential([])
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Should return input unchanged
        output = network(x)
        assert np.allclose(output.data, x.data)


class TestMLPCreation:
    """Test MLP creation functions."""
    
    def test_create_mlp_basic(self):
        """Test basic MLP creation."""
        mlp = create_mlp(input_size=3, hidden_sizes=[4], output_size=2)
        
        assert len(mlp.layers) == 4  # Dense + ReLU + Dense + Sigmoid
        assert isinstance(mlp.layers[0], Dense)
        assert mlp.layers[0].input_size == 3
        assert mlp.layers[0].output_size == 4
        assert isinstance(mlp.layers[1], ReLU)
        assert isinstance(mlp.layers[2], Dense)
        assert mlp.layers[2].input_size == 4
        assert mlp.layers[2].output_size == 2
        assert isinstance(mlp.layers[3], Sigmoid)
    
    def test_create_mlp_multiple_hidden(self):
        """Test MLP creation with multiple hidden layers."""
        mlp = create_mlp(input_size=10, hidden_sizes=[16, 8, 4], output_size=3)
        
        assert len(mlp.layers) == 8  # 3 Dense + 3 ReLU + 1 Dense + 1 Sigmoid
        
        # Check Dense layers
        dense_layers = [layer for layer in mlp.layers if isinstance(layer, Dense)]
        assert len(dense_layers) == 4
        
        assert dense_layers[0].input_size == 10
        assert dense_layers[0].output_size == 16
        assert dense_layers[1].input_size == 16
        assert dense_layers[1].output_size == 8
        assert dense_layers[2].input_size == 8
        assert dense_layers[2].output_size == 4
        assert dense_layers[3].input_size == 4
        assert dense_layers[3].output_size == 3
    
    def test_create_mlp_no_hidden(self):
        """Test MLP creation with no hidden layers."""
        mlp = create_mlp(input_size=5, hidden_sizes=[], output_size=2)
        
        assert len(mlp.layers) == 2  # Dense + Sigmoid
        assert isinstance(mlp.layers[0], Dense)
        assert mlp.layers[0].input_size == 5
        assert mlp.layers[0].output_size == 2
        assert isinstance(mlp.layers[1], Sigmoid)
    
    def test_create_mlp_custom_activation(self):
        """Test MLP creation with custom activation functions."""
        mlp = create_mlp(
            input_size=3, 
            hidden_sizes=[4], 
            output_size=2,
            activation=Tanh,
            output_activation=Tanh
        )
        
        assert len(mlp.layers) == 4
        assert isinstance(mlp.layers[1], Tanh)  # Hidden activation
        assert isinstance(mlp.layers[3], Tanh)  # Output activation


class TestSpecializedNetworks:
    """Test specialized network creation functions."""
    
    def test_create_classification_network(self):
        """Test classification network creation."""
        classifier = create_classification_network(
            input_size=100, 
            num_classes=5,
            hidden_sizes=[32, 16]
        )
        
        assert len(classifier.layers) == 6  # Dense(100→32) + ReLU + Dense(32→16) + ReLU + Dense(16→5) + Softmax
        
        # Check output layer
        dense_layers = [layer for layer in classifier.layers if isinstance(layer, Dense)]
        assert dense_layers[-1].output_size == 5
        # Should use Softmax for multi-class classification
        from tinytorch.core.activations import Softmax
        assert isinstance(classifier.layers[-1], Softmax)
    
    def test_create_classification_network_default(self):
        """Test classification network with default hidden sizes."""
        classifier = create_classification_network(input_size=50, num_classes=3)
        
        # Should use default hidden size of input_size // 2
        expected_hidden = 50 // 2
        dense_layers = [layer for layer in classifier.layers if isinstance(layer, Dense)]
        assert dense_layers[0].output_size == expected_hidden
        assert dense_layers[1].output_size == 3
    
    def test_create_regression_network(self):
        """Test regression network creation."""
        regressor = create_regression_network(
            input_size=13, 
            output_size=1,
            hidden_sizes=[8, 4]
        )
        
        assert len(regressor.layers) == 6  # Dense(13→8) + ReLU + Dense(8→4) + ReLU + Dense(4→1) + Tanh
        
        # Check output layer
        dense_layers = [layer for layer in regressor.layers if isinstance(layer, Dense)]
        assert dense_layers[-1].output_size == 1
        assert isinstance(regressor.layers[-1], Tanh)
    
    def test_create_regression_network_default(self):
        """Test regression network with default parameters."""
        regressor = create_regression_network(input_size=20)
        
        # Should use default output_size=1 and hidden_size=input_size//2
        expected_hidden = 20 // 2
        dense_layers = [layer for layer in regressor.layers if isinstance(layer, Dense)]
        assert dense_layers[0].output_size == expected_hidden
        assert dense_layers[1].output_size == 1


class TestNetworkBehavior:
    """Test network behavior and functionality."""
    
    def test_network_shape_transformations(self):
        """Test that networks properly transform tensor shapes."""
        network = Sequential([
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
            Sigmoid()
        ])
        
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = network(x)
        
        assert x.shape == (2, 3)
        assert output.shape == (2, 2)
    
    def test_network_activations(self):
        """Test that activation functions are properly applied."""
        network = Sequential([
            Dense(2, 3),
            ReLU(),
            Dense(3, 1),
            Sigmoid()
        ])
        
        x = Tensor([[-1.0, 1.0]])
        output = network(x)
        
        # ReLU should zero out negative values
        # Sigmoid should output values between 0 and 1
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
    
    def test_network_parameter_count(self):
        """Test that networks have the expected number of parameters."""
        network = Sequential([
            Dense(3, 4),  # 3*4 + 4 = 16 parameters
            ReLU(),
            Dense(4, 2),  # 4*2 + 2 = 10 parameters
            Sigmoid()
        ])
        
        # Count parameters (weights + biases)
        total_params = 0
        for layer in network.layers:
            if hasattr(layer, 'weights'):
                total_params += layer.weights.data.size
                if hasattr(layer, 'bias') and layer.bias is not None:
                    total_params += layer.bias.data.size
        
        assert total_params == 26  # 16 + 10


class TestVisualizationFunctions:
    """Test visualization functions (basic functionality, not visual output)."""
    
    def test_visualize_network_architecture_exists(self):
        """Test that visualization function exists and is callable."""
        network = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
        
        # Should not raise an error
        try:
            visualize_network_architecture(network, "Test Network")
        except Exception as e:
            pytest.fail(f"visualize_network_architecture raised {e}")
    
    def test_visualize_data_flow_exists(self):
        """Test that data flow visualization function exists and is callable."""
        network = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Should not raise an error
        try:
            visualize_data_flow(network, x, "Test Data Flow")
        except Exception as e:
            pytest.fail(f"visualize_data_flow raised {e}")
    
    def test_compare_networks_exists(self):
        """Test that network comparison function exists and is callable."""
        network1 = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
        network2 = Sequential([Dense(3, 8), ReLU(), Dense(8, 2), Sigmoid()])
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Should not raise an error
        try:
            compare_networks([network1, network2], ["Small", "Large"], x, "Test Comparison")
        except Exception as e:
            pytest.fail(f"compare_networks raised {e}")
    
    def test_analyze_network_behavior_exists(self):
        """Test that behavior analysis function exists and is callable."""
        network = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Should not raise an error
        try:
            analyze_network_behavior(network, x, "Test Behavior")
        except Exception as e:
            pytest.fail(f"analyze_network_behavior raised {e}")


class TestPracticalApplications:
    """Test practical network applications."""
    
    def test_digit_classification_network(self):
        """Test creating a network for digit classification."""
        classifier = create_classification_network(
            input_size=784,  # 28x28 image
            num_classes=10,   # 10 digits
            hidden_sizes=[128, 64]
        )
        
        # Test with fake image data
        fake_image = Tensor(np.random.randn(1, 784).astype(np.float32))
        output = classifier(fake_image)
        
        assert output.shape == (1, 10)
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
        # Should sum to approximately 1 (probability distribution)
        assert np.abs(np.sum(output.data) - 1.0) < 0.1
    
    def test_sentiment_analysis_network(self):
        """Test creating a network for sentiment analysis."""
        classifier = create_classification_network(
            input_size=100,  # 100-dimensional embeddings
            num_classes=2,    # Positive/Negative
            hidden_sizes=[32, 16]
        )
        
        # Test with fake text embeddings
        fake_embeddings = Tensor(np.random.randn(1, 100).astype(np.float32))
        output = classifier(fake_embeddings)
        
        assert output.shape == (1, 2)
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
    
    def test_house_price_prediction_network(self):
        """Test creating a network for house price prediction."""
        regressor = create_regression_network(
            input_size=13,   # 13 house features
            output_size=1,   # 1 price prediction
            hidden_sizes=[8, 4]
        )
        
        # Test with fake house features
        fake_features = Tensor(np.random.randn(1, 13).astype(np.float32))
        output = regressor(fake_features)
        
        assert output.shape == (1, 1)
        # Tanh output should be between -1 and 1
        assert np.all(output.data >= -1) and np.all(output.data <= 1)


class TestNetworkIntegration:
    """Test integration with other modules."""
    
    def test_network_with_tensor_operations(self):
        """Test that networks work with tensor operations."""
        network = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
        
        # Create input using tensor operations
        x1 = Tensor([[1.0, 2.0, 3.0]])
        x2 = Tensor([[4.0, 5.0, 6.0]])
        x_combined = Tensor(np.vstack([x1.data, x2.data]))
        
        output = network(x_combined)
        assert output.shape == (2, 2)
    
    def test_network_with_activations_module(self):
        """Test that networks properly use activations from the activations module."""
        # This test ensures we're using the activations from the activations module
        # rather than re-implementing them
        network = Sequential([
            Dense(2, 3),
            ReLU(),  # From activations module
            Dense(3, 1),
            Sigmoid()  # From activations module
        ])
        
        x = Tensor([[-1.0, 1.0]])
        output = network(x)
        
        # Test that activations work correctly
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
    
    def test_network_with_layers_module(self):
        """Test that networks properly use layers from the layers module."""
        # This test ensures we're using the Dense layers from the layers module
        network = Sequential([
            Dense(3, 4),  # From layers module
            ReLU(),
            Dense(4, 2),  # From layers module
            Sigmoid()
        ])
        
        x = Tensor([[1.0, 2.0, 3.0]])
        output = network(x)
        
        # Test that layers work correctly
        assert output.shape == (1, 2)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 
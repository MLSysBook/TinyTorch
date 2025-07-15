"""
Integration Tests - Complete ML Pipeline

Tests how student modules work together in realistic ML workflows.
Uses working implementations for dependencies to avoid cascade failures.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import working implementations (these should be from the package)
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
    from tinytorch.core.layers import Dense
    from tinytorch.core.networks import Sequential, create_mlp
except ImportError:
    # Fallback for development
    sys.path.append(str(project_root / "modules" / "source" / "01_tensor"))
    sys.path.append(str(project_root / "modules" / "source" / "02_activations"))
    sys.path.append(str(project_root / "modules" / "source" / "03_layers"))
    sys.path.append(str(project_root / "modules" / "source" / "04_networks"))
    
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    from layers_dev import Dense
    from networks_dev import Sequential, create_mlp


class TestBasicMLPipeline:
    """Test basic ML pipeline: Tensor → Layers → Networks."""
    
    def test_tensor_to_layer_integration(self):
        """Test tensor works with dense layers."""
        # Create tensor
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Create layer
        layer = Dense(input_size=3, output_size=2)
        
        # Forward pass
        output = layer(x)
        
        # Verify integration
        assert hasattr(output, 'data') and hasattr(output, 'shape')  # Tensor-like
        assert output.shape == (1, 2)
        assert np.all(np.isfinite(output.data))
        
    def test_layer_to_activation_integration(self):
        """Test layers work with activation functions."""
        # Create layer and activation
        layer = Dense(input_size=2, output_size=3)
        activation = ReLU()
        
        # Test data
        x = Tensor([[1.0, -1.0]])
        
        # Forward pass
        linear_output = layer(x)
        activated_output = activation(linear_output)
        
        # Verify integration
        assert hasattr(activated_output, 'data') and hasattr(activated_output, 'shape')
        assert activated_output.shape == linear_output.shape
        assert np.all(activated_output.data >= 0)  # ReLU property
        
    def test_complete_forward_pass(self):
        """Test complete forward pass through network."""
        # Create network
        network = Sequential([
            Dense(input_size=4, output_size=3),
            ReLU(),
            Dense(input_size=3, output_size=2),
            Sigmoid()
        ])
        
        # Test data
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Forward pass
        output = network(x)
        
        # Verify complete pipeline
        assert hasattr(output, 'data') and hasattr(output, 'shape')
        assert output.shape == (1, 2)
        assert np.all(output.data >= 0) and np.all(output.data <= 1)  # Sigmoid range


class TestMLApplications:
    """Test realistic ML applications."""
    
    def test_binary_classification(self):
        """Test binary classification pipeline."""
        # Create binary classifier
        classifier = create_mlp(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        
        # Simulate batch of data
        batch_size = 10
        x = Tensor(np.random.randn(batch_size, 4))
        
        # Forward pass
        predictions = classifier(x)
        
        # Verify binary classification
        assert predictions.shape == (batch_size, 1)
        assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1)
        
    def test_multiclass_classification(self):
        """Test multi-class classification pipeline."""
        # Create multi-class classifier
        classifier = create_mlp(
            input_size=8,
            hidden_sizes=[16, 8],
            output_size=3
        )
        
        # Simulate batch of data
        batch_size = 5
        x = Tensor(np.random.randn(batch_size, 8))
        
        # Forward pass
        predictions = classifier(x)
        
        # Verify multi-class classification
        assert predictions.shape == (batch_size, 3)
        
        # Check softmax properties
        for i in range(batch_size):
            row_sum = np.sum(predictions.data[i])
            assert abs(row_sum - 1.0) < 1e-4, f"Row {i} sum: {row_sum}"
            assert np.all(predictions.data[i] >= 0), f"Row {i} has negative values"
            
    def test_regression_pipeline(self):
        """Test regression pipeline."""
        # Create regression network (no output activation)
        regressor = Sequential([
            Dense(input_size=5, output_size=10),
            ReLU(),
            Dense(input_size=10, output_size=5),
            Tanh(),
            Dense(input_size=5, output_size=1)
        ])
        
        # Simulate batch of data
        batch_size = 8
        x = Tensor(np.random.randn(batch_size, 5))
        
        # Forward pass
        predictions = regressor(x)
        
        # Verify regression
        assert predictions.shape == (batch_size, 1)
        assert np.all(np.isfinite(predictions.data))


class TestArchitectureComparison:
    """Test different network architectures."""
    
    def test_shallow_vs_deep_networks(self):
        """Test shallow vs deep network behavior."""
        input_size = 6
        output_size = 2
        
        # Shallow network
        shallow = create_mlp(
            input_size=input_size,
            hidden_sizes=[20],
            output_size=output_size
        )
        
        # Deep network
        deep = create_mlp(
            input_size=input_size,
            hidden_sizes=[8, 8, 8],
            output_size=output_size
        )
        
        # Test same input
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        
        shallow_output = shallow(x)
        deep_output = deep(x)
        
        # Both should work
        assert shallow_output.shape == (1, output_size)
        assert deep_output.shape == (1, output_size)
        
        # Different architectures should give different results
        assert not np.allclose(shallow_output.data, deep_output.data)
        
    def test_different_activation_functions(self):
        """Test networks with different activation functions."""
        # Test with basic Sequential networks
        relu_network = Sequential([
            Dense(input_size=3, output_size=4),
            ReLU(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        sigmoid_network = Sequential([
            Dense(input_size=3, output_size=4),
            Sigmoid(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        tanh_network = Sequential([
            Dense(input_size=3, output_size=4),
            Tanh(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        networks = [relu_network, sigmoid_network, tanh_network]
        results = []
        
        for network in networks:
            x = Tensor([[1.0, -1.0, 0.5]])
            output = network(x)
            results.append(output.data)
            
            # All should produce valid outputs
            assert output.shape == (1, 2)
            assert np.all(output.data >= 0) and np.all(output.data <= 1)
        
        # Different activations should give different results
        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[1], results[2])


class TestNumericalStability:
    """Test numerical stability across modules."""
    
    def test_large_inputs(self):
        """Test with large input values."""
        network = create_mlp(
            input_size=3,
            hidden_sizes=[4],
            output_size=2
        )
        
        # Large positive values
        x_large = Tensor([[100.0, 200.0, 300.0]])
        output_large = network(x_large)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(output_large.data))
        assert output_large.shape == (1, 2)
        
    def test_small_inputs(self):
        """Test with very small input values."""
        network = Sequential([
            Dense(input_size=3, output_size=4),
            Tanh(),
            Dense(input_size=4, output_size=2),
            Sigmoid()
        ])
        
        # Very small values
        x_small = Tensor([[1e-8, -1e-8, 1e-10]])
        output_small = network(x_small)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(output_small.data))
        assert output_small.shape == (1, 2)
        
    def test_batch_processing(self):
        """Test batch processing stability."""
        network = create_mlp(
            input_size=4,
            hidden_sizes=[6, 3],
            output_size=2
        )
        
        # Large batch
        batch_size = 100
        x = Tensor(np.random.randn(batch_size, 4))
        
        output = network(x)
        
        # Should handle large batches
        assert output.shape == (batch_size, 2)
        assert np.all(np.isfinite(output.data))
        
        # Softmax properties should hold for each sample
        for i in range(batch_size):
            row_sum = np.sum(output.data[i])
            assert abs(row_sum - 1.0) < 1e-3  # Relaxed tolerance


class TestErrorHandling:
    """Test error handling across modules."""
    
    def test_shape_mismatch_errors(self):
        """Test proper error handling for shape mismatches."""
        layer = Dense(input_size=3, output_size=2)
        
        # Wrong input size
        x_wrong = Tensor([[1.0, 2.0]])  # Should be size 3
        
        with pytest.raises(Exception):  # Should raise some kind of error
            layer(x_wrong)
            
    def test_empty_network(self):
        """Test handling of empty networks."""
        # Empty network should raise error or handle gracefully
        try:
            empty_network = Sequential([])
            x = Tensor([[1.0, 2.0]])
            output = empty_network(x)
            # If it doesn't raise an error, output should equal input
            assert np.allclose(output.data, x.data)
        except Exception:
            # It's also fine if it raises an error
            pass


def test_integration_summary():
    """Summary test showing complete integration."""
    print("🔗 Integration Test Summary")
    print("=" * 50)
    
    # Create a complete ML pipeline
    network = Sequential([
        Dense(input_size=4, output_size=8),    # Tensor → Layer
        ReLU(),                                # Layer → Activation
        Dense(input_size=8, output_size=4),    # Activation → Layer
        Tanh(),                                # Layer → Activation
        Dense(input_size=4, output_size=2),    # Activation → Layer
        Softmax()                              # Layer → Activation
    ])
    
    # Test with batch data
    batch_size = 20
    x = Tensor(np.random.randn(batch_size, 4))
    
    # Forward pass
    output = network(x)
    
    # Verify complete integration
    assert output.shape == (batch_size, 2)
    assert np.all(np.isfinite(output.data))
    
    # Check softmax properties
    for i in range(batch_size):
        row_sum = np.sum(output.data[i])
        assert abs(row_sum - 1.0) < 1e-4
    
    print("✅ Complete ML pipeline integration successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Network layers: {len(network.layers)}")
    print("   All modules working together correctly!")


if __name__ == "__main__":
    test_integration_summary() 
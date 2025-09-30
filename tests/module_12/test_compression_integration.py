"""
Integration Tests - Compression Module

Tests real integration between compression techniques and other TinyTorch modules.
Uses actual TinyTorch components to verify model compression works correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.compression import (
    CompressionMetrics, prune_weights_by_magnitude, calculate_sparsity,
    quantize_layer_weights, DistillationLoss, compute_neuron_importance,
    prune_layer_neurons, compare_compression_techniques
)


class TestCompressionMetricsIntegration:
    """Test real integration between compression metrics and TinyTorch components."""
    
    def test_compression_metrics_with_real_tensors(self):
        """Test CompressionMetrics works with real Tensor-based networks."""
        # Create network with real Dense layers (which use real Tensors)
        layers = [
            Dense(784, 128),
            Dense(128, 64),
            Dense(64, 10)
        ]
        network = Sequential(layers)
        
        # Test compression metrics
        metrics = CompressionMetrics()
        param_counts = metrics.count_parameters(network)
        
        # Verify it works with real components
        assert 'total_parameters' in param_counts
        assert param_counts['total_parameters'] > 0
        
        # Test model size calculation
        size_info = metrics.calculate_model_size(network)
        assert 'size_mb' in size_info  # Fixed key name
        assert size_info['size_mb'] > 0
    
    def test_compression_metrics_with_different_networks(self):
        """Test compression metrics work with different network architectures."""
        # Test with small network
        small_network = Sequential([Dense(10, 5), Dense(5, 2)])
        
        # Test with larger network
        large_network = Sequential([
            Dense(100, 80),
            Dense(80, 60),
            Dense(60, 40),
            Dense(40, 10)
        ])
        
        metrics = CompressionMetrics()
        
        # Test both networks
        small_params = metrics.count_parameters(small_network)
        large_params = metrics.count_parameters(large_network)
        
        # Verify larger network has more parameters
        assert large_params['total_parameters'] > small_params['total_parameters']
        
        # Test size calculations
        small_size = metrics.calculate_model_size(small_network)
        large_size = metrics.calculate_model_size(large_network)
        
        assert large_size['size_mb'] > small_size['size_mb']


class TestCompressionNetworkIntegration:
    """Test compression techniques work with Sequential networks."""
    
    def test_comprehensive_comparison_integration(self):
        """Test comprehensive comparison works with real networks."""
        # Create real network
        layers = [
            Dense(784, 256),
            Dense(256, 128),
            Dense(128, 10)
        ]
        network = Sequential(layers)
        
        # Test comprehensive comparison
        results = compare_compression_techniques(network)
        
        # Verify all compression techniques worked (with correct key names)
        assert 'baseline' in results
        assert 'magnitude_pruning' in results  # Fixed key name
        assert 'quantization' in results  # Fixed key name
        assert 'structured_pruning' in results  # Fixed key name
        assert 'combined' in results
        
        # Verify metrics make sense
        baseline_params = results['baseline']['parameters']
        combined_params = results['combined']['parameters']
        
        # Combined should have fewer parameters than baseline
        assert combined_params <= baseline_params
        
        # Test compression ratios
        assert results['combined']['compression_ratio'] >= 1.0
        assert results['quantization']['compression_ratio'] >= 1.0
    
    def test_network_parameter_counting_consistency(self):
        """Test that parameter counting is consistent across compression techniques."""
        # Create network
        layers = [Dense(50, 30), Dense(30, 10)]
        network = Sequential(layers)
        
        # Test metrics before compression
        metrics = CompressionMetrics()
        original_params = metrics.count_parameters(network)
        
        # Test comprehensive comparison  
        results = compare_compression_techniques(network)
        
        # Verify baseline matches original count
        assert results['baseline']['parameters'] == original_params['total_parameters']
        
        # Verify structured pruning reduces parameters
        assert results['structured_pruning']['parameters'] < original_params['total_parameters']


class TestDistillationLossIntegration:
    """Test knowledge distillation works with real network components."""
    
    def test_distillation_loss_with_real_networks(self):
        """Test knowledge distillation with real network components."""
        # Create teacher and student networks
        teacher_layers = [Dense(10, 8), Dense(8, 5)]
        student_layers = [Dense(10, 4), Dense(4, 5)]
        
        teacher_network = Sequential(teacher_layers)
        student_network = Sequential(student_layers)
        
        # Test distillation loss
        distill_loss = DistillationLoss(temperature=3.0, alpha=0.7)
        
        # Create sample data
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        true_labels = Tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        
        # Forward passes
        teacher_output = teacher_network(x)
        student_output = student_network(x)
        
        # Compute distillation loss
        loss = distill_loss(student_output.data, teacher_output.data, true_labels.data)
        
        # Verify distillation works
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
        
        # Test with different temperatures
        high_temp_loss = DistillationLoss(temperature=5.0, alpha=0.5)
        loss_high_temp = high_temp_loss(student_output.data, teacher_output.data, true_labels.data)
        
        assert isinstance(loss_high_temp, (float, np.floating))
        assert loss_high_temp >= 0
    
    def test_distillation_loss_components(self):
        """Test individual components of distillation loss."""
        distill_loss = DistillationLoss(temperature=3.0, alpha=0.5)
        
        # Test with simple logits
        student_logits = np.array([[2.0, 1.0, 0.5]])
        teacher_logits = np.array([[1.8, 1.2, 0.3]])
        true_labels = np.array([[1.0, 0.0, 0.0]])
        
        # Test softmax computation
        student_probs = distill_loss._softmax(student_logits / distill_loss.temperature)
        teacher_probs = distill_loss._softmax(teacher_logits / distill_loss.temperature)
        
        # Verify softmax properties
        assert np.allclose(np.sum(student_probs, axis=1), 1.0)
        assert np.allclose(np.sum(teacher_probs, axis=1), 1.0)
        assert np.all(student_probs >= 0)
        assert np.all(teacher_probs >= 0)
        
        # Test cross entropy computation
        ce_loss = distill_loss._cross_entropy_loss(student_logits, true_labels)
        assert isinstance(ce_loss, (float, np.floating))
        assert ce_loss >= 0


class TestCompressionEdgeCases:
    """Test compression techniques handle edge cases with real components."""
    
    def test_compression_with_small_networks(self):
        """Test compression works with very small networks."""
        # Test with minimal network
        tiny_network = Sequential([Dense(3, 2), Dense(2, 1)])
        
        # Test compression metrics
        metrics = CompressionMetrics()
        params = metrics.count_parameters(tiny_network)
        assert params['total_parameters'] > 0
        
        # Test comprehensive comparison
        results = compare_compression_techniques(tiny_network)
        assert 'baseline' in results
        assert 'combined' in results
        assert results['baseline']['parameters'] > 0
    
    def test_compression_preserves_network_types(self):
        """Test that compression preserves network structure types."""
        # Create network with different layer types
        layers = [
            Dense(10, 8),
            ReLU(),
            Dense(8, 5),
            Softmax()
        ]
        network = Sequential(layers)
        
        # Test that network structure info is preserved
        assert len(network.layers) == 4
        
        # Test that compression metrics still work
        metrics = CompressionMetrics()
        params = metrics.count_parameters(network)
        assert params['total_parameters'] > 0
        
        # Only Dense layers should contribute to parameter count
        dense_count = sum(1 for layer in network.layers if isinstance(layer, Dense))
        assert dense_count == 2  # Should have 2 Dense layers 
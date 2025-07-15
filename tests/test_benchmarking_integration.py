"""
Integration Tests - Benchmarking Module

Tests real integration between benchmarking framework and other TinyTorch modules.
Uses actual TinyTorch components to verify systematic evaluation works correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential
from tinytorch.core.kernels import vectorized_relu
from tinytorch.core.benchmarking import BenchmarkScenarios, StatisticalValidator, TinyTorchPerf


class TestBenchmarkingIntegration:
    """Test real integration between benchmarking framework and TinyTorch components."""
    
    def test_benchmark_scenarios_with_real_model(self):
        """Test BenchmarkScenarios works with real TinyTorch models."""
        # Create real model
        model = Sequential([
            Dense(4, 8),
            ReLU(),
            Dense(8, 2),
            Softmax()
        ])
        
        # Create benchmark scenarios
        scenarios = BenchmarkScenarios()
        
        # Create simple dataset
        dataset = [Tensor(np.random.randn(4).tolist()) for _ in range(10)]
        
        # Test single stream scenario
        results = scenarios.single_stream(model, dataset, num_queries=5)
        
        # Verify integration
        assert hasattr(results, 'latency')
        assert hasattr(results, 'throughput')
        assert hasattr(results, 'accuracy')
        assert len(results.latency) == 5
        assert results.throughput > 0
        assert 0.0 <= results.accuracy <= 1.0
    
    def test_statistical_validator_with_benchmark_results(self):
        """Test StatisticalValidator works with benchmark results."""
        # Create validator
        validator = StatisticalValidator()
        
        # Create sample benchmark results
        results_a = [0.01, 0.012, 0.011, 0.013, 0.009]
        results_b = [0.015, 0.017, 0.016, 0.018, 0.014]
        
        # Test statistical validation
        stats = validator.validate_comparison(results_a, results_b)
        
        # Verify statistical analysis
        assert hasattr(stats, 'significant')
        assert hasattr(stats, 'p_value')
        assert hasattr(stats, 'effect_size')
        assert hasattr(stats, 'recommendation')
        
        # Verify reasonable values
        assert isinstance(stats.significant, bool)
        assert stats.p_value >= 0.0
        assert isinstance(stats.effect_size, (int, float))
        assert isinstance(stats.recommendation, str)
    
    def test_tinytorch_perf_with_basic_models(self):
        """Test TinyTorchPerf framework with basic models."""
        # Create real model
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        
        # Create benchmarking framework
        perf = TinyTorchPerf()
        perf.set_model(model)
        
        # Create dataset
        dataset = [Tensor(np.random.randn(10).tolist()) for _ in range(8)]
        perf.set_dataset(dataset)
        
        # Test benchmarking
        results = perf.run_single_stream(num_queries=5)
        
        # Verify basic benchmarking integration
        assert hasattr(results, 'latency')
        assert hasattr(results, 'throughput')
        assert hasattr(results, 'accuracy')
        assert len(results.latency) == 5
        assert results.throughput > 0


class TestBenchmarkingWithKernels:
    """Test benchmarking integration with optimized kernels."""
    
    def test_benchmarking_kernel_optimized_operations(self):
        """Test benchmarking framework with kernel-optimized operations."""
        # Create model using kernel operations
        def kernel_model(x):
            # Use kernel operations in model
            return vectorized_relu(x)
        
        # Create benchmarking framework
        perf = TinyTorchPerf()
        perf.set_model(kernel_model)
        
        # Create dataset
        dataset = [Tensor(np.random.randn(5).tolist()) for _ in range(8)]
        perf.set_dataset(dataset)
        
        # Benchmark kernel operations
        results = perf.run_single_stream(num_queries=6)
        
        # Verify kernel + benchmarking integration
        assert hasattr(results, 'latency')
        assert hasattr(results, 'throughput')
        assert len(results.latency) == 6
        assert results.throughput > 0
    
    def test_performance_comparison_with_kernels(self):
        """Test performance comparison between standard and kernel operations."""
        # Create standard model
        standard_model = Sequential([Dense(4, 4), ReLU()])
        
        # Create dataset
        dataset = [Tensor(np.random.randn(4).tolist()) for _ in range(10)]
        
        # Benchmark standard model
        perf_standard = TinyTorchPerf()
        perf_standard.set_model(standard_model)
        perf_standard.set_dataset(dataset)
        
        standard_results = perf_standard.run_single_stream(num_queries=5)
        
        # Verify we can benchmark different implementations
        assert hasattr(standard_results, 'latency')
        assert hasattr(standard_results, 'throughput')
        assert len(standard_results.latency) == 5
        
        # Test that benchmarking framework can handle different model types
        def kernel_relu_model(x):
            return vectorized_relu(x)
        
        perf_kernel = TinyTorchPerf()
        perf_kernel.set_model(kernel_relu_model)
        perf_kernel.set_dataset(dataset)
        
        kernel_results = perf_kernel.run_single_stream(num_queries=5)
        
        assert hasattr(kernel_results, 'latency')
        assert hasattr(kernel_results, 'throughput')
        assert len(kernel_results.latency) == 5


class TestBenchmarkingWithNetworks:
    """Test benchmarking framework with neural networks."""
    
    def test_benchmarking_sequential_networks(self):
        """Test benchmarking with Sequential networks."""
        # Create realistic network
        network = Sequential([
            Dense(8, 16),
            ReLU(),
            Dense(16, 8),
            ReLU(),
            Dense(8, 3),
            Softmax()
        ])
        
        # Create benchmarking framework
        perf = TinyTorchPerf()
        perf.set_model(network)
        
        # Create dataset
        dataset = [Tensor(np.random.randn(8).tolist()) for _ in range(12)]
        perf.set_dataset(dataset)
        
        # Test all benchmark scenarios
        single_stream = perf.run_single_stream(num_queries=6)
        server_results = perf.run_server(target_qps=10.0, duration=2.0)
        offline_results = perf.run_offline(batch_size=4)
        
        # Verify all scenarios work
        for results in [single_stream, server_results, offline_results]:
            assert hasattr(results, 'latency')
            assert hasattr(results, 'throughput')
            assert hasattr(results, 'accuracy')
            assert len(results.latency) > 0
            assert results.throughput > 0
    
    def test_benchmarking_with_different_network_sizes(self):
        """Test benchmarking scales with network complexity."""
        # Create small network
        small_network = Sequential([Dense(4, 2)])
        
        # Create large network
        large_network = Sequential([
            Dense(4, 32),
            ReLU(),
            Dense(32, 16),
            ReLU(),
            Dense(16, 2)
        ])
        
        # Create dataset
        dataset = [Tensor(np.random.randn(4).tolist()) for _ in range(10)]
        
        # Benchmark both networks
        for network in [small_network, large_network]:
            perf = TinyTorchPerf()
            perf.set_model(network)
            perf.set_dataset(dataset)
            
            results = perf.run_single_stream(num_queries=5)
            
            # Verify benchmarking works regardless of network size
            assert hasattr(results, 'latency')
            assert hasattr(results, 'throughput')
            assert len(results.latency) == 5
            assert results.throughput > 0


def test_integration_summary():
    """Summary test demonstrating complete benchmarking integration."""
    print("🎯 Integration Summary: Benchmarking ↔ TinyTorch Components")
    print("=" * 60)
    
    # Create comprehensive test
    print("🏗️  Testing benchmarking integration...")
    
    # Test 1: Create model with multiple components
    model = Sequential([
        Dense(6, 12),
        ReLU(),
        Dense(12, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    ])
    
    # Test 2: Create benchmarking framework
    perf = TinyTorchPerf()
    perf.set_model(model)
    
    # Test 3: Create dataset
    dataset = [Tensor(np.random.randn(6).tolist()) for _ in range(15)]
    perf.set_dataset(dataset)
    
    # Test 4: Run comprehensive benchmarking
    single_stream = perf.run_single_stream(num_queries=8)
    server_results = perf.run_server(target_qps=10.0, duration=2.0)
    offline_results = perf.run_offline(batch_size=5)
    
    # Test 5: Statistical validation
    validator = StatisticalValidator()
    
    # Create comparison data
    results_a = single_stream.latency[:5]
    results_b = [x * 1.1 for x in results_a]  # Slightly slower
    
    stats = validator.validate_comparison(results_a, results_b)
    
    # Verify complete integration
    assert hasattr(single_stream, 'latency')
    assert hasattr(server_results, 'throughput')
    assert hasattr(offline_results, 'accuracy')
    assert hasattr(stats, 'significant')
    assert hasattr(stats, 'recommendation')
    
    print("✅ Benchmarking integration successful!")
    print(f"   Single stream queries: {len(single_stream.latency)}")
    print(f"   Server throughput: {server_results.throughput:.1f} QPS")
    print(f"   Offline accuracy: {offline_results.accuracy:.3f}")
    print(f"   Statistical comparison: {stats.recommendation}")
    print("   Components: Networks → Layers → Activations → Tensors → Benchmarking")
    print("🎉 Systematic ML performance evaluation ready for production!") 
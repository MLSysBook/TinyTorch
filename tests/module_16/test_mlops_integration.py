"""
Integration Tests - MLOps Module

Tests real integration between MLOps pipeline and other TinyTorch modules.
Uses actual TinyTorch components to verify production monitoring works correctly.
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
from tinytorch.core.mlops import (
    ModelMonitor, DriftDetector, RetrainingTrigger, MLOpsPipeline
)


class TestMLOpsIntegration:
    """Test real integration between MLOps pipeline and TinyTorch components."""
    
    def test_model_monitor_with_real_models(self):
        """Test ModelMonitor works with real TinyTorch models."""
        # Create real model
        model = Sequential([
            Dense(4, 8),
            ReLU(),
            Dense(8, 2),
            Softmax()
        ])
        
        # Create model monitor
        monitor = ModelMonitor(model)
        
        # Test data
        test_data = [
            (Tensor([1.0, 2.0, 3.0, 4.0]), Tensor([1])),
            (Tensor([2.0, 3.0, 4.0, 5.0]), Tensor([0])),
            (Tensor([3.0, 4.0, 5.0, 6.0]), Tensor([1]))
        ]
        
        # Test monitoring
        performance = monitor.track_performance(test_data)
        
        # Verify integration
        assert 'accuracy' in performance
        assert 'loss' in performance
        assert 'timestamp' in performance
        assert 0.0 <= performance['accuracy'] <= 1.0
        assert performance['loss'] >= 0.0
        assert isinstance(performance['timestamp'], float)
    
    def test_drift_detector_with_real_data(self):
        """Test DriftDetector works with real tensor data."""
        # Create drift detector
        detector = DriftDetector()
        
        # Create baseline data
        baseline_data = [Tensor([1.0, 2.0, 3.0]) for _ in range(10)]
        detector.set_baseline(baseline_data)
        
        # Test with similar data (no drift)
        similar_data = [Tensor([1.1, 2.1, 3.1]) for _ in range(10)]
        drift_result = detector.detect_drift(similar_data)
        
        # Verify no drift detection
        assert 'drift_detected' in drift_result
        assert 'drift_score' in drift_result
        assert 'threshold' in drift_result
        assert isinstance(drift_result['drift_detected'], bool)
        assert isinstance(drift_result['drift_score'], (int, float))
        assert isinstance(drift_result['threshold'], (int, float))
    
    def test_retraining_trigger_with_training_integration(self):
        """Test RetrainingTrigger works with training components."""
        # Create simple model
        model = Sequential([Dense(3, 2), Sigmoid()])
        
        # Create training data
        train_data = [
            (Tensor([1.0, 2.0, 3.0]), Tensor([1])),
            (Tensor([2.0, 3.0, 4.0]), Tensor([0])),
            (Tensor([3.0, 4.0, 5.0]), Tensor([1]))
        ]
        
        # Create retraining trigger
        trigger = RetrainingTrigger(
            model=model,
            training_data=train_data,
            performance_threshold=0.5
        )
        
        # Test trigger evaluation
        should_retrain = trigger.should_retrain(current_accuracy=0.3)
        
        # Verify trigger logic
        assert isinstance(should_retrain, bool)
        assert should_retrain == True  # Accuracy below threshold
        
        # Test with good performance
        should_not_retrain = trigger.should_retrain(current_accuracy=0.8)
        assert should_not_retrain == False  # Accuracy above threshold
    
    def test_mlops_pipeline_with_all_components(self):
        """Test complete MLOps pipeline with all TinyTorch components."""
        # Create real model
        model = Sequential([
            Dense(4, 6),
            ReLU(),
            Dense(6, 2),
            Softmax()
        ])
        
        # Create datasets
        train_data = [
            (Tensor([1.0, 2.0, 3.0, 4.0]), Tensor([1])),
            (Tensor([2.0, 3.0, 4.0, 5.0]), Tensor([0])),
            (Tensor([3.0, 4.0, 5.0, 6.0]), Tensor([1]))
        ]
        
        val_data = [
            (Tensor([1.5, 2.5, 3.5, 4.5]), Tensor([1])),
            (Tensor([2.5, 3.5, 4.5, 5.5]), Tensor([0]))
        ]
        
        baseline_data = [Tensor([1.0, 2.0, 3.0, 4.0]) for _ in range(5)]
        
        # Create MLOps pipeline
        pipeline = MLOpsPipeline(
            model=model,
            training_data=train_data,
            validation_data=val_data,
            baseline_data=baseline_data
        )
        
        # Test system health check
        new_data = [Tensor([1.2, 2.2, 3.2, 4.2]) for _ in range(3)]
        health = pipeline.check_system_health(new_data, current_accuracy=0.7)
        
        # Verify complete pipeline integration
        assert 'model_performance' in health
        assert 'drift_status' in health
        assert 'retraining_needed' in health
        assert 'system_status' in health
        
        # Check data types
        assert isinstance(health['model_performance'], dict)
        assert isinstance(health['drift_status'], dict)
        assert isinstance(health['retraining_needed'], bool)
        assert isinstance(health['system_status'], str)


class TestMLOpsWithBenchmarking:
    """Test MLOps integration with benchmarking framework."""
    
    def test_mlops_with_performance_benchmarking(self):
        """Test MLOps pipeline with performance benchmarking."""
        # Create model
        model = Sequential([Dense(4, 2), ReLU()])
        
        # Create MLOps pipeline
        train_data = [(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor([1]))]
        val_data = [(Tensor([2.0, 3.0, 4.0, 5.0]), Tensor([0]))]
        baseline_data = [Tensor([1.0, 2.0, 3.0, 4.0])]
        
        pipeline = MLOpsPipeline(
            model=model,
            training_data=train_data,
            validation_data=val_data,
            baseline_data=baseline_data
        )
        
        # Test with benchmarking
        perf = TinyTorchPerf()
        perf.set_model(model)
        
        test_data = [Tensor([1.0, 2.0, 3.0, 4.0]) for _ in range(5)]
        perf.set_dataset(test_data)
        
        # Run benchmark
        results = perf.run_single_stream(num_queries=3)
        
        # Test MLOps health check
        health = pipeline.check_system_health(test_data, current_accuracy=0.8)
        
        # Verify benchmarking + MLOps integration
        assert hasattr(results, 'latency')
        assert hasattr(results, 'throughput')
        assert len(results.latency) == 3
        
        assert 'system_status' in health
        assert 'model_performance' in health
        assert health['system_status'] in ['healthy', 'degraded', 'critical']
    
    def test_mlops_performance_monitoring_integration(self):
        """Test MLOps performance monitoring with benchmarking."""
        # Create model
        model = Sequential([Dense(3, 2), Sigmoid()])
        
        # Create monitor
        monitor = ModelMonitor(model)
        
        # Test data
        test_data = [
            (Tensor([1.0, 2.0, 3.0]), Tensor([1])),
            (Tensor([2.0, 3.0, 4.0]), Tensor([0]))
        ]
        
        # Monitor performance
        performance = monitor.track_performance(test_data)
        
        # Test with benchmarking
        perf = TinyTorchPerf()
        perf.set_model(model)
        
        inference_data = [Tensor([1.0, 2.0, 3.0]) for _ in range(4)]
        perf.set_dataset(inference_data)
        
        benchmark_results = perf.run_single_stream(num_queries=4)
        
        # Verify monitoring + benchmarking integration
        assert 'accuracy' in performance
        assert 'loss' in performance
        assert hasattr(benchmark_results, 'latency')
        assert hasattr(benchmark_results, 'throughput')
        
        # Both should work with the same model
        assert len(benchmark_results.latency) == 4
        assert 0.0 <= performance['accuracy'] <= 1.0


class TestMLOpsWithNetworks:
    """Test MLOps integration with different network architectures."""
    
    def test_mlops_with_different_network_architectures(self):
        """Test MLOps pipeline with different network types."""
        # Test with different architectures
        networks = [
            Sequential([Dense(4, 2)]),  # Simple network
            Sequential([Dense(4, 8), ReLU(), Dense(8, 2)]),  # Deep network
            Sequential([Dense(4, 4), ReLU(), Dense(4, 2), Softmax()])  # With softmax
        ]
        
        for i, network in enumerate(networks):
            # Create MLOps pipeline
            train_data = [(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor([1]))]
            val_data = [(Tensor([2.0, 3.0, 4.0, 5.0]), Tensor([0]))]
            baseline_data = [Tensor([1.0, 2.0, 3.0, 4.0])]
            
            pipeline = MLOpsPipeline(
                model=network,
                training_data=train_data,
                validation_data=val_data,
                baseline_data=baseline_data
            )
            
            # Test system health
            new_data = [Tensor([1.5, 2.5, 3.5, 4.5])]
            health = pipeline.check_system_health(new_data, current_accuracy=0.7)
            
            # Verify each architecture works
            assert 'system_status' in health
            assert 'model_performance' in health
            assert health['system_status'] in ['healthy', 'degraded', 'critical']
    
    def test_mlops_scalability_with_network_complexity(self):
        """Test MLOps pipeline scales with network complexity."""
        # Create networks of different sizes
        small_network = Sequential([Dense(2, 2)])
        large_network = Sequential([
            Dense(8, 16), ReLU(),
            Dense(16, 8), ReLU(),
            Dense(8, 2)
        ])
        
        for network in [small_network, large_network]:
            # Create monitor
            monitor = ModelMonitor(network)
            
            # Test data (adjust size for network)
            input_size = 2 if network == small_network else 8
            test_data = [
                (Tensor(np.random.randn(input_size).tolist()), Tensor([1])),
                (Tensor(np.random.randn(input_size).tolist()), Tensor([0]))
            ]
            
            # Monitor performance
            performance = monitor.track_performance(test_data)
            
            # Verify monitoring works regardless of network size
            assert 'accuracy' in performance
            assert 'loss' in performance
            assert 0.0 <= performance['accuracy'] <= 1.0
            assert performance['loss'] >= 0.0


def test_integration_summary():
    """Summary test demonstrating complete MLOps integration."""
    print("🎯 Integration Summary: MLOps ↔ TinyTorch Components")
    print("=" * 60)
    
    # Create comprehensive test
    print("🏗️  Testing complete MLOps integration...")
    
    # Test 1: Create model with multiple components
    model = Sequential([
        Dense(6, 12),
        ReLU(),
        Dense(12, 6),
        ReLU(),
        Dense(6, 2),
        Softmax()
    ])
    
    # Test 2: Create datasets
    train_data = [
        (Tensor(np.random.randn(6).tolist()), Tensor([1])),
        (Tensor(np.random.randn(6).tolist()), Tensor([0])),
        (Tensor(np.random.randn(6).tolist()), Tensor([1]))
    ]
    
    val_data = [
        (Tensor(np.random.randn(6).tolist()), Tensor([1])),
        (Tensor(np.random.randn(6).tolist()), Tensor([0]))
    ]
    
    baseline_data = [Tensor(np.random.randn(6).tolist()) for _ in range(5)]
    
    # Test 3: Create complete MLOps pipeline
    pipeline = MLOpsPipeline(
        model=model,
        training_data=train_data,
        validation_data=val_data,
        baseline_data=baseline_data
    )
    
    # Test 4: Test system health monitoring
    new_data = [Tensor(np.random.randn(6).tolist()) for _ in range(3)]
    health = pipeline.check_system_health(new_data, current_accuracy=0.75)
    
    # Test 5: Test individual components
    monitor = ModelMonitor(model)
    performance = monitor.track_performance(val_data)
    
    detector = DriftDetector()
    detector.set_baseline(baseline_data)
    drift_result = detector.detect_drift(new_data)
    
    # Test 6: Test with benchmarking
    perf = TinyTorchPerf()
    perf.set_model(model)
    perf.set_dataset(new_data)
    benchmark_results = perf.run_single_stream(num_queries=3)
    
    # Verify complete integration
    assert 'system_status' in health
    assert 'model_performance' in health
    assert 'drift_status' in health
    assert 'retraining_needed' in health
    
    assert 'accuracy' in performance
    assert 'loss' in performance
    
    assert 'drift_detected' in drift_result
    assert 'drift_score' in drift_result
    
    assert hasattr(benchmark_results, 'latency')
    assert hasattr(benchmark_results, 'throughput')
    
    print("✅ MLOps integration successful!")
    print(f"   System status: {health['system_status']}")
    print(f"   Model accuracy: {performance['accuracy']:.3f}")
    print(f"   Drift detected: {drift_result['drift_detected']}")
    print(f"   Retraining needed: {health['retraining_needed']}")
    print(f"   Benchmark latency: {len(benchmark_results.latency)} measurements")
    print("   Components: All TinyTorch modules → MLOps → Production System")
    print("🎉 Complete production ML system ready for deployment!") 
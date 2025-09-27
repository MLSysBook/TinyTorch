#!/usr/bin/env python3
"""
Performance Integration Tests for TinyTorch
Tests memory usage, scaling behavior, and cross-module performance
"""

import sys
import time
import tracemalloc
import gc
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class PerformanceIntegrationTester:
    def __init__(self):
        self.results = {}
        
    def print_section(self, title):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
    def test_all_performance_integration(self):
        """Run all performance integration tests"""
        self.print_section("PERFORMANCE INTEGRATION TESTS")
        
        # Test 1: Memory usage patterns
        self.test_memory_patterns()
        
        # Test 2: Cross-module performance
        self.test_cross_module_performance()
        
        # Test 3: Scaling behavior
        self.test_scaling_behavior()
        
        # Test 4: Training loop efficiency
        self.test_training_efficiency()
        
        # Generate report
        self.generate_performance_report()
        
    def test_memory_patterns(self):
        """Test memory usage patterns across modules"""
        print("\n1. Testing Memory Usage Patterns...")
        
        tracemalloc.start()
        
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            
            # Baseline measurement
            current, peak = tracemalloc.get_traced_memory()
            baseline_memory = peak / 1024 / 1024  # MB
            
            # Test tensor creation memory
            tensors = []
            for i in range(100):
                tensors.append(Tensor([[1.0, 2.0, 3.0]] * 10))
                
            current, peak = tracemalloc.get_traced_memory()
            tensor_memory = peak / 1024 / 1024 - baseline_memory
            
            # Test layer creation memory
            layers = []
            for i in range(50):
                layers.append(Linear(10, 5))
                layers.append(ReLU())
                
            current, peak = tracemalloc.get_traced_memory()
            layer_memory = peak / 1024 / 1024 - baseline_memory - tensor_memory
            
            # Test CNN components memory
            cnn_components = []
            for i in range(20):
                cnn_components.append(Conv2D((3, 3)))
                cnn_components.append(MaxPool2D((2, 2)))
                
            current, peak = tracemalloc.get_traced_memory()
            cnn_memory = peak / 1024 / 1024 - baseline_memory - tensor_memory - layer_memory
            
            print(f"  Baseline memory: {baseline_memory:.2f} MB")
            print(f"  Tensor creation (100 tensors): {tensor_memory:.2f} MB")
            print(f"  Layer creation (100 layers): {layer_memory:.2f} MB")
            print(f"  CNN components (40 components): {cnn_memory:.2f} MB")
            
            # Cleanup and measure memory reduction
            del tensors, layers, cnn_components
            gc.collect()
            
            current, peak = tracemalloc.get_traced_memory()
            final_memory = current / 1024 / 1024
            
            print(f"  Memory after cleanup: {final_memory:.2f} MB")
            
            self.results["memory_patterns"] = {
                "baseline": baseline_memory,
                "tensor_overhead": tensor_memory,
                "layer_overhead": layer_memory,
                "cnn_overhead": cnn_memory,
                "cleanup_efficiency": (peak/1024/1024 - final_memory) / (peak/1024/1024) * 100
            }
            
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results["memory_patterns"] = {"error": str(e)}
            
        tracemalloc.stop()
        
    def test_cross_module_performance(self):
        """Test performance when using multiple modules together"""
        print("\n2. Testing Cross-Module Performance...")
        
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.spatial import flatten
            
            # Create components
            linear1 = Linear(100, 50)
            linear2 = Linear(50, 10)
            relu = ReLU()
            
            # Test data
            x = Tensor([[1.0] * 100] * 32)  # Batch size 32
            
            # Time forward passes
            times = []
            for _ in range(10):
                start = time.time()
                
                h1 = linear1.forward(x)
                h1_relu = relu.forward(h1)
                h2 = linear2.forward(h1_relu)
                output = relu.forward(h2)
                
                end = time.time()
                times.append((end - start) * 1000)  # ms
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  Forward pass timing (10 runs):")
            print(f"    Average: {avg_time:.2f} ms")
            print(f"    Min: {min_time:.2f} ms") 
            print(f"    Max: {max_time:.2f} ms")
            print(f"    Std dev: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")
            
            # Test with larger batches
            batch_sizes = [1, 8, 32, 128]
            batch_times = {}
            
            for batch_size in batch_sizes:
                x_batch = Tensor([[1.0] * 100] * batch_size)
                
                start = time.time()
                h1 = linear1.forward(x_batch)
                h1_relu = relu.forward(h1)
                h2 = linear2.forward(h1_relu)
                output = relu.forward(h2)
                end = time.time()
                
                batch_times[batch_size] = (end - start) * 1000
                
            print(f"  Batch size scaling:")
            for batch_size, time_ms in batch_times.items():
                per_sample = time_ms / batch_size
                print(f"    Batch {batch_size}: {time_ms:.2f} ms total, {per_sample:.2f} ms/sample")
                
            self.results["cross_module_performance"] = {
                "avg_forward_time": avg_time,
                "batch_times": batch_times,
                "scaling_efficiency": batch_times[128] / (128 * batch_times[1] / 1) if 1 in batch_times and 128 in batch_times else 0
            }
            
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results["cross_module_performance"] = {"error": str(e)}
            
    def test_scaling_behavior(self):
        """Test scaling behavior across different input sizes"""
        print("\n3. Testing Scaling Behavior...")
        
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            
            # Test different input sizes
            input_sizes = [10, 50, 100, 500, 1000]
            scaling_results = {}
            
            for input_size in input_sizes:
                layer = Linear(input_size, input_size // 2)
                x = Tensor([[1.0] * input_size])
                
                # Time multiple runs
                times = []
                for _ in range(5):
                    start = time.time()
                    output = layer.forward(x)
                    end = time.time()
                    times.append((end - start) * 1000)
                    
                avg_time = sum(times) / len(times)
                scaling_results[input_size] = avg_time
                
                print(f"  Input size {input_size}: {avg_time:.3f} ms")
                
            # Analyze scaling
            if len(scaling_results) >= 2:
                sizes = list(scaling_results.keys())
                times = list(scaling_results.values())
                
                # Simple linear scaling check
                size_ratio = sizes[-1] / sizes[0]
                time_ratio = times[-1] / times[0]
                
                print(f"  Scaling analysis:")
                print(f"    Size ratio: {size_ratio:.1f}x")
                print(f"    Time ratio: {time_ratio:.1f}x")
                print(f"    Scaling efficiency: {size_ratio / time_ratio:.2f}")
                
            self.results["scaling_behavior"] = {
                "input_sizes": scaling_results,
                "efficiency_score": size_ratio / time_ratio if 'size_ratio' in locals() else 0
            }
            
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results["scaling_behavior"] = {"error": str(e)}
            
    def test_training_efficiency(self):
        """Test training loop efficiency with multiple modules"""
        print("\n4. Testing Training Loop Efficiency...")
        
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            # Simple training simulation
            model_layers = [
                Linear(20, 15),
                ReLU(),
                Linear(15, 10),
                ReLU(),
                Linear(10, 1)
            ]
            
            # Training data
            X = Tensor([[1.0] * 20] * 16)  # Batch of 16
            y = Tensor([[1.0]] * 16)
            
            # Simulate training epochs
            epoch_times = []
            
            for epoch in range(5):
                start = time.time()
                
                # Forward pass through all layers
                h = X
                for layer in model_layers:
                    h = layer.forward(h)
                    
                # Simulate loss computation
                loss = h  # Simple placeholder
                
                end = time.time()
                epoch_times.append((end - start) * 1000)
                
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            
            print(f"  Training simulation (5 epochs):")
            print(f"    Average epoch time: {avg_epoch_time:.2f} ms")
            print(f"    Forward pass only (no gradients)")
            print(f"    Batch size: 16, Model: 20→15→10→1")
            
            # Test different batch sizes
            batch_sizes = [4, 16, 64]
            batch_efficiency = {}
            
            for batch_size in batch_sizes:
                X_batch = Tensor([[1.0] * 20] * batch_size)
                
                start = time.time()
                h = X_batch
                for layer in model_layers:
                    h = layer.forward(h)
                end = time.time()
                
                time_per_sample = ((end - start) * 1000) / batch_size
                batch_efficiency[batch_size] = time_per_sample
                
                print(f"    Batch {batch_size}: {time_per_sample:.3f} ms/sample")
                
            self.results["training_efficiency"] = {
                "avg_epoch_time": avg_epoch_time,
                "batch_efficiency": batch_efficiency
            }
            
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results["training_efficiency"] = {"error": str(e)}
            
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        self.print_section("PERFORMANCE INTEGRATION SUMMARY")
        
        print(f"\nPERFORMANCE ANALYSIS RESULTS:")
        print(f"{'='*40}")
        
        # Memory patterns analysis
        if "memory_patterns" in self.results and "error" not in self.results["memory_patterns"]:
            mem = self.results["memory_patterns"]
            print(f"\nMEMORY USAGE PATTERNS:")
            print(f"  Baseline memory: {mem['baseline']:.2f} MB")
            print(f"  Tensor overhead: {mem['tensor_overhead']:.2f} MB")
            print(f"  Layer overhead: {mem['layer_overhead']:.2f} MB")
            print(f"  CNN overhead: {mem['cnn_overhead']:.2f} MB")
            print(f"  Cleanup efficiency: {mem['cleanup_efficiency']:.1f}%")
            
        # Cross-module performance
        if "cross_module_performance" in self.results and "error" not in self.results["cross_module_performance"]:
            perf = self.results["cross_module_performance"]
            print(f"\nCROSS-MODULE PERFORMANCE:")
            print(f"  Average forward pass: {perf['avg_forward_time']:.2f} ms")
            print(f"  Scaling efficiency: {perf['scaling_efficiency']:.2f}")
            
        # Scaling behavior
        if "scaling_behavior" in self.results and "error" not in self.results["scaling_behavior"]:
            scale = self.results["scaling_behavior"]
            print(f"\nSCALING BEHAVIOR:")
            print(f"  Efficiency score: {scale['efficiency_score']:.2f}")
            
        # Training efficiency
        if "training_efficiency" in self.results and "error" not in self.results["training_efficiency"]:
            train = self.results["training_efficiency"]
            print(f"\nTRAINING EFFICIENCY:")
            print(f"  Average epoch time: {train['avg_epoch_time']:.2f} ms")
            
        # Overall assessment
        error_count = sum(1 for result in self.results.values() if "error" in result)
        success_rate = (len(self.results) - error_count) / len(self.results) * 100 if self.results else 0
        
        print(f"\n{'='*40}")
        print(f"PERFORMANCE INTEGRATION SCORE: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🚀 EXCELLENT: Outstanding performance integration")
        elif success_rate >= 70:
            print("✅ GOOD: Solid performance characteristics")
        elif success_rate >= 50:
            print("⚠️  MODERATE: Some performance concerns")
        else:
            print("❌ NEEDS WORK: Performance integration issues")
            
        return success_rate >= 70

if __name__ == "__main__":
    print("Starting Performance Integration Tests...")
    
    tester = PerformanceIntegrationTester()
    success = tester.test_all_performance_integration()
    
    sys.exit(0 if success else 1)
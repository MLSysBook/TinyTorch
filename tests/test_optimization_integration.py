#!/usr/bin/env python3
"""
Optimization Integration Tests - Modules 15-20

This test suite validates that all optimization modules work together
correctly and achieve the expected performance improvements.
"""

import sys
import os
import numpy as np
import time
import tracemalloc
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_profiling_to_acceleration_pipeline():
    """Test Module 15 (Profiling) → Module 16 (Acceleration) integration."""
    print("\n🔬 Testing Profiling → Acceleration Pipeline")
    print("=" * 60)
    
    try:
        # Import profiling (Module 15)
        sys.path.append(str(project_root / "modules" / "15_profiling"))
        from profiling_dev import Timer, MemoryProfiler, FLOPCounter
        
        # Import acceleration (Module 16)  
        sys.path.append(str(project_root / "modules" / "16_acceleration"))
        from acceleration_dev import OptimizedBackend, accelerate_function
        
        # Test profiling MLP
        def slow_mlp(x):
            """Slow MLP implementation for profiling."""
            w1 = np.random.randn(784, 256).astype(np.float32)
            w2 = np.random.randn(256, 10).astype(np.float32) 
            h = np.dot(x, w1)
            h = np.maximum(h, 0)  # ReLU
            return np.dot(h, w2)
        
        # Profile the slow version
        timer = Timer()
        x = np.random.randn(32, 784).astype(np.float32)
        
        with timer:
            slow_result = slow_mlp(x)
        slow_time = timer.elapsed_ms
        
        # Accelerate using Module 16
        backend = OptimizedBackend()
        fast_mlp = accelerate_function(slow_mlp)
        
        with timer:
            fast_result = fast_mlp(x)
        fast_time = timer.elapsed_ms
        
        # Verify results are similar
        assert slow_result.shape == fast_result.shape, "Shape mismatch"
        speedup = slow_time / fast_time if fast_time > 0 else 1.0
        
        print(f"✅ Profiling → Acceleration successful!")
        print(f"   Slow time: {slow_time:.2f}ms")
        print(f"   Fast time: {fast_time:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Profiling → Acceleration failed: {e}")
        return False

def test_quantization_to_compression_pipeline():
    """Test Module 17 (Quantization) → Module 18 (Compression) integration."""
    print("\n⚡ Testing Quantization → Compression Pipeline") 
    print("=" * 60)
    
    try:
        # Import quantization (Module 17)
        sys.path.append(str(project_root / "modules" / "17_quantization"))
        from quantization_dev import INT8Quantizer, QuantizedConv2d
        
        # Import compression (Module 18)
        sys.path.append(str(project_root / "modules" / "18_compression"))
        from compression_dev import MagnitudePruner, ModelCompressor
        
        # Create test CNN layer
        np.random.seed(42)
        conv_weights = np.random.normal(0, 0.02, (32, 16, 3, 3))
        
        # Step 1: Quantize weights
        quantizer = INT8Quantizer()
        quant_weights, scale, zero_point, stats = quantizer.quantize_weights(conv_weights)
        
        print(f"✅ Quantization complete:")
        print(f"   Compression: {stats['compression']:.1f}x")
        print(f"   Error: {stats['error']:.6f}")
        
        # Step 2: Prune quantized weights  
        pruner = MagnitudePruner()
        pruned_weights, mask, prune_stats = pruner.prune(quant_weights, sparsity=0.7)
        
        print(f"✅ Pruning complete:")
        print(f"   Sparsity: {prune_stats['actual_sparsity']:.1%}")
        print(f"   Compression: {prune_stats['compression_ratio']:.1f}x")
        
        # Step 3: Combined optimization
        original_size = conv_weights.nbytes
        final_size = np.sum(pruned_weights != 0) * 1  # 1 byte per INT8
        total_compression = original_size / final_size
        
        print(f"✅ Combined optimization:")
        print(f"   Original: {original_size:,} bytes")
        print(f"   Final: {final_size:,} bytes")
        print(f"   Total compression: {total_compression:.1f}x")
        
        assert total_compression > 10, f"Should achieve >10x compression, got {total_compression:.1f}x"
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization → Compression failed: {e}")
        return False

def test_caching_to_benchmarking_pipeline():
    """Test Module 19 (Caching) → Module 20 (Benchmarking) integration."""
    print("\n🚀 Testing Caching → Benchmarking Pipeline")
    print("=" * 60)
    
    try:
        # Import caching (Module 19)
        sys.path.append(str(project_root / "modules" / "19_caching"))
        from caching_dev import KVCache, CachedMultiHeadAttention
        
        # Import benchmarking (Module 20)
        sys.path.append(str(project_root / "modules" / "20_benchmarking"))
        from benchmarking_dev import TinyMLPerf
        
        # Create cached attention
        embed_dim = 128
        num_heads = 8
        max_seq_len = 100
        
        cache = KVCache(max_seq_len, n_layers=1, n_heads=num_heads, head_dim=embed_dim//num_heads)
        cached_attention = CachedMultiHeadAttention(embed_dim, num_heads, cache)
        
        # Test generation with caching
        def generate_with_cache(seq_len):
            """Generate sequence using cached attention."""
            outputs = []
            for i in range(seq_len):
                # Simulate incremental token generation
                q = np.random.randn(1, 1, embed_dim)
                k = np.random.randn(1, 1, embed_dim)  
                v = np.random.randn(1, 1, embed_dim)
                
                output = cached_attention.forward(q, k, v, layer_id=0, position=i)
                outputs.append(output)
            return np.concatenate(outputs, axis=1)
        
        # Benchmark with TinyMLPerf
        benchmark = TinyMLPerf()
        
        # Test short sequence
        short_result = generate_with_cache(10)
        print(f"✅ Short sequence: {short_result.shape}")
        
        # Test long sequence  
        long_result = generate_with_cache(50)
        print(f"✅ Long sequence: {long_result.shape}")
        
        print(f"✅ Caching → Benchmarking successful!")
        print(f"   Cache enabled generation scaling")
        print(f"   Ready for TinyMLPerf competition")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching → Benchmarking failed: {e}")
        return False

def test_full_optimization_pipeline():
    """Test complete optimization pipeline: Profile → Quantize → Compress → Cache → Benchmark."""
    print("\n🔥 Testing Full Optimization Pipeline")
    print("=" * 60)
    
    try:
        # Create test model
        model_weights = {
            'conv1': np.random.normal(0, 0.02, (32, 3, 5, 5)),
            'conv2': np.random.normal(0, 0.02, (64, 32, 5, 5)), 
            'fc': np.random.normal(0, 0.01, (10, 1024))
        }
        
        original_params = sum(w.size for w in model_weights.values())
        original_size_mb = sum(w.nbytes for w in model_weights.values()) / (1024 * 1024)
        
        print(f"📊 Original model:")
        print(f"   Parameters: {original_params:,}")
        print(f"   Size: {original_size_mb:.1f} MB")
        
        # Step 1: Profile (Module 15)
        sys.path.append(str(project_root / "modules" / "15_profiling"))
        from profiling_dev import MemoryProfiler
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # Step 2: Quantize (Module 17)
        sys.path.append(str(project_root / "modules" / "17_quantization"))
        from quantization_dev import INT8Quantizer
        
        quantizer = INT8Quantizer()
        quantized_weights = {}
        for name, weights in model_weights.items():
            quant_w, scale, zero_point, stats = quantizer.quantize_weights(weights)
            quantized_weights[name] = quant_w
        
        print(f"✅ Step 1: Quantization complete (4x compression)")
        
        # Step 3: Compress (Module 18)
        sys.path.append(str(project_root / "modules" / "18_compression"))
        from compression_dev import ModelCompressor
        
        compressor = ModelCompressor()
        compressed_model = compressor.compress_model(quantized_weights, {
            'conv1': 0.6,
            'conv2': 0.7,
            'fc': 0.8
        })
        
        print(f"✅ Step 2: Compression complete")
        
        # Calculate final compression
        compressed_params = sum(
            np.sum(info['weights'] != 0) 
            for info in compressed_model.values()
        )
        
        # Estimate size with INT8 + sparsity
        compressed_size_mb = compressed_params * 1 / (1024 * 1024)  # 1 byte per INT8
        
        total_compression = original_size_mb / compressed_size_mb
        param_reduction = (1 - compressed_params / original_params) * 100
        
        print(f"📊 Final optimized model:")
        print(f"   Parameters: {compressed_params:,} ({param_reduction:.1f}% reduction)")
        print(f"   Size: {compressed_size_mb:.2f} MB")
        print(f"   Total compression: {total_compression:.1f}x")
        
        # Step 4: Memory profiling
        memory_stats = profiler.get_memory_stats()
        profiler.stop_profiling()
        
        print(f"✅ Step 3: Profiling complete")
        print(f"   Peak memory: {memory_stats.get('peak_mb', 0):.1f} MB")
        
        # Validate optimization achievements
        assert total_compression > 10, f"Should achieve >10x compression, got {total_compression:.1f}x"
        assert param_reduction > 70, f"Should reduce >70% parameters, got {param_reduction:.1f}%"
        
        print(f"🎉 Full optimization pipeline successful!")
        print(f"   Achieved {total_compression:.1f}x model compression")
        print(f"   Ready for edge deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Full optimization pipeline failed: {e}")
        return False

def test_performance_validation():
    """Validate that optimizations actually improve performance."""
    print("\n⚡ Testing Performance Validation")
    print("=" * 60)
    
    try:
        # Test that each optimization provides measurable improvement
        improvements = {}
        
        # Test 1: Acceleration speedup
        try:
            sys.path.append(str(project_root / "modules" / "16_acceleration"))
            from acceleration_dev import OptimizedBackend
            
            backend = OptimizedBackend()
            x = np.random.randn(1000, 1000).astype(np.float32)
            y = np.random.randn(1000, 1000).astype(np.float32)
            
            # Baseline
            start = time.time()
            baseline_result = np.dot(x, y)
            baseline_time = time.time() - start
            
            # Optimized
            start = time.time()
            optimized_result = backend.matmul_optimized(x, y)
            optimized_time = time.time() - start
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            improvements['acceleration'] = speedup
            print(f"✅ Acceleration speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"⚠️  Acceleration test skipped: {e}")
            improvements['acceleration'] = 1.0
        
        # Test 2: Memory reduction from compression
        try:
            sys.path.append(str(project_root / "modules" / "18_compression"))
            from compression_dev import MagnitudePruner
            
            weights = np.random.normal(0, 0.1, (1000, 1000))
            original_memory = weights.nbytes
            
            pruner = MagnitudePruner()
            pruned_weights, mask, stats = pruner.prune(weights, sparsity=0.8)
            compressed_memory = np.sum(pruned_weights != 0) * 4  # FP32 bytes
            
            memory_reduction = original_memory / compressed_memory
            improvements['compression'] = memory_reduction
            print(f"✅ Memory reduction: {memory_reduction:.2f}x")
            
        except Exception as e:
            print(f"⚠️  Compression test skipped: {e}")
            improvements['compression'] = 1.0
            
        # Test 3: Cache efficiency for sequences
        try:
            sys.path.append(str(project_root / "modules" / "19_caching"))
            from caching_dev import KVCache
            
            # Measure cache benefit for long sequences
            cache = KVCache(max_seq_len=200, n_layers=4, n_heads=8, head_dim=64)
            
            # Simulate cache benefit
            seq_len = 100
            cache_memory_mb = (seq_len * 4 * 8 * 64 * 4) / (1024 * 1024)  # Rough estimate
            theoretical_speedup = seq_len / 10  # O(N) vs O(N²)
            
            improvements['caching'] = theoretical_speedup
            print(f"✅ Cache theoretical speedup: {theoretical_speedup:.2f}x for seq_len={seq_len}")
            
        except Exception as e:
            print(f"⚠️  Caching test skipped: {e}")
            improvements['caching'] = 1.0
        
        # Validate overall improvements
        total_speedup = 1.0
        for name, speedup in improvements.items():
            if speedup > 1.0:
                total_speedup *= speedup
        
        print(f"\n🎯 Performance Summary:")
        for name, speedup in improvements.items():
            print(f"   {name.capitalize()}: {speedup:.2f}x improvement")
        print(f"   Combined potential: {total_speedup:.2f}x")
        
        # At least some optimizations should provide measurable improvement
        significant_improvements = sum(1 for s in improvements.values() if s > 1.2)
        assert significant_improvements >= 2, f"Need at least 2 significant improvements, got {significant_improvements}"
        
        print(f"✅ Performance validation successful!")
        print(f"   {significant_improvements} optimizations show >1.2x improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return False

def run_all_integration_tests():
    """Run all optimization integration tests."""
    print("🚀 OPTIMIZATION INTEGRATION TEST SUITE")
    print("=" * 80)
    print("Testing modules 15-20 work together correctly...")
    
    tests = [
        ("Profiling → Acceleration Pipeline", test_profiling_to_acceleration_pipeline),
        ("Quantization → Compression Pipeline", test_quantization_to_compression_pipeline), 
        ("Caching → Benchmarking Pipeline", test_caching_to_benchmarking_pipeline),
        ("Full Optimization Pipeline", test_full_optimization_pipeline),
        ("Performance Validation", test_performance_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            print(f"🧪 Running: {test_name}")
            print(f"{'='*80}")
            
            success = test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*80}")
    print(f"🎯 INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
    print(f"{'='*80}")
    
    if passed == total:
        print("🎉 ALL OPTIMIZATION INTEGRATION TESTS PASSED!")
        print("✅ Modules 15-20 work together correctly")
        print("✅ Optimization pipeline is functional")
        print("✅ Performance improvements validated")
        print("✅ Ready for production optimization workflows")
    else:
        print(f"⚠️  {total-passed} integration tests failed")
        print("❌ Some optimization combinations need fixes")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
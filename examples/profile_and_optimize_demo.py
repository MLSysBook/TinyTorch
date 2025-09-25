#!/usr/bin/env python3
"""
Profile → Optimize Demo

Simple demonstration of the Profile → Optimize cycle using TinyTorch modules.
Shows how Module 15 (Profiling) identifies bottlenecks and Module 16 (Acceleration)
fixes them with measurable improvements.

Perfect for students learning the optimization workflow.
"""

import numpy as np
from tinytorch.utils.profiler import Timer, MemoryProfiler
from tinytorch.core.acceleration import matmul_naive, matmul_blocked


def demonstrate_matrix_multiplication_optimization():
    """Show how profiling guides matrix multiplication optimization."""
    print("🔬 PROFILE → OPTIMIZE DEMONSTRATION")
    print("=" * 50)
    print("Using TinyTorch Module 15 (Profiling) and Module 16 (Acceleration)")
    
    # Create test matrices
    sizes = [50, 100, 200, 400]
    print("\\n📊 Profiling matrix multiplication performance...")
    
    timer = Timer()
    results = {}
    
    for size in sizes:
        print(f"\\n🧮 Testing {size}×{size} matrices:")
        
        # Create random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Profile naive implementation
        naive_stats = timer.measure(matmul_naive, warmup=2, runs=10, args=(A, B))
        
        # Profile blocked implementation  
        blocked_stats = timer.measure(matmul_blocked, warmup=2, runs=10, args=(A, B))
        
        # Calculate speedup
        speedup = naive_stats['mean_ms'] / blocked_stats['mean_ms']
        
        print(f"   Naive:   {naive_stats['mean_ms']:.2f} ± {naive_stats['std_ms']:.2f} ms")
        print(f"   Blocked: {blocked_stats['mean_ms']:.2f} ± {blocked_stats['std_ms']:.2f} ms")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        
        results[size] = {
            'naive_ms': naive_stats['mean_ms'],
            'blocked_ms': blocked_stats['mean_ms'],
            'speedup': speedup
        }
        
        # Verify correctness
        naive_result = matmul_naive(A, B)
        blocked_result = matmul_blocked(A, B)
        correctness = np.allclose(naive_result, blocked_result, atol=1e-4)
        print(f"   ✅ Correctness: {'PASS' if correctness else 'FAIL'}")
    
    # Analysis
    print("\\n📈 PERFORMANCE ANALYSIS")
    print("=" * 30)
    best_speedup = max(results[size]['speedup'] for size in sizes)
    worst_speedup = min(results[size]['speedup'] for size in sizes)
    
    print(f"Best speedup: {best_speedup:.2f}x (larger matrices benefit more)")
    print(f"Worst speedup: {worst_speedup:.2f}x (overhead for small matrices)")
    
    print("\\n🎯 KEY INSIGHTS:")
    print("• Blocked matrix multiplication improves cache locality")
    print("• Larger matrices see bigger improvements") 
    print("• Always profile before optimizing!")
    print("• Verify correctness after optimization")


def demonstrate_memory_profiling():
    """Show memory profiling capabilities."""
    print("\\n\\n💾 MEMORY PROFILING DEMONSTRATION")
    print("=" * 50)
    
    memory_profiler = MemoryProfiler()
    
    def memory_intensive_operation():
        """Operation that uses significant memory."""
        # Create large arrays
        large_arrays = []
        for i in range(5):
            array = np.random.randn(1000, 1000).astype(np.float32)
            large_arrays.append(array)
        
        # Do some computation
        result = sum(arr.sum() for arr in large_arrays)
        return result
    
    print("\\n🔍 Profiling memory usage...")
    memory_stats = memory_profiler.profile(memory_intensive_operation)
    
    print(f"📊 Memory Profile:")
    print(f"   Baseline: {memory_stats['baseline_mb']:.2f} MB")
    print(f"   Peak Usage: {memory_stats['peak_mb']:.2f} MB") 
    print(f"   Memory Allocated: {memory_stats['allocated_mb']:.2f} MB")
    
    print(f"\\n💡 Memory Insights:")
    print(f"   • Operation used {memory_stats['peak_mb']:.1f} MB at peak")
    print(f"   • This helps identify memory bottlenecks")
    print(f"   • Critical for optimizing large model training")


def main():
    """Run profile and optimize demonstration."""
    print("🚀 Starting Profile → Optimize demonstration...")
    print("This shows the fundamental optimization workflow:")
    print("1. Profile to identify bottlenecks")
    print("2. Apply targeted optimizations")
    print("3. Measure improvements")
    print("4. Verify correctness")
    
    try:
        # Demonstrate the core workflow
        demonstrate_matrix_multiplication_optimization()
        demonstrate_memory_profiling()
        
        print("\\n\\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 50)
        print("You've learned the essential optimization workflow:")
        print("✓ Use profiling to find bottlenecks")
        print("✓ Apply specific optimizations")  
        print("✓ Measure performance improvements")
        print("✓ Always verify correctness")
        
        print("\\n📚 Next steps:")
        print("• Try profiling your own TinyTorch models")
        print("• Experiment with different optimization techniques")
        print("• Use TinyMLPerf to benchmark your improvements")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
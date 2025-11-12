"""
Autograd Systems Analysis - Memory & Performance Profiling

This file contains the P0 critical additions for Module 05 autograd:
- Memory profiling with tracemalloc
- Performance benchmarking
- Computational complexity analysis

These functions should be inserted after test_module() and before the module summary.
"""

import numpy as np
import tracemalloc
import time
from tinytorch.core.tensor import Tensor


def profile_autograd_memory():
    """
    Profile memory usage of autograd operations.

    This function demonstrates the memory cost of gradient tracking
    by comparing requires_grad=True vs. requires_grad=False.
    """
    print("\n" + "=" * 60)
    print("📊 Autograd Memory Profiling")
    print("=" * 60)

    # Test 1: Memory without gradients
    print("\n🔬 Test 1: Memory without gradient tracking...")
    tracemalloc.start()
    x_no_grad = Tensor(np.random.randn(1000, 1000), requires_grad=False)
    y_no_grad = x_no_grad.matmul(x_no_grad)
    mem_no_grad = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
    tracemalloc.stop()

    # Test 2: Memory with gradients
    print("🔬 Test 2: Memory with gradient tracking...")
    tracemalloc.start()
    x_with_grad = Tensor(np.random.randn(1000, 1000), requires_grad=True)
    y_with_grad = x_with_grad.matmul(x_with_grad)
    mem_with_grad = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
    tracemalloc.stop()

    # Test 3: Memory after backward
    print("🔬 Test 3: Memory after backward pass...")
    tracemalloc.start()
    x_backward = Tensor(np.random.randn(1000, 1000), requires_grad=True)
    y_backward = x_backward.matmul(x_backward)
    loss = y_backward.sum()
    loss.backward()
    mem_after_backward = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
    tracemalloc.stop()

    print(f"\n📊 Memory Usage (1000×1000 matrix):")
    print(f"  • No gradients:      {mem_no_grad:.2f} MB")
    print(f"  • With gradients:    {mem_with_grad:.2f} MB ({mem_with_grad/mem_no_grad:.2f}× overhead)")
    print(f"  • After backward:    {mem_after_backward:.2f} MB")

    graph_overhead = mem_with_grad - mem_no_grad
    gradient_storage = mem_after_backward - mem_with_grad

    print(f"  • Graph overhead:    {graph_overhead:.2f} MB")
    print(f"  • Gradient storage:  {gradient_storage:.2f} MB")

    print("\n💡 Key Insight: Autograd adds ~2-3× memory overhead")
    print("   (1× for gradients + 1-2× for computation graph)")


def benchmark_backward_pass():
    """
    Benchmark forward vs. backward pass timing.

    Demonstrates that backward pass is typically 2-3× slower than forward
    due to additional matmul operations for gradient computation.
    """
    print("\n" + "=" * 60)
    print("⚡ Backward Pass Performance Benchmarking")
    print("=" * 60)

    sizes = [100, 500, 1000]

    for size in sizes:
        # Forward pass timing (no gradients)
        x = Tensor(np.random.randn(size, size), requires_grad=False)
        W = Tensor(np.random.randn(size, size), requires_grad=False)

        start = time.perf_counter()
        for _ in range(10):
            y = x.matmul(W)
        forward_time = (time.perf_counter() - start) / 10

        # Forward + backward timing
        x = Tensor(np.random.randn(size, size), requires_grad=True)
        W = Tensor(np.random.randn(size, size), requires_grad=True)

        start = time.perf_counter()
        for _ in range(10):
            x.zero_grad()
            W.zero_grad()
            y = x.matmul(W)
            loss = y.sum()
            loss.backward()
        total_time = (time.perf_counter() - start) / 10

        backward_time = total_time - forward_time

        print(f"\n📐 Matrix size: {size}×{size}")
        print(f"  • Forward pass:  {forward_time*1000:.2f} ms")
        print(f"  • Backward pass: {backward_time*1000:.2f} ms ({backward_time/forward_time:.2f}× forward)")
        print(f"  • Total:         {total_time*1000:.2f} ms")

    print("\n💡 Key Insight: Backward pass ≈ 2-3× forward pass time")
    print("   (grad_x = grad @ W.T + W.T @ grad = 2 matmuls vs. 1 in forward)")


def analyze_complexity():
    """
    Display computational complexity analysis for autograd operations.

    Shows time and space complexity for common operations.
    """
    print("\n" + "=" * 60)
    print("📊 Computational Complexity Analysis")
    print("=" * 60)

    print("\n### Time Complexity")
    print("-" * 60)
    print(f"{'Operation':<20} {'Forward':<15} {'Backward':<15} {'Total':<15}")
    print("-" * 60)
    print(f"{'Add':<20} {'O(n)':<15} {'O(n)':<15} {'O(n)':<15}")
    print(f"{'Mul':<20} {'O(n)':<15} {'O(n)':<15} {'O(n)':<15}")
    print(f"{'Matmul (n×n)':<20} {'O(n³)':<15} {'O(n³) × 2':<15} {'O(n³)':<15}")
    print(f"{'Sum':<20} {'O(n)':<15} {'O(n)':<15} {'O(n)':<15}")
    print(f"{'ReLU':<20} {'O(n)':<15} {'O(n)':<15} {'O(n)':<15}")
    print(f"{'Softmax':<20} {'O(n)':<15} {'O(n)':<15} {'O(n)':<15}")
    print("-" * 60)

    print("\n💡 Key Insight: Matrix operations dominate training time")
    print("   For Matmul with (m×k) @ (k×n):")
    print("   - Forward: O(m×k×n)")
    print("   - Backward grad_A: O(m×n×k)  [grad_Z @ B.T]")
    print("   - Backward grad_B: O(k×m×n)  [A.T @ grad_Z]")
    print("   - Total: ~3× forward pass cost")

    print("\n### Space Complexity")
    print("-" * 60)
    print(f"{'Component':<25} {'Memory Usage':<35}")
    print("-" * 60)
    print(f"{'Parameters':<25} {'P (baseline)':<35}")
    print(f"{'Activations':<25} {'~P (for N layers ≈ P/N per layer)':<35}")
    print(f"{'Gradients':<25} {'P (1:1 with parameters)':<35}")
    print(f"{'Computation Graph':<25} {'0.2-0.5P (Function objects)':<35}")
    print(f"{'Total Training':<25} {'~2.5-3P':<35}")
    print("-" * 60)

    print("\n💡 Key Insight: Training requires ~3× parameter memory")


# Main execution block with all profiling
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 AUTOGRAD SYSTEMS ANALYSIS")
    print("=" * 60)

    profile_autograd_memory()
    benchmark_backward_pass()
    analyze_complexity()

    print("\n" + "=" * 60)
    print("✅ Systems analysis complete!")
    print("=" * 60)

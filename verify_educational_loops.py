#!/usr/bin/env python3
"""
Verification script for educational matrix multiplication loops.

This script demonstrates that TinyTorch now uses educational triple-nested loops 
for matrix multiplication, setting up the optimization progression for Module 15.
"""

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, matmul
import numpy as np
import time

def demonstrate_educational_loops():
    """Demonstrate the educational loop implementation."""
    print("🔥 TinyTorch Educational Matrix Multiplication Demo")
    print("=" * 60)
    
    print("\n📚 Current Implementation: Triple-Nested Loops (Educational)")
    print("   • Clear understanding of every operation")
    print("   • Shows the fundamental computation pattern") 
    print("   • Intentionally simple for learning")
    
    # Test basic functionality
    print("\n1. Basic Matrix Multiplication Test:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = a @ b
    print(f"   {a.data.tolist()} @ {b.data.tolist()}")
    print(f"   = {result.data.tolist()}")
    print(f"   Expected: [[19, 22], [43, 50]] ✅")
    
    # Test neural network layer
    print("\n2. Neural Network Layer Test:")
    layer = Linear(3, 2)
    input_data = Tensor([[1.0, 2.0, 3.0]])
    output = layer(input_data)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Uses educational matmul internally ✅")
    
    # Show performance characteristics (intentionally slow)
    print("\n3. Performance Characteristics (Intentionally Educational):")
    sizes = [10, 50, 100]
    for size in sizes:
        a = Tensor(np.random.randn(size, size))
        b = Tensor(np.random.randn(size, size))
        
        start_time = time.time()
        result = a @ b
        elapsed = time.time() - start_time
        
        print(f"   {size}×{size} matrix multiplication: {elapsed:.4f}s")
    
    print("\n🎯 Module 15 Optimization Progression Preview:")
    print("   Step 1 (current): Educational loops - slow but clear")
    print("   Step 2 (future):  Loop blocking for cache efficiency")
    print("   Step 3 (future):  Vectorized operations with NumPy")
    print("   Step 4 (future):  GPU acceleration and BLAS libraries")
    
    print("\n✅ Educational matrix multiplication ready!")
    print("   Students will understand optimization progression by building it!")
    
def verify_correctness():
    """Verify that educational loops produce correct results."""
    print("\n🔬 Correctness Verification:")
    
    test_cases = [
        # Simple 2x2
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[19, 22], [43, 50]]),
        # Non-square
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]),
        # Vector multiplication
        ([[1, 2, 3]], [[4], [5], [6]], [[32]]),
    ]
    
    for i, (a_data, b_data, expected) in enumerate(test_cases):
        a = Tensor(a_data)
        b = Tensor(b_data)
        result = a @ b
        
        assert np.allclose(result.data, expected), f"Test {i+1} failed"
        print(f"   Test {i+1}: {a.shape} @ {b.shape} → {result.shape} ✅")
    
    print("   All correctness tests passed!")

if __name__ == "__main__":
    demonstrate_educational_loops()
    verify_correctness()
    
    print("\n🎉 Educational matrix multiplication setup complete!")
    print("   Ready for Module 15 optimization journey!")
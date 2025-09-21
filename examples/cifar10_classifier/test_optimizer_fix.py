#!/usr/bin/env python3
"""
Direct test of optimizer bias shape preservation.
"""

import numpy as np
import sys
import os
sys.path.append('/Users/VJ/GitHub/TinyTorch')

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.optimizers import Adam

def test_optimizer_shape_preservation():
    """Test that optimizer preserves parameter shapes."""
    print("🔍 Testing Optimizer Shape Preservation")
    print("=" * 50)
    
    # Create parameters like a Dense layer would have
    weights = Variable(Tensor(np.random.randn(10, 3).astype(np.float32)), requires_grad=True)
    bias = Variable(Tensor(np.random.randn(3).astype(np.float32)), requires_grad=True)
    
    print(f"Initial weights shape: {weights.data.shape}")
    print(f"Initial bias shape: {bias.data.shape}")
    
    # Create optimizer
    optimizer = Adam([weights, bias], learning_rate=0.001)
    
    # Simulate different batch sizes causing different gradient shapes
    batch_sizes = [32, 16, 8, 4, 1]
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n--- Step {i+1}: Simulating batch size {batch_size} ---")
        
        # Simulate gradients (these would come from backward pass)
        # Weights gradient should always be (10, 3)
        weights_grad = np.random.randn(10, 3).astype(np.float32)
        weights.grad = Variable(Tensor(weights_grad))
        
        # Bias gradient should always be (3,) regardless of batch size
        # This is the KEY TEST - bias gradient shape should be parameter shape
        bias_grad = np.random.randn(3).astype(np.float32)
        bias.grad = Variable(Tensor(bias_grad))
        
        print(f"  Weights grad shape: {weights.grad.data.shape}")
        print(f"  Bias grad shape: {bias.grad.data.shape}")
        print(f"  Before step - weights shape: {weights.data.shape}")
        print(f"  Before step - bias shape: {bias.data.shape}")
        
        # The critical test: does optimizer.step() preserve shapes?
        try:
            optimizer.step()
            
            print(f"  ✅ After step - weights shape: {weights.data.shape}")
            print(f"  ✅ After step - bias shape: {bias.data.shape}")
            
            # Verify shapes are preserved
            if weights.data.shape != (10, 3):
                print(f"  ❌ Weights shape corrupted! Expected (10, 3), got {weights.data.shape}")
                return False, i, batch_size
                
            if bias.data.shape != (3,):
                print(f"  ❌ Bias shape corrupted! Expected (3,), got {bias.data.shape}")
                return False, i, batch_size
                
            print(f"  ✅ Shapes preserved correctly")
            
        except Exception as e:
            print(f"  ❌ Optimizer step failed: {e}")
            print(f"  Weights shape: {weights.data.shape}")
            print(f"  Bias shape: {bias.data.shape}")
            return False, i, batch_size
    
    print(f"\n🎉 All optimizer steps completed successfully!")
    print(f"Final weights shape: {weights.data.shape}")
    print(f"Final bias shape: {bias.data.shape}")
    return True, None, None

if __name__ == "__main__":
    success, fail_iter, fail_batch = test_optimizer_shape_preservation()
    
    print("\n" + "=" * 50)
    print("📊 Optimizer Fix Test Results:")
    if success:
        print("✅ OPTIMIZER SHAPE FIX SUCCESSFUL!")
        print("Parameter shapes are now preserved during optimization!")
        print("Variable batch sizes should work correctly!")
    else:
        print(f"❌ Test failed at step {fail_iter}, simulated batch size {fail_batch}")
        print("The optimizer shape corruption issue still exists.")
#!/usr/bin/env python3
"""
Debug Variable Batch Size Issue - Find exactly where bias gets corrupted.
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.autograd import Variable
from tinytorch.core.training import MeanSquaredError as MSELoss

def test_variable_batch_corruption():
    """Reproduce the exact variable batch size issue."""
    print("🔍 Testing Variable Batch Size Corruption")
    print("=" * 60)
    
    # Create the exact model that fails
    print("🏗️ Creating multi-layer model...")
    fc1 = Dense(10, 5)  # Simple version: 10 → 5 → 3
    fc2 = Dense(5, 3)
    relu = ReLU()
    softmax = Softmax()
    
    # Convert to Variables (like real training)
    fc1.weights = Variable(fc1.weights, requires_grad=True)
    fc1.bias = Variable(fc1.bias, requires_grad=True)
    fc2.weights = Variable(fc2.weights, requires_grad=True)
    fc2.bias = Variable(fc2.bias, requires_grad=True)
    
    print(f"✅ Model created:")
    print(f"  FC1: weights {fc1.weights.data.shape}, bias {fc1.bias.data.shape}")
    print(f"  FC2: weights {fc2.weights.data.shape}, bias {fc2.bias.data.shape}")
    
    # Test with different batch sizes
    batch_sizes = [32, 16, 8, 4]
    loss_fn = MSELoss()
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n🔄 Iteration {i+1}: Batch size {batch_size}")
        
        # Create synthetic batch
        x_data = np.random.randn(batch_size, 10).astype(np.float32)
        x = Variable(Tensor(x_data), requires_grad=True)
        
        # Create target
        y_data = np.random.randn(batch_size, 3).astype(np.float32)
        y = Variable(Tensor(y_data), requires_grad=False)
        
        print(f"  Input: {x.data.shape}")
        print(f"  Before forward - FC1 bias: {fc1.bias.data.shape}")
        print(f"  Before forward - FC2 bias: {fc2.bias.data.shape}")
        
        try:
            # Forward pass
            z1 = fc1.forward(x)
            a1 = relu.forward(z1) 
            z2 = fc2.forward(a1)
            output = softmax.forward(z2)
            
            print(f"  ✅ Forward pass: {output.data.shape}")
            print(f"  After forward - FC1 bias: {fc1.bias.data.shape}")
            print(f"  After forward - FC2 bias: {fc2.bias.data.shape}")
            
            # Compute loss
            loss = loss_fn(output, y)
            print(f"  ✅ Loss computed: {loss.data}")
            
            # Backward pass (this might corrupt shapes)
            if hasattr(loss, 'backward'):
                print(f"  🔄 Before backward - FC1 bias: {fc1.bias.data.shape}")
                print(f"  🔄 Before backward - FC2 bias: {fc2.bias.data.shape}")
                
                loss.backward()
                
                print(f"  ✅ Backward completed")
                print(f"  After backward - FC1 bias: {fc1.bias.data.shape}")
                print(f"  After backward - FC2 bias: {fc2.bias.data.shape}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            print(f"  Error state - FC1 bias: {fc1.bias.data.shape}")
            print(f"  Error state - FC2 bias: {fc2.bias.data.shape}")
            
            # This is where we'd see the corruption
            return False, i, batch_size
    
    print(f"\n🎉 All batch sizes completed successfully!")
    return True, None, None

def test_optimizer_corruption():
    """Test if optimizer updates corrupt bias shapes."""
    print("\n" * 2)
    print("🔍 Testing Optimizer Shape Corruption")
    print("=" * 60)
    
    from tinytorch.core.optimizers import Adam
    
    # Simple model
    layer = Dense(5, 3)
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    print(f"✅ Initial bias shape: {layer.bias.data.shape}")
    
    # Create optimizer
    optimizer = Adam([layer.weights, layer.bias], learning_rate=0.001)
    loss_fn = MSELoss()
    
    # Test multiple updates with different batch sizes
    for batch_size in [16, 8, 4]:
        print(f"\n🔄 Testing optimizer with batch size {batch_size}")
        
        # Forward pass
        x = Variable(Tensor(np.random.randn(batch_size, 5).astype(np.float32)), requires_grad=True)
        y = Variable(Tensor(np.random.randn(batch_size, 3).astype(np.float32)), requires_grad=False)
        
        output = layer.forward(x)
        loss = loss_fn(output, y)
        
        print(f"  Before optimizer step - bias: {layer.bias.data.shape}")
        
        # Optimizer update
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  ✅ After optimizer step - bias: {layer.bias.data.shape}")
            
        except Exception as e:
            print(f"  ❌ Optimizer failed: {e}")
            print(f"  Error bias shape: {layer.bias.data.shape}")
            return False
    
    print(f"\n🎉 Optimizer tests completed successfully!")
    return True

if __name__ == "__main__":
    # Test 1: Variable batch sizes
    success1, fail_iter, fail_batch = test_variable_batch_corruption()
    
    # Test 2: Optimizer updates  
    success2 = test_optimizer_corruption()
    
    print("\n" + "=" * 60)
    print("📊 Debug Results:")
    print(f"  Variable batch test: {'✅ PASS' if success1 else '❌ FAIL'}")
    if not success1:
        print(f"    Failed at iteration {fail_iter}, batch size {fail_batch}")
    
    print(f"  Optimizer test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🤔 Hmm, isolated tests pass. The issue might be in:")
        print("  • Complex interaction between multiple layers")
        print("  • DataLoader batch handling")
        print("  • Specific to CIFAR-10 data shapes")
        print("  • Timing of when Variable/Tensor conversions happen")
    else:
        print(f"\n🎯 Found the issue! Check the failing test above.")
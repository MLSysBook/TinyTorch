#!/usr/bin/env python3
"""
Debug the bias broadcasting issue - find exactly where shapes get corrupted.
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.autograd import Variable

def debug_bias_shapes():
    """Debug exactly where bias shapes get corrupted."""
    print("🔍 Debugging Bias Shape Corruption")
    print("=" * 50)
    
    # Create a Dense layer
    layer = Dense(10, 5)  # 10 inputs → 5 outputs
    
    print("🏗️ Initial Dense Layer State:")
    print(f"  Weights shape: {layer.weights.shape}")
    print(f"  Bias shape: {layer.bias.shape}")
    print(f"  Bias data: {layer.bias.data}")
    print()
    
    # Convert to Variables (like our model does)
    print("🔄 Converting to Variables...")
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    print("After Variable conversion:")
    print(f"  Weights shape: {layer.weights.data.shape}")
    print(f"  Bias shape: {layer.bias.data.shape}")
    print(f"  Bias type: {type(layer.bias.data)}")
    print()
    
    # Test with different batch sizes
    for batch_size in [32, 16, 8]:
        print(f"📦 Testing with batch size {batch_size}:")
        
        # Create input
        input_data = np.random.randn(batch_size, 10).astype(np.float32)
        x = Variable(Tensor(input_data), requires_grad=True)
        
        print(f"  Input shape: {x.data.shape}")
        print(f"  Bias shape before forward: {layer.bias.data.shape}")
        
        try:
            # Forward pass
            output = layer.forward(x)
            print(f"  ✅ Forward pass succeeded: {output.data.shape}")
            print(f"  Bias shape after forward: {layer.bias.data.shape}")
            
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            print(f"  Bias shape when failed: {layer.bias.data.shape}")
            
            # Let's see what happened inside
            print(f"  Debug info:")
            print(f"    Input to layer: {x.data.shape}")
            print(f"    Weights: {layer.weights.data.shape}")
            print(f"    Expected output: ({batch_size}, 5)")
            print(f"    Actual bias: {layer.bias.data.shape}")
            break
        
        print()

def debug_manual_forward():
    """Debug the forward pass step by step."""
    print("🔧 Manual Forward Pass Debug")
    print("=" * 50)
    
    # Create simple case
    layer = Dense(3, 2)  # 3 → 2
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    # Test data
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2 samples
    x = Variable(Tensor(x_data), requires_grad=True)
    
    print(f"Input: {x.data.shape} = {x_data}")
    print(f"Weights: {layer.weights.data.shape}")
    print(f"Bias: {layer.bias.data.shape} = {layer.bias.data.data}")
    print()
    
    # Manual matrix multiplication
    print("Step 1: Matrix multiplication")
    weights_data = layer.weights.data.data
    result = x_data @ weights_data
    print(f"  x @ weights = {result.shape}")
    print(f"  Result: {result}")
    print()
    
    print("Step 2: Bias addition")
    bias_data = layer.bias.data.data
    print(f"  Bias data: {bias_data.shape} = {bias_data}")
    
    try:
        final = result + bias_data
        print(f"  ✅ Manual addition works: {final.shape}")
        print(f"  Final result: {final}")
    except Exception as e:
        print(f"  ❌ Manual addition fails: {e}")
    
    print()
    print("Step 3: Try TinyTorch forward")
    try:
        output = layer.forward(x)
        print(f"  ✅ TinyTorch forward works: {output.data.shape}")
    except Exception as e:
        print(f"  ❌ TinyTorch forward fails: {e}")

if __name__ == "__main__":
    debug_bias_shapes()
    print()
    debug_manual_forward()
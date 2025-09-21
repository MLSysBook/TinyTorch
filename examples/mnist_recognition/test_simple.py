#!/usr/bin/env python3
"""
Simple test to isolate the broadcasting issue.
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.autograd import Variable

def test_dense_layer_batches():
    """Test Dense layer with different batch sizes."""
    print("Testing Dense layer with variable batch sizes...")
    
    # Create a simple Dense layer
    layer = Dense(784, 128)
    activation = ReLU()
    
    print(f"Layer weights shape: {layer.weights.shape}")
    print(f"Layer bias shape: {layer.bias.shape}")
    
    # Test with different batch sizes
    for batch_size in [32, 16, 1]:
        print(f"\n--- Testing batch size {batch_size} ---")
        
        # Create input data
        input_data = np.random.randn(batch_size, 784).astype(np.float32)
        x = Tensor(input_data)
        
        print(f"Input shape: {x.shape}")
        
        try:
            # Test with regular Tensor
            output = layer.forward(x)
            print(f"✅ Tensor output shape: {output.shape}")
            
            # Test with Variable
            x_var = Variable(x, requires_grad=True)
            print(f"Variable input shape: {x_var.data.shape}")
            
            output_var = layer.forward(x_var)
            print(f"✅ Variable output shape: {output_var.data.shape}")
            
            # Test with ReLU
            activated = activation.forward(output_var)
            print(f"✅ After ReLU shape: {activated.data.shape}")
            
        except Exception as e:
            print(f"❌ Error with batch size {batch_size}: {e}")
            return False
    
    return True

def test_model_conversion():
    """Test the Variable conversion in our model."""
    print("\n" + "="*50)
    print("Testing model Variable conversion...")
    
    from examples.mnist_recognition.train_simple import MNISTClassifier
    
    model = MNISTClassifier()
    
    # Test different batch sizes
    for batch_size in [32, 16, 1]:
        print(f"\n--- Testing model with batch size {batch_size} ---")
        
        input_data = np.random.randn(batch_size, 784).astype(np.float32)
        x = Tensor(input_data)
        
        print(f"Input shape: {x.shape}")
        
        try:
            output = model.forward(x)
            print(f"✅ Model output shape: {output.data.shape}")
            
        except Exception as e:
            print(f"❌ Model error with batch size {batch_size}: {e}")
            print(f"   This is where our MNIST training fails!")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Debugging TinyTorch MNIST Training Issues")
    print("="*60)
    
    # Test 1: Basic Dense layer
    success1 = test_dense_layer_batches()
    
    # Test 2: Full model
    success2 = test_model_conversion()
    
    print("\n" + "="*60)
    print("📊 Test Results:")
    print(f"  Dense layer: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Full model:  {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! MNIST training should work.")
    else:
        print("\n🔧 Found the issue - debugging info provided above.")
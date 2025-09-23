#!/usr/bin/env python3
"""
Quick Example Validation - Test that all examples can at least run their core functionality
without long training loops or large data downloads.
"""

import sys
import os
sys.path.append('.')

def test_xor_example():
    """Test XOR example core functionality."""
    print("🔬 Testing XOR Example Core Functionality...")
    try:
        from examples.xornet.train_xor_modern_api import XORNet, create_xor_dataset
        import tinytorch.nn as nn
        import tinytorch.optim as optim
        import numpy as np
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import Variable
        from tinytorch.core.training import MeanSquaredError as MSELoss
        
        # Test network creation
        model = XORNet()
        optimizer = optim.SGD(model.parameters(), learning_rate=0.1)
        criterion = MSELoss()
        
        # Test data creation
        X, y = create_xor_dataset()
        
        # Test single forward pass
        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y), requires_grad=False)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Extract loss value properly
        if hasattr(loss, 'data'):
            if hasattr(loss.data, 'data'):
                loss_val = float(loss.data.data.flat[0])
            else:
                loss_val = float(loss.data.flat[0])
        else:
            loss_val = float(loss.flat[0])
        
        print(f"  ✅ XOR network created successfully")
        print(f"  ✅ Forward pass works, loss: {loss_val:.4f}")
        print(f"  ✅ Output shape: {outputs.data.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ XOR example failed: {e}")
        return False

def test_mnist_example():
    """Test MNIST MLP example core functionality."""
    print("🔬 Testing MNIST MLP Example Core Functionality...")
    try:
        from examples.mnist.train_mlp_modern_api import SimpleMLP, create_sample_mnist_data
        import tinytorch.nn as nn
        import tinytorch.optim as optim
        import numpy as np
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import Variable
        from tinytorch.core.training import CrossEntropyLoss
        
        # Test network creation
        model = SimpleMLP()
        optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
        criterion = CrossEntropyLoss()
        
        # Test data creation
        X, y = create_sample_mnist_data()
        
        # Test single forward pass
        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y.astype(np.float32)), requires_grad=False)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Extract loss value properly
        if hasattr(loss, 'data'):
            if hasattr(loss.data, 'data'):
                loss_val = float(loss.data.data.flat[0])
            else:
                loss_val = float(loss.data.flat[0])
        else:
            loss_val = float(loss.flat[0])
        
        print(f"  ✅ MNIST MLP created successfully")
        print(f"  ✅ Forward pass works, loss: {loss_val:.4f}")
        print(f"  ✅ Output shape: {outputs.data.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ MNIST example failed: {e}")
        return False

def test_cifar10_example_structure():
    """Test CIFAR-10 CNN example structure (without data download)."""
    print("🔬 Testing CIFAR-10 CNN Example Structure...")
    try:
        from examples.cifar10.train_cnn_modern_api import ModernCNN
        import tinytorch.nn as nn
        import tinytorch.optim as optim
        import numpy as np
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import Variable
        from tinytorch.core.training import CrossEntropyLoss
        
        # Test network creation
        model = ModernCNN()
        optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
        criterion = CrossEntropyLoss()
        
        # Test with sample CIFAR-like data (avoid download)
        batch_size = 4
        X = np.random.randn(batch_size, 3, 32, 32).astype(np.float32) * 0.1
        y = np.random.randint(0, 10, batch_size).astype(np.int64)
        
        # Test single forward pass
        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y.astype(np.float32)), requires_grad=False)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Extract loss value properly
        if hasattr(loss, 'data'):
            if hasattr(loss.data, 'data'):
                loss_val = float(loss.data.data.flat[0])
            else:
                loss_val = float(loss.data.flat[0])
        else:
            loss_val = float(loss.flat[0])
        
        print(f"  ✅ CIFAR-10 CNN created successfully")
        print(f"  ✅ Forward pass works, loss: {loss_val:.4f}")
        print(f"  ✅ Output shape: {outputs.data.shape}")
        print(f"  ✅ Handles 3D image data correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ CIFAR-10 example failed: {e}")
        return False

def main():
    """Run all example validation tests."""
    print("🧪 Quick Example Validation")
    print("=" * 50)
    print("Testing core functionality of all examples without long training...")
    print()
    
    results = []
    
    # Test each example
    tests = [
        ("XOR Network", test_xor_example),
        ("MNIST MLP", test_mnist_example), 
        ("CIFAR-10 CNN", test_cifar10_example_structure)
    ]
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Summary
    print("📊 Example Validation Results")
    print("=" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    print()
    print(f"Summary: {passed}/{total} examples working")
    
    if passed == total:
        print("🎉 All examples are working!")
        print("✅ Ready for training rounds!")
    else:
        print("⚠️  Some examples need fixes before training")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
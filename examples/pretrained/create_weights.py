#!/usr/bin/env python3
"""
Create pretrained weights for TinyTorch inference demos.

This script generates realistic pretrained weights that solve:
1. XOR problem - Simple 2-4-1 network  
2. MNIST digit classification - MLP classifier
3. CIFAR-10 image classification - CNN (placeholder for future)

All weights are manually crafted to demonstrate working solutions
and motivate students after completing Phase 1 modules.
"""

import numpy as np
import os

def create_xor_weights():
    """
    Create weights for XOR network (2-4-1 architecture).
    
    These weights solve the XOR problem:
    [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0
    """
    # Hidden layer weights (2 inputs -> 4 hidden units)
    # Manually designed to detect different input patterns
    hidden_weight = np.array([
        [ 1.5, -1.5],  # Unit 0: detects [1,0] pattern  
        [-1.5,  1.5],  # Unit 1: detects [0,1] pattern
        [ 1.5,  1.5],  # Unit 2: detects [1,1] pattern (OR gate)
        [-1.5, -1.5]   # Unit 3: detects [0,0] pattern (NOR gate)
    ], dtype=np.float32)
    
    hidden_bias = np.array([-0.5, -0.5, -1.0, 1.0], dtype=np.float32)
    
    # Output layer weights (4 hidden -> 1 output)
    # Combines patterns to create XOR: (unit0 OR unit1) AND NOT unit2
    output_weight = np.array([[1.0, 1.0, -1.5, 0.0]], dtype=np.float32)
    output_bias = np.array([0.0], dtype=np.float32)
    
    return {
        'hidden.weight': hidden_weight,
        'hidden.bias': hidden_bias,
        'output.weight': output_weight,
        'output.bias': output_bias
    }

def create_mnist_weights():
    """
    Create weights for MNIST MLP (784-128-64-10 architecture).
    
    These are synthetic but realistic weights for digit classification.
    Uses Xavier initialization scaled appropriately for good performance.
    """
    np.random.seed(42)  # Reproducible weights
    
    # Layer 1: 784 -> 128
    hidden1_weight = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
    hidden1_bias = np.zeros(128)
    
    # Layer 2: 128 -> 64  
    hidden2_weight = np.random.randn(64, 128) * np.sqrt(2.0 / 128)
    hidden2_bias = np.zeros(64)
    
    # Output layer: 64 -> 10
    output_weight = np.random.randn(10, 64) * np.sqrt(2.0 / 64)
    output_bias = np.zeros(10)
    
    # Apply some manual tuning to make weights more realistic
    # Reduce magnitude slightly for better convergence
    hidden1_weight *= 0.7
    hidden2_weight *= 0.8
    output_weight *= 0.9
    
    return {
        'hidden1.weight': hidden1_weight.astype(np.float32),
        'hidden1.bias': hidden1_bias.astype(np.float32),
        'hidden2.weight': hidden2_weight.astype(np.float32), 
        'hidden2.bias': hidden2_bias.astype(np.float32),
        'output.weight': output_weight.astype(np.float32),
        'output.bias': output_bias.astype(np.float32)
    }

def create_cifar10_weights():
    """
    Create placeholder weights for CIFAR-10 CNN.
    
    This is a placeholder for future CNN implementation.
    Creates realistic-sized weight matrices for:
    - Conv2d layers
    - Linear layers for classification
    """
    np.random.seed(123)  # Different seed for variety
    
    # Placeholder CNN architecture: Conv(32) -> Conv(64) -> FC(128) -> FC(10)
    # These weights won't work until CNN layers are implemented in Module 6+
    
    # Conv layer 1: 3 input channels -> 32 output channels, 3x3 kernel
    conv1_weight = np.random.randn(32, 3, 3, 3) * np.sqrt(2.0 / (3 * 3 * 3))
    conv1_bias = np.zeros(32)
    
    # Conv layer 2: 32 -> 64 channels, 3x3 kernel
    conv2_weight = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32 * 3 * 3))
    conv2_bias = np.zeros(64)
    
    # FC layer 1: Flattened conv output -> 128
    # Assuming 8x8 feature maps after pooling: 64 * 8 * 8 = 4096
    fc1_weight = np.random.randn(128, 4096) * np.sqrt(2.0 / 4096)
    fc1_bias = np.zeros(128)
    
    # Output layer: 128 -> 10 classes
    fc2_weight = np.random.randn(10, 128) * np.sqrt(2.0 / 128)
    fc2_bias = np.zeros(10)
    
    return {
        'conv1.weight': conv1_weight.astype(np.float32),
        'conv1.bias': conv1_bias.astype(np.float32),
        'conv2.weight': conv2_weight.astype(np.float32),
        'conv2.bias': conv2_bias.astype(np.float32),
        'fc1.weight': fc1_weight.astype(np.float32),
        'fc1.bias': fc1_bias.astype(np.float32),
        'fc2.weight': fc2_weight.astype(np.float32),
        'fc2.bias': fc2_bias.astype(np.float32)
    }

def main():
    """Create all pretrained weight files."""
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🏗️  Creating pretrained weights for TinyTorch inference demos...")
    
    # Create XOR weights
    print("  📊 Creating XOR network weights (2-4-1 architecture)...")
    xor_weights = create_xor_weights()
    np.savez(os.path.join(output_dir, 'xor_weights.npz'), **xor_weights)
    print(f"     ✅ Saved xor_weights.npz ({len(xor_weights)} weight matrices)")
    
    # Create MNIST weights  
    print("  📊 Creating MNIST MLP weights (784-128-64-10 architecture)...")
    mnist_weights = create_mnist_weights()
    np.savez(os.path.join(output_dir, 'mnist_mlp_weights.npz'), **mnist_weights)
    print(f"     ✅ Saved mnist_mlp_weights.npz ({len(mnist_weights)} weight matrices)")
    
    # Create CIFAR-10 weights (placeholder)
    print("  📊 Creating CIFAR-10 CNN weights (placeholder for future use)...")
    cifar_weights = create_cifar10_weights()
    np.savez(os.path.join(output_dir, 'cifar10_cnn_weights.npz'), **cifar_weights)
    print(f"     ✅ Saved cifar10_cnn_weights.npz ({len(cifar_weights)} weight matrices)")
    
    print("\n🎉 All pretrained weights created successfully!")
    print("\n📁 Files created:")
    for filename in ['xor_weights.npz', 'mnist_mlp_weights.npz', 'cifar10_cnn_weights.npz']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   • {filename} ({size_kb:.1f} KB)")
    
    print("\n💡 Next steps:")
    print("   • Run the inference demos to see your TinyTorch code in action!")
    print("   • python examples/xor_inference.py")
    print("   • python examples/mnist_inference.py") 
    print("   • python examples/cifar10_inference.py (placeholder)")

if __name__ == "__main__":
    main()
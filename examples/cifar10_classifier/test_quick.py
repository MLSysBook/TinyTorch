#!/usr/bin/env python3
"""
Quick CIFAR-10 MLP Test - Minimal example to prove the pipeline works
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

def test_cifar10_pipeline():
    """Test minimal CIFAR-10 → MLP pipeline without training."""
    print("🧪 Testing CIFAR-10 MLP Pipeline")
    print("=" * 40)
    
    # Load small subset of CIFAR-10
    dataset = CIFAR10Dataset(root="./data", train=False, download=False)  # Test set
    loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Fixed batch size
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    print(f"✅ Sample shape: {dataset[0][0].shape}")
    
    # Build simple MLP
    model_layers = [
        Dense(3072, 256),  # 32*32*3 → 256
        ReLU(),
        Dense(256, 10),    # 256 → 10 classes
        Softmax()
    ]
    
    print(f"✅ Model created: 3072 → 256 → 10")
    
    # Test forward pass with one batch
    for images, labels in loader:
        print(f"✅ Batch loaded: {images.shape}")
        
        # Flatten images
        batch_size = images.shape[0]
        flattened = images.data.reshape(batch_size, -1)
        x = Tensor(flattened)
        print(f"✅ Images flattened: {x.shape}")
        
        # Forward pass through model
        for i, layer in enumerate(model_layers):
            x = layer(x)
            print(f"✅ Layer {i+1} output: {x.shape}")
        
        # Check predictions
        predictions = x.data
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = labels.data
        
        accuracy = np.mean(pred_classes == true_classes)
        print(f"✅ Random accuracy: {accuracy:.1%} (expected ~10%)")
        
        break  # Just test one batch
    
    print("\n🎉 CIFAR-10 → MLP pipeline works!")
    print("Ready for full training implementation.")
    return True

if __name__ == "__main__":
    test_cifar10_pipeline()
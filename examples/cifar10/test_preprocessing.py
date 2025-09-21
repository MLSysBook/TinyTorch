#!/usr/bin/env python3
"""
Test the preprocessing function specifically
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

def preprocess_images(images, training=True):
    """Copy of the preprocessing function from train_cifar10_mlp.py"""
    print(f"    Preprocessing batch of size {images.shape[0]}, training={training}")
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    print(f"    Extracted numpy array: {images_np.shape}")
    
    if training:
        print("    Applying data augmentation...")
        # Data augmentation - prevents overfitting
        augmented = np.copy(images_np)
        print(f"    Copied data for augmentation: {augmented.shape}")
        
        for i in range(batch_size):
            print(f"      Processing image {i+1}/{batch_size}")
            # Random horizontal flip (50% chance)
            if np.random.random() > 0.5:
                augmented[i] = np.flip(augmented[i], axis=2)
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            augmented[i] = np.clip(augmented[i] * brightness, 0, 1)
            
            # Small random translations
            if np.random.random() > 0.5:
                shift_x = np.random.randint(-2, 3)
                shift_y = np.random.randint(-2, 3)
                augmented[i] = np.roll(augmented[i], shift_x, axis=2)
                augmented[i] = np.roll(augmented[i], shift_y, axis=1)
        
        images_np = augmented
        print("    ✅ Data augmentation complete")
    
    print("    Flattening and normalizing...")
    # Flatten to (batch_size, 3072)
    flat = images_np.reshape(batch_size, -1)
    
    # Optimized normalization: scale to [-2, 2] range
    normalized = (flat - 0.5) / 0.25
    
    result = Tensor(normalized.astype(np.float32))
    print(f"    ✅ Preprocessing complete: {result.shape}")
    return result

def test_preprocessing():
    """Test preprocessing function with different batch sizes"""
    print("🔧 Testing preprocessing function...")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    # Get first batch
    print("Getting first batch...")
    images, labels = next(iter(train_loader))
    print(f"Batch: images {images.shape}, labels {labels.shape}")
    
    # Test preprocessing without augmentation
    print("\n1. Testing preprocessing without augmentation...")
    start_time = time.time()
    result1 = preprocess_images(images, training=False)
    time1 = time.time() - start_time
    print(f"✅ No augmentation: {time1:.4f}s, output shape {result1.shape}")
    
    # Test preprocessing with augmentation
    print("\n2. Testing preprocessing with augmentation...")
    start_time = time.time()
    result2 = preprocess_images(images, training=True)
    time2 = time.time() - start_time
    print(f"✅ With augmentation: {time2:.4f}s, output shape {result2.shape}")
    
    # Test with larger batch
    print("\n3. Testing with larger batch (32)...")
    train_loader_large = DataLoader(train_dataset, batch_size=32, shuffle=False)
    images_large, labels_large = next(iter(train_loader_large))
    print(f"Large batch: images {images_large.shape}, labels {labels_large.shape}")
    
    start_time = time.time()
    result3 = preprocess_images(images_large, training=True)
    time3 = time.time() - start_time
    print(f"✅ Large batch with augmentation: {time3:.4f}s, output shape {result3.shape}")
    
    # Check if timing scales linearly
    if time3 > time2 * 10:  # Should be roughly 8x slower (32/4), but allowing 10x
        print(f"⚠️  Preprocessing may be inefficient: {time2:.4f}s -> {time3:.4f}s")
    else:
        print("✅ Preprocessing timing looks reasonable")

def main():
    print("🧪 Preprocessing Function Test")
    print("=" * 50)
    
    try:
        test_preprocessing()
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
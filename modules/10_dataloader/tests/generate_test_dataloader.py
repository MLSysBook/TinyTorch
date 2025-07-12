#!/usr/bin/env python3
"""
Generate small test data for data module testing.

This creates a small mock dataset that mimics CIFAR-10 structure but is tiny
and doesn't require downloading anything.
"""

import numpy as np
import pickle
import os
from pathlib import Path

def generate_test_cifar10_data():
    """Generate small test data that mimics CIFAR-10 structure."""
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create small test dataset
    train_size = 50  # Small training set
    test_size = 20   # Small test set
    
    # Generate random image data (3x32x32, values 0-255)
    train_data = np.random.randint(0, 256, size=(train_size, 3, 32, 32), dtype=np.uint8)
    train_labels = np.random.randint(0, 10, size=(train_size,), dtype=np.uint8)
    
    test_data = np.random.randint(0, 256, size=(test_size, 3, 32, 32), dtype=np.uint8)
    test_labels = np.random.randint(0, 10, size=(test_size,), dtype=np.uint8)
    
    # Create the data directory
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    
    # Save training data (mimics CIFAR-10 format)
    train_dict = {
        b'data': train_data.reshape(train_size, -1),  # Flatten to (N, 3072)
        b'labels': train_labels.tolist(),
        b'batch_label': b'training batch 1 of 1',
        b'filenames': [f'train_image_{i}.png'.encode() for i in range(train_size)]
    }
    
    with open(data_dir / "data_batch_1", "wb") as f:
        pickle.dump(train_dict, f)
    
    # Save test data
    test_dict = {
        b'data': test_data.reshape(test_size, -1),  # Flatten to (N, 3072)
        b'labels': test_labels.tolist(),
        b'batch_label': b'testing batch 1 of 1',
        b'filenames': [f'test_image_{i}.png'.encode() for i in range(test_size)]
    }
    
    with open(data_dir / "test_batch", "wb") as f:
        pickle.dump(test_dict, f)
    
    # Save metadata
    meta_dict = {
        b'label_names': [name.encode() for name in class_names],
        b'num_cases_per_batch': [train_size],
        b'num_vis': 3072  # 32*32*3
    }
    
    with open(data_dir / "batches.meta", "wb") as f:
        pickle.dump(meta_dict, f)
    
    print(f"✅ Generated test data:")
    print(f"   - Training samples: {train_size}")
    print(f"   - Test samples: {test_size}")
    print(f"   - Image shape: (3, 32, 32)")
    print(f"   - Classes: {len(class_names)}")
    print(f"   - Saved to: {data_dir}")
    
    return data_dir

if __name__ == "__main__":
    generate_test_cifar10_data() 
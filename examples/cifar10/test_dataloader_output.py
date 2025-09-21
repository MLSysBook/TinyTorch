#!/usr/bin/env python3
"""
Test what the DataLoader actually returns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

def main():
    print("🔍 DataLoader Output Investigation")
    print("=" * 50)
    
    # Load dataset
    train_dataset = CIFAR10Dataset(train=True, root='data')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    # Get first batch
    images, labels = next(iter(train_loader))
    
    print(f"Images type: {type(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Images has reshape: {hasattr(images, 'reshape')}")
    print(f"Images has data: {hasattr(images, 'data')}")
    print(f"Images has _data: {hasattr(images, '_data')}")
    
    if hasattr(images, 'data'):
        print(f"Images.data type: {type(images.data)}")
        print(f"Images.data shape: {images.data.shape}")
        print(f"Images.data has reshape: {hasattr(images.data, 'reshape')}")
    
    if hasattr(images, '_data'):
        print(f"Images._data type: {type(images._data)}")
        print(f"Images._data shape: {images._data.shape}")
        print(f"Images._data has reshape: {hasattr(images._data, 'reshape')}")
    
    print(f"\nLabels type: {type(labels)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels has data: {hasattr(labels, 'data')}")
    print(f"Labels has _data: {hasattr(labels, '_data')}")
    
    if hasattr(labels, 'data'):
        print(f"Labels.data type: {type(labels.data)}")
    
    if hasattr(labels, '_data'):
        print(f"Labels._data type: {type(labels._data)}")

if __name__ == "__main__":
    main()
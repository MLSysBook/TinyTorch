#!/usr/bin/env python3
"""
Test CIFAR-10 components individually to isolate issues
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

def test_basic_components():
    """Test basic components work"""
    print("🔧 Testing basic components...")
    
    # Test Tensor creation
    print("1. Testing Tensor creation...")
    x = Tensor([[1, 2], [3, 4]])
    print(f"✅ Tensor created: {x.shape}")
    
    # Test Variable creation
    print("2. Testing Variable creation...")
    v = Variable(x, requires_grad=True)
    print(f"✅ Variable created: requires_grad={v.requires_grad}")
    
    # Test Dense layer
    print("3. Testing Dense layer...")
    fc = Dense(2, 3)
    print(f"✅ Dense layer created: {fc.weights.shape}")
    
    # Test ReLU
    print("4. Testing ReLU...")
    relu = ReLU()
    out = relu(v)
    print(f"✅ ReLU works: output shape {out.data.shape}")
    
    print("✅ All basic components work!\n")

def test_loss_function():
    """Test loss function works"""
    print("🔧 Testing loss function...")
    
    loss_fn = CrossEntropyLoss()
    
    # Create test data
    pred = Variable(Tensor([[1.0, 2.0, 0.5]]), requires_grad=True)
    true = Variable(Tensor([[1]]), requires_grad=False)  # Class 1
    
    print("Computing loss...")
    loss = loss_fn(pred, true)
    
    # Extract loss value properly
    if hasattr(loss.data, 'data'):
        loss_val = float(loss.data.data)
    elif hasattr(loss.data, '_data'):
        loss_val = float(loss.data._data)
    else:
        loss_val = float(loss.data)
    
    print(f"✅ Loss computed: {loss_val:.4f}")
    print("✅ Loss function works!\n")

def test_dataset_creation():
    """Test dataset creation (without loading data)"""
    print("🔧 Testing dataset creation...")
    
    try:
        print("Creating train dataset...")
        start_time = time.time()
        train_dataset = CIFAR10Dataset(train=True, root='data')
        creation_time = time.time() - start_time
        print(f"✅ Train dataset created in {creation_time:.2f}s")
        print(f"   Size: {len(train_dataset)} samples")
        
        print("Creating test dataset...")
        start_time = time.time()
        test_dataset = CIFAR10Dataset(train=False, root='data')
        creation_time = time.time() - start_time
        print(f"✅ Test dataset created in {creation_time:.2f}s")
        print(f"   Size: {len(test_dataset)} samples")
        
        print("✅ Dataset creation works!\n")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return None, None

def test_dataloader_first_batch(train_dataset):
    """Test loading first batch from dataloader"""
    print("🔧 Testing DataLoader first batch...")
    
    if train_dataset is None:
        print("❌ Skipping - no dataset available")
        return
    
    try:
        print("Creating DataLoader...")
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        
        print("Getting first batch...")
        start_time = time.time()
        
        # Get first batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_time = time.time() - start_time
            print(f"✅ First batch loaded in {batch_time:.2f}s")
            print(f"   Images shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels: {labels.data[:4] if hasattr(labels, 'data') else labels[:4]}")
            break
        
        print("✅ DataLoader first batch works!\n")
        
    except Exception as e:
        print(f"❌ DataLoader failed: {e}\n")

def test_simple_forward_pass():
    """Test simple forward pass with dummy data"""
    print("🔧 Testing simple forward pass...")
    
    try:
        # Create simple model
        fc1 = Dense(10, 5)
        fc2 = Dense(5, 3)
        relu = ReLU()
        
        # Initialize properly as Variables
        fc1.weights = Variable(fc1.weights.data, requires_grad=True)
        fc1.bias = Variable(fc1.bias.data, requires_grad=True)
        fc2.weights = Variable(fc2.weights.data, requires_grad=True)
        fc2.bias = Variable(fc2.bias.data, requires_grad=True)
        
        # Create dummy input
        x = Variable(Tensor(np.random.randn(2, 10)), requires_grad=False)
        
        print("Forward pass...")
        start_time = time.time()
        
        h1 = fc1(x)
        h1_act = relu(h1)
        logits = fc2(h1_act)
        
        forward_time = time.time() - start_time
        print(f"✅ Forward pass completed in {forward_time:.4f}s")
        print(f"   Output shape: {logits.data.shape}")
        
        # Test loss
        loss_fn = CrossEntropyLoss()
        targets = Variable(Tensor([[1], [2]]), requires_grad=False)
        loss = loss_fn(logits, targets)
        
        if hasattr(loss.data, 'data'):
            loss_val = loss.data.data
        elif hasattr(loss.data, '_data'):
            loss_val = loss.data._data
        else:
            loss_val = loss.data
            
        print(f"✅ Loss computed: {loss_val}")
        print("✅ Simple forward pass works!\n")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}\n")

def main():
    print("🧪 CIFAR-10 Component Testing")
    print("=" * 50)
    
    test_basic_components()
    test_loss_function()
    
    train_dataset, test_dataset = test_dataset_creation()
    test_dataloader_first_batch(train_dataset)
    
    test_simple_forward_pass()
    
    print("🎯 Component testing complete!")
    print("If all tests pass, the issue is likely in the training loop logic.")

if __name__ == "__main__":
    main()
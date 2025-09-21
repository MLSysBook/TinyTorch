#!/usr/bin/env python3
"""
Test just the training loop with minimal data to isolate the hang
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

def preprocess_images_simple(images):
    """Simplified preprocessing without augmentation"""
    batch_size = images.shape[0]
    flat = images.reshape(batch_size, -1)
    normalized = (flat - 0.5) / 0.25
    return Tensor(normalized.astype(np.float32))

def create_simple_model():
    """Create and initialize a simple model"""
    fc1 = Dense(3072, 64)   # Much smaller than original
    fc2 = Dense(64, 10)
    
    # Initialize with reasonable values
    for layer in [fc1, fc2]:
        fan_in = layer.weights.shape[0]
        std = np.sqrt(2.0 / fan_in) * 0.5
        layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
        layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
        
        layer.weights = Variable(layer.weights, requires_grad=True)
        layer.bias = Variable(layer.bias, requires_grad=True)
    
    return fc1, fc2

def test_single_batch_training():
    """Test training on just one batch to isolate the issue"""
    print("🔧 Testing single batch training...")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    # Create model
    print("Creating model...")
    fc1, fc2 = create_simple_model()
    relu = ReLU()
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam([fc1.weights, fc1.bias, fc2.weights, fc2.bias], learning_rate=0.001)
    
    print("Getting first batch...")
    images, labels = next(iter(train_loader))
    print(f"Batch loaded: images {images.shape}, labels {labels.shape}")
    
    print("Starting training step...")
    step_start = time.time()
    
    # Preprocessing
    print("  Preprocessing...")
    preprocess_start = time.time()
    x = Variable(preprocess_images_simple(images), requires_grad=False)
    y_true = Variable(labels, requires_grad=False)
    preprocess_time = time.time() - preprocess_start
    print(f"  ✅ Preprocessing: {preprocess_time:.4f}s")
    
    # Forward pass
    print("  Forward pass...")
    forward_start = time.time()
    h1 = fc1(x)
    h1_act = relu(h1)
    logits = fc2(h1_act)
    forward_time = time.time() - forward_start
    print(f"  ✅ Forward pass: {forward_time:.4f}s")
    print(f"     Logits shape: {logits.data.shape}")
    
    # Loss computation
    print("  Computing loss...")
    loss_start = time.time()
    loss = loss_fn(logits, y_true)
    loss_time = time.time() - loss_start
    
    # Extract loss value
    if hasattr(loss.data, 'data'):
        loss_val = float(loss.data.data)
    elif hasattr(loss.data, '_data'):
        loss_val = float(loss.data._data)
    else:
        loss_val = float(loss.data)
    
    print(f"  ✅ Loss computation: {loss_time:.4f}s, Loss: {loss_val:.4f}")
    
    # Backward pass
    print("  Backward pass...")
    backward_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    backward_time = time.time() - backward_start
    print(f"  ✅ Backward pass: {backward_time:.4f}s")
    
    # Optimizer step  
    print("  Optimizer step...")
    step_start_time = time.time()
    optimizer.step()
    step_time = time.time() - step_start_time
    print(f"  ✅ Optimizer step: {step_time:.4f}s")
    
    total_time = time.time() - step_start
    print(f"✅ Single batch training: {total_time:.4f}s total")
    
    return True

def test_multiple_batches():
    """Test multiple batches to see if there's a memory leak or accumulation issue"""
    print("\n🔧 Testing multiple batch training...")
    
    # Load dataset
    train_dataset = CIFAR10Dataset(train=True, root='data')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    # Create model
    fc1, fc2 = create_simple_model()
    relu = ReLU()
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam([fc1.weights, fc1.bias, fc2.weights, fc2.bias], learning_rate=0.001)
    
    print("Training on 5 batches...")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 5:  # Only 5 batches
            break
            
        print(f"  Batch {batch_idx + 1}/5...")
        batch_start = time.time()
        
        # Simple training step
        x = Variable(preprocess_images_simple(images), requires_grad=False)
        y_true = Variable(labels, requires_grad=False)
        
        # Forward
        h1 = fc1(x)
        h1_act = relu(h1)
        logits = fc2(h1_act)
        
        # Loss
        loss = loss_fn(logits, y_true)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        
        # Extract loss
        if hasattr(loss.data, 'data'):
            loss_val = float(loss.data.data)
        elif hasattr(loss.data, '_data'):
            loss_val = float(loss.data._data)
        else:
            loss_val = float(loss.data)
            
        print(f"    ✅ Batch {batch_idx + 1}: {batch_time:.4f}s, Loss: {loss_val:.4f}")
        
        # Check if it's getting slower (memory leak indicator)
        if batch_time > 1.0:  # If any batch takes over 1 second, something's wrong
            print(f"    ⚠️  Batch taking too long: {batch_time:.4f}s")
            break
    
    print("✅ Multiple batch training completed")

def main():
    print("🧪 Training Loop Diagnostic")
    print("=" * 50)
    
    try:
        success = test_single_batch_training()
        if success:
            test_multiple_batches()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
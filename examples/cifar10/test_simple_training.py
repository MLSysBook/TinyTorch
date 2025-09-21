#!/usr/bin/env python3
"""
Test simple CIFAR-10 training with just a few batches to see what works
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

def preprocess_images(images, training=True):
    """Simplified preprocessing to avoid potential issues"""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    # Skip augmentation for now to test core training
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.5) / 0.25
    return Tensor(normalized.astype(np.float32))

class SimpleCIFAR10_MLP:
    """Much simpler model for testing"""
    
    def __init__(self):
        print("🏗️ Building Simple MLP for CIFAR-10...")
        
        # Simple architecture
        self.fc1 = Dense(3072, 128)  # Much smaller
        self.fc2 = Dense(128, 10)
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2]
        
        # Initialize weights
        self._initialize_weights()
        
        total_params = sum(np.prod(layer.weights.shape) + np.prod(layer.bias.shape) 
                          for layer in self.layers)
        print(f"✅ Model: 3072 → 128 → 10")
        print(f"   Parameters: {total_params:,}")
    
    def _initialize_weights(self):
        """Simple He initialization"""
        for i, layer in enumerate(self.layers):
            fan_in = layer.weights.shape[0]
            std = np.sqrt(2.0 / fan_in) * 0.5
            
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            
            # Make trainable
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through the network."""
        h1 = self.relu(self.fc1(x))
        logits = self.fc2(h1)
        return logits
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
        return params

def test_simple_cifar10_training():
    """Test the simplest possible CIFAR-10 training"""
    print("🚀 Simple CIFAR-10 Training Test")
    print("=" * 50)
    
    # Load data - just small batch
    print("📚 Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)  # Very small batch
    
    print(f"✅ Loaded {len(train_dataset):,} train samples")
    
    # Create simple model
    print("\n🏗️ Creating simple model...")
    model = SimpleCIFAR10_MLP()
    
    # Setup training
    print("\n⚙️ Setting up training...")
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    
    print("✅ Training setup complete")
    
    # Test training on just a few batches
    print("\n📊 Training on 3 batches...")
    
    total_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 3:  # Only 3 batches
            break
        
        print(f"\n  🔄 Batch {batch_idx + 1}/3")
        batch_start = time.time()
        
        # Preprocess
        print("    Preprocessing...")
        preprocess_start = time.time()
        x = Variable(preprocess_images(images, training=False), requires_grad=False)  # No augmentation
        y_true = Variable(labels, requires_grad=False)
        preprocess_time = time.time() - preprocess_start
        print(f"    ✅ Preprocess: {preprocess_time:.4f}s")
        
        # Forward pass
        print("    Forward pass...")
        forward_start = time.time()
        logits = model.forward(x)
        forward_time = time.time() - forward_start
        print(f"    ✅ Forward: {forward_time:.4f}s")
        
        # Loss
        print("    Computing loss...")
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
        
        print(f"    ✅ Loss: {loss_time:.4f}s, Value: {loss_val:.4f}")
        
        # Backward
        print("    Backward pass...")
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - backward_start
        print(f"    ✅ Backward: {backward_time:.4f}s")
        
        # Update
        print("    Parameter update...")
        update_start = time.time()
        optimizer.step()
        update_time = time.time() - update_start
        print(f"    ✅ Update: {update_time:.4f}s")
        
        batch_time = time.time() - batch_start
        print(f"  ✅ Batch {batch_idx + 1} total: {batch_time:.4f}s")
        
        # If any step takes too long, report it
        if batch_time > 5.0:
            print(f"    ⚠️  Batch taking very long: {batch_time:.4f}s")
        
        # Calculate accuracy for this batch
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        preds = np.argmax(logits_np, axis=1)
        labels_np = y_true.data._data if hasattr(y_true.data, '_data') else y_true.data
        accuracy = np.mean(preds == labels_np)
        print(f"    📊 Batch accuracy: {accuracy:.1%}")
    
    total_time = time.time() - total_start
    print(f"\n✅ 3 batches completed in {total_time:.4f}s")
    print(f"   Average per batch: {total_time/3:.4f}s")
    
    if total_time < 10.0:
        print("🎉 Training speed looks good!")
        return True
    else:
        print("⚠️  Training seems slow")
        return False

def main():
    try:
        success = test_simple_cifar10_training()
        if success:
            print("\n💡 Core training works! The issue might be:")
            print("   - Too many batches per epoch (500)")
            print("   - Large batch size (64)")
            print("   - Complex data augmentation")
            print("   - Memory accumulation over many batches")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
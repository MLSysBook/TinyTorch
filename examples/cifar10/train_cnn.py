#!/usr/bin/env python3
"""
CIFAR-10 CNN Training - Using MultiChannelConv2D

Demonstrates the power of convolutions for image classification.
Uses TinyTorch's multi-channel Conv2D implementation.
Should achieve better accuracy than MLP version (~60% vs 55%).
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
from tinytorch.core.spatial import MultiChannelConv2D, MaxPool2D, flatten
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

class SimpleCNN:
    """CNN for CIFAR-10 using multi-channel Conv2D layers.
    
    Architecture:
    - Conv(3→32) → ReLU → Pool(2x2) → 32@15x15
    - Conv(32→64) → ReLU → Pool(2x2) → 64@6x6  
    - Flatten → Dense(2304→128) → ReLU
    - Dense(128→10) → Softmax (via CrossEntropyLoss)
    """
    
    def __init__(self):
        # Convolutional layers using MultiChannelConv2D
        # Note: No padding support yet, so output sizes will be smaller
        self.conv1 = MultiChannelConv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = MultiChannelConv2D(in_channels=32, out_channels=64, kernel_size=(3, 3))
        
        # Pooling layers
        self.pool = MaxPool2D(pool_size=(2, 2))
        
        # Calculate size after convolutions and pooling
        # Input: 3@32x32
        # After conv1 (3x3): 32@30x30
        # After pool1 (2x2): 32@15x15
        # After conv2 (3x3): 64@13x13
        # After pool2 (2x2): 64@6x6
        # Flattened: 64 * 6 * 6 = 2304
        
        # Fully connected layers
        self.fc1 = Dense(64 * 6 * 6, 128)
        self.fc2 = Dense(128, 10)
        
        self.relu = ReLU()
        
        # Collect all layers with parameters
        self.conv_layers = [self.conv1, self.conv2]
        self.fc_layers = [self.fc1, self.fc2]
        
        # Initialize weights (already done in MultiChannelConv2D with He init)
        self._initialize_fc_weights()
    
    def _initialize_fc_weights(self):
        """Initialize fully connected layer weights."""
        for i, layer in enumerate(self.fc_layers):
            fan_in = layer.weights.shape[0]
            # Use smaller std for output layer
            std = 0.01 if i == len(self.fc_layers) - 1 else np.sqrt(2.0 / fan_in)
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch, 3, 32, 32) or flattened
            
        Returns:
            Logits of shape (batch, 10)
        """
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        
        # Reshape from flattened to image format if needed
        if len(x.shape) == 2 and x.shape[1] == 3072:
            # Reshape from (batch, 3072) to (batch, 3, 32, 32)
            x_data = x.data if hasattr(x, 'data') else x._data
            x_reshaped = x_data.reshape(batch_size, 3, 32, 32)
            x = Tensor(x_reshaped) if not isinstance(x, Variable) else Variable(x_reshaped, x.requires_grad)
        elif len(x.shape) == 2:
            # Single flattened image
            x_data = x.data if hasattr(x, 'data') else x._data
            x_reshaped = x_data.reshape(3, 32, 32)
            x = Tensor(x_reshaped) if not isinstance(x, Variable) else Variable(x_reshaped, x.requires_grad)
        
        # Conv block 1: 3@32x32 → 32@30x30 → 32@15x15
        h = self.conv1(x)
        h = self.relu(h)
        h = self.pool(h)
        
        # Conv block 2: 32@15x15 → 64@13x13 → 64@6x6
        h = self.conv2(h)
        h = self.relu(h)
        h = self.pool(h)
        
        # Flatten for FC layers: 64@6x6 → 2304
        h = flatten(h)
        
        # FC layers: 2304 → 128 → 10
        h = self.relu(self.fc1(h))
        return self.fc2(h)
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        # Conv layer parameters
        for conv in self.conv_layers:
            params.append(conv.weights)
            if conv.bias is not None:
                params.append(conv.bias)
        # FC layer parameters
        for fc in self.fc_layers:
            params.extend([fc.weights, fc.bias])
        return params
    
    def count_parameters(self):
        """Count total number of parameters."""
        total = 0
        for p in self.parameters():
            if hasattr(p, 'data'):
                data = p.data if not hasattr(p.data, '_data') else p.data._data
                total += np.prod(data.shape)
        return total

def preprocess(images, training=True):
    """Preprocess CIFAR-10 images.
    
    Args:
        images: Raw image tensor
        training: Whether to apply data augmentation
        
    Returns:
        Preprocessed tensor ready for CNN
    """
    images_np = images.data if hasattr(images, 'data') else images._data
    batch_size = images_np.shape[0]
    
    # Data augmentation for training (horizontal flip)
    if training:
        augmented = np.copy(images_np)
        for i in range(batch_size):
            if np.random.random() > 0.5:
                # Flip the spatial dimensions (last axis for flattened, axis 2 for image format)
                if len(augmented.shape) == 2:
                    # Flattened format: reshape, flip, flatten
                    img = augmented[i].reshape(3, 32, 32)
                    img = np.flip(img, axis=2)
                    augmented[i] = img.flatten()
                else:
                    augmented[i] = np.flip(augmented[i], axis=2)
        images_np = augmented
    
    # Normalize (using CIFAR-10 statistics)
    normalized = (images_np - 0.485) / 0.229
    
    # Ensure correct shape for CNN
    if len(normalized.shape) == 2:
        # From flat (batch, 3072) to image format (batch, 3, 32, 32)
        normalized = normalized.reshape(batch_size, 3, 32, 32)
    
    return Tensor(normalized.astype(np.float32))

def evaluate(model, dataloader, max_batches=30):
    """Evaluate model accuracy.
    
    Args:
        model: CNN model
        dataloader: Data loader
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Accuracy as float between 0 and 1
    """
    correct = total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        # Preprocess and create Variable
        x = Variable(preprocess(images, training=False), requires_grad=False)
        
        # Forward pass
        logits = model.forward(x)
        
        # Get predictions
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        # Count correct predictions
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
    
    return correct / total if total > 0 else 0

def main():
    print("="*60)
    print("CIFAR-10 CNN Training - MultiChannelConv2D")
    print("="*60)
    print("\n🧠 Using TinyTorch's multi-channel convolutions!")
    print("Architecture: Conv(3→32) → Pool → Conv(32→64) → Pool → Dense")
    
    # Load data
    print("\n📚 Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    # Smaller batch size for memory efficiency with convolutions
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Create model
    print("\n🔧 Initializing CNN model...")
    model = SimpleCNN()
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"  - Conv layers: {32*3*3*3 + 32 + 64*32*3*3 + 64:,} parameters")
    print(f"  - FC layers: {64*6*6*128 + 128 + 128*10 + 10:,} parameters")
    
    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training settings (reduced for demo)
    epochs = 5  # Reduced for faster demo
    eval_every = 50
    max_batches = 200  # Limit batches per epoch for demo
    
    print(f"\n🚀 Training for {epochs} epochs (limited to {max_batches} batches/epoch)...")
    print("-" * 40)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0
        batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            
            # Forward pass
            x = Variable(preprocess(images, training=True), requires_grad=True)
            y = Variable(labels, requires_grad=False)
            
            logits = model.forward(x)
            loss = loss_fn(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data
            batches += 1
            
            # Periodic evaluation
            if (batch_idx + 1) % eval_every == 0:
                train_acc = evaluate(model, train_loader, max_batches=5)
                test_acc = evaluate(model, test_loader, max_batches=10)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"Loss={running_loss/batches:.3f}, "
                      f"Train={train_acc:.1%}, Test={test_acc:.1%}")
                
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
        
        # End of epoch
        epoch_time = time.time() - start_time
        test_accuracy = evaluate(model, test_loader, max_batches=50)
        print(f"\n✓ Epoch {epoch+1} complete in {epoch_time:.1f}s")
        print(f"  Test Accuracy: {test_accuracy:.1%}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
    # Final evaluation
    print("\n" + "="*60)
    print("📊 Final Evaluation")
    print("-" * 40)
    
    final_accuracy = evaluate(model, test_loader, max_batches=100)
    print(f"Final Test Accuracy: {final_accuracy:.1%}")
    print(f"Best Accuracy Achieved: {best_accuracy:.1%}")
    
    # Performance comparison
    print("\n🎯 Performance Comparison:")
    print(f"  Random Baseline:  ~10%")
    print(f"  MLP (no conv):    ~55%")
    print(f"  CNN (with Conv2D): {final_accuracy:.1%} {'✅' if final_accuracy > 0.55 else ''}")
    
    if final_accuracy > 0.55:
        print("\n🎉 Success! CNN outperforms MLP!")
        print("   Convolutions extract spatial features effectively!")
    
    print("\n💡 Why CNNs work better for images:")
    print("  - Conv2D learns spatial feature detectors")
    print("  - Parameter sharing (same filter across image)")
    print("  - Translation invariance from pooling")
    print("  - Hierarchical feature learning (edges → shapes → objects)")
    print("\n📈 Systems Insight:")
    print(f"  - Conv parameters: {32*3*3*3 + 64*32*3*3:,} (~{(32*3*3*3 + 64*32*3*3)*4/1024:.1f} KB)")
    print(f"  - MLP equivalent: {3072*1024:,} (~{3072*1024*4/1024/1024:.1f} MB)")
    print("  - Parameter reduction: {(1 - (32*3*3*3 + 64*32*3*3)/(3072*1024)):.1%}")

if __name__ == "__main__":
    main()
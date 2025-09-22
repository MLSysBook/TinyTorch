#!/usr/bin/env python3
"""
CIFAR-10 CNN Training - Using Conv2D

Demonstrates the power of convolutions for image classification.
Should achieve better accuracy than MLP version.
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
from tinytorch.core.spatial import Conv2D, MaxPool2D
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

class SimpleCNN:
    """CNN for CIFAR-10 using Conv2D layers."""
    
    def __init__(self):
        # Convolutional layers
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        
        # Calculate size after convolutions and pooling
        # 32x32 -> pool -> 16x16 -> pool -> 8x8 -> pool -> 4x4
        # 128 channels * 4 * 4 = 2048
        
        # Fully connected layers
        self.fc1 = Dense(128 * 4 * 4, 256)
        self.fc2 = Dense(256, 10)
        
        self.relu = ReLU()
        
        # Collect all layers with parameters
        self.conv_layers = [self.conv1, self.conv2, self.conv3]
        self.fc_layers = [self.fc1, self.fc2]
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        # Conv layers - He initialization
        for conv in self.conv_layers:
            fan_in = conv.weight.shape[1] * conv.weight.shape[2] * conv.weight.shape[3]
            std = np.sqrt(2.0 / fan_in)
            conv.weight._data = np.random.randn(*conv.weight.shape).astype(np.float32) * std
            if conv.bias is not None:
                conv.bias._data = np.zeros(conv.bias.shape, dtype=np.float32)
            conv.weight = Variable(conv.weight.data, requires_grad=True)
            if conv.bias is not None:
                conv.bias = Variable(conv.bias.data, requires_grad=True)
        
        # FC layers
        for i, layer in enumerate(self.fc_layers):
            fan_in = layer.weights.shape[0]
            std = 0.01 if i == len(self.fc_layers) - 1 else np.sqrt(2.0 / fan_in)
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through CNN."""
        # Reshape from (batch, 3072) to (batch, 3, 32, 32) if needed
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.reshape(batch_size, 3, 32, 32)
        
        # Conv block 1
        h = self.relu(self.conv1(x))
        h = self.pool(h)  # 32x32 -> 16x16
        
        # Conv block 2  
        h = self.relu(self.conv2(h))
        h = self.pool(h)  # 16x16 -> 8x8
        
        # Conv block 3
        h = self.relu(self.conv3(h))
        h = self.pool(h)  # 8x8 -> 4x4
        
        # Flatten for FC layers
        h = h.reshape(batch_size, -1)
        
        # FC layers
        h = self.relu(self.fc1(h))
        return self.fc2(h)
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        for conv in self.conv_layers:
            params.append(conv.weight)
            if conv.bias is not None:
                params.append(conv.bias)
        for fc in self.fc_layers:
            params.extend([fc.weights, fc.bias])
        return params

def preprocess(images, training=True):
    """Preprocess CIFAR-10 images."""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    # Data augmentation for training
    if training:
        augmented = np.copy(images_np)
        for i in range(batch_size):
            if np.random.random() > 0.5:
                # Horizontal flip
                augmented[i] = np.flip(augmented[i], axis=2)
        images_np = augmented
    
    # Normalize
    normalized = (images_np - 0.485) / 0.229
    
    # Ensure correct shape for CNN: (batch, 3, 32, 32)
    if len(normalized.shape) == 2:
        # From flat to image format
        batch_size = normalized.shape[0]
        normalized = normalized.reshape(batch_size, 3, 32, 32)
    
    return Tensor(normalized.astype(np.float32))

def evaluate(model, dataloader, max_batches=30):
    """Evaluate model accuracy."""
    correct = total = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        x = Variable(preprocess(images, training=False), requires_grad=False)
        logits = model.forward(x)
        
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
    
    return correct / total if total > 0 else 0

def main():
    print("="*60)
    print("CIFAR-10 CNN Training - Convolutional Neural Network")
    print("="*60)
    print("\nUsing Conv2D layers for spatial feature extraction!")
    print("Architecture: Conv2D -> Pool -> Conv2D -> Pool -> Conv2D -> Pool -> FC")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nInitializing CNN model...")
    model = SimpleCNN()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training settings
    epochs = 10
    eval_every = 50
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0
        batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit batches for quick demo
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
            
            # Evaluate periodically
            if (batch_idx + 1) % eval_every == 0:
                train_acc = evaluate(model, train_loader, max_batches=10)
                test_acc = evaluate(model, test_loader, max_batches=20)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"Loss={running_loss/batches:.3f}, "
                      f"Train={train_acc:.1%}, Test={test_acc:.1%}")
        
        # End of epoch evaluation
        epoch_time = time.time() - start_time
        test_accuracy = evaluate(model, test_loader, max_batches=50)
        print(f"\nEpoch {epoch+1} complete in {epoch_time:.1f}s - Test Accuracy: {test_accuracy:.1%}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("-" * 40)
    
    final_accuracy = evaluate(model, test_loader, max_batches=100)
    print(f"Final Test Accuracy: {final_accuracy:.1%}")
    
    # Compare with baselines
    print("\n📊 Performance Comparison:")
    print(f"  Random Baseline: ~10%")
    print(f"  MLP (no conv):   ~55%")
    print(f"  CNN (with Conv2D): {final_accuracy:.1%} {'✅' if final_accuracy > 0.55 else ''}")
    
    if final_accuracy > 0.55:
        print("\n🎉 CNN outperforms MLP! Convolutions work!")
    
    print("\n💡 Why CNNs work better for images:")
    print("  - Conv2D learns spatial features")
    print("  - Pooling provides translation invariance")
    print("  - Hierarchical feature learning")
    print("  - Parameter sharing reduces overfitting")

if __name__ == "__main__":
    main()
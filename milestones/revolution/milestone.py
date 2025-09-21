#!/usr/bin/env python3
"""
Revolution Milestone: CIFAR-10 Object Recognition with CNN
Trains a CNN to 65%+ accuracy recognizing real-world objects using YOUR TinyTorch.

This demonstrates your complete training pipeline - data loading, forward/backward
passes, optimization, and convergence tracking.
"""

import tinytorch
from tinytorch.core import Tensor
from tinytorch.core.layers import Dense, Conv2d, MaxPool2d
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.data import DataLoader, CIFAR10Dataset
from tinytorch.core.optimizers import Adam
from tinytorch.core.training import Trainer
import numpy as np

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
train_dataset = CIFAR10Dataset(train=True)
test_dataset = CIFAR10Dataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Build the CNN - exactly like you would in PyTorch
class CIFAR10CNN:
    """CNN for CIFAR-10 classification."""
    
    def __init__(self):
        # Convolutional layers
        self.conv_layers = [
            Conv2d(3, 32, kernel_size=3, padding=1),  # 3x32x32 -> 32x32x32
            ReLU(),
            MaxPool2d(2),  # 32x32x32 -> 32x16x16
            
            Conv2d(32, 64, kernel_size=3, padding=1),  # 32x16x16 -> 64x16x16
            ReLU(),
            MaxPool2d(2),  # 64x16x16 -> 64x8x8
        ]
        
        # Fully connected layers
        self.fc_layers = [
            Dense(64 * 8 * 8, 128),
            ReLU(),
            Dense(128, 10),
            Softmax()
        ]
        
        self.all_layers = self.conv_layers + self.fc_layers
    
    def forward(self, x):
        """Forward pass through the CNN."""
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten for fully connected layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        for layer in self.all_layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    params.append(layer.bias)
        return params

# Create model and optimizer
model = CIFAR10CNN()
optimizer = Adam(model.parameters(), learning_rate=0.001)

# Training function
def train_epoch(model, train_loader, optimizer):
    """Train for one epoch."""
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model.forward(images)
        
        # Calculate loss (cross-entropy)
        loss = cross_entropy_loss(outputs, labels)
        
        # Backward pass
        gradients = loss.backward()
        
        # Update weights
        optimizer.step(gradients)
        
        # Track accuracy
        predictions = np.argmax(outputs.data, axis=1)
        correct += np.sum(predictions == labels.data)
        total += len(labels)
        total_loss += loss.data
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.data:.4f}, Acc={100*correct/total:.1f}%")
    
    return total_loss / len(train_loader), correct / total

# Evaluation function
def evaluate(model, test_loader):
    """Evaluate model on test set."""
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        outputs = model.forward(images)
        predictions = np.argmax(outputs.data, axis=1)
        correct += np.sum(predictions == labels.data)
        total += len(labels)
    
    return correct / total

# Cross-entropy loss
def cross_entropy_loss(predictions, targets):
    """Calculate cross-entropy loss."""
    batch_size = predictions.shape[0]
    
    # Convert targets to one-hot
    targets_onehot = np.zeros_like(predictions.data)
    targets_onehot[np.arange(batch_size), targets.data] = 1
    
    # Calculate loss
    epsilon = 1e-7
    pred_clipped = np.clip(predictions.data, epsilon, 1 - epsilon)
    loss = -np.sum(targets_onehot * np.log(pred_clipped)) / batch_size
    
    return Tensor(loss)

# Training loop
print("\n🚀 Training YOUR TinyTorch CNN on CIFAR-10...")
print("=" * 50)

num_epochs = 5
best_accuracy = 0

for epoch in range(num_epochs):
    print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    
    # Evaluate
    test_acc = evaluate(model, test_loader)
    
    print(f"📊 Summary: Train Loss={train_loss:.4f}, "
          f"Train Acc={train_acc*100:.1f}%, Test Acc={test_acc*100:.1f}%")
    
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        print(f"🎯 New best accuracy: {best_accuracy*100:.1f}%")

print("\n" + "=" * 50)
print("🎯 FINAL RESULTS:")
print(f"Best Test Accuracy: {best_accuracy*100:.1f}%")
print(f"Target: 65%+")

if best_accuracy >= 0.65:
    print("\n🎉 REVOLUTION MILESTONE ACHIEVED!")
    print("YOUR TinyTorch CNN recognizes real-world objects!")
    print("You've sparked the deep learning revolution from scratch!")
    print("\nYour training pipeline includes:")
    print("  ✅ Convolutional feature extraction")
    print("  ✅ Automatic differentiation")
    print("  ✅ Adam optimization")
    print("  ✅ Complete training loop")
else:
    print("\n⚠️ Keep training...")
    print("Try more epochs or adjusting hyperparameters.")

print("\n📦 Modules Used:")
print("  • tinytorch.core.layers.{Conv2d, MaxPool2d} - Spatial processing")
print("  • tinytorch.core.optimizers.Adam - Adaptive optimization")
print("  • tinytorch.core.training - Complete training pipeline")
print("  • tinytorch.data.CIFAR10Dataset - Real-world data")
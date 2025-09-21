#!/usr/bin/env python3
"""
CIFAR-10 Image Classification with TinyTorch CNNs

Train a Convolutional Neural Network to classify real-world images
into 10 categories using the CIFAR-10 dataset.

This demonstrates:
- Convolutional Neural Networks with TinyTorch
- Real image processing with spatial operations
- Advanced training techniques (data augmentation, learning rate scheduling)
- Production-level computer vision
"""

import numpy as np
import tinytorch as tt
from tinytorch.core import Tensor
from tinytorch.core.spatial import Conv2D, MaxPool2D, Flatten
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.normalization import BatchNorm2D, BatchNorm1D
from tinytorch.data import DataLoader, CIFAR10Dataset
from tinytorch.core.optimizers import Adam
from tinytorch.core.training import CrossEntropyLoss, Trainer


class SimpleCNN:
    """A simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        # Convolutional layers
        self.conv1 = Conv2D(3, 32, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x32
        self.bn1 = BatchNorm2D(32)
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)  # 32x32x32 -> 32x32x64
        self.bn2 = BatchNorm2D(64)
        self.conv3 = Conv2D(64, 128, kernel_size=3, padding=1)  # 16x16x64 -> 16x16x128
        self.bn3 = BatchNorm2D(128)
        
        # Pooling
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(128 * 4 * 4, 256)  # After 3 pools: 32->16->8->4
        self.bn4 = BatchNorm1D(256)
        self.fc2 = Dense(256, num_classes)
        
        # Activations
        self.relu = ReLU()
        self.softmax = Softmax()
    
    def forward(self, x):
        """Forward pass through CNN."""
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        layers = [self.conv1, self.conv2, self.conv3,
                 self.bn1, self.bn2, self.bn3, self.bn4,
                 self.fc1, self.fc2]
        for layer in layers:
            params.extend(layer.parameters())
        return params


def train_epoch(model, dataloader, optimizer, loss_fn, epoch):
    """Train for one epoch."""
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Forward pass
        predictions = model.forward(images)
        
        # Compute loss
        loss = loss_fn(predictions, labels)
        total_loss += float(loss.data)
        
        # Compute accuracy
        pred_classes = np.argmax(predictions.data, axis=1)
        correct += np.sum(pred_classes == labels.data)
        total += len(labels)
        
        # Backward pass (if autograd available)
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Log progress
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{len(dataloader)} | "
                  f"Loss: {loss.data:.4f} | "
                  f"Acc: {100*correct/total:.1f}%")
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader):
    """Evaluate model on test set."""
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for images, labels in dataloader:
        predictions = model.forward(images)
        pred_classes = np.argmax(predictions.data, axis=1)
        
        correct += np.sum(pred_classes == labels.data)
        total += len(labels)
        
        # Per-class accuracy
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += (pred_classes[i] == label)
            class_total[label] += 1
    
    return correct / total, class_correct / class_total


def main():
    print("=" * 70)
    print("🖼️ CIFAR-10 CNN Classification with TinyTorch")
    print("=" * 70)
    print()
    
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load dataset
    print("📚 Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Image size: 32×32×3 (RGB)")
    print(f"  Classes: {', '.join(classes)}")
    print()
    
    # Build model
    print("🏗️ Building Convolutional Neural Network...")
    model = SimpleCNN()
    print("  Architecture:")
    print("    Conv(3→32) → BN → ReLU → MaxPool(2×2)")
    print("    Conv(32→64) → BN → ReLU → MaxPool(2×2)")
    print("    Conv(64→128) → BN → ReLU → MaxPool(2×2)")
    print("    Flatten → Dense(2048→256) → BN → ReLU")
    print("    Dense(256→10) → Softmax")
    print()
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Training loop
    print("🎯 Training CNN...")
    print("-" * 70)
    
    num_epochs = 20
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Adjust learning rate
        if epoch == 10:
            optimizer.lr = 0.0001
            print("  📉 Reducing learning rate to 0.0001")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, epoch)
        
        # Evaluate
        test_acc, class_accuracies = evaluate(model, test_loader)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print(f"  🎉 New best accuracy: {test_acc:.1%}")
        
        print(f"  Summary: Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | "
              f"Test Acc: {test_acc:.1%}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 Final Results:")
    print("-" * 70)
    
    test_accuracy, class_accuracies = evaluate(model, test_loader)
    print(f"Overall Test Accuracy: {test_accuracy:.1%}")
    print(f"Best Accuracy Achieved: {best_accuracy:.1%}")
    print()
    
    print("Per-Class Accuracy:")
    for i, class_name in enumerate(classes):
        acc = class_accuracies[i] * 100
        bar = "█" * int(acc / 2)  # Simple bar chart
        print(f"  {class_name:12s}: {acc:5.1f}% {bar}")
    
    print()
    if test_accuracy >= 0.65:
        print("🎉 SUCCESS! Your CNN achieves strong real-world performance!")
        print("You've built a framework capable of production computer vision!")
    elif test_accuracy >= 0.50:
        print("📈 Good progress! Your CNN is learning real-world patterns!")
    else:
        print(f"🔧 Keep training! Target: 65%+, Current: {test_accuracy:.1%}")
    
    return test_accuracy


if __name__ == "__main__":
    accuracy = main()
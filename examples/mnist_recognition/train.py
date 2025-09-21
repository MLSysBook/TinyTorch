#!/usr/bin/env python3
"""
MNIST Handwritten Digit Recognition with TinyTorch

Train a neural network to recognize handwritten digits (0-9) using
the MNIST dataset - the "Hello World" of computer vision.

This demonstrates:
- Real dataset loading with TinyTorch
- Multi-layer perceptron for image classification
- Training loop with batch processing
- Model evaluation and accuracy metrics
"""

import numpy as np
import tinytorch as tt
from tinytorch.core import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.data import DataLoader, MNISTDataset
from tinytorch.core.optimizers import Adam
from tinytorch.core.training import CrossEntropyLoss


class MNISTClassifier:
    """Multi-layer perceptron for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        # Three-layer architecture
        self.fc1 = Dense(input_size, hidden_size)
        self.fc2 = Dense(hidden_size, 64)
        self.fc3 = Dense(64, num_classes)
        
        # Activations
        self.relu = ReLU()
        self.softmax = Softmax()
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten images: (batch, 28, 28) -> (batch, 784)
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        return (self.fc1.parameters() + 
                self.fc2.parameters() + 
                self.fc3.parameters())


def train_epoch(model, dataloader, optimizer, loss_fn):
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
        true_classes = np.argmax(labels.data, axis=1) if len(labels.shape) > 1 else labels.data
        correct += np.sum(pred_classes == true_classes)
        total += len(labels)
        
        # Backward pass (if autograd available)
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx:3d}/{len(dataloader)} | "
                  f"Loss: {loss.data:.4f} | "
                  f"Acc: {100*correct/total:.1f}%")
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader):
    """Evaluate model on test set."""
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        predictions = model.forward(images)
        
        pred_classes = np.argmax(predictions.data, axis=1)
        true_classes = np.argmax(labels.data, axis=1) if len(labels.shape) > 1 else labels.data
        
        correct += np.sum(pred_classes == true_classes)
        total += len(labels)
    
    return correct / total


def main():
    print("=" * 60)
    print("👁️ MNIST Digit Recognition with TinyTorch")
    print("=" * 60)
    print()
    
    # Load dataset
    print("📚 Loading MNIST dataset...")
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print()
    
    # Build model
    print("🏗️ Building neural network...")
    model = MNISTClassifier()
    print("  Architecture: 784 → 128 → 64 → 10")
    print()
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Training loop
    print("🎯 Training...")
    print("-" * 60)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn)
        
        # Evaluate
        test_acc = evaluate(model, test_loader)
        
        print(f"  Summary: Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | "
              f"Test Acc: {test_acc:.1%}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    print("-" * 60)
    
    test_accuracy = evaluate(model, test_loader)
    print(f"Test Set Accuracy: {test_accuracy:.1%}")
    
    if test_accuracy >= 0.85:
        print("\n🎉 SUCCESS! Your TinyTorch achieves production-level accuracy!")
        print("You've built a framework that can do real computer vision!")
    else:
        print(f"\n📈 Getting there! Target: 85%+, Current: {test_accuracy:.1%}")
    
    # Show some predictions
    print("\n🔍 Sample Predictions:")
    print("-" * 40)
    
    # Get one batch for demonstration
    for images, labels in test_loader:
        predictions = model.forward(images)
        pred_classes = np.argmax(predictions.data, axis=1)
        true_classes = np.argmax(labels.data, axis=1) if len(labels.shape) > 1 else labels.data
        
        # Show first 5
        for i in range(min(5, len(images))):
            status = "✅" if pred_classes[i] == true_classes[i] else "❌"
            print(f"  True: {true_classes[i]}, Predicted: {pred_classes[i]} {status}")
        break
    
    return test_accuracy


if __name__ == "__main__":
    accuracy = main()
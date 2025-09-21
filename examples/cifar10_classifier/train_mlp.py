#!/usr/bin/env python3
"""
CIFAR-10 Image Recognition with TinyTorch MLP

This example demonstrates Milestone 1: "Machines Can See"
Train a Multi-Layer Perceptron to recognize real RGB images from CIFAR-10.

This shows:
- Real dataset loading with TinyTorch
- Multi-layer perceptron for RGB image classification
- Training loop with batch processing
- Model evaluation and accuracy metrics
- ML Systems insights: scaling challenges and performance implications

Target: 45%+ accuracy (proves framework works on real data)
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset
from tinytorch.core.optimizers import Adam
from tinytorch.core.training import MeanSquaredError as MSELoss
from tinytorch.core.autograd import Variable


class CIFAR10MLPClassifier:
    """Multi-layer perceptron for CIFAR-10 classification.
    
    Architecture designed for RGB images (32x32x3 = 3072 input features).
    This demonstrates the scaling challenges when moving from toy problems
    to real-world data complexity.
    """
    
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10):
        print(f"🏗️ Building MLP: {input_size} → {hidden_size} → 256 → {num_classes}")
        
        # Three-layer architecture: 3072 → 512 → 256 → 10
        self.fc1 = Dense(input_size, hidden_size)
        self.fc2 = Dense(hidden_size, 256)
        self.fc3 = Dense(256, num_classes)
        
        # Activations
        self.relu = ReLU()
        self.softmax = Softmax()
        
        # Convert to Variables for training
        self._make_trainable()
        
        # Report system implications
        total_params = sum(p.data.size for p in self.parameters())
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"📊 Model size: {total_params:,} parameters ({memory_mb:.1f} MB)")
    
    def _make_trainable(self):
        """Convert parameters to Variables for autograd."""
        self.fc1.weights = Variable(self.fc1.weights, requires_grad=True)
        self.fc1.bias = Variable(self.fc1.bias, requires_grad=True)
        self.fc2.weights = Variable(self.fc2.weights, requires_grad=True)
        self.fc2.bias = Variable(self.fc2.bias, requires_grad=True)
        self.fc3.weights = Variable(self.fc3.weights, requires_grad=True)
        self.fc3.bias = Variable(self.fc3.bias, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Convert input to Variable if needed
        if not hasattr(x, 'requires_grad'):
            x = Variable(x, requires_grad=True)
        
        # Flatten RGB images: (batch, 3, 32, 32) → (batch, 3072)
        if len(x.data.shape) > 2:
            batch_size = x.data.shape[0]
            x = Variable(Tensor(x.data.data.reshape(batch_size, -1)), requires_grad=True)
        
        # Layer 1: 3072 → 512
        x = self.fc1(x)
        x = self.relu(x)
        
        # Layer 2: 512 → 256
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output layer: 256 → 10
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        return [
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias,
            self.fc3.weights, self.fc3.bias
        ]


def train_epoch(model, dataloader, optimizer, loss_fn, epoch):
    """Train for one epoch."""
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n--- Epoch {epoch + 1} Training ---")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Forward pass
        predictions = model.forward(images)
        
        # Convert labels to one-hot for MSE loss
        batch_size = labels.shape[0]
        num_classes = 10
        labels_onehot = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            label_idx = int(labels.data[i])
            labels_onehot[i, label_idx] = 1
        labels_var = Variable(Tensor(labels_onehot), requires_grad=False)
        
        # Compute loss
        loss = loss_fn(predictions, labels_var)
        total_loss += float(loss.data.data if hasattr(loss.data, 'data') else loss.data)
        
        # Compute accuracy
        pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
        pred_classes = np.argmax(pred_data, axis=1)
        true_classes = labels.data
        correct += np.sum(pred_classes == true_classes)
        total += labels.shape[0]
        
        # Backward pass
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Log progress every few batches
        if batch_idx % 10 == 0:
            curr_acc = 100 * correct / total if total > 0 else 0
            print(f"  Batch {batch_idx:2d}/{len(dataloader)} | "
                  f"Loss: {loss.data.data if hasattr(loss.data, 'data') else loss.data:.4f} | "
                  f"Acc: {curr_acc:.1f}%")
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader):
    """Evaluate model on test set."""
    correct = 0
    total = 0
    
    print("\n--- Evaluation ---")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        predictions = model.forward(images)
        
        pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
        pred_classes = np.argmax(pred_data, axis=1)
        true_classes = labels.data
        
        correct += np.sum(pred_classes == true_classes)
        total += labels.shape[0]
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}: {100*correct/total:.1f}% accuracy")
    
    return correct / total


def main():
    print("=" * 60)
    print("🖼️  CIFAR-10 Image Recognition with TinyTorch")
    print("=" * 60)
    print()
    
    # Load real CIFAR-10 dataset
    print("📚 Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(root="./data", train=True, download=True)
    test_dataset = CIFAR10Dataset(root="./data", train=False, download=False)
    
    # Use batch sizes that divide evenly (50,000 % 125 = 0, 10,000 % 125 = 0)
    train_loader = DataLoader(train_dataset, batch_size=125, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=125, shuffle=False)
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Image shape: {train_dataset[0][0].shape}")
    print()
    
    # Build model
    print("🏗️ Building neural network...")
    model = CIFAR10MLPClassifier()
    print()
    
    # Setup training
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    loss_fn = MSELoss()
    
    # Training loop
    print("🎯 Training...")
    print("-" * 60)
    
    num_epochs = 3  # Short training for demonstration
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, epoch)
        
        # Evaluate
        test_acc = evaluate(model, test_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy: {test_acc:.1%}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print(f"  🎯 New best accuracy!")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    print("-" * 60)
    
    final_accuracy = evaluate(model, test_loader)
    print(f"\nFinal Test Accuracy: {final_accuracy:.1%}")
    print(f"Best Accuracy Achieved: {best_accuracy:.1%}")
    
    # Milestone check
    target_accuracy = 0.45  # 45% for CIFAR-10 MLP
    if final_accuracy >= target_accuracy:
        print(f"\n🎉 MILESTONE 1 ACHIEVED!")
        print(f"Your TinyTorch achieves {final_accuracy:.1%} accuracy on real RGB images!")
        print("You've built a framework that handles real-world data complexity!")
    else:
        print(f"\n📈 Progress: {final_accuracy:.1%} (Target: {target_accuracy:.1%})")
        print("Keep training or try architectural improvements!")
    
    # Show some predictions with class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n🔍 Sample Predictions:")
    print("-" * 50)
    
    for images, labels in test_loader:
        predictions = model.forward(images)
        pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
        pred_classes = np.argmax(pred_data, axis=1)
        true_classes = labels.data
        
        # Show first 5
        for i in range(min(5, images.shape[0])):
            true_name = class_names[true_classes[i]]
            pred_name = class_names[pred_classes[i]]
            status = "✅" if pred_classes[i] == true_classes[i] else "❌"
            print(f"  True: {true_name:>10}, Predicted: {pred_name:>10} {status}")
        break
    
    # ML Systems Analysis
    print("\n" + "=" * 60)
    print("⚡ ML Systems Analysis:")
    print("-" * 60)
    print("🔍 Key Systems Insights:")
    print(f"  • Model parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"  • Memory footprint: {sum(p.data.size for p in model.parameters()) * 4 / 1024 / 1024:.1f} MB")
    print(f"  • Input complexity: 3,072 features (vs 784 for MNIST)")
    print(f"  • Scaling challenge: 4× data → 16× parameters → slower training")
    print(f"  • Performance: MLPs struggle with spatial data (CNNs will be better!)")
    
    print("\n📦 Components Used:")
    print("  ✅ Dense layers with autograd")
    print("  ✅ ReLU and Softmax activations") 
    print("  ✅ Adam optimizer")
    print("  ✅ MSE loss (CrossEntropy coming soon)")
    print("  ✅ CIFAR-10 dataset with real RGB images")
    print("  ✅ Complete training pipeline")
    
    return final_accuracy


if __name__ == "__main__":
    accuracy = main()
#!/usr/bin/env python3
"""
CIFAR-10 Training with TinyTorch Universal Dashboard

This script demonstrates training a real neural network on CIFAR-10 images
using the beautiful TinyTorch dashboard with real-time visualization.

Target: 55%+ accuracy with gorgeous training visualization
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

# Import the universal dashboard
from examples.common.training_dashboard import create_cifar10_dashboard

class OptimizedCIFAR10_MLP:
    """Optimized MLP for CIFAR-10 with dashboard integration"""
    
    def __init__(self):
        # Good balance of accuracy and speed
        self.fc1 = Dense(3072, 1024)
        self.fc2 = Dense(1024, 512)
        self.fc3 = Dense(512, 256)
        self.fc4 = Dense(256, 10)
        
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        self._initialize_weights()
        
        # Calculate total parameters
        self.total_params = sum(np.prod(layer.weights.shape) + np.prod(layer.bias.shape) 
                               for layer in self.layers)
    
    def _initialize_weights(self):
        """Improved weight initialization"""
        for i, layer in enumerate(self.layers):
            fan_in = layer.weights.shape[0]
            
            if i == len(self.layers) - 1:  # Output layer
                std = 0.01
            else:  # Hidden layers
                std = np.sqrt(2.0 / fan_in) * 0.6
            
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass"""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        logits = self.fc4(h3)
        return logits
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
        return params

def preprocess_images_fast(images, training=True):
    """Fast preprocessing optimized for speed"""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    if training:
        # Simple augmentation: horizontal flip
        augmented = np.copy(images_np)
        for i in range(batch_size):
            if np.random.random() > 0.5:
                augmented[i] = np.flip(augmented[i], axis=2)
        images_np = augmented
    
    # Normalize
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.485) / 0.229
    
    return Tensor(normalized.astype(np.float32))

def evaluate_model_with_metrics(model, dataloader, max_batches=60):
    """Enhanced evaluation with additional metrics"""
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    confidence_sum = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        x = Variable(preprocess_images_fast(images, training=False), requires_grad=False)
        logits = model.forward(x)
        
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        
        # Apply softmax for confidence
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        predictions = np.argmax(logits_np, axis=1)
        max_probs = np.max(probs, axis=1)
        
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
        confidence_sum += np.sum(max_probs)
        
        # Per-class accuracy
        for i in range(len(labels_np)):
            label = labels_np[i]
            class_total[label] += 1
            if predictions[i] == label:
                class_correct[label] += 1
    
    accuracy = correct / total if total > 0 else 0
    avg_confidence = confidence_sum / total if total > 0 else 0
    class_accuracies = class_correct / np.maximum(class_total, 1)
    
    return accuracy, avg_confidence, class_accuracies

def main():
    """Main training with enhanced dashboard"""
    
    # Create dashboard
    dashboard = create_cifar10_dashboard()
    
    # Load dataset
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = OptimizedCIFAR10_MLP()
    
    # Show welcome screen
    dashboard.show_welcome(
        model_info={
            "Architecture": "3072 → 1024 → 512 → 256 → 10",
            "Parameters": f"{model.total_params:,}",
            "Task": "10-class image classification",
            "Dataset": "CIFAR-10 (32×32 RGB images)"
        },
        config={
            "Optimizer": "Adam",
            "Learning Rate": "0.002 (with decay)",
            "Batch Size": "64",
            "Augmentation": "Horizontal flip",
            "Target Accuracy": "55%+"
        }
    )
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.002)
    
    # Training parameters
    num_epochs = 15  # Good balance of time and accuracy
    batches_per_epoch = 250  # Reasonable training time
    
    # Start training
    dashboard.start_training(num_epochs=num_epochs, target_accuracy=0.55)
    
    # Training loop
    for epoch in range(num_epochs):
        
        # Training phase with progress bar
        train_losses = []
        train_correct = 0
        train_total = 0
        
        with dashboard.show_batch_progress(epoch+1, "Training", batches_per_epoch) as progress:
            train_task = progress.add_task("Training...", total=batches_per_epoch)
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= batches_per_epoch:
                    break
                
                # Training step
                x = Variable(preprocess_images_fast(images, training=True), requires_grad=False)
                y_true = Variable(labels, requires_grad=False)
                
                logits = model.forward(x)
                loss = loss_fn(logits, y_true)
                
                # Track metrics
                loss_val = float(loss.data.data) if hasattr(loss.data, 'data') else float(loss.data._data)
                train_losses.append(loss_val)
                
                logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
                preds = np.argmax(logits_np, axis=1)
                labels_np = y_true.data._data if hasattr(y_true.data, '_data') else y_true.data
                train_correct += np.sum(preds == labels_np)
                train_total += len(labels_np)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                progress.update(train_task, advance=1, 
                              description=f"Training (Loss: {loss_val:.3f})")
        
        # Evaluation with enhanced metrics
        train_accuracy = train_correct / train_total
        test_accuracy, avg_confidence, class_accuracies = evaluate_model_with_metrics(
            model, test_loader, max_batches=60
        )
        
        # Calculate additional metrics
        avg_loss = np.mean(train_losses)
        learning_rate = optimizer.learning_rate
        
        # Extra metrics for dashboard
        extra_metrics = {
            "Confidence": avg_confidence,
            "Learning_Rate": learning_rate,
            "Min_Class_Acc": np.min(class_accuracies),
            "Max_Class_Acc": np.max(class_accuracies)
        }
        
        # Update dashboard
        dashboard.update_epoch(
            epoch + 1,
            train_accuracy,
            test_accuracy,
            avg_loss,
            extra_metrics
        )
        
        # Learning rate scheduling
        if epoch == 8:
            optimizer.learning_rate *= 0.5
        elif epoch == 12:
            optimizer.learning_rate *= 0.5
        
        # Early stopping if we hit high accuracy
        if test_accuracy >= 0.60:
            print(f"\n🎊 Exceptional performance ({test_accuracy:.1%})! Stopping early.")
            break
    
    # Final comprehensive evaluation
    final_accuracy, final_confidence, final_class_acc = evaluate_model_with_metrics(
        model, test_loader, max_batches=None
    )
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"\n" + "="*60)
    print(f"🔍 PER-CLASS PERFORMANCE:")
    print(f"-"*60)
    for i, (name, acc) in enumerate(zip(class_names, final_class_acc)):
        print(f"{name:>12}: {acc:.1%}")
    print(f"-"*60)
    print(f"{'Average':>12}: {np.mean(final_class_acc):.1%}")
    
    # Finish training
    results = dashboard.finish_training(final_accuracy)
    
    # Educational insights
    print(f"\n💡 Key Insights:")
    print(f"   • Neural networks can learn complex visual patterns")
    print(f"   • Achieved {final_accuracy:.1%} accuracy on real 32×32 images")
    print(f"   • Training time: {results['total_time']:.1f}s (practical for education)")
    print(f"   • Path to higher accuracy: CNN layers, more data, longer training")
    
    return results

if __name__ == "__main__":
    main()
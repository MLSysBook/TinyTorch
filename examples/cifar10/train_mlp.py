#!/usr/bin/env python3
"""
Simple CIFAR-10 Training

Basic training script without fancy UI - just results.
Achieves ~55% accuracy in about 1 minute.
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

class SimpleMLP:
    def __init__(self):
        # Standard architecture
        self.fc1 = Dense(3072, 1024)
        self.fc2 = Dense(1024, 512)
        self.fc3 = Dense(512, 256)
        self.fc4 = Dense(256, 10)
        
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        # Initialize weights
        for i, layer in enumerate(self.layers):
            fan_in = layer.weights.shape[0]
            std = 0.01 if i == len(self.layers) - 1 else np.sqrt(2.0 / fan_in) * 0.6
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.fc4(h3)
    
    def parameters(self):
        return [p for layer in self.layers for p in [layer.weights, layer.bias]]

def preprocess(images, training=True):
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    if training:
        augmented = np.copy(images_np)
        for i in range(batch_size):
            if np.random.random() > 0.5:
                augmented[i] = np.flip(augmented[i], axis=2)
        images_np = augmented
    
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.485) / 0.229
    return Tensor(normalized.astype(np.float32))

def evaluate(model, dataloader, max_batches=30):
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
    print("="*50)
    print("CIFAR-10 Training - Simple Version")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = SimpleMLP()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.002)
    
    print("Model: 3072 → 1024 → 512 → 256 → 10")
    print("Starting training...\n")
    
    # Training
    epochs = 15
    batches_per_epoch = 200
    best_acc = 0.0
    
    for epoch in range(epochs):
        # LR decay
        if epoch == 8:
            optimizer.learning_rate *= 0.5
        elif epoch == 12:
            optimizer.learning_rate *= 0.5
        
        # Training loop
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= batches_per_epoch:
                break
            
            x = Variable(preprocess(images, training=True), requires_grad=False)
            y_true = Variable(labels, requires_grad=False)
            
            logits = model.forward(x)
            loss = loss_fn(logits, y_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluation
        test_acc = evaluate(model, test_loader)
        best_acc = max(best_acc, test_acc)
        
        print(f"Epoch {epoch+1:2d}: {test_acc:.1%} (best: {best_acc:.1%})")
    
    print("\n" + "="*50)
    print(f"Final Result: {best_acc:.1%}")
    print(f"Improvement over random: {best_acc/0.10:.1f}×")
    print("="*50)
    
    if best_acc >= 0.55:
        print("\n✅ Success: Achieved 55%+ accuracy!")
    elif best_acc >= 0.50:
        print("\n📈 Good: Achieved 50%+ accuracy")
    
    return best_acc

if __name__ == "__main__":
    main()
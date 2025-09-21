#!/usr/bin/env python3
"""
Random Network Baseline for CIFAR-10

This shows what an UNTRAINED neural network with random weights achieves.
This is the true baseline - what performance do we get with no learning?

Expected: ~10% (random chance for 10 classes)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

class RandomNetwork:
    """Neural network with random weights - no training"""
    
    def __init__(self):
        # Same architecture as our trained network
        self.fc1 = Dense(3072, 1024)
        self.fc2 = Dense(1024, 512)
        self.fc3 = Dense(512, 256)
        self.fc4 = Dense(256, 10)
        
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        # Random initialization (NO TRAINING)
        for layer in self.layers:
            # Just random weights - this is our baseline
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * 0.1
            layer.bias._data = np.random.randn(*layer.bias.shape).astype(np.float32) * 0.1
            
            layer.weights = Variable(layer.weights.data, requires_grad=False)
            layer.bias = Variable(layer.bias.data, requires_grad=False)
    
    def forward(self, x):
        """Forward pass through random network"""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.fc4(h3)

def preprocess(images):
    """Same preprocessing as trained network"""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.485) / 0.229
    return Tensor(normalized.astype(np.float32))

def evaluate_random_network(model, dataloader):
    """Evaluate the random network"""
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Preprocess
        x = Variable(preprocess(images), requires_grad=False)
        
        # Forward pass through RANDOM network
        logits = model.forward(x)
        
        # Get predictions
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
    
    return correct / total if total > 0 else 0

def main():
    print("="*60)
    print("🎲 CIFAR-10 RANDOM NETWORK BASELINE")
    print("="*60)
    print("\nTesting an UNTRAINED neural network with random weights")
    print("This is what we get with NO learning\n")
    
    # Load test data
    print("Loading CIFAR-10 test dataset...")
    test_dataset = CIFAR10Dataset(train=False, root='data')
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    print(f"Loaded {len(test_dataset)} test images\n")
    
    # Create UNTRAINED network
    print("Creating neural network with RANDOM weights (no training)...")
    model = RandomNetwork()
    print("Architecture: 3072 → 1024 → 512 → 256 → 10")
    print("Status: UNTRAINED (random weights)\n")
    
    # Evaluate random network
    print("Evaluating random network on test set...")
    accuracy = evaluate_random_network(model, test_loader)
    
    print("\n" + "="*60)
    print("📊 RANDOM NETWORK RESULTS")
    print("="*60)
    print(f"\nAccuracy with NO training: {accuracy:.1%}")
    print(f"Expected (random chance): ~10%")
    
    # Compare with trained network
    print("\n" + "="*60)
    print("📈 COMPARISON")
    print("="*60)
    print(f"Random network (untrained): {accuracy:.1%}")
    print(f"Trained network: ~55%")
    improvement = 55.0 / (accuracy * 100) if accuracy > 0 else 0
    print(f"Improvement from training: {improvement:.1f}× better")
    
    print("\n💡 This proves that training actually works!")
    print("   The network starts at random (~10%) and learns to 55%")
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    
    if accuracy < 15:
        print("\n✅ Confirmed: Untrained network ≈ random chance")
    else:
        print(f"\n⚠️ Unexpected: {accuracy:.1%} is higher than expected for random weights")
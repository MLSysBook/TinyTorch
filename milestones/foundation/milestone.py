#!/usr/bin/env python3
"""
Foundation Milestone: MNIST Digit Recognition
Achieves 85%+ accuracy recognizing handwritten digits using YOUR TinyTorch.

This is what real ML code looks like - clean, professional, and using
the framework you built from scratch.
"""

import tinytorch
from tinytorch.core import Tensor, Dense, ReLU, Softmax
from tinytorch.data import DataLoader, MNISTDataset
from tinytorch.core.optimizers import SGD
import numpy as np

# Load MNIST dataset
print("Loading MNIST dataset...")
train_dataset = MNISTDataset(train=True)
test_dataset = MNISTDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Build the network - exactly like you would in PyTorch
class MNISTClassifier:
    """Simple MLP for MNIST classification."""
    
    def __init__(self):
        self.layers = [
            Dense(784, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            Softmax()
        ]
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten images from 28x28 to 784
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Pass through each layer
        for layer in self.layers:
            x = layer(x)
        return x
    
    def load_pretrained(self, checkpoint_path='foundation_weights.npz'):
        """Load pre-trained weights that achieve 85%+ accuracy."""
        weights = np.load(checkpoint_path)
        
        # Load weights into Dense layers
        dense_layers = [l for l in self.layers if isinstance(l, Dense)]
        dense_layers[0].weights = Tensor(weights['dense1_w'])
        dense_layers[0].bias = Tensor(weights['dense1_b'])
        dense_layers[1].weights = Tensor(weights['dense2_w'])
        dense_layers[1].bias = Tensor(weights['dense2_b'])
        dense_layers[2].weights = Tensor(weights['dense3_w'])
        dense_layers[2].bias = Tensor(weights['dense3_b'])

# Create and load the model
model = MNISTClassifier()
model.load_pretrained()

# Evaluate the model
def evaluate(model, test_loader):
    """Evaluate model accuracy on test set."""
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        # Forward pass
        outputs = model.forward(images)
        
        # Get predictions
        predictions = np.argmax(outputs.data, axis=1)
        correct += np.sum(predictions == labels.data)
        total += len(labels)
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(test_loader)}: "
                  f"Accuracy {100 * correct / total:.1f}%")
    
    accuracy = 100 * correct / total
    return accuracy

# Run evaluation
print("\n🧪 Testing YOUR TinyTorch MLP on MNIST...")
print("=" * 50)

accuracy = evaluate(model, test_loader)

print("\n🎯 RESULTS:")
print(f"Test Accuracy: {accuracy:.1f}%")
print(f"Target: 85%+")

if accuracy >= 85:
    print("\n🎉 MILESTONE ACHIEVED!")
    print("YOUR TinyTorch recognizes handwritten digits with production accuracy!")
    print("You've built the foundation of computer vision from scratch!")
else:
    print("\n⚠️ Not quite there yet...")
    print("Check that all modules are properly exported and working together.")

print("\n📦 Modules Used:")
print("  • tinytorch.core.Tensor - Mathematical foundation")
print("  • tinytorch.core.Dense - Fully connected layers")
print("  • tinytorch.core.{ReLU, Softmax} - Activation functions")
print("  • tinytorch.data.{DataLoader, MNISTDataset} - Data pipeline")
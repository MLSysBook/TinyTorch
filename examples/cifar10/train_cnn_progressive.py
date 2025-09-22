#!/usr/bin/env python3
"""
CIFAR-10 CNN Training - Progressive Improvements with Conv2D

This example shows progressive improvements using our actual Conv2D implementation.
We'll demonstrate how to get better performance step by step.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Tanh
from tinytorch.core.spatial import MultiChannelConv2D, MaxPool2D, flatten
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.optimizers import Adam, SGD
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset


class ProgressiveCNN:
    """Progressive CNN architecture showing incremental improvements.
    
    This model demonstrates how each architectural choice improves performance:
    1. Basic: Single conv layer per block
    2. Deeper: Double conv layers (VGG-style)
    3. Wider: More filters
    4. Regularized: Dropout-like regularization
    """
    
    def __init__(self, version='v1'):
        """
        Initialize CNN with different architectural versions.
        
        Versions:
        - v1: Basic (2 conv blocks) ~58-60%
        - v2: Deeper (4 conv blocks) ~62-65%
        - v3: Wider (more filters) ~65-68%
        - v4: All improvements ~68-70%
        """
        self.version = version
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=(2, 2))
        
        if version == 'v1':
            # Basic: Minimal CNN
            # Expected: ~58-60% accuracy
            self.conv1 = MultiChannelConv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
            self.conv2 = MultiChannelConv2D(in_channels=32, out_channels=64, kernel_size=(3, 3))
            # After conv1: 32@30x30, pool: 32@15x15
            # After conv2: 64@13x13, pool: 64@6x6
            self.fc1 = Dense(64 * 6 * 6, 128)
            self.fc2 = Dense(128, 10)
            self.dropout_rate = 0.0
            self.conv_layers = [self.conv1, self.conv2]
            
        elif version == 'v2':
            # Deeper: Add more conv layers (VGG-style)
            # Expected: ~62-65% accuracy
            self.conv1a = MultiChannelConv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
            self.conv1b = MultiChannelConv2D(in_channels=32, out_channels=32, kernel_size=(3, 3))
            self.conv2a = MultiChannelConv2D(in_channels=32, out_channels=64, kernel_size=(3, 3))
            self.conv2b = MultiChannelConv2D(in_channels=64, out_channels=64, kernel_size=(3, 3))
            # After conv1a: 32@30x30, conv1b: 32@28x28, pool: 32@14x14
            # After conv2a: 64@12x12, conv2b: 64@10x10, pool: 64@5x5
            self.fc1 = Dense(64 * 5 * 5, 128)
            self.fc2 = Dense(128, 10)
            self.dropout_rate = 0.0
            self.conv_layers = [self.conv1a, self.conv1b, self.conv2a, self.conv2b]
            
        elif version == 'v3':
            # Wider: More filters per layer
            # Expected: ~65-68% accuracy
            self.conv1a = MultiChannelConv2D(in_channels=3, out_channels=64, kernel_size=(3, 3))
            self.conv1b = MultiChannelConv2D(in_channels=64, out_channels=64, kernel_size=(3, 3))
            self.conv2a = MultiChannelConv2D(in_channels=64, out_channels=128, kernel_size=(3, 3))
            self.conv2b = MultiChannelConv2D(in_channels=128, out_channels=128, kernel_size=(3, 3))
            # After conv1a: 64@30x30, conv1b: 64@28x28, pool: 64@14x14
            # After conv2a: 128@12x12, conv2b: 128@10x10, pool: 128@5x5
            self.fc1 = Dense(128 * 5 * 5, 256)
            self.fc2 = Dense(256, 10)
            self.dropout_rate = 0.3
            self.conv_layers = [self.conv1a, self.conv1b, self.conv2a, self.conv2b]
            
        elif version == 'v4':
            # All improvements: Deeper + Wider + Regularized
            # Expected: ~68-72% accuracy
            self.conv1a = MultiChannelConv2D(in_channels=3, out_channels=64, kernel_size=(3, 3))
            self.conv1b = MultiChannelConv2D(in_channels=64, out_channels=64, kernel_size=(3, 3))
            self.conv2a = MultiChannelConv2D(in_channels=64, out_channels=128, kernel_size=(3, 3))
            self.conv2b = MultiChannelConv2D(in_channels=128, out_channels=128, kernel_size=(3, 3))
            self.conv3 = MultiChannelConv2D(in_channels=128, out_channels=256, kernel_size=(3, 3))
            # After conv1a: 64@30x30, conv1b: 64@28x28, pool: 64@14x14
            # After conv2a: 128@12x12, conv2b: 128@10x10, pool: 128@5x5
            # After conv3: 256@3x3
            self.fc1 = Dense(256 * 3 * 3, 512)
            self.fc2 = Dense(512, 256)
            self.fc3 = Dense(256, 10)
            self.dropout_rate = 0.5
            self.conv_layers = [self.conv1a, self.conv1b, self.conv2a, self.conv2b, self.conv3]
            
        # Collect FC layers based on version
        if version == 'v4':
            self.fc_layers = [self.fc1, self.fc2, self.fc3]
        else:
            self.fc_layers = [self.fc1, self.fc2]
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Smart initialization based on layer depth."""
        # Conv layers - He initialization
        for i, conv in enumerate(self.conv_layers):
            # Scale initialization based on depth
            depth_scale = 1.0 / (1.0 + i * 0.05)
            
            fan_in = conv.weights.shape[1] * conv.weights.shape[2] * conv.weights.shape[3]
            std = np.sqrt(2.0 / fan_in) * depth_scale
            
            conv.weights._data = np.random.randn(*conv.weights.shape).astype(np.float32) * std
            if conv.bias is not None:
                conv.bias._data = np.zeros(conv.bias.shape, dtype=np.float32)
            
            conv.weights = Variable(conv.weights.data, requires_grad=True)
            if conv.bias is not None:
                conv.bias = Variable(conv.bias.data, requires_grad=True)
        
        # FC layers - Xavier initialization
        for i, layer in enumerate(self.fc_layers):
            fan_in = layer.weights.shape[0]
            fan_out = layer.weights.shape[1]
            
            # Output layer gets smaller initialization
            if i == len(self.fc_layers) - 1:
                std = 0.01
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def dropout(self, x, training=True):
        """Simple dropout implementation."""
        if not training or self.dropout_rate == 0:
            return x
        
        x_data = x.data if hasattr(x, 'data') else x._data
        keep_prob = 1 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, size=x_data.shape) / keep_prob
        dropped = x_data * mask
        
        if isinstance(x, Variable):
            return Variable(dropped.astype(np.float32), requires_grad=x.requires_grad)
        return Tensor(dropped.astype(np.float32))
    
    def forward(self, x, training=True):
        """Forward pass through the network."""
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        
        # Reshape if flattened
        if len(x.shape) == 2 and x.shape[1] == 3072:
            x_data = x.data if hasattr(x, 'data') else x._data
            x_reshaped = x_data.reshape(batch_size, 3, 32, 32)
            x = Tensor(x_reshaped) if not isinstance(x, Variable) else Variable(x_reshaped, x.requires_grad)
        
        # Forward through conv layers based on version
        if self.version == 'v1':
            # Basic: Conv → Pool → Conv → Pool
            h = self.relu(self.conv1(x))
            h = self.pool(h)
            h = self.relu(self.conv2(h))
            h = self.pool(h)
            
        elif self.version == 'v2':
            # Deeper: Conv → Conv → Pool → Conv → Conv → Pool
            h = self.relu(self.conv1a(x))
            h = self.relu(self.conv1b(h))
            h = self.pool(h)
            h = self.relu(self.conv2a(h))
            h = self.relu(self.conv2b(h))
            h = self.pool(h)
            
        elif self.version == 'v3':
            # Wider: Same as v2 but more filters
            h = self.relu(self.conv1a(x))
            h = self.relu(self.conv1b(h))
            h = self.pool(h)
            h = self.relu(self.conv2a(h))
            h = self.relu(self.conv2b(h))
            h = self.pool(h)
            
        elif self.version == 'v4':
            # All improvements
            h = self.relu(self.conv1a(x))
            h = self.relu(self.conv1b(h))
            h = self.pool(h)
            h = self.relu(self.conv2a(h))
            h = self.relu(self.conv2b(h))
            h = self.pool(h)
            h = self.relu(self.conv3(h))
        
        # Flatten for FC layers
        h = flatten(h)
        
        # FC layers with dropout
        if self.version == 'v4':
            h = self.relu(self.fc1(h))
            h = self.dropout(h, training)
            h = self.relu(self.fc2(h))
            h = self.dropout(h, training)
            return self.fc3(h)
        else:
            h = self.relu(self.fc1(h))
            if self.dropout_rate > 0:
                h = self.dropout(h, training)
            return self.fc2(h)
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        for conv in self.conv_layers:
            params.append(conv.weights)
            if conv.bias is not None:
                params.append(conv.bias)
        for fc in self.fc_layers:
            params.extend([fc.weights, fc.bias])
        return params
    
    def count_parameters(self):
        """Count total parameters."""
        total = 0
        for p in self.parameters():
            if hasattr(p, 'data'):
                data = p.data if not hasattr(p.data, '_data') else p.data._data
                total += np.prod(data.shape)
        return total


def preprocess(images, training=True, augmentation_level=1):
    """
    Preprocessing with progressive augmentation.
    
    augmentation_level:
    0: No augmentation
    1: Basic (flip only)
    2: Moderate (flip + brightness)
    3: Strong (flip + brightness + contrast)
    """
    images_np = images.data if hasattr(images, 'data') else images._data
    batch_size = images_np.shape[0]
    
    if training and augmentation_level > 0:
        augmented = np.copy(images_np)
        for i in range(batch_size):
            # Level 1: Horizontal flip
            if augmentation_level >= 1 and np.random.random() > 0.5:
                if len(augmented.shape) == 2:
                    img = augmented[i].reshape(3, 32, 32)
                    img = np.flip(img, axis=2)
                    augmented[i] = img.flatten()
                else:
                    augmented[i] = np.flip(augmented[i], axis=2)
            
            # Level 2: Brightness adjustment
            if augmentation_level >= 2 and np.random.random() > 0.5:
                brightness = np.random.uniform(0.9, 1.1)
                augmented[i] = augmented[i] * brightness
            
            # Level 3: Contrast adjustment
            if augmentation_level >= 3 and np.random.random() > 0.5:
                contrast = np.random.uniform(0.9, 1.1)
                mean = np.mean(augmented[i])
                augmented[i] = (augmented[i] - mean) * contrast + mean
        
        images_np = augmented
    
    # Normalize
    normalized = (images_np - 0.485) / 0.229
    
    # Ensure correct shape for CNN
    if len(normalized.shape) == 2:
        normalized = normalized.reshape(batch_size, 3, 32, 32)
    
    return Tensor(normalized.astype(np.float32))


def evaluate(model, dataloader, max_batches=30):
    """Evaluate model accuracy."""
    correct = total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        x = Variable(preprocess(images, training=False), requires_grad=False)
        logits = model.forward(x, training=False)
        
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
    
    return correct / total if total > 0 else 0


def train_version(version, epochs=5, show_details=True):
    """Train a specific version of the CNN."""
    if show_details:
        print(f"\n{'='*60}")
        print(f"Training CNN {version}")
        print(f"{'='*60}")
    
    # Configuration based on version
    configs = {
        'v1': {'lr': 0.001, 'batch_size': 64, 'augmentation': 1, 'desc': 'Basic CNN'},
        'v2': {'lr': 0.001, 'batch_size': 64, 'augmentation': 2, 'desc': 'Deeper CNN'},
        'v3': {'lr': 0.0008, 'batch_size': 32, 'augmentation': 2, 'desc': 'Wider CNN'},
        'v4': {'lr': 0.0005, 'batch_size': 32, 'augmentation': 3, 'desc': 'Full CNN'},
    }
    
    config = configs[version]
    
    if show_details:
        print(f"Configuration: {config['desc']}")
        print(f"  Learning rate: {config['lr']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Augmentation level: {config['augmentation']}")
    
    # Load data
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = ProgressiveCNN(version=version)
    if show_details:
        print(f"  Parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    
    # Training
    best_accuracy = 0
    batches_per_epoch = 300 if version in ['v3', 'v4'] else 200
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0
        batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= batches_per_epoch:
                break
            
            # Forward pass
            x = Variable(preprocess(images, training=True, 
                                   augmentation_level=config['augmentation']), 
                        requires_grad=True)
            y = Variable(labels, requires_grad=False)
            
            logits = model.forward(x, training=True)
            loss = loss_fn(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data
            batches += 1
        
        # Evaluation
        test_acc = evaluate(model, test_loader, max_batches=50)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        if show_details:
            epoch_time = time.time() - start_time
            print(f"  Epoch {epoch+1}: Test Acc={test_acc:.1%}, "
                  f"Best={best_accuracy:.1%}, Time={epoch_time:.1f}s")
    
    return best_accuracy


def main():
    """Train all versions progressively to show improvements."""
    print("="*70)
    print("CIFAR-10 Progressive CNN Training")
    print("Demonstrating incremental improvements with Conv2D")
    print("="*70)
    
    print("\n📊 Expected Performance Progression:")
    print("  Random:     10% (baseline)")
    print("  MLP:        55% (no convolutions)")
    print("  v1 Basic:   ~60% (minimal CNN)")
    print("  v2 Deeper:  ~63% (more layers)")
    print("  v3 Wider:   ~66% (more filters)")
    print("  v4 Full:    ~70% (all improvements)")
    
    print("\n🚀 Starting Progressive Training...")
    
    # Train each version
    results = {}
    
    # Quick training for demonstration
    print("\n" + "="*70)
    print("PHASE 1: Quick Training (3 epochs each)")
    print("="*70)
    
    for version in ['v1', 'v2', 'v3', 'v4']:
        accuracy = train_version(version, epochs=3, show_details=True)
        results[version] = accuracy
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - Progressive Improvements")
    print("="*70)
    
    print("\nAccuracy Progression:")
    print(f"  Baseline (Random): 10.0%")
    print(f"  MLP (No Conv):     55.0%")
    print(f"  v1 Basic CNN:      {results['v1']:.1%} (+{results['v1']-0.55:.1%} vs MLP)")
    print(f"  v2 Deeper CNN:     {results['v2']:.1%} (+{results['v2']-results['v1']:.1%} vs v1)")
    print(f"  v3 Wider CNN:      {results['v3']:.1%} (+{results['v3']-results['v2']:.1%} vs v2)")
    print(f"  v4 Full CNN:       {results['v4']:.1%} (+{results['v4']-results['v3']:.1%} vs v3)")
    
    print(f"\n🎯 Total Improvement: {results['v4']-0.10:.1%} over random!")
    print(f"   Conv2D Advantage: {results['v4']-0.55:.1%} over MLP!")
    
    print("\n💡 Key Insights:")
    print("1. Basic Conv2D immediately beats MLP (spatial processing)")
    print("2. Deeper networks learn hierarchical features")
    print("3. More filters capture richer representations")
    print("4. Regularization (dropout) prevents overfitting")
    print("5. Each improvement is incremental but compounds!")
    
    print("\n🏗️ Architecture Evolution:")
    print("  v1: 2 conv layers  → Learn edges")
    print("  v2: 4 conv layers  → Learn shapes")
    print("  v3: Wider filters  → Learn textures")
    print("  v4: All + dropout  → Learn objects")
    
    print("\n📈 To reach 70%+ consistently:")
    print("  - Train for 10+ epochs")
    print("  - Use learning rate scheduling")
    print("  - Add batch normalization (when available)")
    print("  - More aggressive augmentation")
    
    if results['v4'] >= 0.68:
        print("\n🏆 SUCCESS! Approaching 70% with our Conv2D implementation!")
    elif results['v4'] >= 0.65:
        print("\n📈 Great progress! Close to 70% target!")
    else:
        print("\n💪 Solid CNN performance! More epochs will improve results.")


if __name__ == "__main__":
    main()
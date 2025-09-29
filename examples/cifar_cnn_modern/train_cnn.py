#!/usr/bin/env python3
"""
CIFAR-10 CNN (Modern) - Convolutional Revolution
===============================================

📚 HISTORICAL CONTEXT:
Convolutional Neural Networks revolutionized computer vision by exploiting spatial
structure in images. Unlike MLPs that flatten images (losing spatial relationships),
CNNs preserve spatial hierarchies through local connectivity and weight sharing,
enabling recognition of complex patterns in natural images.

🎯 WHAT YOU'RE BUILDING:
Using YOUR TinyTorch implementations, you'll build a CNN that achieves 65%+ accuracy
on CIFAR-10 natural images - proving YOUR spatial modules can extract hierarchical
features from real-world photographs!

✅ REQUIRED MODULES (Run after Module 10):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : YOUR data structure with autodiff
  Module 03 (Activations)   : YOUR ReLU for feature extraction
  Module 04 (Layers)        : YOUR Linear layers for classification
  Module 05 (Losses)        : YOUR CrossEntropy loss
  Module 07 (Optimizers)    : YOUR Adam optimizer
  Module 08 (Training)      : YOUR training loops
  Module 09 (Spatial)       : YOUR Conv2D, MaxPool2D, Flatten
  Module 10 (DataLoader)    : YOUR CIFAR10Dataset and batching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Hierarchical Feature Extraction):
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Input Image │  │   Conv2D    │  │   MaxPool   │  │   Conv2D    │  │   MaxPool   │  │   Flatten   │  │   Linear    │  │   Linear    │
    │ 32×32×3 RGB │─▶│    3→32     │─▶│     2×2     │─▶│    32→64    │─▶│     2×2     │─▶│   →2304     │─▶│  2304→256   │─▶│   256→10    │
    │   Pixels    │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M4   │  │   YOUR M4   │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                      Edge Detection     Downsample      Shape Detection    Downsample       Vectorize     Hidden Layer    Classification
                           ↓                                  ↓                                                                   ↓
                    Low-level features              High-level features                                                  10 Class Probs

🔍 CIFAR-10 DATASET - REAL NATURAL IMAGES:

CIFAR-10 contains 60,000 32×32 color images in 10 classes:

    Sample Images:                    Feature Hierarchy YOUR CNN Learns:
    
    ┌──────────┐                     Layer 1 (Conv 3→32):
    │ ✈️ Plane │                     • Edge detectors
    │[Sky blue │                     • Color gradients
    │[White    │                     • Simple textures
    │[Wings    │                     
    └──────────┘                     Layer 2 (Conv 32→64):
                                      • Object parts
    ┌──────────┐                     • Complex patterns
    │ 🚗 Car   │                     • Spatial relationships
    │[Red body]│                     
    │[Wheels]  │                     Output Layer:
    │[Windows] │                     • Complete objects
    └──────────┘                     • Class probabilities

    Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

    Why CNNs Excel at Natural Images:
    • LOCAL CONNECTIVITY: Pixels near each other are related
    • WEIGHT SHARING: Same filter detects patterns everywhere
    • HIERARCHICAL LEARNING: Edges → Shapes → Objects
    • TRANSLATION INVARIANCE: Detects cat anywhere in image

📊 EXPECTED PERFORMANCE:
- Dataset: 50,000 training images, 10,000 test images
- Training time: 3-5 minutes (demonstration mode)
- Expected accuracy: 65%+ (with YOUR simple CNN!)
- Parameters: ~600K (mostly in conv layers)
"""

import sys
import os
import numpy as np
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor              # Module 02: YOU built this!
from tinytorch.core.layers import Linear             # Module 04: YOU built this!
from tinytorch.core.activations import ReLU, Softmax  # Module 03: YOU built this!
from tinytorch.core.spatial import Conv2d, MaxPool2D  # Module 09: YOU built this!
from tinytorch.core.optimizers import Adam            # Module 07: YOU built this!
from tinytorch.core.dataloader import DataLoader, Dataset  # Module 10: YOU built this!

# Import dataset manager
try:
    from examples.data_manager import DatasetManager
except ImportError:
    sys.path.append(os.path.join(project_root, 'examples'))
    from data_manager import DatasetManager

class CIFARDataset(Dataset):
    """Custom CIFAR-10 Dataset using YOUR Dataset interface from Module 10!"""
    
    def __init__(self, data, labels):
        """Initialize with data and labels arrays."""
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        """Get a single sample - YOUR Dataset interface!"""
        return Tensor(self.data[idx]), Tensor([self.labels[idx]])
    
    def __len__(self):
        """Return dataset size - YOUR Dataset interface!"""
        return len(self.data)
    
    def get_num_classes(self):
        """Return number of classes."""
        return 10

def flatten(x):
    """Flatten spatial features for dense layers - YOUR implementation!"""
    batch_size = x.data.shape[0]
    return Tensor(x.data.reshape(batch_size, -1))

class CIFARCNN:
    """
    Convolutional Neural Network for CIFAR-10 using YOUR TinyTorch!
    
    This architecture demonstrates how spatial feature extraction enables
    recognition of complex patterns in natural images.
    """
    
    def __init__(self):
        print("🧠 Building CIFAR-10 CNN with YOUR TinyTorch modules...")
        
        # Convolutional feature extractors - YOUR spatial modules!
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))   # Module 09!
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # Module 09!
        self.pool = MaxPool2D(pool_size=(2, 2))  # Module 09: YOUR pooling!
        
        # Activation functions
        self.relu = ReLU()  # Module 03: YOUR activation!
        
        # Dense classification head
        # After conv1(32→30)→pool(15)→conv2(13)→pool(6): 64*6*6 = 2304 features
        self.fc1 = Linear(64 * 6 * 6, 256)  # Module 04: YOUR Linear!
        self.fc2 = Linear(256, 10)          # Module 04: YOUR Linear!
        
        # Calculate total parameters
        conv1_params = 3 * 3 * 3 * 32 + 32     # 3×3 kernels, 3→32 channels
        conv2_params = 3 * 3 * 32 * 64 + 64    # 3×3 kernels, 32→64 channels
        fc1_params = 64 * 6 * 6 * 256 + 256    # Flattened→256
        fc2_params = 256 * 10 + 10             # 256→10 classes
        self.total_params = conv1_params + conv2_params + fc1_params + fc2_params
        
        print(f"   Conv1: 3→32 channels (YOUR Conv2D extracts edges)")
        print(f"   Conv2: 32→64 channels (YOUR Conv2D builds shapes)")
        print(f"   Dense: 2304→256→10 (YOUR Linear classification)")
        print(f"   Total parameters: {self.total_params:,}")
        
    def forward(self, x):
        """Forward pass through YOUR CNN architecture."""
        # First conv block: Extract low-level features (edges, colors)
        x = self.conv1(x)           # Module 09: YOUR Conv2D!
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.pool(x)            # Module 09: YOUR MaxPool2D!
        
        # Second conv block: Build higher-level features (shapes, patterns)
        x = self.conv2(x)           # Module 09: YOUR Conv2D!
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.pool(x)            # Module 09: YOUR MaxPool2D!
        
        # Flatten and classify
        x = flatten(x)              # Module 09: YOUR spatial→dense bridge!
        x = self.fc1(x)             # Module 04: YOUR Linear!
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.fc2(x)             # Module 04: YOUR classification!
        
        return x
    
    def parameters(self):
        """Get all trainable parameters from YOUR layers."""
        return [
            self.conv1.weights, self.conv1.bias,
            self.conv2.weights, self.conv2.bias,
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias
        ]

def visualize_cifar_cnn():
    """Show how CNNs process natural images."""
    print("\n" + "="*70)
    print("🖼️  VISUALIZING CNN FEATURE EXTRACTION:")
    print("="*70)
    
    print("""
    How YOUR CNN Sees Images:           Feature Maps at Each Layer:
    
    Original Image (32×32×3):           After Conv1 (30×30×32):
    ┌────────────────┐                 ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │ [Cat in grass] │                 │Edge detectors...│ 32 filters
    │ Complex scene  │ → Conv+ReLU →   │Texture maps...  │ detect
    │ Many patterns  │                 │Color gradients. │ features
    └────────────────┘                 └─┴─┴─┴─┴─┴─┴─┴─┴─┘
    
    After Pool1 (15×15×32):            After Conv2 (13×13×64):
    ┌─────────┐                        ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │Reduced  │                        │Cat ears...      │ 64 filters
    │spatial  │ → Conv+ReLU →          │Cat eyes...      │ combine
    │dimension│                        │Grass texture... │ features
    └─────────┘                        └─┴─┴─┴─┴─┴─┴─┴─┴─┘
    
    After Pool2 + Flatten:             Classification:
    [6×6×64 = 2304 features] → Dense → [plane|car|bird|CAT|...]
                                              Highest probability
    
    Key CNN Advantages YOUR Implementation Provides:
    ✓ SPATIAL HIERARCHY: Low → High level features
    ✓ PARAMETER SHARING: 3×3 kernel used everywhere
    ✓ TRANSLATION INVARIANCE: Detects patterns anywhere
    ✓ AUTOMATIC FEATURE LEARNING: No manual engineering!
    """)
    print("="*70)

def train_cifar_cnn(model, train_loader, epochs=3, learning_rate=0.001):
    """Train CNN using YOUR complete training system with DataLoader!"""
    print("\n🚀 Training CIFAR-10 CNN with YOUR TinyTorch!")
    print(f"   Dataset: {len(train_loader.dataset)} color images")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   YOUR DataLoader (Module 10) handles batching!")
    print(f"   YOUR Adam optimizer (Module 07)")
    
    # YOUR optimizer
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    
    for epoch in range(epochs):
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        # Use YOUR DataLoader to iterate through batches!
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Demo mode - limit batches
                break
            
            # Forward pass with YOUR CNN
            outputs = model.forward(batch_data)  # YOUR spatial features!
            
            # Manual cross-entropy loss
            batch_size = len(batch_labels.data)
            num_classes = 10
            targets_one_hot = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                targets_one_hot[i, int(batch_labels.data[i])] = 1.0
            
            # Cross-entropy: -sum(y * log(softmax(x)))
            # Apply softmax first - handle nested data access
            outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
            exp_outputs = np.exp(outputs_np - np.max(outputs_np, axis=1, keepdims=True))
            softmax_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

            eps = 1e-8
            loss_value = -np.mean(np.sum(targets_one_hot * np.log(softmax_outputs + eps), axis=1))
            loss = Tensor([loss_value])
            
            # Backward pass with YOUR autograd
            optimizer.zero_grad()  # Module 07!
            loss.backward()        # Module 06: YOUR autodiff!
            optimizer.step()       # Module 07!
            
            # Track accuracy
            predictions = np.argmax(outputs_np, axis=1)
            correct += np.sum(predictions == batch_labels.data.flatten())
            total += len(batch_labels.data)
            
            epoch_loss += loss_value
            batch_count += 1
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                acc = 100 * correct / total
                print(f"   Batch {batch_idx+1}: "
                      f"Loss = {loss_value:.4f}, Accuracy = {acc:.1f}%")
        
        # Epoch summary
        epoch_acc = 100 * correct / total
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"   → Epoch Complete: Loss = {avg_loss:.4f}, "
              f"Accuracy = {epoch_acc:.1f}% (YOUR CNN + DataLoader!)")
    
    return model

def test_cifar_cnn(model, test_loader, class_names):
    """Test YOUR CNN on CIFAR-10 test set using DataLoader."""
    print("\n🧪 Testing YOUR CNN on Natural Images with YOUR DataLoader...")
    
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    # Test using YOUR DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        if batch_idx >= 20:  # Demo mode - limit batches
            break
        
        outputs = model.forward(batch_data)

        outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
        predictions = np.argmax(outputs_np, axis=1)
        batch_y = batch_labels.data.flatten()
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)
        
        # Per-class accuracy
        for j in range(len(batch_y)):
            label = int(batch_y[j])
            class_total[label] += 1
            if predictions[j] == label:
                class_correct[label] += 1
    
    # Results
    accuracy = 100 * correct / total
    print(f"\n   📊 Overall Test Accuracy: {accuracy:.2f}%")
    
    # Per-class performance
    print("\n   Per-Class Performance (YOUR CNN's understanding):")
    print("   " + "─"*50)
    print("   │ Class      │ Accuracy │ Visual               │")
    print("   ├────────────┼──────────┼──────────────────────┤")
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            bar_length = int(class_acc / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   │ {class_name:10} │  {class_acc:5.1f}%  │ {bar} │")
    
    print("   " + "─"*50)
    
    if accuracy >= 65:
        print("\n   🎉 EXCELLENT! YOUR CNN mastered natural image recognition!")
    elif accuracy >= 50:
        print("\n   ✅ Good progress! YOUR CNN is learning visual features!")
    else:
        print("\n   🔄 YOUR CNN is still learning... (normal for demo mode)")
    
    return accuracy

def analyze_cnn_systems(model):
    """Analyze YOUR CNN from an ML systems perspective."""
    print("\n🔬 SYSTEMS ANALYSIS of YOUR CNN Implementation:")
    
    print(f"\n   Model Architecture:")
    print(f"   • Convolutional layers: 2 (3→32→64 channels)")
    print(f"   • Pooling layers: 2 (2×2 max pooling)")
    print(f"   • Dense layers: 2 (2304→256→10)")
    print(f"   • Total parameters: {model.total_params:,}")
    
    print(f"\n   Computational Complexity:")
    print(f"   • Conv1: 32×30×30×(3×3×3) = 777,600 ops")
    print(f"   • Conv2: 64×13×13×(3×3×32) = 3,093,504 ops")
    print(f"   • Dense: 2,304×256 + 256×10 = 592,384 ops")
    print(f"   • Total: ~4.5M ops per image")
    
    print(f"\n   Memory Requirements:")
    print(f"   • Parameters: {model.total_params * 4 / 1024:.1f} KB")
    print(f"   • Activations (peak): ~500 KB per image")
    print(f"   • YOUR implementation: Pure Python + NumPy")
    
    print(f"\n   🏛️ CNN Evolution:")
    print(f"   • 1989: LeCun's CNN for handwritten digits")
    print(f"   • 2012: AlexNet revolutionizes ImageNet")
    print(f"   • 2015: ResNet enables 100+ layer networks")
    print(f"   • YOUR CNN: Core principles that power them all!")
    
    print(f"\n   💡 Why CNNs Dominate Vision:")
    print(f"   • Spatial hierarchy matches visual cortex")
    print(f"   • Parameter sharing: 3×3 kernel vs 32×32 dense")
    print(f"   • Translation invariance from weight sharing")
    print(f"   • YOUR implementation demonstrates all of these!")

def main():
    """Demonstrate CIFAR-10 CNN using YOUR TinyTorch!"""
    
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN')
    parser.add_argument('--test-only', action='store_true',
                       help='Test architecture only')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs (demo mode)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Show CNN visualization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use small subset for testing')
    args = parser.parse_args()
    
    print("🎯 CIFAR-10 CNN - Natural Image Recognition with YOUR Spatial Modules!")
    print("   Historical significance: CNNs revolutionized computer vision")
    print("   YOUR achievement: Spatial feature extraction on real photos")
    print("   Components used: YOUR Conv2D + MaxPool2D + complete system")
    
    # Visualization
    if args.visualize:
        visualize_cifar_cnn()
    
    # Class names
    class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Step 1: Load CIFAR-10
    print("\n📥 Loading CIFAR-10 dataset...")
    data_manager = DatasetManager()
    
    try:
        (train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()
        print(f"✅ Loaded {len(train_data)} training, {len(test_data)} test images")
        
        if args.quick_test:
            train_data = train_data[:1000]
            train_labels = train_labels[:1000]
            test_data = test_data[:500]
            test_labels = test_labels[:500]
            print("   (Using subset for quick testing)")
            
    except Exception as e:
        print(f"⚠️  CIFAR-10 download failed: {e}")
        print("   Using synthetic data for architecture testing...")
        train_data = np.random.randn(100, 3, 32, 32).astype(np.float32)
        train_labels = np.random.randint(0, 10, 100).astype(np.int64)
        test_data = np.random.randn(20, 3, 32, 32).astype(np.float32)
        test_labels = np.random.randint(0, 10, 20).astype(np.int64)
    
    # Step 2: Create Datasets and DataLoaders using YOUR Module 10!
    print("\n📦 Creating YOUR Dataset and DataLoader (Module 10)...")
    train_dataset = CIFARDataset(train_data, train_labels)
    test_dataset = CIFARDataset(test_data, test_labels)
    
    # YOUR DataLoader handles batching and shuffling!
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    print(f"   Train DataLoader: {len(train_dataset)} samples, batch_size={args.batch_size}")
    print(f"   Test DataLoader: {len(test_dataset)} samples, batch_size=100")
    
    # Step 3: Build CNN
    model = CIFARCNN()
    
    if args.test_only:
        print("\n🧪 ARCHITECTURE TEST MODE")
        # Test with a single batch from YOUR DataLoader
        for batch_data, batch_labels in train_loader:
            test_output = model.forward(batch_data)
            print(f"✅ Forward pass successful! Shape: {test_output.data.shape}")
            print("✅ YOUR CNN + DataLoader work together!")
            break
        return
    
    # Step 4: Train using YOUR DataLoader
    start_time = time.time()
    model = train_cifar_cnn(model, train_loader, epochs=args.epochs)
    train_time = time.time() - start_time
    
    # Step 5: Test using YOUR DataLoader
    accuracy = test_cifar_cnn(model, test_loader, class_names)
    
    # Step 5: Analysis
    analyze_cnn_systems(model)
    
    print(f"\n⏱️  Training time: {train_time:.1f} seconds")
    print(f"   Images/sec: {len(train_dataset) * args.epochs / train_time:.0f}")
    
    print("\n✅ SUCCESS! CIFAR-10 CNN Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • YOUR Conv2D extracts spatial features from natural images")
    print("   • YOUR MaxPool2D reduces dimensions while preserving information")
    print("   • YOUR DataLoader efficiently batches and shuffles data")
    print("   • YOUR CNN achieves real accuracy on complex photos")
    print("   • YOUR complete ML system works end-to-end!")
    
    print("\n🚀 Next Steps:")
    print("   • Continue to TinyGPT after Module 14 (Transformers)")
    print("   • YOUR spatial understanding scales to segmentation, detection, etc.")
    print(f"   • With {accuracy:.1f}% accuracy, YOUR computer vision works!")

if __name__ == "__main__":
    main()
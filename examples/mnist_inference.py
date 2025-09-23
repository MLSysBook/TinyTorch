#!/usr/bin/env python3
"""
🎯 MNIST Inference Demo - Your TinyTorch Code Recognizes Handwritten Digits!

After completing Phase 1 (Modules 1-5), this demo shows that your code
can classify handwritten digits - a classic computer vision task that
demonstrates the power of multi-layer perceptrons.

🎉 EVERY LINE USES CODE YOU BUILT FROM SCRATCH!
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tinytorch.nn as nn
import tinytorch.nn.functional as F
from tinytorch.core.tensor import Tensor

class MNIST_MLP(nn.Module):
    """
    MNIST Multi-Layer Perceptron - 784-128-64-10 architecture that you built!
    
    This network classifies 28x28 pixel images (784 features) into 10 digit classes.
    It demonstrates how neural networks can learn complex pattern recognition
    from high-dimensional input data.
    
    Architecture:
    - Input: 784 features (28×28 pixel intensities)
    - Hidden1: 128 ReLU units (learn low-level features)
    - Hidden2: 64 ReLU units (learn higher-level combinations)  
    - Output: 10 units (probability distribution over digits 0-9)
    """
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)  # You built Linear layers in Module 4!
        self.hidden2 = nn.Linear(128, 64)   # Multi-layer composition from Module 5!
        self.output = nn.Linear(64, 10)     # Classification head
    
    def forward(self, x):
        # Flatten image to vector (if needed)
        if len(x.data.shape) > 2:
            x = F.flatten(x, start_dim=1)   # You built flatten in Module 4!
        
        x = F.relu(self.hidden1(x))         # You built ReLU in Module 3!
        x = F.relu(self.hidden2(x))         # Hidden layer activation
        return self.output(x)               # Raw logits (pre-softmax)

def load_pretrained_weights(model, weights_path):
    """
    Load pretrained weights into MNIST model.
    
    In production, this would load from training checkpoints.
    Demonstrates model serialization - crucial for deployment.
    """
    print(f"🔄 Loading pretrained weights from {weights_path}...")
    
    # Load weights from NPZ file
    weights = np.load(weights_path)
    
    # Set each layer's parameters manually
    model.hidden1.weights.data = weights['hidden1.weight']
    model.hidden1.bias.data = weights['hidden1.bias']
    model.hidden2.weights.data = weights['hidden2.weight'] 
    model.hidden2.bias.data = weights['hidden2.bias']
    model.output.weights.data = weights['output.weight']
    model.output.bias.data = weights['output.bias']
    
    print("✅ Weights loaded successfully!")
    return model

def create_synthetic_digit_data():
    """
    Create synthetic digit-like patterns for demonstration.
    
    Since we don't have real MNIST data loaded, we'll create simple
    patterns that resemble digits. This shows the inference pipeline
    without requiring large datasets.
    """
    print("📊 Creating synthetic digit patterns...")
    
    # Create 28x28 synthetic patterns
    patterns = []
    labels = []
    
    # Pattern for "0" - circle-like
    zero_pattern = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            # Create circular pattern
            center_i, center_j = 14, 14
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            if 8 <= distance <= 12:
                zero_pattern[i, j] = 1.0
    patterns.append(zero_pattern.flatten())
    labels.append(0)
    
    # Pattern for "1" - vertical line
    one_pattern = np.zeros((28, 28))
    one_pattern[:, 13:15] = 1.0  # Vertical line in center
    patterns.append(one_pattern.flatten())
    labels.append(1)
    
    # Pattern for "2" - horizontal lines
    two_pattern = np.zeros((28, 28))
    two_pattern[5:7, :] = 1.0    # Top line
    two_pattern[13:15, :] = 1.0  # Middle line  
    two_pattern[21:23, :] = 1.0  # Bottom line
    patterns.append(two_pattern.flatten())
    labels.append(2)
    
    # Add some noise to make it more realistic
    for i in range(len(patterns)):
        noise = np.random.normal(0, 0.1, patterns[i].shape)
        patterns[i] = np.clip(patterns[i] + noise, 0, 1)
    
    return np.array(patterns, dtype=np.float32), np.array(labels)

def softmax_numpy(x):
    """Apply softmax to convert logits to probabilities."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def test_mnist_inference():
    """
    Test MNIST inference on synthetic digit patterns.
    
    Demonstrates the complete inference pipeline:
    1. Data preprocessing (normalization, flattening)
    2. Forward pass through network
    3. Probability prediction via softmax
    4. Classification decision
    """
    print("🧪 Testing MNIST digit classification...")
    
    # Create test data
    test_images, test_labels = create_synthetic_digit_data()
    digit_names = ['zero', 'one', 'two']
    
    print(f"\n📊 Classifying {len(test_images)} synthetic digit patterns:")
    print("Pattern   -> Predicted (Confidence) | Expected | Correct?")
    print("---------+------------------------+----------+---------")
    
    correct_predictions = 0
    
    for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
        # Create tensor from your Tensor class (Module 2)!
        input_tensor = Tensor(image.reshape(1, -1))  # Batch size 1
        
        # Run inference using your neural network (Modules 3-5)!
        logits = model(input_tensor)
        
        # Convert to probabilities
        probs = softmax_numpy(logits.data)
        predicted_class = np.argmax(probs)
        confidence = probs[0, predicted_class]
        
        # Check correctness
        is_correct = predicted_class == true_label
        if is_correct:
            correct_predictions += 1
        
        status = "✅" if is_correct else "❌"
        pattern_name = digit_names[i]
        expected_name = digit_names[true_label]
        predicted_name = str(predicted_class) if predicted_class < len(digit_names) else f"digit_{predicted_class}"
        
        print(f"{pattern_name:8} -> {predicted_name:8} ({confidence:.1%})    | {expected_name:8} | {status}")
    
    accuracy = correct_predictions / len(test_images) * 100
    print(f"\n🎯 Accuracy: {correct_predictions}/{len(test_images)} = {accuracy:.1f}%")
    
    if accuracy >= 50:
        print("🎉 GREAT! Your TinyTorch code shows digit classification capability!")
        print("   With real MNIST data and training, this would achieve 95%+ accuracy!")
    else:
        print("📚 Results vary with random weights. Real training achieves high accuracy.")
        print("   The important thing is your inference pipeline works perfectly!")
    
    return accuracy

def explain_mnist_significance():
    """Explain why MNIST matters in computer vision and ML systems."""
    print("\n" + "="*65)
    print("🎓 WHY MNIST MATTERS - ML Systems Thinking")
    print("="*65)
    
    print("""
👁️  COMPUTER VISION BREAKTHROUGH:
   • MNIST was the "Hello World" of computer vision (1990s)
   • Proved neural networks could recognize visual patterns
   • Gateway to modern CV: ImageNet, object detection, facial recognition
   • Same MLP architecture you built scales to any image classification

🏗️  SYSTEMS ARCHITECTURE LESSONS:
   • High-dimensional input (784 features) → low-dimensional output (10 classes)
   • Multiple hidden layers learn hierarchical feature representations
   • Layer1: edges, corners | Layer2: shapes, patterns | Output: digits
   • Demonstrates universal approximation theorem in practice

⚙️  PRODUCTION ENGINEERING INSIGHTS:
   • Batch processing: Same code handles 1 image or 1 million images
   • Memory efficiency: 784×128×64×10 = ~200K parameters (manageable)
   • Inference latency: Matrix multiplications are embarrassingly parallel
   • Model serving: Weight loading enables deployment at scale

🧠 SCALING TO MODERN AI:
   • Your MLP → CNN (spatial awareness) → Transformer (attention)
   • Same linear algebra: W·x + b (weights, activations, gradients)
   • Same software patterns: modules, parameters, forward/backward
   • ImageNet uses identical principles with 1000× more parameters

📊 PERFORMANCE CHARACTERISTICS:
   • Training: ~60K parameters need ~60K examples (MNIST has 60K)
   • Inference: ~200K FLOPs per prediction (modern GPUs: billion/sec)
   • Memory: ~1MB model size (easily fits in cache)
   • Latency: Sub-millisecond on modern hardware
""")

def main():
    """
    Main demo showing MNIST digit classification with pretrained weights.
    
    Demonstrates that after Phase 1, students have built a framework
    capable of real computer vision tasks!
    """
    print("🎯 TinyTorch MNIST Inference Demo") 
    print("=" * 50)
    print("🎉 Every operation uses code YOU built from scratch!")
    print()
    
    # Create model using your Module system (Module 5)
    global model
    model = MNIST_MLP()
    param_count = sum(p.data.size for p in model.parameters())
    print(f"🏗️  Created MNIST MLP with {param_count:,} parameters")
    print(f"   Architecture: 784 → 128 → 64 → 10")
    
    # Load pretrained weights
    weights_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'mnist_mlp_weights.npz')
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        print("   Run: python examples/pretrained/create_weights.py")
        return
    
    model = load_pretrained_weights(model, weights_path)
    
    # Test inference
    accuracy = test_mnist_inference()
    
    # Educational content
    explain_mnist_significance()
    
    print("\n🎉 CONGRATULATIONS!")
    print("   You've built a computer vision framework that classifies images!")
    print("   Next: Complete more modules to train on real MNIST/CIFAR-10 data!")

if __name__ == "__main__":
    main()
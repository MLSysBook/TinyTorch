#!/usr/bin/env python3
"""
Milestone 2: CNN/CIFAR-10 Training Capability Test

This tests whether TinyTorch can build and train CNN architectures
by validating core components and training a simple CNN on toy data.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear, Module
from tinytorch.core.spatial import Conv2d, MaxPool2D, flatten
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import Adam

class SimpleCNN(Module):
    """Simple CNN for testing CNN training capability."""
    
    def __init__(self, num_classes=2, input_size=(1, 8, 8)):
        super().__init__()
        # Simple CNN architecture: Conv -> ReLU -> Pool -> Flatten -> Dense
        self.conv1 = Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.flatten = flatten
        
        # Calculate flattened size dynamically
        # Input: (1, 8, 8)
        # After conv: (4, 6, 6) - conv reduces by kernel_size-1 on each side
        # After pool: (4, 3, 3) - pool reduces by factor of 2
        # Flattened: 4 * 3 * 3 = 36
        conv_out_channels = 4
        conv_out_h = input_size[1] - 3 + 1  # 8 - 3 + 1 = 6
        conv_out_w = input_size[2] - 3 + 1  # 8 - 3 + 1 = 6
        pool_out_h = conv_out_h // 2        # 6 // 2 = 3
        pool_out_w = conv_out_w // 2        # 6 // 2 = 3
        flattened_size = conv_out_channels * pool_out_h * pool_out_w  # 4 * 3 * 3 = 36
        
        self.fc1 = Linear(flattened_size, num_classes)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        """Forward pass through CNN."""
        # Convolutional features
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Flatten for dense layers
        x = self.flatten(x)
        
        # Dense prediction
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    
    def parameters(self):
        """Collect all parameters for optimizer."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.fc1.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.parameters():
            param.grad = None

def test_cnn_components():
    """Test CNN components individually."""
    print("🔧 Testing CNN Components...")
    
    # Test Conv2d layer
    print("  Testing Conv2d layer...")
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3))
    test_input = Variable(np.random.randn(1, 8, 8).astype(np.float32), requires_grad=True)  # Single channel 8x8
    conv_output = conv(test_input)
    print(f"    Input shape: {test_input.shape}")
    print(f"    Conv output shape: {conv_output.shape}")
    assert conv_output.shape == (2, 6, 6), f"Expected (2, 6, 6), got {conv_output.shape}"
    
    # Test ReLU activation with Variable
    print("  Testing ReLU with Variable...")
    relu = ReLU()
    relu_input = Variable(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32), requires_grad=True)
    relu_output = relu(relu_input)
    print(f"    ReLU input: {relu_input.data}")
    print(f"    ReLU output: {relu_output.data}")
    expected = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    assert np.allclose(relu_output.data, expected), f"ReLU failed: expected {expected}, got {relu_output.data}"
    
    # Test MaxPool2D
    print("  Testing MaxPool2D...")
    pool = MaxPool2D(pool_size=(2, 2))
    pool_input = Variable(np.random.randn(2, 6, 6).astype(np.float32), requires_grad=True)  # 2 channels, 6x6
    pool_output = pool(pool_input)
    print(f"    Pool input shape: {pool_input.shape}")
    print(f"    Pool output shape: {pool_output.shape}")
    assert pool_output.shape == (2, 3, 3), f"Expected (2, 3, 3), got {pool_output.shape}"
    
    # Test flatten
    print("  Testing flatten...")
    flat_input = Variable(np.random.randn(2, 3, 3).astype(np.float32), requires_grad=True)  # 2 channels, 3x3
    flattened = flatten(flat_input)
    print(f"    Flatten input shape: {flat_input.shape}")
    print(f"    Flatten output shape: {flattened.shape}")
    expected_flat_size = 2 * 3 * 3  # 18 features
    assert flattened.shape[1] == expected_flat_size, f"Expected {expected_flat_size} features, got {flattened.shape[1]}"
    
    print("  ✅ All CNN components working!")
    
def test_gradient_flow():
    """Test that gradients flow through CNN properly."""
    print("🔄 Testing Gradient Flow Through CNN...")
    
    # Create simple CNN
    model = SimpleCNN(num_classes=1, input_size=(1, 8, 8))
    
    # Create test input
    x = Variable(np.random.randn(1, 8, 8).astype(np.float32), requires_grad=True)  # Single image, 1 channel, 8x8
    target = Variable(np.array([[0.7]], dtype=np.float32), requires_grad=False)  # Target output
    
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    prediction = model.forward(x)
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Prediction: {prediction.data}")
    
    # Compute loss
    loss_fn = MeanSquaredError()
    loss = loss_fn(prediction, target)
    print(f"  Loss: {loss.data}")
    
    # Check parameter gradients before backward
    conv_weight_before = model.conv1.weight.grad
    fc_weight_before = model.fc1.weights.grad
    
    print(f"  Conv weight grad before backward: {conv_weight_before}")
    print(f"  FC weight grad before backward: {fc_weight_before}")
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check parameter gradients after backward
    conv_weight_after = model.conv1.weight.grad
    fc_weight_after = model.fc1.weights.grad
    
    print(f"  Conv weight grad after backward: {conv_weight_after is not None}")
    print(f"  FC weight grad after backward: {fc_weight_after is not None}")
    
    # Verify gradients exist
    if conv_weight_after is not None:
        print(f"    Conv grad shape: {conv_weight_after.shape}")
        print(f"    Conv grad magnitude: {np.linalg.norm(conv_weight_after.data):.6f}")
    
    if fc_weight_after is not None:
        print(f"    FC grad shape: {fc_weight_after.shape}")
        print(f"    FC grad magnitude: {np.linalg.norm(fc_weight_after.data):.6f}")
    
    # Test passes if we get gradients
    gradients_exist = (conv_weight_after is not None) and (fc_weight_after is not None)
    if gradients_exist:
        print("  ✅ Gradient flow through CNN working!")
    else:
        print("  ❌ Gradient flow through CNN broken!")
    
    return gradients_exist

def test_cnn_training():
    """Test CNN training on toy binary classification problem."""
    print("🎯 Testing CNN Training...")
    
    # Create toy dataset: simple pattern detection
    # Pattern 1: bright center (class 1)
    # Pattern 0: dark center (class 0)
    X_train = []
    y_train = []
    
    for i in range(20):
        if i < 10:
            # Class 0: dark center
            img = np.random.randn(1, 8, 8).astype(np.float32) * 0.1  # Low noise
            img[0, 3:5, 3:5] = -1.0  # Dark center
            label = [0.0]
        else:
            # Class 1: bright center
            img = np.random.randn(1, 8, 8).astype(np.float32) * 0.1  # Low noise  
            img[0, 3:5, 3:5] = 1.0  # Bright center
            label = [1.0]
        
        X_train.append(img)
        y_train.append(label)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    print(f"  Training data: {X_train.shape}, Labels: {y_train.shape}")
    
    # Create CNN model
    model = SimpleCNN(num_classes=1, input_size=(1, 8, 8))
    loss_fn = MeanSquaredError()
    optimizer = Adam(model.parameters(), learning_rate=0.01)
    
    print("  Training CNN...")
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        
        for i in range(len(X_train)):
            # Convert to Variables
            x_var = Variable(X_train[i], requires_grad=False)
            y_var = Variable(y_train[i], requires_grad=False)
            
            # Forward pass
            prediction = model.forward(x_var)
            loss = loss_fn(prediction, y_var)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.data.data if hasattr(loss.data, 'data') else loss.data
            pred_class = 1.0 if prediction.data.data > 0.5 else 0.0
            true_class = y_train[i][0]
            if abs(pred_class - true_class) < 0.1:
                correct += 1
        
        accuracy = correct / len(X_train) * 100
        avg_loss = total_loss / len(X_train)
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:2d}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:5.1f}%")
    
    # Final evaluation
    print("  Final test results:")
    correct = 0
    for i in range(len(X_train)):
        x_var = Variable(X_train[i], requires_grad=False)
        prediction = model.forward(x_var)
        pred_class = 1.0 if prediction.data.data > 0.5 else 0.0
        true_class = y_train[i][0]
        
        is_correct = abs(pred_class - true_class) < 0.1
        if is_correct:
            correct += 1
        
        if i < 5:  # Show first few examples
            print(f"    Sample {i}: pred={pred_class:.0f}, true={true_class:.0f} {'✅' if is_correct else '❌'}")
    
    final_accuracy = correct / len(X_train) * 100
    print(f"  Final Accuracy: {final_accuracy:.1f}%")
    
    # Success if we achieve reasonable accuracy
    success = final_accuracy >= 80.0
    if success:
        print("  ✅ CNN training successful!")
    else:
        print(f"  ⚠️ CNN training achieved {final_accuracy:.1f}% accuracy (target: 80%+)")
    
    return success

def main():
    """Run CNN training capability tests."""
    print("🔥 Milestone 2: CNN/CIFAR-10 Training Capability Test")
    print("=" * 60)
    
    try:
        # Test 1: Components
        test_cnn_components()
        print()
        
        # Test 2: Gradient flow
        gradient_success = test_gradient_flow()
        print()
        
        if not gradient_success:
            print("❌ Gradient flow test failed - cannot proceed with training")
            return False
        
        # Test 3: Training
        training_success = test_cnn_training()
        print()
        
        # Summary
        print("=" * 60)
        print("📊 MILESTONE 2 SUMMARY")
        print(f"Component Tests:  ✅ PASSED")
        print(f"Gradient Flow:    {'✅ PASSED' if gradient_success else '❌ FAILED'}")
        print(f"CNN Training:     {'✅ PASSED' if training_success else '❌ FAILED'}")
        
        overall_success = gradient_success and training_success
        
        if overall_success:
            print("\n🎉 MILESTONE 2 SUCCESS!")
            print("TinyTorch CNN training capability validated:")
            print("  ✅ Conv2d layers work with Variable gradients")
            print("  ✅ MaxPool2D and flatten preserve gradient flow")
            print("  ✅ ReLU activation works with Variables")
            print("  ✅ CNN can train on spatial pattern recognition")
            print("  ✅ Complete CNN pipeline functional")
        else:
            print("\n⚠️ MILESTONE 2 INCOMPLETE")
            print("Issues found - CNN training capability needs fixes")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ MILESTONE 2 FAILED")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("🚀 Ready for CIFAR-10 CNN training!")
    else:
        print("🔧 CNN components need fixes before CIFAR-10 training")
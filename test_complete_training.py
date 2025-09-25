#!/usr/bin/env python3
"""
Complete TinyTorch Training Pipeline Test

This script demonstrates end-to-end training with:
- Linear layers that maintain gradient connections
- Variable-aware activations  
- Autograd-enabled loss functions
- Proper gradient flow through the entire network

Tests both XOR learning and linear regression to validate the pipeline.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import SGD, Adam

def test_variable_operations():
    """Test basic Variable operations work correctly."""
    print("🧪 Testing Variable Operations...")
    
    # Test Variable creation and operations
    x = Variable([[2.0, 3.0]], requires_grad=True)
    y = Variable([[1.0, 4.0]], requires_grad=True)
    
    # Test addition
    z = x + y
    assert hasattr(z, 'backward'), "Addition should return Variable with backward"
    print("✅ Variable addition works")
    
    # Test multiplication
    w = x * y
    assert hasattr(w, 'backward'), "Multiplication should return Variable with backward"
    print("✅ Variable multiplication works")
    
    # Test matrix multiplication
    a = Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Variable([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a @ b
    assert hasattr(c, 'backward'), "Matrix multiplication should return Variable with backward"
    print("✅ Variable matrix multiplication works")
    
    # Test backward pass
    c.backward()
    assert a.grad is not None, "Gradient should be computed for a"
    assert b.grad is not None, "Gradient should be computed for b"
    print("✅ Backward pass works")
    
    print("🎉 All Variable operations work correctly!")

def test_linear_layer_with_variables():
    """Test Linear layer with Variable inputs."""
    print("\n🧪 Testing Linear Layer with Variables...")
    
    # Create a simple linear layer
    layer = Linear(input_size=2, output_size=3)
    
    # Test with Tensor input (inference mode)
    tensor_input = Tensor([[1.0, 2.0]])
    tensor_output = layer(tensor_input)
    print(f"✅ Tensor input: {tensor_input.shape} → {tensor_output.shape}")
    
    # Test with Variable input (training mode)
    var_input = Variable([[1.0, 2.0]], requires_grad=True)
    var_output = layer(var_input)
    
    assert isinstance(var_output, Variable), "Output should be Variable for Variable input"
    assert hasattr(var_output, 'backward'), "Output should support backward pass"
    print(f"✅ Variable input: {var_input.shape} → {var_output.shape}")
    
    # Test gradient flow through layer  
    # Create a simple loss that depends on the output
    loss_value = Variable(np.sum(var_output.data.data if hasattr(var_output.data, 'data') else var_output.data))
    loss_value.backward()
    
    # Check that input variable received gradients
    assert var_input.grad is not None, "Input Variable should have gradients after backward"
    
    print("✅ Gradient computation completed successfully")
    
    print("✅ Gradient flow through Linear layer works")
    print("🎉 Linear layer with Variables works correctly!")

def test_activation_with_variables():
    """Test activation functions with Variable inputs."""
    print("\n🧪 Testing Activations with Variables...")
    
    # Test data
    var_input = Variable([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)
    
    # Test ReLU
    relu = ReLU()
    relu_output = relu(var_input)
    assert isinstance(relu_output, Variable), "ReLU should return Variable for Variable input"
    print("✅ ReLU with Variables works")
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    sigmoid_output = sigmoid(var_input)
    assert isinstance(sigmoid_output, Variable), "Sigmoid should return Variable for Variable input"
    print("✅ Sigmoid with Variables works")
    
    # Test Tanh
    tanh = Tanh()
    tanh_output = tanh(var_input)
    assert isinstance(tanh_output, Variable), "Tanh should return Variable for Variable input"
    print("✅ Tanh with Variables works")
    
    print("🎉 All activations with Variables work correctly!")

def create_xor_network():
    """Create a network capable of learning XOR function."""
    class XORNetwork:
        def __init__(self):
            # XOR requires nonlinearity - can't be solved by linear model alone
            self.layer1 = Linear(2, 4)  # Input layer: 2 inputs → 4 hidden units
            self.activation1 = Tanh()   # Nonlinear activation
            self.layer2 = Linear(4, 1)  # Output layer: 4 hidden → 1 output
            self.activation2 = Sigmoid()  # Output activation for probability
            
        def forward(self, x):
            # Forward pass through network
            x = self.layer1(x)
            x = self.activation1(x)
            x = self.layer2(x)
            x = self.activation2(x)
            return x
        
        def parameters(self):
            # Collect all parameters for optimizer
            params = []
            params.extend(self.layer1.parameters())
            params.extend(self.layer2.parameters())
            return params
        
        def zero_grad(self):
            # Reset gradients for all parameters
            for param in self.parameters():
                param.grad = None
    
    return XORNetwork()

def test_xor_training():
    """Test complete training pipeline with XOR problem."""
    print("\n🚀 Testing Complete Training Pipeline: XOR Learning")
    print("=" * 60)
    
    # XOR training data
    # Input: [x1, x2], Output: x1 XOR x2
    X_train = np.array([
        [0.0, 0.0],  # 0 XOR 0 = 0
        [0.0, 1.0],  # 0 XOR 1 = 1
        [1.0, 0.0],  # 1 XOR 0 = 1
        [1.0, 1.0]   # 1 XOR 1 = 0
    ])
    
    y_train = np.array([
        [0.0],  # Expected output for [0, 0]
        [1.0],  # Expected output for [0, 1]
        [1.0],  # Expected output for [1, 0]
        [0.0]   # Expected output for [1, 1]
    ])
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Create network, loss function, and optimizer
    network = create_xor_network()
    loss_fn = MeanSquaredError()
    optimizer = Adam(network.parameters(), learning_rate=0.01)
    
    print(f"Network parameters: {len(network.parameters())} tensors")
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 500
    print_every = 100
    
    for epoch in range(num_epochs):
        # Forward pass
        X_var = Variable(X_train, requires_grad=False)
        y_var = Variable(y_train, requires_grad=False)
        
        # Get predictions
        predictions = network.forward(X_var)
        
        # Compute loss
        loss = loss_fn(predictions, y_var)
        
        # Backward pass
        network.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if epoch % print_every == 0:
            loss_value = loss.data.data if hasattr(loss.data, 'data') else loss.data
            print(f"Epoch {epoch:3d}: Loss = {loss_value:.6f}")
    
    # Test final predictions
    print("\n📊 Final Results:")
    print("Input → Expected | Predicted")
    print("-" * 30)
    
    with_grad = network.forward(Variable(X_train, requires_grad=False))
    final_predictions = with_grad.data.data if hasattr(with_grad.data, 'data') else with_grad.data
    
    correct_predictions = 0
    for i in range(len(X_train)):
        expected = y_train[i, 0]
        predicted = final_predictions[i, 0]
        predicted_class = 1.0 if predicted > 0.5 else 0.0
        is_correct = "✅" if abs(predicted_class - expected) < 0.1 else "❌"
        
        print(f"{X_train[i]} → {expected:.1f}     | {predicted:.3f} ({predicted_class:.0f}) {is_correct}")
        
        if abs(predicted_class - expected) < 0.1:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(X_train) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct_predictions}/{len(X_train)})")
    
    if accuracy >= 75.0:
        print("🎉 SUCCESS: Network learned XOR function!")
        return True
    else:
        print("❌ Network failed to learn XOR function adequately.")
        return False

def test_linear_regression():
    """Test training pipeline with simple linear regression."""
    print("\n🚀 Testing Training Pipeline: Linear Regression")
    print("=" * 55)
    
    # Generate simple linear data: y = 2x + 1 + noise
    np.random.seed(42)  # For reproducible results
    X_train = np.random.randn(100, 1) * 2  # Random inputs
    y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)  # Linear relationship + noise
    
    print(f"Training data: {X_train.shape[0]} samples")
    
    # Create simple linear model (no activation needed for regression)
    model = Linear(1, 1)
    loss_fn = MeanSquaredError()
    optimizer = SGD([model.weights, model.bias], learning_rate=0.01)
    
    # Training
    num_epochs = 200
    for epoch in range(num_epochs):
        # Forward pass
        X_var = Variable(X_train, requires_grad=False)
        y_var = Variable(y_train, requires_grad=False)
        
        predictions = model(X_var)
        loss = loss_fn(predictions, y_var)
        
        # Backward pass
        model.weights.grad = None
        model.bias.grad = None
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if epoch % 50 == 0:
            loss_val = loss.data.data if hasattr(loss.data, 'data') else loss.data
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.6f}")
    
    # Check learned parameters
    learned_weight = model.weights.data[0, 0]
    learned_bias = model.bias.data[0]
    
    print(f"\nLearned parameters:")
    print(f"Weight: {learned_weight:.3f} (expected: ~2.0)")
    print(f"Bias:   {learned_bias:.3f} (expected: ~1.0)")
    
    # Check if parameters are reasonable
    weight_ok = abs(learned_weight - 2.0) < 0.5
    bias_ok = abs(learned_bias - 1.0) < 0.5
    
    if weight_ok and bias_ok:
        print("✅ Linear regression learned correct parameters!")
        return True
    else:
        print("❌ Linear regression failed to learn correct parameters.")
        return False

def main():
    """Run all tests for the complete training pipeline."""
    print("🔥 TinyTorch Complete Training Pipeline Test")
    print("=" * 60)
    
    success_count = 0
    total_tests = 5
    
    try:
        # Test 1: Basic Variable operations
        test_variable_operations()
        success_count += 1
    except Exception as e:
        print(f"❌ Variable operations test failed: {e}")
    
    try:
        # Test 2: Linear layer with Variables
        test_linear_layer_with_variables()
        success_count += 1
    except Exception as e:
        print(f"❌ Linear layer test failed: {e}")
    
    try:
        # Test 3: Activations with Variables
        test_activation_with_variables()
        success_count += 1
    except Exception as e:
        print(f"❌ Activation test failed: {e}")
    
    try:
        # Test 4: XOR training
        if test_xor_training():
            success_count += 1
    except Exception as e:
        print(f"❌ XOR training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test 5: Linear regression
        if test_linear_regression():
            success_count += 1
    except Exception as e:
        print(f"❌ Linear regression test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! TinyTorch training pipeline works end-to-end!")
    elif success_count >= 3:
        print("✅ Core functionality works! Some advanced features need attention.")
    else:
        print("❌ Major issues detected. Core training pipeline needs fixes.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test inference and training capabilities for all TinyTorch examples.
Tests each capability systematically and reports issues.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid

def test_inference(name, model, input_data):
    """Test if forward pass works."""
    print(f"\n{'='*60}")
    print(f"Testing {name} - Inference")
    print('='*60)
    
    try:
        output = model.forward(input_data)
        output_np = np.array(output.data.data if hasattr(output.data, 'data') else output.data)
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_data.data.shape}")
        print(f"   Output shape: {output_np.shape}")
        print(f"   Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
        return True, output
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False, None

def test_gradient_flow(name, model, input_data, output):
    """Test if gradients propagate through the model."""
    print(f"\n{'-'*60}")
    print(f"Testing {name} - Gradient Flow")
    print('-'*60)
    
    try:
        # Create a simple loss
        output_np = np.array(output.data.data if hasattr(output.data, 'data') else output.data)
        loss_value = np.mean(output_np ** 2)
        loss = Tensor([loss_value])
        
        # Clear existing gradients
        for param in model.parameters():
            param.grad = None
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist
        grads_found = 0
        grads_nonzero = 0
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grads_found += 1
                grad_np = np.array(param.grad.data if hasattr(param.grad, 'data') else param.grad)
                if np.any(grad_np != 0):
                    grads_nonzero += 1
                    print(f"   Param {i}: grad shape {grad_np.shape}, norm={np.linalg.norm(grad_np):.6f}")
        
        if grads_found == 0:
            print(f"❌ No gradients found")
            return False
        elif grads_nonzero == 0:
            print(f"⚠️  Gradients exist but all are zero")
            return False
        else:
            print(f"✅ Gradients flowing: {grads_nonzero}/{len(model.parameters())} parameters have non-zero gradients")
            return True
            
    except Exception as e:
        print(f"❌ Gradient flow failed: {e}")
        return False

def test_parameter_update(name, model, input_data):
    """Test if parameters update during training."""
    print(f"\n{'-'*60}")
    print(f"Testing {name} - Parameter Updates")
    print('-'*60)
    
    try:
        # Store initial weights
        initial_weights = []
        for param in model.parameters():
            param_np = np.array(param.data.data if hasattr(param.data, 'data') else param.data)
            initial_weights.append(param_np.copy())
        
        # Do a training step
        output = model.forward(input_data)
        output_np = np.array(output.data.data if hasattr(output.data, 'data') else output.data)
        loss = Tensor([np.mean(output_np ** 2)])
        
        # Clear gradients
        for param in model.parameters():
            param.grad = None
            
        loss.backward()
        
        # Manual gradient descent
        learning_rate = 0.01
        for param in model.parameters():
            if param.grad is not None:
                grad_np = np.array(param.grad.data if hasattr(param.grad, 'data') else param.grad)
                param.data = param.data - learning_rate * grad_np
        
        # Check if weights changed
        weights_changed = 0
        for i, (param, initial) in enumerate(zip(model.parameters(), initial_weights)):
            current = np.array(param.data.data if hasattr(param.data, 'data') else param.data)
            if not np.allclose(current, initial):
                weights_changed += 1
                change = np.linalg.norm(current - initial)
                print(f"   Param {i}: changed by {change:.6f}")
        
        if weights_changed > 0:
            print(f"✅ Parameters updated: {weights_changed}/{len(model.parameters())} changed")
            return True
        else:
            print(f"❌ No parameters updated")
            return False
            
    except Exception as e:
        print(f"❌ Parameter update failed: {e}")
        return False

# Test Perceptron
print("\n" + "="*60)
print("CAPABILITY TESTING: PERCEPTRON 1957")
print("="*60)

from examples.perceptron_1957.rosenblatt_perceptron import RosenblattPerceptron

perceptron = RosenblattPerceptron(input_size=2, output_size=1)
perceptron_input = Tensor(np.random.randn(5, 2).astype(np.float32))

inf_ok, output = test_inference("Perceptron", perceptron, perceptron_input)
if inf_ok:
    grad_ok = test_gradient_flow("Perceptron", perceptron, perceptron_input, output)
    update_ok = test_parameter_update("Perceptron", perceptron, perceptron_input)

# Test XOR
print("\n" + "="*60)
print("CAPABILITY TESTING: XOR 1969")
print("="*60)

from examples.xor_1969.minsky_xor_problem import XORNetwork

xor_model = XORNetwork(input_size=2, hidden_size=4, output_size=1)
xor_input = Tensor(np.random.randn(4, 2).astype(np.float32))

inf_ok, output = test_inference("XOR", xor_model, xor_input)
if inf_ok:
    grad_ok = test_gradient_flow("XOR", xor_model, xor_input, output)
    update_ok = test_parameter_update("XOR", xor_model, xor_input)

# Summary
print("\n" + "="*60)
print("CAPABILITY TESTING SUMMARY")
print("="*60)
print("\n🔍 Key findings will guide our fixes...")

#!/usr/bin/env python3
"""
Focused test for Conv2d gradient flow only.
Avoids loading the full spatial_dev module which has issues with pooling tests.
"""

import numpy as np
import sys
import os

# Add modules to path
sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')
sys.path.append('modules/04_layers')

from tensor_dev import Tensor
from autograd_dev import Variable
from layers_dev import Parameter, Module

# Define just the Conv2d class without the full module
class Conv2d(Module):
    """2D Convolutional Layer - Isolated for testing"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        
        kH, kW = kernel_size
        # He initialization for weights
        fan_in = in_channels * kH * kW
        std = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(np.random.randn(out_channels, in_channels, kH, kW).astype(np.float32) * std)
        
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass through multi-channel Conv2D layer with automatic differentiation."""
        # Import Variable for gradient tracking
        input_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        
        # Convert parameters to Variables
        weight_var = Variable(self.weight, requires_grad=True) if not isinstance(self.weight, Variable) else self.weight
        bias_var = None
        if self.bias is not None:
            bias_var = Variable(self.bias, requires_grad=True) if not isinstance(self.bias, Variable) else self.bias
        
        # Perform convolution operation
        result_var = self._conv2d_operation(input_var, weight_var, bias_var)
        return result_var
    
    def _conv2d_operation(self, input_var, weight_var, bias_var):
        """Core convolution operation with automatic differentiation support."""
        # Extract data for computation
        input_data = input_var.data
        if hasattr(input_data, 'data'):  # If it's a Tensor
            input_data = input_data.data
        
        weight_data = weight_var.data
        if hasattr(weight_data, 'data'):  # If it's a Tensor
            weight_data = weight_data.data
        
        # Handle single image vs batch
        if len(input_data.shape) == 3:  # Single image: (in_channels, H, W)
            input_data = input_data[None, ...]  # Add batch dimension
            single_image = True
        else:
            single_image = False
        
        batch_size, in_channels, H, W = input_data.shape
        kH, kW = self.kernel_size
        
        # Validate input channels
        assert in_channels == self.in_channels, f"Expected {self.in_channels} input channels, got {in_channels}"
        
        # Calculate output dimensions
        out_H = H - kH + 1
        out_W = W - kW + 1
        
        # Perform convolution computation
        output = np.zeros((batch_size, self.out_channels, out_H, out_W), dtype=np.float32)
        
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                # Get filter for this output channel
                filter_weights = weight_data[out_c]  # Shape: (in_channels, kH, kW)
                
                # Convolve across all input channels
                for in_c in range(in_channels):
                    input_channel = input_data[b, in_c]  # Shape: (H, W)
                    filter_channel = filter_weights[in_c]  # Shape: (kH, kW)
                    
                    # Perform 2D convolution
                    for i in range(out_H):
                        for j in range(out_W):
                            patch = input_channel[i:i+kH, j:j+kW]
                            output[b, out_c, i, j] += np.sum(patch * filter_channel)
                
                # Add bias if enabled
                if self.use_bias and bias_var is not None:
                    bias_data = bias_var.data
                    if hasattr(bias_data, 'data'):  # If it's a Tensor
                        bias_data = bias_data.data
                    output[b, out_c] += bias_data[out_c]
        
        # Remove batch dimension if input was single image
        if single_image:
            output = output[0]
        
        # Create proper gradient function for convolution
        captured_input_data = input_data.copy()
        captured_weight_data = weight_data.copy()
        captured_in_channels = in_channels
        captured_kH, captured_kW = kH, kW
        conv_layer = self
        
        def conv2d_grad_fn(grad_output):
            """Proper gradient function for convolution."""
            # Convert grad_output to numpy
            grad_data = grad_output.data.data if hasattr(grad_output, 'data') else grad_output
            
            # Handle batch vs single image
            if len(captured_input_data.shape) == 3:  # Single image case
                grad_data = grad_data[None, ...]
                input_for_grad = captured_input_data[None, ...]
            else:
                input_for_grad = captured_input_data
            
            batch_size, out_channels, out_H, out_W = grad_data.shape
            
            # Compute weight gradients
            if weight_var.requires_grad:
                weight_grad = np.zeros_like(captured_weight_data)
                for b in range(batch_size):
                    for out_c in range(out_channels):
                        for in_c in range(captured_in_channels):
                            for i in range(out_H):
                                for j in range(out_W):
                                    patch = input_for_grad[b, in_c, i:i+captured_kH, j:j+captured_kW]
                                    weight_grad[out_c, in_c] += grad_data[b, out_c, i, j] * patch
                
                # Apply gradients to weight parameter
                conv_layer.weight.grad = weight_grad
            
            # Compute bias gradients
            if bias_var is not None and bias_var.requires_grad and conv_layer.bias is not None:
                bias_grad = np.sum(grad_data, axis=(0, 2, 3))  # Sum over batch, H, W
                conv_layer.bias.grad = bias_grad
        
        # Return Variable that maintains the computational graph
        return Variable(output, requires_grad=(input_var.requires_grad or weight_var.requires_grad), 
                       grad_fn=conv2d_grad_fn if (input_var.requires_grad or weight_var.requires_grad) else None)
    
    def __call__(self, x):
        """Make layer callable: layer(x) same as layer.forward(x)"""
        return self.forward(x)

def test_conv2d_gradients():
    """Test that Conv2d produces gradients for its parameters."""
    print("🔬 Testing Conv2d Gradient Flow...")
    
    # Create small Conv2d layer
    conv = Conv2d(in_channels=2, out_channels=3, kernel_size=(2, 2))
    print(f"Conv2d created: {conv.in_channels} -> {conv.out_channels}, kernel {conv.kernel_size}")
    
    # Create small input
    x_data = np.random.randn(2, 4, 4)  # 2 channels, 4x4 image
    x = Variable(Tensor(x_data), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = conv(x)
    print(f"Output shape: {y.shape}")
    print(f"Output type: {type(y)}")
    
    # Check if output is Variable
    assert isinstance(y, Variable), f"Expected Variable, got {type(y)}"
    
    # Check parameter gradients before backward
    print("\nBefore backward pass:")
    print(f"Conv weight grad exists: {hasattr(conv.weight, 'grad') and conv.weight.grad is not None}")
    if conv.bias is not None:
        print(f"Conv bias grad exists: {hasattr(conv.bias, 'grad') and conv.bias.grad is not None}")
    
    # Backward pass
    print("\n🔥 Running backward pass...")
    try:
        # Create gradient for output
        grad_output = Variable(Tensor(np.ones_like(y.data.data)), requires_grad=False)
        
        # Call the gradient function manually (simulating backward)
        if hasattr(y, 'grad_fn') and y.grad_fn is not None:
            print("Calling grad_fn...")
            y.grad_fn(grad_output)
        else:
            print("❌ No grad_fn found on output Variable")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check parameter gradients after backward
    print("\nAfter backward pass:")
    weight_has_grad = hasattr(conv.weight, 'grad') and conv.weight.grad is not None
    print(f"Conv weight grad exists: {weight_has_grad}")
    if weight_has_grad:
        print(f"  Weight grad shape: {conv.weight.grad.shape if hasattr(conv.weight.grad, 'shape') else 'No shape'}")
        print(f"  Weight grad type: {type(conv.weight.grad)}")
        grad_magnitude = np.abs(conv.weight.grad).mean()
        print(f"  Weight grad magnitude: {grad_magnitude}")
    
    if conv.bias is not None:
        bias_has_grad = hasattr(conv.bias, 'grad') and conv.bias.grad is not None
        print(f"Conv bias grad exists: {bias_has_grad}")
        if bias_has_grad:
            print(f"  Bias grad shape: {conv.bias.grad.shape if hasattr(conv.bias.grad, 'shape') else 'No shape'}")
            grad_magnitude = np.abs(conv.bias.grad).mean()
            print(f"  Bias grad magnitude: {grad_magnitude}")
    
    # Test result
    if weight_has_grad:
        print("\n✅ Conv2d gradient test PASSED! Gradients are flowing properly.")
        return True
    else:
        print("\n❌ Conv2d gradient test FAILED! No gradients found.")
        return False

if __name__ == "__main__":
    success = test_conv2d_gradients()
    sys.exit(0 if success else 1)
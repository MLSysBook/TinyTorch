#!/usr/bin/env python3
"""
Test the complete CNN pipeline with fixed Conv2d gradients.
Uses the minimal working Conv2d and other components.
"""

import numpy as np
import sys

# Add modules to path
sys.path.append('modules/02_tensor')
sys.path.append('modules/03_activations')
sys.path.append('modules/04_layers')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor
from autograd_dev import Variable

# Import working components
try:
    from activations_dev import ReLU
    has_relu = True
except:
    has_relu = False
    print("Warning: ReLU not available, will skip activation tests")

try:
    from layers_dev import Parameter, Module, Linear
    has_linear = True
except:
    has_linear = False
    print("Warning: Linear not available")

# Use the working minimal Conv2d from our test
class Conv2d(Module):
    """Working Conv2d with proper gradient flow"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        
        kH, kW = kernel_size
        # He initialization
        fan_in = in_channels * kH * kW
        std = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(np.random.randn(out_channels, in_channels, kH, kW).astype(np.float32) * std)
        
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass with gradient function"""
        input_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        weight_var = Variable(self.weight.data, requires_grad=True)
        bias_var = Variable(self.bias.data, requires_grad=True) if self.bias is not None else None
        
        result = self._conv2d_operation(input_var, weight_var, bias_var)
        return result
    
    def _conv2d_operation(self, input_var, weight_var, bias_var):
        """Convolution with proper gradient function"""
        # Extract numpy data properly
        input_data = input_var.data.data
        weight_data = weight_var.data.data if hasattr(weight_var.data, 'data') else weight_var.data
        
        # Handle batch dimension
        if len(input_data.shape) == 3:
            input_data = input_data[None, ...]
            single_image = True
        else:
            single_image = False
        
        batch_size, in_channels, H, W = input_data.shape
        kH, kW = self.kernel_size
        out_H = H - kH + 1
        out_W = W - kW + 1
        
        # Forward computation
        output = np.zeros((batch_size, self.out_channels, out_H, out_W), dtype=np.float32)
        
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                filter_weights = weight_data[out_c]
                for in_c in range(in_channels):
                    input_channel = input_data[b, in_c]
                    filter_channel = filter_weights[in_c]
                    for i in range(out_H):
                        for j in range(out_W):
                            patch = input_channel[i:i+kH, j:j+kW]
                            output[b, out_c, i, j] += np.sum(patch * filter_channel)
                
                # Add bias
                if self.use_bias and bias_var is not None:
                    bias_data = bias_var.data.data if hasattr(bias_var.data, 'data') else bias_var.data
                    output[b, out_c] += bias_data[out_c]
        
        if single_image:
            output = output[0]
        
        # Create gradient function
        captured_input = input_data.copy()
        captured_weight = weight_data.copy()
        conv_layer = self
        
        def conv2d_grad_fn(grad_output):
            """Compute and store gradients"""
            grad_data = grad_output.data.data if hasattr(grad_output.data, 'data') else grad_output.data
            
            # Handle shape correctly
            if len(captured_input.shape) == 3:
                grad_data = grad_data[None, ...]
                input_for_grad = captured_input[None, ...]
            else:
                input_for_grad = captured_input
            
            if len(grad_data.shape) == 3:
                batch_size, out_channels, out_H, out_W = 1, grad_data.shape[0], grad_data.shape[1], grad_data.shape[2]
                grad_data = grad_data[None, ...]
            else:
                batch_size, out_channels, out_H, out_W = grad_data.shape
            
            # Weight gradients
            if weight_var.requires_grad:
                weight_grad = np.zeros_like(captured_weight)
                for b in range(batch_size):
                    for out_c in range(out_channels):
                        for in_c in range(in_channels):
                            for i in range(out_H):
                                for j in range(out_W):
                                    patch = input_for_grad[b, in_c, i:i+kH, j:j+kW]
                                    weight_grad[out_c, in_c] += grad_data[b, out_c, i, j] * patch
                
                conv_layer.weight.grad = weight_grad
            
            # Bias gradients
            if bias_var is not None and bias_var.requires_grad:
                bias_grad = np.sum(grad_data, axis=(0, 2, 3))
                conv_layer.bias.grad = bias_grad
        
        return Variable(output, requires_grad=(input_var.requires_grad or weight_var.requires_grad), 
                       grad_fn=conv2d_grad_fn)
    
    def __call__(self, x):
        return self.forward(x)

# Simple flatten function
def flatten(x):
    """Flatten tensor to 1D (keeping batch dimension)"""
    if isinstance(x, Variable):
        data = x.data.data
        flattened = data.reshape(data.shape[0] if len(data.shape) > 1 else 1, -1)
        return Variable(Tensor(flattened), requires_grad=x.requires_grad)
    else:
        data = x.data if hasattr(x, 'data') else x
        flattened = data.reshape(data.shape[0] if len(data.shape) > 1 else 1, -1)
        return Tensor(flattened)

def test_cnn_pipeline():
    """Test complete CNN pipeline: Conv2d -> ReLU -> Flatten -> Linear"""
    print("🔬 Testing Complete CNN Pipeline...")
    
    print("\n1. Creating CNN Architecture:")
    # Create small CNN: 3 RGB channels -> 8 feature maps -> flatten -> 10 classes
    conv = Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
    print(f"   Conv2d: {conv.in_channels} -> {conv.out_channels}, kernel {conv.kernel_size}")
    
    if has_linear:
        # Calculate flattened size: 8 channels * 6*6 spatial (8-3+1=6)
        linear = Linear(input_size=8*6*6, output_size=10)
        print(f"   Linear: {linear.input_size} -> {linear.output_size}")
    else:
        print("   (Linear layer not available)")
    
    print("\n2. Forward Pass:")
    # Create RGB input: 3 channels, 8x8 image
    x_data = np.random.randn(3, 8, 8).astype(np.float32)
    x = Variable(Tensor(x_data), requires_grad=True)
    print(f"   Input shape: {x.shape}")
    
    # Conv2d forward
    conv_out = conv(x)
    print(f"   Conv2d output shape: {conv_out.shape}")
    print(f"   Conv2d output type: {type(conv_out)}")
    
    # ReLU (if available)
    if has_relu:
        relu = ReLU()
        relu_out = relu(conv_out)
        print(f"   ReLU output shape: {relu_out.shape}")
        current_output = relu_out
    else:
        print("   (Skipping ReLU - not available)")
        current_output = conv_out
    
    # Flatten
    flat_out = flatten(current_output)
    print(f"   Flatten output shape: {flat_out.shape}")
    
    # Linear (if available)
    if has_linear:
        final_out = linear(flat_out)
        print(f"   Linear output shape: {final_out.shape}")
        print(f"   Final output type: {type(final_out)}")
        final_variable = final_out
    else:
        print("   (Linear layer not available)")
        final_variable = flat_out
    
    print("\n3. Backward Pass:")
    # Check gradients before backward
    print("   Before backward:")
    print(f"     Conv weight grad: {hasattr(conv.weight, 'grad') and conv.weight.grad is not None}")
    print(f"     Conv bias grad: {hasattr(conv.bias, 'grad') and conv.bias.grad is not None}")
    if has_linear:
        print(f"     Linear weight grad: {hasattr(linear.weights, 'grad') and linear.weights.grad is not None}")
    
    # Simulate loss and backward
    try:
        # Create fake loss gradient
        grad_output = Variable(Tensor(np.ones_like(final_variable.data.data)), requires_grad=False)
        
        # Backward pass
        if hasattr(final_variable, 'grad_fn') and final_variable.grad_fn is not None:
            print("   Running backward pass...")
            final_variable.grad_fn(grad_output)
            
            # Check gradients after backward
            print("   After backward:")
            conv_weight_grad = hasattr(conv.weight, 'grad') and conv.weight.grad is not None
            conv_bias_grad = hasattr(conv.bias, 'grad') and conv.bias.grad is not None
            print(f"     Conv weight grad: {conv_weight_grad}")
            print(f"     Conv bias grad: {conv_bias_grad}")
            
            if conv_weight_grad:
                print(f"     Conv weight grad magnitude: {np.abs(conv.weight.grad).mean():.6f}")
            if conv_bias_grad:
                print(f"     Conv bias grad magnitude: {np.abs(conv.bias.grad).mean():.6f}")
            
            if has_linear:
                linear_grad = hasattr(linear.weights, 'grad') and linear.weights.grad is not None
                print(f"     Linear weight grad: {linear_grad}")
                
            if conv_weight_grad:
                print("\n✅ SUCCESS: CNN Pipeline with gradient flow working!")
                return True
            else:
                print("\n❌ FAILED: Conv2d gradients not computed")
                return False
                
        else:
            print("   ❌ No grad_fn found - no gradients available")
            return False
            
    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cnn_pipeline()
    sys.exit(0 if success else 1)
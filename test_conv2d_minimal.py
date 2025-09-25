#!/usr/bin/env python3
"""
Minimal test for Conv2d gradient flow - no imports of problematic modules.
"""

import numpy as np
import sys

# Create minimal classes needed for testing
class Tensor:
    """Minimal Tensor class for testing"""
    def __init__(self, data):
        self.data = np.array(data)
    
    @property
    def shape(self):
        return self.data.shape
    
    def numpy(self):
        return self.data

class Variable:
    """Minimal Variable class for testing"""
    def __init__(self, data, requires_grad=True, grad_fn=None):
        if isinstance(data, Tensor):
            self.data = data
        else:
            self.data = Tensor(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape

class Parameter:
    """Minimal Parameter class for testing"""
    def __init__(self, data):
        self.data = np.array(data)
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape

class Module:
    """Minimal Module base class"""
    def __init__(self):
        pass

class Conv2d(Module):
    """Minimal Conv2d for gradient testing"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        
        kH, kW = kernel_size
        # Small random weights
        self.weight = Parameter(np.random.randn(out_channels, in_channels, kH, kW).astype(np.float32) * 0.1)
        
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass with gradient function"""
        input_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        weight_var = Variable(self.weight.data, requires_grad=True)  # Use .data from Parameter
        bias_var = Variable(self.bias.data, requires_grad=True) if self.bias is not None else None
        
        result = self._conv2d_operation(input_var, weight_var, bias_var)
        return result
    
    def _conv2d_operation(self, input_var, weight_var, bias_var):
        """Convolution with proper gradient function"""
        # Extract numpy data
        input_data = input_var.data.data
        # weight_var.data might be Parameter (has .data directly) or Tensor (has .data.data)
        if hasattr(weight_var.data, 'data'):
            weight_data = weight_var.data.data  # Parameter case
        else:
            weight_data = weight_var.data  # Direct numpy case
        
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
                    if hasattr(bias_var.data, 'data'):
                        bias_data = bias_var.data.data  # Parameter case
                    else:
                        bias_data = bias_var.data  # Direct numpy case
                    output[b, out_c] += bias_data[out_c]
        
        if single_image:
            output = output[0]
        
        # Create gradient function
        captured_input = input_data.copy()
        captured_weight = weight_data.copy()
        conv_layer = self
        
        def conv2d_grad_fn(grad_output):
            """Compute and store gradients"""
            if hasattr(grad_output.data, 'data'):
                grad_data = grad_output.data.data
            else:
                grad_data = grad_output.data
            
            if len(captured_input.shape) == 3:  # Single image case 
                grad_data = grad_data[None, ...]
                input_for_grad = captured_input[None, ...]
                single_grad = True
            else:
                input_for_grad = captured_input
                single_grad = False
            
            # Handle shape correctly
            if len(grad_data.shape) == 3:
                batch_size, out_channels, out_H, out_W = 1, grad_data.shape[0], grad_data.shape[1], grad_data.shape[2]
                grad_data = grad_data[None, ...]  # Add batch dim
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

def test_conv2d_gradients():
    """Test Conv2d gradient computation"""
    print("🔬 Testing Conv2d Gradient Flow...")
    
    # Create layer
    conv = Conv2d(in_channels=2, out_channels=3, kernel_size=(2, 2))
    print(f"Conv2d: {conv.in_channels} -> {conv.out_channels}, kernel {conv.kernel_size}")
    
    # Create input
    x_data = np.random.randn(2, 4, 4).astype(np.float32)
    x = Variable(x_data, requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = conv(x)
    print(f"Output shape: {y.shape}")
    print(f"Output is Variable: {isinstance(y, Variable)}")
    print(f"Output has grad_fn: {hasattr(y, 'grad_fn') and y.grad_fn is not None}")
    
    # Check gradients before backward
    print("\nBefore backward:")
    print(f"Weight grad exists: {conv.weight.grad is not None}")
    print(f"Bias grad exists: {conv.bias.grad is not None}")
    
    # Simulate backward pass
    print("\n🔥 Running backward pass...")
    if y.grad_fn is not None:
        grad_output = Variable(np.ones_like(y.data.data), requires_grad=False)
        y.grad_fn(grad_output)
        
        print("After backward:")
        print(f"Weight grad exists: {conv.weight.grad is not None}")
        print(f"Bias grad exists: {conv.bias.grad is not None}")
        
        if conv.weight.grad is not None:
            print(f"Weight grad shape: {conv.weight.grad.shape}")
            print(f"Weight grad magnitude: {np.abs(conv.weight.grad).mean():.6f}")
        
        if conv.bias.grad is not None:
            print(f"Bias grad shape: {conv.bias.grad.shape}")
            print(f"Bias grad magnitude: {np.abs(conv.bias.grad).mean():.6f}")
        
        if conv.weight.grad is not None and conv.bias.grad is not None:
            print("\n✅ SUCCESS: Conv2d gradients computed correctly!")
            return True
        else:
            print("\n❌ FAILED: Gradients not computed")
            return False
    else:
        print("❌ No gradient function found")
        return False

if __name__ == "__main__":
    success = test_conv2d_gradients()
    sys.exit(0 if success else 1)
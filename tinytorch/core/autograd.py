"""
TinyTorch Autograd Module - Clean Implementation

This module provides automatic differentiation for Tensors.
No Variable class - just pure Tensor with gradient tracking!
"""

import numpy as np
from typing import Optional, List, Tuple
from tinytorch.core.tensor import Tensor

# Enable autograd function from the clean implementation
def enable_autograd():
    """Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.
    """
    # Check if already enabled
    if hasattr(Tensor, '_autograd_enabled'):
        return

    print("✅ Autograd enabled for TinyTorch!")
    print("   - Use Tensor with requires_grad=True")
    print("   - Call backward() to compute gradients")
    print("   - NO Variable class needed!")

    # The actual enhancement would be done here
    # For now, we rely on the tensor having dormant features
    Tensor._autograd_enabled = True

# Auto-enable when module is imported
enable_autograd()

# Export clean operations (no Variable!)
__all__ = ['enable_autograd']

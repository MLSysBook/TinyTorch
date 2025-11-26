__version__ = "0.1.0"

# Import core functionality
from . import core

# Make common components easily accessible at top level
from .core.tensor import Tensor
from .core.layers import Linear, Dropout
from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
# from .core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss  # TEMP: removed for testing
from .core.optimizers import SGD, AdamW

# 🔥 CRITICAL: Enable automatic differentiation
# This patches Tensor operations to track gradients
# Use quiet=True when imported by CLI tools to avoid cluttering output
import os
from .core.autograd import enable_autograd
enable_autograd(quiet=os.environ.get('TINYTORCH_QUIET', '').lower() in ('1', 'true', 'yes'))

# Export main public API
__all__ = [
    'core',
    'Tensor',
    'Linear', 'Dropout',
    'Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax',
    # 'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',  # TEMP: removed for testing
    'SGD', 'AdamW'
]

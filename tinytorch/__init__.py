__version__ = "0.1.0"

# Import core functionality
from . import core

# Make common components easily accessible at top level
from .core.tensor import Tensor
from .core.layers import Linear, Dropout
from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
from .core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from .core.optimizers import SGD, AdamW

# 🔥 CRITICAL: Enable automatic differentiation
# This patches Tensor operations to track gradients
from .core.autograd import enable_autograd
enable_autograd()

# Export main public API
__all__ = [
    'core',
    'Tensor',
    'Linear', 'Dropout',
    'Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax',
    'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',
    'SGD', 'AdamW'
]

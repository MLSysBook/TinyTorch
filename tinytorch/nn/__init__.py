"""
TinyTorch Neural Network Module (nn)

This package provides PyTorch-compatible neural network building blocks:

Core Components:
- Module: Base class for all layers (automatic parameter registration)
- Linear: Fully connected layer (renamed from Dense) 
- Conv2d: 2D convolutional layer (renamed from MultiChannelConv2D)

Functional Interface:
- functional (F): Stateless operations like relu, flatten, max_pool2d

Example Usage:
    import tinytorch.nn as nn
    import tinytorch.nn.functional as F
    
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, (3, 3))  # RGB → 32 features
            self.fc1 = nn.Linear(800, 10)           # Classifier
        
        def forward(self, x):
            x = F.relu(self.conv1(x))  # Convolution + activation
            x = F.flatten(x)           # Flatten for dense layer
            return self.fc1(x)         # Classification
    
    model = CNN()
    params = list(model.parameters())  # Auto-collected parameters!

The key insight: Students implement the core algorithms (conv, linear transforms)
while this infrastructure provides the clean API they expect from PyTorch.
"""

# Import Module base class
from ..core.layers import Module

# Import layers from core (these contain the student implementations)  
from ..core.layers import Linear, ReLU, Dropout
from ..core.activations import Sigmoid
from ..core.spatial import Conv2d, MaxPool2d, AvgPool2d

# Import transformer components
from ..text.embeddings import Embedding, PositionalEncoding
from ..core.attention import MultiHeadAttention, scaled_dot_product_attention
from ..models.transformer import LayerNorm, TransformerBlock

# Functional interface (if it exists)
try:
    from . import functional
    F = functional
except ImportError:
    functional = None
    F = None

# Utility functions
def Parameter(data, requires_grad=True):
    """Create a parameter tensor (learnable weight)."""
    from ..core.tensor import Tensor
    import numpy as np
    if not isinstance(data, Tensor):
        data = Tensor(np.array(data))
    data.requires_grad = requires_grad
    return data

class Sequential(Module):
    """Sequential container for stacking layers."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) if hasattr(layer, '__call__') else layer.forward(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

# Export the main public API
__all__ = [
    'Module',
    'Linear', 
    'Conv2d',
    'Embedding',
    'PositionalEncoding',
    'MultiHeadAttention',
    'LayerNorm',
    'TransformerBlock',
    'Sequential',
    'Parameter',
    'scaled_dot_product_attention',
]

# Add functional exports if available
if functional is not None:
    __all__.extend(['functional', 'F'])

# Note: Parameter function will be available after tensor module export
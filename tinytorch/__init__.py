__version__ = "0.1.0"

# Import core functionality
from . import core

# Make common components easily accessible (only what exists)
from .core.tensor import Tensor

# Export main public API (only what works)
__all__ = [
    'core',
    'Tensor'
]

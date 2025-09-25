"""
Core TinyTorch components.

This module contains the fundamental building blocks:
- utils: Utility functions 
- tensor: Core tensor implementation
- autograd: Automatic differentiation
- modules: Neural network layers
- optimizers: Training optimizers
- quantization: INT8 quantization for inference acceleration

All code is auto-generated from notebooks. Do not edit manually.
"""

# 🛡️ STUDENT PROTECTION: Automatic validation on import
# This ensures critical functionality works before students start training
try:
    from ._validation import auto_validate_on_import
    auto_validate_on_import()
except ImportError:
    # Validation module not available, continue silently
    pass
except Exception:
    # Don't crash on import issues, just warn
    import warnings
    warnings.warn(
        "🚨 TinyTorch validation failed. Core functionality may be broken. "
        "Check if you've accidentally edited files in tinytorch/core/",
        UserWarning
    ) 
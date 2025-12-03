"""
TinyTorch - Build ML Systems From First Principles

A complete educational ML framework for learning neural network internals
by implementing everything from scratch.

Top-level exports provide easy access to commonly used components.
For advanced modules (optimization, profiling), import from submodules:
    from tinytorch.profiling.profiler import Profiler
    from tinytorch.optimization.quantization import quantize_int8
    from tinytorch.generation.kv_cache import enable_kv_cache
"""

__version__ = "0.1.0"

# ============================================================================
# Core Functionality (Modules 01-07)
# ============================================================================
from .core.tensor import Tensor
from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
from .core.layers import Layer, Linear, Dropout
from .core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from .core.optimizers import SGD, Adam, AdamW
from .core.training import Trainer, CosineSchedule, clip_grad_norm

# ============================================================================
# Data Loading (Module 08)
# ============================================================================
from .data.loader import Dataset, TensorDataset, DataLoader
from .data.loader import RandomHorizontalFlip, RandomCrop, Compose  # Augmentation

# ============================================================================
# Spatial Operations (Module 09)
# ============================================================================
from .core.spatial import Conv2d, MaxPool2d, AvgPool2d

# ============================================================================
# Text Processing (Modules 10-11)
# ============================================================================
from .text.tokenization import Tokenizer, CharTokenizer, BPETokenizer
from .text.embeddings import Embedding, PositionalEncoding, EmbeddingLayer

# ============================================================================
# Attention & Transformers (Modules 12-13)
# ============================================================================
from .core.attention import MultiHeadAttention, scaled_dot_product_attention
from .core.transformer import LayerNorm, MLP, TransformerBlock, GPT, create_causal_mask

# ============================================================================
# Enable Autograd (CRITICAL - must happen after imports)
# ============================================================================
import os
from .core.autograd import enable_autograd

# Enable autograd quietly when imported by CLI tools
enable_autograd()

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Version
    '__version__',
    
    # Core - Tensor
    'Tensor',
    
    # Core - Activations
    'Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax',
    
    # Core - Layers
    'Layer', 'Linear', 'Dropout',
    
    # Core - Losses
    'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',
    
    # Core - Optimizers
    'SGD', 'Adam', 'AdamW',
    
    # Core - Training
    'Trainer', 'CosineSchedule', 'clip_grad_norm',
    
    # Data Loading
    'Dataset', 'TensorDataset', 'DataLoader',
    'RandomHorizontalFlip', 'RandomCrop', 'Compose',
    
    # Core - Spatial (CNN)
    'Conv2d', 'MaxPool2d', 'AvgPool2d',
    
    # Text/NLP
    'Tokenizer', 'CharTokenizer', 'BPETokenizer',
    'Embedding', 'PositionalEncoding', 'EmbeddingLayer',
    
    # Core - Attention
    'MultiHeadAttention', 'scaled_dot_product_attention',
    
    # Models - Transformers
    'LayerNorm', 'MLP', 'TransformerBlock', 'GPT', 'create_causal_mask',
]

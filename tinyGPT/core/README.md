# TinyGPT Core Components

This directory contains the core components for TinyGPT, a educational implementation of GPT-style language models built on TinyTorch foundations.

## Components

### `tokenizer.py` - Character-Level Tokenization
- **CharTokenizer**: Character-level tokenizer for text processing
- **Key Features**:
  - Simple character-to-token mapping
  - Vocabulary size limiting for computational efficiency
  - Special tokens support (`<UNK>`, `<PAD>`)
  - Batch encoding with padding/truncation
  - Comprehensive text analysis capabilities

**Usage:**
```python
from core.tokenizer import CharTokenizer

tokenizer = CharTokenizer(vocab_size=100)
tokenizer.fit(training_text)
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
```

### `training.py` - Language Model Training Infrastructure
- **LanguageModelTrainer**: Complete training pipeline for language models
- **LanguageModelLoss**: Cross-entropy loss with next-token prediction
- **LanguageModelAccuracy**: Accuracy metrics for language modeling

**Key Features**:
- Text-to-sequence data preparation
- Next-token prediction training
- Autoregressive text generation
- Training/validation splitting
- Comprehensive evaluation metrics

**Usage:**
```python
from core.training import LanguageModelTrainer
from core.models import TinyGPT

model = TinyGPT(vocab_size=50, d_model=128)
trainer = LanguageModelTrainer(model, tokenizer)

history = trainer.fit(text, epochs=5, seq_length=64)
generated = trainer.generate_text("Hello", max_length=50)
```

### `attention.py` - Attention Mechanisms
- **MultiHeadAttention**: Multi-head self-attention implementation
- **SelfAttention**: Simplified single-head attention
- **PositionalEncoding**: Sinusoidal positional embeddings
- **create_causal_mask**: Causal masking for autoregressive models

### `models.py` - Transformer Models
- **TinyGPT**: Complete GPT-style transformer model
- **TransformerBlock**: Individual transformer layer
- **LayerNorm**: Layer normalization implementation
- **SimpleLM**: Simplified language model for comparison

## Integration with TinyTorch

The TinyGPT components are designed to maximize reuse of TinyTorch components:

**Reused Components (70%+)**:
- Dense layers for all linear transformations
- Activation functions (ReLU, Softmax)
- Loss functions (CrossEntropyLoss)
- Optimizers (Adam)
- Training infrastructure patterns
- Tensor operations

**New Components for NLP**:
- Multi-head attention mechanisms
- Positional encoding
- Layer normalization
- Causal masking
- Text tokenization
- Autoregressive generation

## Educational Benefits

1. **Character-Level Simplicity**: Easy to understand tokenization without complex subword algorithms
2. **Transparent Architecture**: All components implemented with clear educational comments
3. **Component Reuse**: Demonstrates how ML foundations generalize across domains
4. **Progressive Complexity**: From simple tokenizer to full transformer model
5. **Mock Implementations**: Works with or without TinyTorch for standalone learning

## Example: Shakespeare Demo

The `examples/shakespeare_demo.py` demonstrates the complete pipeline:

1. Character tokenization of Shakespeare text
2. TinyGPT model creation and training
3. Text generation at different temperatures
4. Performance analysis and comparison with vision models

This shows how the same mathematical foundations (linear layers, attention, optimization) power both computer vision and natural language processing.

## File Dependencies

```
core/
├── tokenizer.py       # Standalone, only requires numpy
├── attention.py       # Uses TinyTorch Tensor and Dense (with mocks)
├── models.py          # Uses attention.py and TinyTorch layers
├── training.py        # Uses tokenizer.py and TinyTorch components
└── README.md          # This file
```

## Design Philosophy

TinyGPT follows the same educational philosophy as TinyTorch:

- **Build → Use → Understand**: Implement each component before using it
- **Educational Clarity**: Clear code with extensive documentation
- **Minimal Dependencies**: NumPy + educational implementations
- **Real-World Relevance**: Patterns used in production frameworks
- **Component Modularity**: Each piece can be understood independently

The goal is to demystify how language models work while showing how they share foundational concepts with computer vision models.
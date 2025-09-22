# TinyGPT - Language Model Example

**The Capstone Achievement: Build GPT from your own components!**

TinyGPT demonstrates that the ML framework you built from scratch can create a working language model. This isn't a toy - it's a real transformer that generates text.

## What TinyGPT Proves

This example shows that everything you built works together:
- **Tensors & Autograd** (Modules 2, 9) - Gradient computation through transformers
- **Layers & Attention** (Modules 4-7) - Multi-head attention and feed-forward networks
- **Optimizers & Training** (Modules 10-11) - Adam optimizer training language models
- **Data Processing** (Module 8) - Tokenization and sequence handling

## Quick Start

```python
# Train TinyGPT on Shakespeare
python train_shakespeare.py

# Generate text interactively
python generate.py

# See a simple working example
python train_simple.py
```

## Architecture

TinyGPT implements a GPT-style transformer:
- Character-level tokenization (simplest approach)
- Multi-head self-attention
- Positional encodings
- Layer normalization
- Autoregressive generation

## Files

- `tinygpt.py` - The complete model implementation
- `train_shakespeare.py` - Train on Shakespeare text
- `train_simple.py` - Simple pattern learning (for debugging)
- `generate.py` - Interactive text generation
- `utils.py` - Tokenization and data utilities

## Performance

With the components you built:
- Learns simple patterns (like "abcabc...") perfectly
- Generates character-level text after training
- Shows clear learning curves on small datasets

This proves your framework is complete and working!
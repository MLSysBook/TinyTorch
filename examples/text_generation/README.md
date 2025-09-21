# Text Generation with TinyGPT

Generate text using a transformer model built with YOUR TinyTorch!

## What This Demonstrates

- **Transformer architecture** - the foundation of ChatGPT
- **Multi-head attention** mechanisms you built
- **Autoregressive generation** - predicting one token at a time
- **The technology behind modern AI** - GPT, BERT, etc.

## How It Works

```
Input Tokens → Embeddings → Transformer Blocks → Output Logits → Next Token
                              ↑__________________|
                              (Autoregressive Loop)
```

## Running the Example

```bash
python generate.py
```

Expected output:
```
🤖 Text Generation with TinyGPT
======================================================================

🎯 Generating Python-like code:
--------------------------------------------------

Prompt: 'def'
Generated: 'def function_name ( self ) : return None'

Prompt: 'class'  
Generated: 'class MyClass : def __init__ ( self ) :'

Prompt: 'for i in'
Generated: 'for i in range ( 10 ) : print ( i )'

💡 What This Demonstrates:
✅ Transformer architecture with self-attention
✅ Multi-head attention you built from scratch
✅ Autoregressive text generation
✅ The foundation of ChatGPT and GitHub Copilot!

🎉 You've built the technology behind modern AI!
```

## Architecture

```
TinyGPT Model:
├── Token Embeddings (vocab_size → embed_dim)
├── Position Embeddings (max_length → embed_dim)
├── Transformer Blocks (×4)
│   ├── Multi-Head Attention
│   ├── Layer Normalization
│   └── Feed-Forward Network (MLP)
└── Output Projection (embed_dim → vocab_size)
```

## Key Components

- **Self-Attention**: Models relationships between all tokens
- **Position Embeddings**: Gives model sense of word order
- **Layer Normalization**: Stabilizes training
- **Autoregressive**: Generates one token at a time

## What You've Built

This is the same architecture as:
- GPT (Generative Pre-trained Transformer)
- ChatGPT (with more layers and parameters)
- GitHub Copilot (for code generation)
- BERT (with bidirectional attention)

## Requirements

- Module 07 (Attention) for multi-head attention
- Module 16 (TinyGPT) for complete transformer
- All TinyTorch modules exported

## Next Steps

The full Module 16 implementation will:
- Generate complete Python functions
- Work with natural language prompts
- Show beam search and sampling strategies
- Demonstrate real code generation!
# 🤖 TinyGPT (2018) - Transformer Architecture

## What This Demonstrates
Complete transformer language model using YOUR TinyTorch! The architecture that powers ChatGPT, built from YOUR implementations.

## Prerequisites
Complete ALL these TinyTorch modules:
- Module 02 (Tensor) - Data structures
- Module 03 (Activations) - ReLU
- Module 04 (Layers) - Linear layers
- Module 05 (Networks) - Module base class
- Module 06 (Autograd) - Backprop through attention
- Module 08 (Optimizers) - Adam optimizer
- Module 12 (Embeddings) - Token embeddings, positional encoding
- Module 13 (Attention) - Multi-head self-attention
- Module 14 (Transformers) - LayerNorm, TransformerBlock

## 🚀 Quick Start

```bash
# Run transformer demo
python train_gpt.py

# This is a validation demo - no real training data needed
```

## 📊 Dataset Information

### Demo Tokens Only
- **No Real Dataset**: Uses random tokens for architecture validation
- **Purpose**: Demonstrates the transformer works, not full training
- **No Download Required**: Synthetic data only

### Why No Real Dataset?
Full language model training requires:
- Large text corpora (GBs of data)
- Significant compute (GPU hours/days)
- This example validates YOUR architecture works

## 🏗️ Architecture

```
                        Output Logits (Vocabulary Predictions)
                                    ↑
                            Output Projection
                                    ↑
                              Layer Norm
                                    ↑
                    ╔══════════════════════════════╗
                    ║   Transformer Block × 4      ║
                    ║  ┌────────────────────┐     ║
                    ║  │    Layer Norm       │     ║
                    ║  │         ↑           │     ║
                    ║  │  Feed Forward Net   │     ║
                    ║  │         ↑           │     ║
                    ║  │    Layer Norm       │     ║
                    ║  │         ↑           │     ║
                    ║  │  Multi-Head Attention│     ║
                    ║  └────────────────────┘     ║
                    ╚══════════════════════════════╝
                                    ↑
                          Positional Encoding
                                    ↑
                           Token Embeddings
                                    ↑
                             Input Tokens
```

## 📈 Demo Configuration
- **Vocab Size**: 100 tokens (tiny for demo)
- **Embedding Dim**: 32
- **Attention Heads**: 4
- **Layers**: 2 transformer blocks
- **Context Length**: 16 tokens

## 💡 What Makes Transformers Special

### Self-Attention
Each token can "look at" all other tokens to understand context:
```
"The cat sat on the [MASK]"
         ↓
   Attention looks at all words
         ↓
   "mat" (understands context!)
```

### Key Innovations YOUR Implementation Shows
- **Attention**: Context-aware representations
- **Positional Encoding**: Order matters in sequences
- **Layer Norm**: Stable deep network training
- **Residual Connections**: Information flow through layers

## 📚 What You Learn
- Complete transformer architecture from scratch
- How attention creates contextual understanding
- YOUR implementations power modern LLMs
- Foundation for GPT, BERT, ChatGPT, etc.

## 🔬 Systems Insights
- **Memory**: O(n²) for attention (sequence length squared)
- **Compute**: Highly parallelizable (unlike RNNs)
- **Scaling**: Stack more layers for more capability
- **YOUR Version**: Core math is identical to production!

## 🚀 Real Training (Advanced)
To train a real language model:
1. Get text dataset (WikiText, BookCorpus, etc.)
2. Tokenize text into vocabulary
3. Create data loader for sequences
4. Train for many epochs (GPU recommended)
5. Generate text autoregressively

This demo validates the architecture - real training is a larger undertaking!
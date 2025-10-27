# 🤖 Milestone 05: Transformer Era (2017) - TinyGPT

**After completing Modules 10-13**, you can build complete transformer language models!

## 🎯 What You'll Build

A character-level transformer trained on Shakespeare's works - the classic "hello world" of language modeling!

### Shakespeare Text Generation
**File**: `vaswani_shakespeare.py`  
**Goal**: Build a transformer that generates Shakespeare-style text

```bash
python vaswani_shakespeare.py
```

**What it does**:
- Downloads Tiny Shakespeare dataset
- Trains character-level transformer (YOUR implementation!)
- Generates coherent Shakespeare-style text

**Demo**:
```
Prompt: 'To be or not to be,'
Output: 'To be or not to be, that is the question
         Whether tis nobler in the mind to suffer...'
```

---

## 🚀 Quick Start

### Prerequisites
Complete these TinyTorch modules:
- ✅ Module 10: Tokenization
- ✅ Module 11: Embeddings
- ✅ Module 12: Attention
- ✅ Module 13: Transformers

### Run the Example

```bash
# Train transformer on Shakespeare (15-20 min)
python vaswani_shakespeare.py
```

---

## 🎓 Learning Outcomes

After completing this milestone, you'll understand:

### Technical Mastery
- ✅ How tokenization bridges text and numbers
- ✅ How embeddings capture semantic meaning
- ✅ How attention enables context-aware processing
- ✅ How transformers generate sequences autoregressively

### Systems Insights
- ✅ Memory scaling: O(n²) attention complexity
- ✅ Compute trade-offs: model size vs inference speed
- ✅ Vocabulary design: characters vs subwords vs words
- ✅ Generation strategies: greedy vs sampling

### Real-World Connection
- ✅ **GitHub Copilot** = transformer on code
- ✅ **ChatGPT** = scaled-up version of your TinyGPT
- ✅ **GPT-4** = same architecture, 1000× more parameters
- ✅ YOU understand the math that powers modern AI!

---

## 🏗️ Architecture You Built

```
Input Tokens
    ↓
Token Embeddings (Module 11)
    ↓
Positional Encoding (Module 11)
    ↓
╔══════════════════════════════╗
║   Transformer Block × N      ║
║  ┌────────────────────┐     ║
║  │ Multi-Head Attention│ ←── Module 12
║  │         ↓           │     ║
║  │    Layer Norm       │ ←── Module 13
║  │         ↓           │     ║
║  │  Feed Forward Net   │ ←── Module 13
║  │         ↓           │     ║
║  │    Layer Norm       │ ←── Module 13
║  └────────────────────┘     ║
╚══════════════════════════════╝
    ↓
Output Projection
    ↓
Generated Text
```

---

## 🔬 Systems Analysis

### Memory Requirements
```python
TinyCoder (100K params):
  • Model weights: ~400KB
  • Activation memory: ~2MB per batch
  • Total: <10MB RAM

ChatGPT (175B params):
  • Model weights: ~350GB
  • Activation memory: ~100GB per batch
  • Total: ~500GB+ GPU RAM
```

### Computational Complexity
```python
For sequence length n:
  • Attention: O(n²) operations
  • Feed-forward: O(n) operations
  • Total: O(n²) dominated by attention

Why this matters:
  • 10 tokens: ~100 ops
  • 100 tokens: ~10,000 ops
  • 1000 tokens: ~1,000,000 ops
  
Quadratic scaling is why context length is expensive!
```

---

## 💡 Production Differences

### Your TinyGPT vs Production GPT

| Feature | Your TinyGPT | Production GPT-4 |
|---------|--------------|------------------|
| **Parameters** | ~100K | ~1.8 Trillion |
| **Layers** | 4 | ~120 |
| **Training Data** | ~50K tokens | ~13 Trillion tokens |
| **Training Time** | 2 minutes | Months on supercomputers |
| **Inference** | CPU, seconds | GPU clusters, <100ms |
| **Memory** | <10MB | ~500GB |
| **Architecture** | ✅ IDENTICAL | ✅ IDENTICAL |

**Key insight**: You built the SAME architecture. Production is just bigger & optimized!

---

## 🚧 Troubleshooting

### Import Errors
```bash
# Make sure modules are exported
cd modules/source/10_tokenization && tito export
cd ../11_embeddings && tito export
cd ../12_attention && tito export
cd ../13_transformers && tito export

# Rebuild package
cd ../../.. && tito nbdev build
```

### Slow Training
```python
# Reduce model size
model = TinyGPT(
    vocab_size=vocab_size,
    embed_dim=64,      # Smaller (was 128)
    num_heads=4,       # Fewer (was 8)
    num_layers=2,      # Fewer (was 4)
    max_length=64      # Shorter (was 128)
)
```

### Poor Generation Quality
- ✅ Train longer (more steps)
- ✅ Increase model size
- ✅ Use more training data
- ✅ Adjust temperature (0.5-1.0 for code, 0.7-1.2 for text)

---

## 🎉 Success Criteria

You've succeeded when:

✅ Model trains without errors  
✅ Loss decreases over training epochs  
✅ Generated Shakespeare text is coherent (even if not perfect)  
✅ You can generate text with custom prompts  

**Don't expect perfection!** Production models train for months on massive data. Your demo proves you understand the architecture!

---

## 📚 What's Next?

After mastering transformers, you can:

1. **Experiment**: Try different model sizes, hyperparameters
2. **Extend**: Add more sophisticated generation (beam search, top-k sampling)
3. **Scale**: Train on larger datasets for better quality
4. **Optimize**: Add KV caching (Module 14) for faster inference
5. **Benchmark**: Profile memory and compute (Module 15)
6. **Quantize**: Reduce model size (Module 17)

---

## 🏆 Achievement Unlocked

**You built the foundation of modern AI!**

The transformer architecture you implemented powers:
- ChatGPT, GPT-4 (OpenAI)
- Claude (Anthropic)
- LLaMA (Meta)
- PaLM (Google)
- GitHub Copilot
- And virtually every modern LLM!

**The only difference**: Scale. The architecture is what YOU built! 🎉

---

**Ready to generate some text?** Run `python vaswani_shakespeare.py`!
# 🔥 Module: Attention

## 📊 Module Info
- **Difficulty**: ⭐⭐⭐⭐ Advanced
- **Time Estimate**: 6-8 hours
- **Prerequisites**: Tensor, Activations, Layers, Networks modules
- **Next Steps**: CNN, Autograd, Training modules

Build attention mechanisms from scratch and understand the core technology powering modern AI systems like ChatGPT, BERT, and GPT-4. This module teaches you that attention is a powerful pattern-matching mechanism that allows models to dynamically focus on relevant parts of input sequences.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Master attention mechanisms**: Understand how Query, Key, Value projections enable dynamic focus
- **Implement self-attention**: Build the core component that powers transformer architectures
- **Create multi-head attention**: Combine multiple attention patterns for richer representations
- **Add positional encoding**: Give transformers the ability to understand sequence order
- **Build transformer blocks**: Compose attention with feed-forward networks and residual connections
- **Compare attention patterns**: Understand when to use self-attention vs cross-attention

## 🧠 Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement attention mechanisms and transformer components from mathematical foundations
2. **Use**: Apply attention to sequence tasks and visualize what the model "pays attention to"
3. **Reflect**: Compare attention's global perspective with CNN's local receptive fields

## 📚 What You'll Build

### Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    The fundamental attention operation:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = softmax(scores)
    return attention_weights @ V, attention_weights
```

### Multi-Head Attention
```python
class MultiHeadAttention:
    """
    Multiple attention heads capture different types of relationships:
    - Head 1: Syntactic relationships
    - Head 2: Semantic relationships  
    - Head 3: Long-range dependencies
    """
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = Dense(d_model, d_model)
        self.W_k = Dense(d_model, d_model)
        self.W_v = Dense(d_model, d_model)
        self.W_o = Dense(d_model, d_model)
```

### Transformer Block
```python
class TransformerBlock:
    """
    Complete transformer layer combining:
    1. Multi-head self-attention
    2. Residual connections
    3. Layer normalization
    4. Feed-forward network
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = Sequential([
            Dense(d_model, d_ff),
            ReLU(),
            Dense(d_ff, d_model)
        ])
```

## 🔬 Key Concepts

### Why Attention Matters
- **Global context**: Unlike CNNs, attention can connect any two positions directly
- **Dynamic weights**: Attention weights adapt based on input content, not fixed patterns
- **Interpretability**: You can visualize what the model pays attention to
- **Scalability**: Attention scales to very long sequences (with modifications)

### Attention vs Convolution
| Aspect | Convolution | Attention |
|--------|-------------|-----------|
| **Receptive field** | Local, grows with depth | Global from layer 1 |
| **Computation** | O(n) with kernel size | O(n²) with sequence length |
| **Inductive bias** | Spatial locality | Sequence relationships |
| **Best for** | Images, spatial data | Text, sequences |

### Real-World Applications
- **Language Models**: GPT, BERT, ChatGPT
- **Machine Translation**: Google Translate 
- **Vision Transformers**: Image classification without convolution
- **Multimodal AI**: CLIP, DALL-E combining text and images

## 🚀 From Attention to Modern AI

This module bridges classical ML and modern AI:

**Classical (pre-2017)**: RNNs + CNNs + LSTMs
**Modern (post-2017)**: Transformers + Attention + Self-Supervision

Understanding attention mechanisms gives you the foundation to understand:
- How ChatGPT generates text
- How BERT understands language
- How Vision Transformers work without convolution
- How DALL-E combines text and images

## 📈 Module Progression

```
Tensors → Activations → Layers → Networks → **ATTENTION** → CNN → Training
  ↑                                              ↑
Foundation                              Modern AI Core
```

After completing this module, you'll understand the mechanism that powers the AI revolution, making you ready to work with state-of-the-art models and architectures.

## 🎯 Success Criteria

You'll know you've mastered this module when you can:
- [ ] Explain why attention enables better long-range dependencies than RNNs
- [ ] Implement multi-head attention from scratch
- [ ] Visualize attention patterns and interpret what the model focuses on
- [ ] Compare computational complexity of attention vs convolution
- [ ] Build a complete transformer block with residual connections
- [ ] Understand why transformers have revolutionized NLP and computer vision 
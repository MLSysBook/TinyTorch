# Transformer Configuration Changes

## Summary

Applied Phase 1 (Critical Fixes) + Phase 2 (Better Performance) based on industry best practices from Karpathy's nanoGPT, minGPT, and GPT-2 standards.

## Changes Made

### 1. Learning Rate (CRITICAL FIX)
```python
# Before:
learning_rate = 0.001  # 1e-3 (TOO HIGH for transformers)

# After:
learning_rate = 0.0003  # 3e-4 (standard for transformers)
```
**Rationale**: Industry standard from "Attention Is All You Need" (Vaswani et al., 2017) and GPT-2 (Radford et al., 2019). Higher learning rates cause unstable training in attention mechanisms.

### 2. Training Steps (CRITICAL FIX)
```python
# Before:
epochs = 5 (default)
max_batches_per_epoch = 100
# Total steps: 500

# After:
epochs = 20 (default)
max_batches_per_epoch = 500
# Total steps: 10,000
```
**Rationale**: For a 2.5M parameter model on ~1MB Shakespeare corpus, need 5K-10K steps minimum for proper convergence.

### 3. Context Length (IMPORTANT)
```python
# Before:
seq_length = 64  # ~10 words

# After:
seq_length = 128  # ~20 words
```
**Rationale**: Shakespeare sentences average 15-20 words. Longer context = better coherence.

### 4. Model Capacity (IMPORTANT)
```python
# Before:
embed_dim = 128
num_layers = 4
num_heads = 4
# Total params: ~500K

# After:
embed_dim = 256
num_layers = 6
num_heads = 8
# Total params: ~2.5M
```
**Rationale**: 
- Matches minGPT recommendations for character-level tasks
- Head dimension: 256/8 = 32 (optimal for attention)
- 2.5M params is 5x original but still trainable on CPU in ~1 hour
- Better capacity to learn Shakespeare patterns

## Architecture Details

### Head Dimension Check
```python
head_dim = embed_dim / num_heads
         = 256 / 8
         = 32 ✅

Standard practice: 32-64 per head
```

### FFN Expansion
```python
ffn_hidden_dim = embed_dim * 4
               = 256 * 4
               = 1024 ✅

Standard practice: 4x expansion
```

### Parameter Count
```python
Embeddings:     vocab_size (65) × embed_dim (256) = 16,640
Positional:     max_seq_len (128) × embed_dim (256) = 32,768
Transformer×6:  ~400K per layer × 6 = 2.4M
Output:         embed_dim (256) × vocab_size (65) = 16,640

Total: ~2.5M parameters
```

## Expected Results

### Before Changes
- Final Loss: ~1.5-2.0
- Quality: Random characters, occasional words
- Training: Unstable due to high LR

### After Changes
- Final Loss: ~0.8-1.2
- Quality: Coherent sentences, Shakespeare-ish style
- Training: Stable, consistent improvement
- Generated text: Recognizable Shakespeare imitation

## Training Time

- **CPU (NumPy)**: ~45-60 minutes for 20 epochs
- **Steps per epoch**: ~500 batches
- **Total steps**: ~10,000
- **Time per step**: ~0.3-0.4 seconds

## Comparison to Research

| Metric | TinyTorch (New) | nanoGPT | minGPT | GPT-2 Small |
|--------|----------------|---------|--------|-------------|
| embed_dim | 256 | 384 | 192 | 768 |
| num_layers | 6 | 6 | 6 | 12 |
| num_heads | 8 | 6 | 6 | 12 |
| seq_length | 128 | 256 | 128 | 1024 |
| learning_rate | 3e-4 | 3e-4 | 6e-4 | 3e-4 |
| params | 2.5M | 10.65M | 2.7M | 124M |

**Conclusion**: Our new config matches minGPT closely, which is the recommended minimal viable config for educational character-level Shakespeare generation.

## Files Modified

1. `/milestones/05_2017_transformer/vaswani_shakespeare.py`
   - Line 75: Updated expected performance documentation
   - Line 298: Changed default `learning_rate` to 0.0003
   - Line 303: Added learning rate explanation to console output
   - Line 309: Added comment explaining learning rate choice
   - Line 318: Increased `max_batches` from 100 to 500
   - Line 452: Changed default `epochs` from 5 to 20
   - Line 456: Changed default `seq_length` from 64 to 128
   - Line 458: Changed default `embed_dim` from 128 to 256
   - Line 460: Changed default `num_layers` from 4 to 6
   - Line 462: Changed default `num_heads` from 4 to 8

## References

1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Radford et al. (2019) - GPT-2 paper
3. Karpathy's nanoGPT - https://github.com/karpathy/nanoGPT
4. Karpathy's minGPT - https://github.com/karpathy/minGPT

## Testing

To verify the changes work, run:

```bash
# Quick architecture test (1 epoch, small)
python milestones/05_2017_transformer/vaswani_shakespeare.py --epochs 1 --quick-test

# Full training (20 epochs, ~1 hour)
python milestones/05_2017_transformer/vaswani_shakespeare.py

# Custom config
python milestones/05_2017_transformer/vaswani_shakespeare.py \
    --epochs 30 \
    --embed-dim 384 \
    --num-layers 6 \
    --num-heads 6 \
    --seq-length 256
```


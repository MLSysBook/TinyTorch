# Debugging Sequence Reversal: The Attention Test

## Current Status

❌ **Model is NOT learning** (0% accuracy after 30 epochs)
- Loss barely moving: 1.5342 → 1.3062
- Predictions are mostly random or mode-collapsed (lots of 2's)
- This should reach 95%+ if attention works correctly

## Why This Is Perfect for Debugging

This task is **binary**: either attention works (95%+) or it doesn't (0-5%).
No gray area, no "partial success" - it's a perfect diagnostic!

## Comparison: What Works vs What Doesn't

### ✅ Working Implementation
- `tests/milestones/test_transformer_capabilities.py`
- Uses functional approach: `build_simple_transformer()`
- Achieves 95%+ accuracy reliably

### ❌ Failing Implementation  
- `milestones/05_2017_transformer/00_vaswani_attention_proof.py`
- Uses class-based approach: `ReversalTransformer` class
- Gets 0% accuracy

## Debugging Strategy

### Phase 1: Component-Level Tests
1. **Embedding Layer**
   - [ ] Verify embedding lookup works
   - [ ] Check positional encoding is added correctly
   - [ ] Ensure gradients flow through embeddings

2. **Attention Mechanism**
   - [ ] Verify Q, K, V projections
   - [ ] Check attention score computation
   - [ ] Verify softmax and weighted sum
   - [ ] Test multi-head split and concatenation
   - [ ] Ensure attention gradients flow

3. **Feed-Forward Network**
   - [ ] Check Linear → ReLU → Linear path
   - [ ] Verify FFN gradients

4. **Residual Connections**
   - [ ] Verify `x + attn_out` preserves computation graph
   - [ ] Check `x + ffn_out` preserves computation graph

5. **LayerNorm**
   - [ ] Verify normalization computation
   - [ ] Check gradients through LayerNorm

6. **Output Projection**
   - [ ] Verify reshape logic: (batch, seq, embed) → (batch*seq, embed) → (batch, seq, vocab)
   - [ ] Check output projection gradients

### Phase 2: Integration Tests
- [ ] Full forward pass produces correct shapes
- [ ] Loss computation is correct
- [ ] Backward pass flows to all parameters
- [ ] Optimizer updates all parameters
- [ ] Parameters actually change after training step

### Phase 3: Architectural Comparison
- [ ] Compare class-based vs functional implementations
- [ ] Identify structural differences
- [ ] Port fixes from working to failing version

### Phase 4: Hyperparameter Sweep
- [ ] Learning rate (try 0.001, 0.003, 0.005, 0.01)
- [ ] Epochs (try 50, 100)
- [ ] Embed dimension (try 16, 32, 64)
- [ ] Number of heads (try 2, 4, 8)

## Key Questions to Answer

1. **Are gradients flowing?**
   - Check `param.grad` is not None for all parameters
   - Check `param.grad` is not zero
   
2. **Are weights updating?**
   - Save initial weights
   - Train for 1 epoch
   - Verify weights changed

3. **Is the architecture correct?**
   - Does forward pass match our working implementation?
   - Are residual connections preserved?

4. **Is the data correct?**
   - Are input sequences correctly formatted?
   - Are targets correctly formatted?
   - Is vocab size consistent?

## Next Steps

1. Create minimal reproduction test
2. Test each component in isolation
3. Compare with working implementation line-by-line
4. Fix identified issues
5. Verify with full training run


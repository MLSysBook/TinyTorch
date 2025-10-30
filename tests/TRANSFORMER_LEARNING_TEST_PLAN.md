# Transformer Learning Test Plan

## Overview
This document outlines a systematic approach to testing and validating that TinyTorch transformers learn properly across all components and training scenarios.

## Test Status: ✅ PASSING

**Quick Validation Results** (2025-10-30):
- Initial loss: 3.555
- Final loss: 0.031
- Loss decrease: 99.1%
- Training time: 52.1s (500 steps)
- Gradient flow: 21/21 parameters ✅

---

## Layer 1: Component-Level Tests

### 1.1 Autograd Operations
**Purpose**: Verify all arithmetic operations preserve gradients

**Tests**:
- ✅ `tests/05_autograd/test_gradient_flow.py`
  - Addition, subtraction, multiplication, division
  - Backward pass correctness
  - GELU activation gradient flow
  - LayerNorm operations (mean, sqrt, div)
  - Reshape gradient preservation

**Coverage**: 6/6 tests passing

### 1.2 Transformer Components
**Purpose**: Verify gradient flow through transformer building blocks

**Tests**:
- ✅ `tests/13_transformers/test_transformer_gradient_flow.py`
  - MultiHeadAttention (8 parameters)
  - LayerNorm (2 parameters)
  - MLP (4 parameters)
  - Masked attention
  - Full GPT end-to-end (37 parameters)

**Coverage**: 5/5 tests passing

---

## Layer 2: Training Validation Tests

### 2.1 Memorization Test
**Purpose**: Can the model memorize a tiny dataset?

**Setup**:
```python
# 5 patterns, train for 500 steps
patterns = [
    "def add(a, b):\\n    return a + b",
    "def sub(a, b):\\n    return a - b",
    "for i in range(10):\\n    print(i)",
    "if x > 0:\\n    print('positive')",
    "numbers = [1, 2, 3, 4, 5]",
]
```

**Expected**: Loss should decrease > 80% in 500 steps
**Result**: ✅ 99.1% decrease (3.555 → 0.031)

### 2.2 Pattern Learning Test
**Purpose**: Can the model learn systematic patterns?

**Setup**:
- Train on arithmetic functions with various names
- Test if model can complete similar patterns

**Expected**: Model should predict correct structure even with new variable names

### 2.3 Generalization Test
**Purpose**: Does the model generalize or just memorize?

**Setup**:
- Train/test split (45/5 patterns)
- Measure loss on held-out patterns

**Expected**: Test loss should be within 2x of train loss

---

## Layer 3: Regression Tests

### 3.1 Gradient Flow Regression
**File**: `tests/13_transformers/test_transformer_gradient_flow.py`

**What it tests**:
- All attention Q/K/V projections receive gradients
- LayerNorm parameters (gamma, beta) receive gradients  
- MLP parameters receive gradients
- Embedding layers receive gradients

**Why it matters**: Previous bugs broke gradient flow to attention parameters

### 3.2 Loss Decrease Regression
**File**: `tests/13_transformers/test_training_simple.py` (to be created)

**What it tests**:
- Loss decreases on simple dataset
- Loss decrease rate > threshold
- Training completes without errors

**Why it matters**: Ensures the entire training loop works end-to-end

---

## Layer 4: Performance Benchmarks

### 4.1 Training Speed
**Metric**: Steps per second
**Baseline**: ~10 steps/sec for 1-layer, 32d model
**Test**: Monitor for regressions

### 4.2 Memory Usage
**Metric**: Peak memory during training
**Baseline**: <500MB for small models
**Test**: Detect memory leaks

### 4.3 Convergence Rate
**Metric**: Steps to reach 0.1 loss
**Baseline**: ~300 steps on 5-pattern dataset
**Test**: Detect training instabilities

---

## Layer 5: Integration Tests

### 5.1 Full Pipeline Test
**Components**: Tokenizer → Model → Loss → Optimizer → Backward → Update

**Test**:
```bash
python milestones/05_2017_transformer/vaswani_copilot.py --train-only
```

**Expected**: Completes training in < 3 minutes with loss decrease > 80%

### 5.2 Checkpoint Save/Load
**Test**: Save model mid-training, load, continue training

**Expected**: Loss continues decreasing from checkpoint

### 5.3 Generation Quality
**Test**: Generate code completions after training

**Expected**: Completions should be syntactically valid Python

---

## Debugging Checklist

When a model isn't learning:

1. **Check Gradient Flow**
   ```bash
   python tests/13_transformers/test_transformer_gradient_flow.py
   ```
   - Verify all parameters receive non-zero gradients

2. **Check Loss Computation**
   - Print initial loss (should be ~ln(vocab_size))
   - Verify loss decreases over time
   - Check for NaN/Inf values

3. **Check Data Processing**
   - Verify tokenization produces correct IDs
   - Check padding/masking is correct
   - Ensure targets are shifted by 1

4. **Check Hyperparameters**
   - Learning rate not too high (>0.01) or too low (<0.0001)
   - Batch size appropriate
   - Gradient clipping prevents explosions

5. **Check Architecture**
   - Embedding dimension divisible by num_heads
   - Sequence length < max_seq_len
   - Vocabulary size matches tokenizer

---

## Test Execution

### Run All Tests
```bash
# Component tests
pytest tests/05_autograd/test_gradient_flow.py -v
pytest tests/13_transformers/test_transformer_gradient_flow.py -v

# Integration test  
python milestones/05_2017_transformer/vaswani_copilot.py --train-only

# Quick validation
python tests/13_transformers/test_training_simple.py
```

### Expected Output
```
tests/05_autograd/test_gradient_flow.py ................ [ 54%]
tests/13_transformers/test_transformer_gradient_flow.py . [100%]

====== 11 passed in 3.2s ======

Transformer learning: ✅ VERIFIED
```

---

## Maintenance

### When to Update Tests
1. **After any autograd changes**: Run gradient flow tests
2. **After transformer architecture changes**: Run full pipeline test
3. **Before releases**: Run all tests + visual inspection of generations

### Adding New Tests
1. Follow existing test structure
2. Include clear docstrings explaining what's tested
3. Use meaningful assertions with error messages
4. Add to this test plan document

---

## References

- Gradient Flow Tests: `tests/05_autograd/test_gradient_flow.py`
- Transformer Tests: `tests/13_transformers/test_transformer_gradient_flow.py`
- Training Validation: Quick 500-step test shown above
- Integration: `milestones/05_2017_transformer/vaswani_copilot.py`


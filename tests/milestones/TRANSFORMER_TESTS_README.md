# Transformer Capability Tests - Quick Start

## What Are These Tests?

Progressive tests that verify your Transformer implementation actually works, from trivial to complex:

```
✅ Level 0: Copy Task          [10 sec]  - Sanity check
⭐ Level 1: Sequence Reversal  [30 sec]  - PROVES ATTENTION WORKS
✅ Level 2: Sequence Sorting   [1 min]   - Tests comparison
✅ Level 3: Modulus Arithmetic [2 min]   - Tests reasoning
```

## Quick Run

### Run All Tests (~4 minutes)
```bash
python3 tests/milestones/test_transformer_capabilities.py
```

### Run Individual Tests
```python
from tests.milestones.test_transformer_capabilities import *

# Quick sanity check (10 sec)
test_copy_task()

# Core attention test (30 sec) ⭐
test_sequence_reversal()

# Advanced tests
test_sequence_sorting()     # 1 min
test_modulus_arithmetic()   # 2 min
```

## The Key Test: Sequence Reversal ⭐

This is **THE** test that proves attention is working:

```
Task: [1, 2, 3, 4] → [4, 3, 2, 1]

Why it matters:
- Cannot be solved without attention
- Each output position must attend to a different input position
- From the original "Attention is All You Need" paper
- If this passes (95%+ accuracy), your Transformer works!
```

## What Each Test Validates

| Test | What It Checks | If It Fails |
|------|----------------|-------------|
| **Copy** | Basic forward pass | Check embeddings, output projection |
| **Reversal ⭐** | **Attention mechanism** | Check Q·K·V computation, positional encoding |
| **Sorting** | Multi-position comparison | Check attention patterns |
| **Modulus** | Symbolic reasoning | Check model capacity |

## Expected Output

```
======================================================================
TRANSFORMER CAPABILITY TESTS
======================================================================

Level 0: Copy Task (Sanity Check)
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 10s
  ✅ PASS: 100% accuracy

Level 1: Sequence Reversal ⭐ Core Attention Test  
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 30s
  ✅ PASS: 98% accuracy
  Example: [1,2,3,4,5] → [5,4,3,2,1] ✓

Level 2: Sequence Sorting
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 60s  
  ✅ PASS: 92% accuracy
  
Level 3: Modulus Arithmetic
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 120s
  ✅ PASS: 85% accuracy

======================================================================
SUMMARY
======================================================================
Total: 4/4 tests passed
✅ All transformer capability tests passed!
======================================================================
```

## Troubleshooting

### All Tests Fail
- Check: Basic gradient flow (`tests/milestones/test_learning_verification.py`)
- Verify: Autograd is enabled
- Check: Module exports are up to date (`tito export`)

### Copy Passes, Reversal Fails
- **Issue**: Attention mechanism broken
- Check: MultiHeadAttention implementation
- Check: Query·Key·Value computation
- Check: Positional encoding

### Reversal Passes, Sorting Fails
- **Not a problem!** Sorting is harder
- May need: More training epochs or larger model

### Only Getting ~50% on Reversal
- Check: Positional encoding is being added
- Check: Attention mask (should be None for these tests)
- Try: Increasing num_heads or embed_dim

## Design Document

See `TRANSFORMER_TEST_SUITE_DESIGN.md` for:
- Complete test hierarchy
- Educational rationale
- Implementation details
- Extension ideas (patterns, Q&A, etc.)

## When to Run These

### During Development
Run **sequence reversal** after implementing:
- MultiHeadAttention
- Positional Encoding  
- Transformer block

### Before Milestones
Run **all tests** to verify full Transformer stack before attempting:
- TinyTalks Q&A (milestone 05)
- TinyGPT (milestone 20)

### In CI/CD
Add to regression suite:
```bash
# Quick check (< 1 min)
python3 tests/milestones/test_transformer_capabilities.py --quick

# Full check (< 5 min)  
python3 tests/milestones/test_transformer_capabilities.py
```

## Success Criteria

**Minimum** (proves it works):
- ✅ Copy: 100%
- ⭐ Reversal: 95%

**Good** (ready for milestones):
- ✅ Copy: 100%
- ✅ Reversal: 95%
- ✅ Sorting: 85%

**Excellent** (strong implementation):
- All tests: 90%+

---

**Remember**: If **sequence reversal** passes, your Transformer attention mechanism is working correctly! 🎉


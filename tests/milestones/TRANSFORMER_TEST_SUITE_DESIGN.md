# Transformer Test Suite Design

A progression of tests from simple to complex, each validating different aspects of the Transformer architecture.

---

## 🎯 Test Hierarchy (Easy → Hard)

```
Level 0: Copy Task          [10 sec]  ← Sanity check (attention not needed)
Level 1: Sequence Reversal  [30 sec]  ← Requires attention to work ⭐ BEST
Level 2: Sequence Sorting   [1 min]   ← Requires comparison across positions
Level 3: Simple Arithmetic  [2 min]   ← Symbolic reasoning
Level 4: Pattern Completion [3 min]   ← Sequence understanding
Level 5: Character Q&A      [5 min]   ← Natural language (existing TinyTalks)
```

---

## Level 0: Copy Task ✅ **Sanity Check**

### Purpose
Verify the model can learn the identity function. If this fails, something is fundamentally broken.

### Task
```
Input:  [1, 2, 3, 4, 5]
Output: [1, 2, 3, 4, 5]
```

### Why This Test
- **Doesn't require attention** - each position only needs to copy itself
- If this fails, check: embeddings, positional encoding, output projection
- Should reach 100% accuracy in ~10 seconds

### Success Criteria
- ✅ 100% exact match accuracy
- ✅ All positions correct

### What It Tests
- Basic forward pass works
- Embeddings → Output projection pipeline
- Gradients flow through full stack

---

## Level 1: Sequence Reversal ⭐ **CORE TEST**

### Purpose
**Requires attention to work** - must look at all positions. This is the gold standard for verifying attention mechanisms.

### Task
```
Input:  [1, 2, 3, 4, 5]
Output: [5, 4, 3, 2, 1]
```

### Why This Test
- **Cannot be solved without attention** - each output position must attend to a different input position
- From the original "Attention is All You Need" paper
- Binary success: either works or doesn't
- Fast convergence (~30 seconds)

### Success Criteria
- ✅ 95%+ exact sequence match accuracy
- ✅ Shows attention is actually computing relationships

### What It Tests
- Multi-head attention mechanism
- Query-Key-Value computation
- Positional information preservation

### Variations
- **Easy**: Length 4-6, vocab size 10
- **Medium**: Length 8-12, vocab size 20
- **Hard**: Length 16-24, vocab size 50

---

## Level 2: Sequence Sorting

### Purpose
Tests comparison and ordering capabilities.

### Task
```
Input:  [3, 1, 4, 1, 5, 9, 2]
Output: [1, 1, 2, 3, 4, 5, 9]
```

### Why This Test
- Requires comparing elements across positions
- Tests if attention can learn comparison operators
- Natural progression from reversal

### Success Criteria
- ✅ 90%+ exact sequence match
- ✅ Monotonically increasing outputs

### What It Tests
- Multi-position reasoning
- Relative value comparison
- Complex attention patterns

---

## Level 3: Simple Arithmetic

### Purpose
Tests symbolic reasoning and operations.

### Task Types

**Addition**:
```
Input:  [2, +, 3, =]
Output: [5]
```

**Multiplication**:
```
Input:  [3, *, 4, =]
Output: [1, 2]  # "12" as two tokens
```

**Multi-step**:
```
Input:  [2, +, 3, *, 4, =]
Output: [1, 4]  # "(2+3)*4=20" → [2, 0]
```

### Success Criteria
- ✅ 85%+ correct answers on single operations
- ✅ 70%+ on two-step operations

### What It Tests
- Symbolic understanding (+ means addition)
- Sequential computation
- Generalization to unseen combinations

---

## Level 4: Pattern Completion

### Purpose
Tests sequence understanding and prediction.

### Task Types

**Arithmetic Sequences**:
```
Input:  [2, 4, 6, 8, ?]
Output: [10]
```

**Repeating Patterns**:
```
Input:  [1, 2, 3, 1, 2, 3, 1, ?]
Output: [2]
```

**Fibonacci**:
```
Input:  [1, 1, 2, 3, 5, 8, ?]
Output: [13]
```

### Success Criteria
- ✅ 80%+ on simple arithmetic progressions
- ✅ 70%+ on repeating patterns
- ✅ 60%+ on Fibonacci

### What It Tests
- Long-range dependencies
- Pattern recognition
- Inductive reasoning

---

## Level 5: Natural Language Tasks

### Purpose
Real-world language understanding (existing TinyTalks milestone).

### Task Types

**Character-level Q&A**:
```
Input:  "Q: What color is the sky? A: "
Output: "blue"
```

**Word-level Q&A** (if vocab expanded):
```
Input:  ["what", "color", "is", "sky", "?"]
Output: ["blue"]
```

### Success Criteria
- ✅ 70%+ accuracy on simple questions
- ✅ Coherent grammar
- ✅ Contextually appropriate answers

### What It Tests
- Language understanding
- Context retention
- Real-world applicability

---

## 🏗️ Recommended Test Suite Structure

### Quick Verification (< 2 minutes total)
```python
def test_transformer_quick():
    """Fast sanity checks"""
    test_copy_task()          # 10 sec - sanity check
    test_sequence_reversal()  # 30 sec - core attention test
    test_sequence_sorting()   # 60 sec - comparison test
```

### Comprehensive Verification (< 10 minutes total)
```python
def test_transformer_comprehensive():
    """Full capability testing"""
    test_copy_task()              # Sanity
    test_sequence_reversal()      # Core attention
    test_sequence_sorting()       # Comparison
    test_simple_arithmetic()      # Symbolic reasoning
    test_pattern_completion()     # Sequence understanding
    test_character_qa()           # Natural language
```

---

## 📊 Test Matrix

| Test | Time | Accuracy Target | Requires Attention | Difficulty |
|------|------|----------------|-------------------|------------|
| Copy | 10s | 100% | No | Trivial |
| Reversal | 30s | 95% | **Yes** ⭐ | Easy |
| Sorting | 1m | 90% | Yes | Medium |
| Arithmetic | 2m | 85% | Yes | Medium |
| Patterns | 3m | 70% | Yes | Hard |
| Q&A | 5m | 70% | Yes | Hard |

---

## 🎓 Educational Value

### For Students
Each test teaches something:
1. **Copy**: "My model can learn something"
2. **Reversal**: "Attention is actually working!"
3. **Sorting**: "It can compare things"
4. **Arithmetic**: "It understands symbols"
5. **Patterns**: "It can reason about sequences"
6. **Q&A**: "It can handle real language!"

### For Debugging
Progressive difficulty helps isolate issues:
- **Copy fails**: Basic architecture broken
- **Reversal fails**: Attention mechanism broken
- **Sorting fails**: Complex attention patterns not working
- **Arithmetic fails**: Symbolic reasoning not working
- **Patterns fails**: Long-range dependencies broken
- **Q&A fails**: Capacity or data issues

---

## 💻 Implementation Plan

### Phase 1: Core Verification (Recommended)
Create: `tests/milestones/test_transformer_capabilities.py`

```python
class TestTransformerCapabilities:
    def test_copy_task(self):
        """10 sec - Sanity check"""
        
    def test_sequence_reversal(self):
        """30 sec - Core attention test ⭐"""
        
    def test_sequence_sorting(self):
        """60 sec - Comparison test"""
```

### Phase 2: Extended Suite (Optional)
Add arithmetic, patterns, and Q&A to comprehensive suite.

---

## 🎯 Minimum Viable Test Suite

**For regression testing**, we need:
1. ✅ **Gradient flow test** (existing) - Ensures backward pass works
2. ✅ **Copy task** - Ensures forward pass works
3. ⭐ **Sequence reversal** - Ensures attention works

These 3 tests (< 1 minute total) give **high confidence** the Transformer is working correctly.

---

## 📝 Sample Test Output

```bash
$ python3 tests/milestones/test_transformer_capabilities.py

======================================================================
TRANSFORMER CAPABILITY TESTS
======================================================================

Test 1: Copy Task (Sanity Check)
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 10s
  ✅ PASS: 100% accuracy (50/50 sequences correct)
  
Test 2: Sequence Reversal (Core Attention Test) ⭐
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 30s
  ✅ PASS: 98% accuracy (49/50 sequences correct)
  Example: [1,2,3,4,5] → [5,4,3,2,1] ✓
  
Test 3: Sequence Sorting
  Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 60s
  ✅ PASS: 92% accuracy (46/50 sequences correct)
  Example: [3,1,4,2] → [1,2,3,4] ✓

======================================================================
Results: 3/3 tests passed
Total time: 100 seconds
✅ Transformer is working correctly!
======================================================================
```

---

## 🚀 Next Steps

1. **Implement Level 0-1** (Copy + Reversal) for quick verification
2. **Add to CI/CD** as fast regression tests
3. **Optionally add Level 2-3** for comprehensive testing
4. **Keep Level 5** (TinyTalks) as showcase demo

The **sequence reversal test** is the single best test to prove the Transformer architecture is working!


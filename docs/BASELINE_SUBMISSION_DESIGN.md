# Baseline & Submission Design: What Makes Sense

## User Concern

**Question**: For baseline and submitting, what makes sense? Worried that running everything can take a while.

## Current Design

### Baseline Benchmark (`tito benchmark baseline`)

**What it does**:
- Quick operations (tensor ops, matmul, forward pass)
- **Time**: ~1 second
- **Purpose**: Setup validation, environment check
- **Normalized**: SPEC-style to reference system

**Current Implementation**:
```python
# Quick operations only
- Tensor operations: ~0.8ms
- Matrix multiply: ~2.5ms  
- Forward pass: ~6.7ms
Total: ~10ms (normalized to reference)
```

### Milestones

**What they do**:
- Full ML workflows (training, evaluation)
- **Time**: Minutes (3-30 minutes per milestone)
- **Purpose**: Historical recreations, student validation
- **Requires**: Completed modules (student code)

## Recommendation: Keep Baseline Quick, Milestones Optional

### ✅ Baseline at Setup (Fast)

**Keep current approach**:
- ✅ Quick benchmark (~1 second)
- ✅ Validates environment works
- ✅ Normalized to reference system
- ✅ Good for "Hello World" moment
- ✅ Submit to community immediately

**Why this works**:
- Fast setup validation
- Doesn't require student code
- Meaningful baseline (normalized)
- Community submission ready

### ⚠️ Milestones Later (Optional)

**Run milestones as students complete modules**:
- ⚠️ Takes minutes (not seconds)
- ⚠️ Requires completed modules
- ⚠️ Optional for community submission
- ✅ Better for student validation

**Why milestones shouldn't be at setup**:
- Too slow (minutes vs seconds)
- Requires student code (doesn't exist yet)
- Better for progressive validation

## Submission Strategy

### Setup Phase: Baseline Only

**What to submit**:
- ✅ Baseline benchmark results (normalized)
- ✅ System info (country, institution, etc.)
- ✅ Reference implementation results

**Why**:
- Fast (1 second)
- Meaningful (normalized to reference)
- Works immediately (no student code needed)

### Later Phase: Milestones Optional

**What to submit (optional)**:
- ⚠️ Milestone results (as students complete modules)
- ⚠️ Student code performance vs reference
- ⚠️ Progress tracking

**Why optional**:
- Takes time (minutes per milestone)
- Requires completed modules
- Better for personal tracking than community

## Final Recommendation

**✅ Keep baseline quick** (current approach is correct):
- Fast setup validation (~1 second)
- Submit baseline to community
- Normalized to reference system

**✅ Milestones stay separate**:
- Run as students complete modules
- Optional for community submission
- Better for personal progress tracking

**Result**:
- Setup is fast (1 second baseline)
- Community gets meaningful data (normalized baseline)
- Students can optionally submit milestones later
- No time concerns at setup

## Implementation

**Current `tito benchmark baseline`**:
- ✅ Already fast (~1 second)
- ✅ Already normalized
- ✅ Already prompts for submission
- ✅ Perfect for setup phase

**No changes needed!** Current design is correct.


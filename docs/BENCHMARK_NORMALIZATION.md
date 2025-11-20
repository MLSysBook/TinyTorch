# Benchmark Normalization - SPEC-Style Reference System

## Overview

TinyTorch baseline benchmarks use **SPEC-style normalization** to ensure fair comparison across different hardware. Results are normalized to a reference system, making scores comparable regardless of your hardware.

## How It Works

### Reference System

**Reference Hardware:**
- CPU: Intel i5-8th generation
- RAM: 16GB
- Platform: Mid-range laptop

**Reference Times:**
- Tensor Operations: 0.8ms
- Matrix Multiply: 2.5ms
- Forward Pass: 6.7ms
- **Total: 10.0ms**

### Normalization Formula

**SPEC-style normalization:**
```
normalized_score = reference_time / actual_time
```

**Score Calculation:**
```
score = min(100, 100 * normalized_score)
```

### Examples

**Fast System (M3 Mac):**
- Actual time: 5.0ms
- Normalized: 10.0 / 5.0 = 2.0x
- Score: min(100, 100 * 2.0) = **100** (capped at 100)

**Reference System:**
- Actual time: 10.0ms
- Normalized: 10.0 / 10.0 = 1.0x
- Score: min(100, 100 * 1.0) = **100**

**Slower System (Older Laptop):**
- Actual time: 20.0ms
- Normalized: 10.0 / 20.0 = 0.5x
- Score: min(100, 100 * 0.5) = **50**

## Why Normalization Matters

### Without Normalization
- Fast hardware gets high scores unfairly
- Slow hardware gets low scores unfairly
- Can't compare optimization skill across systems

### With Normalization
- ✅ Scores are comparable across hardware
- ✅ Focus on optimization skill, not hardware
- ✅ Fair comparison (like SPEC benchmarks)

## Score Interpretation

**Score Range:**
- **100**: Reference system performance or better
- **80-99**: Slightly slower than reference
- **60-79**: Moderately slower than reference
- **40-59**: Significantly slower than reference
- **<40**: Very slow (may indicate setup issues)

**Normalized Multiplier:**
- **>1.0x**: Faster than reference system
- **1.0x**: Same as reference system
- **<1.0x**: Slower than reference system

## Technical Details

### Reference Times Selection

Reference times are based on:
- Mid-range consumer hardware (common student setup)
- Conservative estimates (most systems should meet or exceed)
- Real-world performance expectations

### Score Capping

Scores are capped at 100 to:
- Prevent unfair advantage for very fast hardware
- Keep focus on "setup validation" not "hardware competition"
- Maintain educational focus

### Future Adjustments

Reference times can be updated if:
- Hardware landscape changes significantly
- Better baseline data becomes available
- Community feedback suggests adjustment needed

## Comparison to SPEC

**Similarities:**
- ✅ Normalize to reference system
- ✅ Hardware-independent scores
- ✅ Fair comparison across systems

**Differences:**
- SPEC: Multiple benchmarks, complex scoring
- TinyTorch: Simple baseline validation, educational focus
- SPEC: Competitive benchmarking
- TinyTorch: Setup validation and learning

## Implementation

Reference times are defined in `tito/commands/benchmark.py`:

```python
def _get_reference_times(self) -> Dict[str, float]:
    """Get reference times for normalization (SPEC-style)."""
    return {
        "tensor_ops": 0.8,
        "matmul": 2.5,
        "forward_pass": 6.7,
        "total": 10.0
    }
```

Normalization happens automatically in `_run_baseline()`.

## Benefits

1. **Fair Comparison**: Scores mean the same thing on any hardware
2. **Educational Focus**: Emphasizes setup validation, not hardware
3. **Industry Standard**: Follows SPEC/MLPerf normalization principles
4. **Motivation**: Students can achieve good scores regardless of hardware


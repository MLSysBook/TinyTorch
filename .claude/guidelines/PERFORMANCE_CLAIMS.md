# TinyTorch Performance Claims Guidelines

## 🎯 Core Principle

**Only claim what you have measured and verified. Honesty builds trust.**

## ✅ Verified Performance Standards

### The Three-Step Verification

1. **Measure Baseline**
```python
# Random/untrained performance
random_model = create_untrained_model()
baseline_accuracy = evaluate(random_model, test_data)
print(f"Baseline: {baseline_accuracy:.1%}")  # Measured: 10%
```

2. **Measure Actual Performance**
```python
# Trained model performance
trained_model = train_model(epochs=15)
actual_accuracy = evaluate(trained_model, test_data)
print(f"Actual: {actual_accuracy:.1%}")  # Measured: 55%
```

3. **Calculate Real Improvement**
```python
improvement = actual_accuracy / baseline_accuracy
print(f"Improvement: {improvement:.1f}×")  # Measured: 5.5×
```

### Reporting Requirements

**ALWAYS include:**
- Exact accuracy percentage
- Training time
- Hardware used
- Number of epochs
- Dataset size

**Example:**
```markdown
✅ GOOD:
- Accuracy: 55% on CIFAR-10 test set
- Training time: 2 minutes on M1 MacBook
- Epochs: 15
- Batch size: 64

❌ BAD:
- "State-of-the-art performance"
- "Can achieve 60-70% with optimization"
- "Approaches CNN-level accuracy"
```

## 📊 The CIFAR-10 Lesson

### What We Claimed vs Reality

**Initial Claims (unverified):**
- "60-70% accuracy achievable with optimization"
- "Advanced techniques push beyond baseline"
- "Sophisticated MLPs rival simple CNNs"

**Actual Results (verified):**
- Baseline: 51-55% consistently
- With optimization attempts: Still ~55%
- Deep networks: Too slow, no improvement
- **Honest conclusion: MLPs achieve 55% reliably**

### The Right Response

When results don't match expectations:

✅ **CORRECT Approach:**
- Test thoroughly
- Report actual results
- Update documentation
- Explain limitations

❌ **WRONG Approach:**
- Keep unverified claims
- Hide negative results
- Blame implementation
- Make excuses

## 🔬 Performance Testing Protocol

### Minimum Testing Requirements

```python
def verify_performance_claim():
    """
    Every performance claim must pass this verification.
    """
    results = []
    
    # Run multiple trials
    for trial in range(3):
        model = create_model()
        accuracy = train_and_evaluate(model)
        results.append(accuracy)
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    # Report with confidence intervals
    print(f"Performance: {mean_acc:.1%} ± {std_acc:.1%}")
    
    # Only claim if consistent
    if std_acc > 0.02:  # >2% variance
        print("⚠️ High variance - need more testing")
        return False
    
    return True
```

### Time Complexity Reporting

```python
# ✅ GOOD: Measured complexity
def measure_scalability():
    sizes = [100, 1000, 10000]
    times = []
    
    for size in sizes:
        data = create_data(size)
        start = time.time()
        process(data)
        times.append(time.time() - start)
    
    # Analyze scaling
    print("Scaling behavior:")
    for size, time in zip(sizes, times):
        print(f"  n={size}: {time:.2f}s")
    
    # Determine complexity
    if times[2] / times[1] > 90:  # 10x data → 100x time
        print("Complexity: O(n²)")

# ❌ BAD: Theoretical claims
def theoretical_complexity():
    print("Should be O(n log n)")  # Not measured
```

## 📝 Documentation Standards

### Performance Tables

```markdown
✅ GOOD Table:

| Model | Dataset | Accuracy | Time | Hardware |
|-------|---------|----------|------|----------|
| MLP-4-layer | CIFAR-10 | 55% | 2 min | M1 CPU |
| Random baseline | CIFAR-10 | 10% | 0 sec | N/A |
| MLP-4-layer | MNIST | 98% | 30 sec | M1 CPU |

❌ BAD Table:

| Model | Performance |
|-------|------------|
| Our MLP | State-of-the-art |
| With optimization | Up to 70% |
| Best case | Rivals CNNs |
```

### Comparison Claims

```markdown
✅ GOOD Comparisons:
- "5.5× better than random baseline (10% → 55%)"
- "Matches typical educational MLP benchmarks"
- "20% below simple CNN performance"

❌ BAD Comparisons:
- "Competitive with modern architectures"
- "Approaching state-of-the-art"
- "Best-in-class for educational frameworks"
```

## ⚠️ Red Flags to Avoid

### Weasel Words
- "Can achieve..." (but didn't)
- "Up to..." (theoretical maximum)
- "Potentially..." (unverified)
- "Should be able to..." (untested)
- "With proper tuning..." (hand-waving)

### Unverified Optimizations
- "With these 10 techniques..." (didn't implement)
- "Research shows..." (not our research)
- "In theory..." (not in practice)
- "Could reach..." (but didn't)

### Vague Metrics
- "Good performance"
- "Impressive results"
- "Significant improvement"
- "Fast training"

## 🎯 The Integrity Test

Before making any performance claim, ask:

1. **Did I measure this myself?**
   - If no → Don't claim it

2. **Can someone reproduce this?**
   - If no → Don't publish it

3. **Is this the typical case?**
   - If no → Note it's exceptional

4. **Would I bet money on this?**
   - If no → Reconsider the claim

## 📌 Remember

> "It's better to under-promise and over-deliver than the opposite."

**Trust is earned through:**
- Honest reporting
- Reproducible results  
- Clear limitations
- Verified claims

**Trust is lost through:**
- Exaggerated claims
- Unverified results
- Hidden failures
- Theoretical promises

## 🏆 Good Examples from TinyTorch

### CIFAR-10 Cleanup
**Before:** "60-70% achievable with optimization"
**After:** "55% verified performance"
**Result:** Honest, trustworthy documentation

### XOR Network
**Claim:** "100% accuracy on XOR"
**Verified:** Yes, consistently achieves 100%
**Result:** Credible claim that builds trust
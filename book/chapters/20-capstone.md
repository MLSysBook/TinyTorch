---
title: "TinyMLPerf Competition - Your Capstone Challenge"
description: "Apply all optimizations in a standardized MLPerf-inspired competition"
difficulty: 5
time_estimate: "10-20 hours"
prerequisites: ["All modules 01-19"]
next_steps: []
learning_objectives:
  - "Apply all Performance Tier optimizations to a standardized benchmark"
  - "Implement either Closed Division (optimize given model) or Open Division (innovate architecture)"
  - "Generate validated submission with normalized metrics"
  - "Demonstrate complete ML systems engineering skills"
  - "Compete fairly across different hardware platforms"
---

# 20. TinyMLPerf Competition

**🏆 CAPSTONE** | Difficulty: ⭐⭐⭐⭐⭐ (5/5 - Ninja Level) | Time: 10-20 hours

## Overview

Your capstone challenge: optimize a CIFAR-10 CNN using everything you've learned. Choose between Closed Division (optimize our CNN) or Open Division (design your own). Compete on a level playing field with normalized metrics that account for hardware differences.

## Learning Objectives

By completing this capstone, you will be able to:

1. **Apply all Performance Tier optimizations** (caching, profiling, acceleration, quantization, compression, benchmarking)
2. **Implement either Closed Division** (optimize given CNN; pure optimization challenge) or **Open Division** (design novel architecture; innovation challenge)
3. **Generate validated submission** with standardized metrics, honor code attestation, and GitHub repo
4. **Demonstrate complete ML systems skills** from implementation through optimization to deployment
5. **Compete fairly** using normalized metrics (speedup, compression ratio) that work across hardware

## Why This Matters

### Production Context

This competition simulates real ML systems engineering:

- **MLPerf** is the industry standard for ML benchmarking; this follows the same principles
- **Production optimization** requires choosing what to optimize and measuring impact
- **Hardware diversity** in production demands normalized comparison metrics
- **Documentation** of optimization choices matters for team collaboration

### Competition Philosophy

This capstone teaches:
- **Optimization discipline**: Profile first, optimize bottlenecks, measure impact
- **Trade-off analysis**: Speed vs accuracy vs memory - what matters for your use case?
- **Fair comparison**: Normalized metrics ensure your M1 MacBook competes fairly with AWS GPU
- **Real constraints**: Must maintain >70% accuracy; actual production requirement

## Competition Structure

### Two Tracks

**Closed Division - Optimization Challenge**
- **Task**: Optimize provided CNN architecture
- **Rules**: Cannot change model architecture, training, or dataset
- **Focus**: Pure systems optimization (caching, quantization, pruning, acceleration)
- **Goal**: Maximum speedup with minimal accuracy loss

**Open Division - Innovation Challenge**  
- **Task**: Design your own architecture
- **Rules**: Can change anything (architecture, training, data augmentation)
- **Focus**: Novel approaches, architectural innovations, creative solutions
- **Goal**: Best efficiency score balancing speed, size, and accuracy

### Metrics (Both Divisions)

**Normalized for Fair Hardware Comparison:**
- **Speedup**: your_inference_time / baseline_inference_time (on YOUR hardware)
- **Compression Ratio**: baseline_params / your_params
- **Accuracy Delta**: your_accuracy - baseline_accuracy (must be ≥ -5%)
- **Efficiency Score**: (speedup × compression) / (1 + |accuracy_loss|)

## Implementation Guide

### Step 1: Validate Your Installation

```bash
tito setup --validate
# Ensures all modules work before starting
```

### Step 2: Generate Baseline

```python
from tinytorch.competition import generate_baseline

# This runs the unoptimized CNN and records your baseline
baseline = generate_baseline()
# Saves: baseline_submission.json with your hardware specs
```

### Step 3: Choose Your Track

**Option A: Closed Division (Recommended for first-time)**
```python
from tinytorch.competition import optimize_closed_division

# Optimize the provided CNN
optimized_model = optimize_closed_division(
    baseline_model,
    techniques=['kvcaching', 'quantization', 'pruning']
)
```

**Option B: Open Division (For advanced students)**
```python
from tinytorch.competition import design_open_division

# Design your own architecture
my_model = MyCustomCNN(...)
# Train it
trained_model = train(my_model, train_loader)
```

### Step 4: Generate Submission

```python
from tinytorch.competition import generate_submission

submission = generate_submission(
    model=optimized_model,
    division='closed',  # or 'open'
    github_repo='https://github.com/yourname/tinytorch-submission',
    techniques_used=['INT8 quantization', '90% magnitude pruning', 'KV caching'],
    athlete_name='Your Name'
)

# This creates: submission.json with all required fields
```

### Step 5: Validate and Submit

```bash
# Local validation
tito submit --file submission.json --validate-only

# Official submission (when ready)
tito submit --file submission.json
```

## Submission Requirements

### Required Fields

- **division**: 'closed' or 'open'
- **athlete_name**: Your name
- **github_repo**: Link to your code (public or private with access)
- **baseline_metrics**: From Step 2
- **optimized_metrics**: From Step 4
- **normalized_scores**: Speedup, compression, accuracy delta
- **techniques_used**: List of optimizations applied
- **honor_code**: "I certify that this submission follows the rules" 
- **hardware**: CPU/GPU specs, RAM (for reference, not ranking)
- **tinytorch_version**: Automatically captured
- **timestamp**: Automatically captured

### Validation Checks

The submission system performs sanity checks:
- ✅ Speedup between 0.5× and 100× (realistic range)
- ✅ Compression between 1× and 100× (realistic range)
- ✅ Accuracy drop < 10% (must maintain reasonable performance)
- ✅ GitHub repo exists and contains code
- ✅ Techniques used are documented
- ✅ No training modifications in Closed Division

### Honor Code

This is an honor-based system with light validation:
- We trust you followed the rules
- Automated checks catch accidental errors
- If something seems wrong, we may ask for clarification
- GitHub repo allows others to learn from your work

## Example Optimizations (Closed Division)

**Beginner**: 
- Apply INT8 quantization: ~4× compression, ~2× speedup
- Result: Speedup=2×, Compression=4×, Efficiency≈8

**Intermediate**:
- Quantization + 50% pruning: ~8× compression, ~3× speedup
- Result: Speedup=3×, Compression=8×, Efficiency≈24

**Advanced**:
- Quantization + 90% pruning + operator fusion: ~40× compression, ~5× speedup
- Result: Speedup=5×, Compression=40×, Efficiency≈200

## Testing

```bash
# Run everything end-to-end
cd modules/source/20_competition
python competition_dev.py

# Export and test
tito export 20_competition
tito test 20_competition

# Generate baseline
python -c "from tinytorch.competition import generate_baseline; generate_baseline()"

# Validate submission
tito submit --file submission.json --validate-only
```

## Where This Code Lives

```
tinytorch/
├── competition/
│   ├── baseline.py             # Baseline model
│   ├── submission.py           # Submission generation
│   └── validate.py             # Validation logic
└── __init__.py

Generated files:
- baseline_submission.json      # Your baseline metrics
- submission.json               # Your final submission
```

## Systems Thinking Questions

1. **Optimization Priority**: You have limited time. Profile shows attention=40%, FFN=35%, embedding=15%, other=10%. Where do you start and why?

2. **Accuracy Trade-off**: Closed Division allows up to 5% accuracy loss. How do you decide what's acceptable? What if you could get 10× speedup for 6% loss?

3. **Hardware Fairness**: Student A has M1 Max, Student B has i5 laptop. Normalized metrics show both achieved 3× speedup. Who optimized better?

4. **Open Division Strategy**: You could design a tiny 100K-param model (fast but potentially less accurate) or optimize a 1M-param model. What's your strategy?

5. **Verification Challenge**: How would you verify submissions without running everyone's code? What checks are sufficient?

## Real-World Connections

### MLPerf

This competition mirrors MLPerf principles:
- Closed Division = MLPerf Closed (fixed model/training)
- Open Division = MLPerf Open (anything goes)
- Normalized metrics for fair hardware comparison
- Honor-based with validation checks

### Industry Applications

**Model Deployment Engineer** (your future job):
- Given: Slow model from research team
- Goal: Deploy at production scale
- Constraints: Latency SLA, accuracy requirements, hardware budget
- Skills: Profiling, optimization, trade-off analysis (this capstone!)

**ML Competition Platforms**: Kaggle, DrivenData use similar structures
- Leaderboards drive innovation
- Standardized metrics ensure fairness
- Open sharing advances the field

## What's Next?

**You've completed TinyTorch!** You've built:
- **Foundation Tier**: All ML building blocks from scratch
- **Intelligence Tier**: Vision and language systems
- **Performance Tier**: Production optimization techniques
- **Capstone**: Real-world ML systems engineering

**Where to go from here:**
- Deploy your optimized model to production
- Contribute to open-source ML frameworks
- Join ML systems research or engineering teams
- Build the next generation of ML infrastructure

---

**Ready for your capstone challenge?** Open `modules/source/20_competition/competition_dev.py` and start optimizing!

**Compete. Optimize. Dominate.** 🏆

# ⊕ XOR Problem (1969) - Minsky & Papert

## Historical Significance

In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," mathematically proving that single-layer perceptrons **cannot** solve the XOR problem. This revelation killed neural network research funding for over a decade - the infamous "AI Winter."

In 1986, Rumelhart, Hinton, and Williams published the backpropagation algorithm for multi-layer networks, and XOR became trivial. This milestone recreates both the crisis and the solution using YOUR TinyTorch!

## Prerequisites

Complete these TinyTorch modules first:

**For Part 1 (xor_crisis.py):**
- Module 01 (Tensor)
- Module 02 (Activations) 
- Module 03 (Layers)
- Module 04 (Losses)
- Module 05 (Autograd)
- Module 06 (Optimizers)

**For Part 2 (xor_solved.py):**
- All of the above ✓

## Quick Start

### Part 1: The Crisis (1969)
Watch a single-layer perceptron **fail** to learn XOR:

```bash
python milestones/02_xor_crisis_1969/xor_crisis.py
```

**Expected:** ~50% accuracy (random guessing) - proves Minsky was right!

### Part 2: The Solution (1986)
Watch a multi-layer network **solve** the "impossible" problem:

```bash
python milestones/02_xor_crisis_1969/xor_solved.py
```

**Expected:** 75%+ accuracy (problem solved!) - proves hidden layers work!

## The XOR Problem

### What is XOR?

XOR (Exclusive OR) outputs 1 when inputs **differ**, 0 when they're the **same**:

```
┌────┬────┬─────┐
│ x₁ │ x₂ │ XOR │
├────┼────┼─────┤
│ 0  │ 0  │  0  │ ← same
│ 0  │ 1  │  1  │ ← different
│ 1  │ 0  │  1  │ ← different
│ 1  │ 1  │  0  │ ← same
└────┴────┴─────┘
```

### Why It's Impossible for Single Layers

The problem is **non-linearly separable** - no single straight line can separate the points:

```
Visual Representation:

1 │ ○ (0,1)    ● (1,1)      Try drawing a line:
  │   [1]       [0]          ANY line fails!
  │
0 │ ● (0,0)    ○ (1,0)       
  │   [0]       [1]         
  └─────────────────
    0          1
```

This fundamental limitation ended the first era of neural networks.

## The Solution

Hidden layers create a **new feature space** where XOR becomes linearly separable!

### Original 1986 Architecture
```
Input (2) → Hidden (2) + Sigmoid → Output (1) + Sigmoid

Total: Only 9 parameters!
```

The 2 hidden units learn:
- `h₁ ≈ x₁ AND NOT x₂`
- `h₂ ≈ x₂ AND NOT x₁`
- `output ≈ h₁ OR h₂` = XOR

### Our Implementation
```
Input (2) → Hidden (4-8) + ReLU → Output (1) + Sigmoid

Modern activation, slightly larger for robustness
```

## Expected Results

### Part 1: The Crisis
- **Accuracy:** ~50% (random guessing)
- **Loss:** Stuck around 0.69 (not decreasing)
- **Weights:** Don't converge to meaningful values
- **Conclusion:** Single-layer perceptrons **cannot** solve XOR

### Part 2: The Solution
- **Accuracy:** 75-100% (problem solved!)
- **Loss:** Decreases to ~0.35 or lower
- **Weights:** Learn meaningful features
- **Conclusion:** Multi-layer networks **can** solve XOR

## What You Learn

1. **Why depth matters** - Hidden layers enable non-linear functions
2. **Historical context** - The XOR crisis that stopped AI research
3. **The breakthrough** - Backpropagation through hidden layers
4. **Your autograd works!** - Multi-layer gradients flow correctly

## Files in This Milestone

- `xor_crisis.py` - Single-layer perceptron **failing** on XOR (1969 crisis)
- `xor_solved.py` - Multi-layer network **solving** XOR (1986 breakthrough)
- `README.md` - This file

## Historical Timeline

- **1969:** Minsky & Papert prove single-layer networks can't solve XOR
- **1970-1986:** AI Winter - 17 years of minimal neural network research
- **1986:** Rumelhart, Hinton, Williams publish backpropagation for multi-layer nets
- **1986+:** AI Renaissance begins
- **TODAY:** Deep learning powers GPT, AlphaGo, autonomous vehicles, etc.

## Next Steps

After completing this milestone:

- **Milestone 03:** MLP Revival (1986) - Train deeper networks on real data
- **Module 08:** DataLoaders for batch processing
- **Module 09:** CNNs for image recognition

Every modern AI architecture builds on what you just learned - hidden layers + backpropagation!
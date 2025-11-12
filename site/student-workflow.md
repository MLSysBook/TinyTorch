# Student Workflow

This guide explains the actual day-to-day workflow for building your ML framework with TinyTorch.

## The Core Workflow

TinyTorch follows a simple three-step cycle:

```{mermaid}
graph LR
    A[Edit Modules<br/>modules/NN_name/] --> B[Export to Package<br/>tito module complete N]
    B --> C[Validate with Milestones<br/>Run milestone scripts]
    C --> A

    style A fill:#e3f2fd
    style B fill:#f0fdf4
    style C fill:#fef3c7
```

### Step 1: Edit Modules

Work on module notebooks in `modules/`:

```bash
# Example: Working on Module 03 (Layers)
cd modules/03_layers
jupyter lab layers_dev.ipynb
```

Each module is a Jupyter notebook that you edit interactively. You'll:
- Implement the required functionality
- Add docstrings and comments
- Run and test your code inline
- See immediate feedback

### Step 2: Export to Package

Once your module implementation is complete, export it to the main TinyTorch package:

```bash
tito module complete MODULE_NUMBER
```

This command:
- Converts your source files to the `tinytorch/` package
- Validates NBGrader metadata
- Makes your implementation available for import

**Example:**
```bash
tito module complete 03  # Export Module 03 (Layers)
```

After export, your code is importable:
```python
from tinytorch.layers import Linear  # YOUR implementation!
```

### Step 3: Validate with Milestones

Run milestone scripts to prove your implementation works:

```bash
cd milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Uses YOUR Tensor (M01)
python 02_rosenblatt_trained.py  # Uses YOUR layers (M01-M07)
```

Each milestone has a README explaining:
- Required modules
- Historical context
- Expected results
- What you're learning

See [Milestones Guide](chapters/milestones.md) for the full progression.

## Module Progression

TinyTorch has 20 modules organized in three tiers:

### Foundation (Modules 01-07)
Core ML infrastructure - tensors, autograd, training loops

**Milestones unlocked:**
- M01: Perceptron (after Module 07)
- M02: XOR Crisis (after Module 07)

### Architecture (Modules 08-13)
Neural network architectures - data loading, CNNs, transformers

**Milestones unlocked:**
- M03: MLPs (after Module 08)
- M04: CNNs (after Module 09)
- M05: Transformers (after Module 13)

### Optimization (Modules 14-19)
Production optimization - profiling, quantization, benchmarking

**Milestones unlocked:**
- M06: MLPerf (after Module 18)

### Capstone Competition (Module 20)
Apply all optimizations in the MLPerf® Edu Competition

## Typical Development Session

Here's what a typical session looks like:

```bash
# 1. Work on a module
cd modules/05_autograd
jupyter lab autograd_dev.ipynb
# Edit your implementation interactively

# 2. Export when ready
tito module complete 05

# 3. Validate with existing milestones
cd ../milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Should still work!

# 4. Continue to next module or milestone
```

## TITO Commands Reference

The most important commands you'll use:

```bash
# Export module to package
tito module complete MODULE_NUMBER

# Check module status (optional capability tracking)
tito checkpoint status

# System information
tito system info
```

For complete command documentation, see [TITO Essentials](tito-essentials.md).

## Checkpoint System (Optional)

TinyTorch includes an optional checkpoint system for tracking progress:

```bash
tito checkpoint status  # View completion tracking
```

This is helpful for self-assessment but **not required** for the core workflow. The essential cycle remains: edit → export → validate.

## Instructor Integration (Coming Soon)

TinyTorch supports NBGrader for classroom use. Documentation for instructors using the autograding features will be available in future releases.

For now, focus on the student workflow: building your implementations and validating them with milestones.

## What's Next?

1. **Start with Module 01**: See [Getting Started](intro.md)
2. **Follow the progression**: Each module builds on previous ones
3. **Run milestones**: Prove your implementations work
4. **Build intuition**: Understand ML systems from first principles

The goal isn't just to write code - it's to **understand** how modern ML frameworks work by building one yourself.

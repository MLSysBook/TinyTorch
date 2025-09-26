# ⊕ XOR Problem (1969) - Minsky & Papert

## What This Demonstrates
The "impossible" problem that killed neural networks for a decade! Shows why hidden layers are essential for non-linear problems.

## Prerequisites
Complete these TinyTorch modules first:
- Module 02 (Tensor) - Data structures
- Module 03 (Activations) - ReLU activation
- Module 04 (Layers) - Linear layers
- Module 06 (Autograd) - Backward propagation

## 🚀 Quick Start

```bash
# Solve XOR with hidden layers
python minsky_xor_problem.py

# Test architecture only
python minsky_xor_problem.py --test-only

# More training epochs for better accuracy
python minsky_xor_problem.py --epochs 2000
```

## 📊 Dataset Information

### XOR Truth Table
```
x1 | x2 | XOR
---|----|----- 
0  | 0  | 0 (same → 0)
0  | 1  | 1 (diff → 1)
1  | 0  | 1 (diff → 1)
1  | 1  | 0 (same → 0)
```

### Generated XOR Data
- **Size**: 1,000 samples with slight noise
- **Property**: NOT linearly separable
- **No Download Required**: Generated on-the-fly

## 🏗️ Architecture
```
Input (2) → Linear (2→4) → ReLU → Linear (4→1) → Sigmoid → Output
              ↑                      ↑
         Hidden Layer!          Output Layer
```

The hidden layer is the KEY - it learns features that make XOR separable!

## 📈 Expected Results
- **Training Time**: ~1 minute
- **Accuracy**: 90%+ (non-linear problem solved!)
- **Parameters**: 17 (compared to perceptron's 3)

## 💡 Historical Significance
- **1969**: Minsky proved single-layer perceptrons can't solve XOR
- **AI Winter**: Neural network research stopped for a decade
- **1986**: Backprop + hidden layers solved it (what YOU built!)
- **Insight**: Depth enables non-linear decision boundaries

## 🎨 Why XOR is Special
```
Single Layer Fails:          Multi-Layer Succeeds:
   
1 │ ○      ●                Hidden units learn:
  │  ╲                       - Unit 1: x1 AND NOT x2
  │   ╲ (No line works!)     - Unit 2: x2 AND NOT x1
0 │ ●  ╲   ○                Then combine: Unit1 OR Unit2
  └───────────
    0      1
```

## 🔧 Command Line Options
- `--test-only`: Test architecture without training
- `--epochs N`: Training epochs (default: 1000)
- `--visualize`: Show XOR visualization (default: True)

## 📚 What You Learn
- Why neural networks need hidden layers
- How non-linearity (ReLU) enables complex functions
- YOUR autograd handles multi-layer backprop
- Foundation principle for all deep learning
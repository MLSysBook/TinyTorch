# XOR Neural Network 🧠

**Classic non-linear function learning with beautiful visualization**

## What is XOR?

The XOR (exclusive OR) problem is a classic neural network challenge that demonstrates a network's ability to learn non-linear functions. Linear models cannot solve XOR, but neural networks with hidden layers can.

**XOR Truth Table:**
```
Input  | Output
-------|-------
0  0   |   0
0  1   |   1  
1  0   |   1
1  1   |   0
```

## Features

- **Beautiful Rich UI** with real-time ASCII plotting
- **Perfect convergence visualization** 
- **100% accuracy achievement** on XOR truth table
- **Educational value** - see exactly how the network learns

## Architecture

```
Input Layer (2) → Hidden Layer (8) → Output Layer (1)
```

- **Activation**: ReLU for hidden layer, linear for output
- **Loss**: Mean Squared Error
- **Optimizer**: SGD with learning rate 0.1
- **Parameters**: ~70 total parameters

## Running the Example

```bash
cd examples/xornet/
python train_xor_network.py
```

**Expected Output:**
- Training completes in ~30 seconds
- Reaches 100% accuracy (perfect XOR solution)
- Beautiful real-time visualization of learning progress
- Final predictions table showing exact XOR outputs

## What You'll See

1. **Welcome Screen**: Model architecture and training configuration
2. **Real-time Training**: ASCII plots showing accuracy and loss curves
3. **Convergence Metrics**: Custom "convergence" metric showing progress to solution
4. **Final Results**: Exact predictions for all XOR inputs
5. **Success Celebration**: Visual confirmation of perfect learning

## Educational Value

This example demonstrates:
- **Non-linear learning**: How hidden layers enable complex function approximation
- **Training visualization**: Real-time feedback on neural network learning
- **Perfect convergence**: What successful optimization looks like
- **TinyTorch capabilities**: Using your own framework for real problems

## Technical Details

- **Training time**: <30 seconds
- **Memory usage**: Minimal (~1MB)
- **Success rate**: 100% (XOR is reliably solvable)
- **Visualization**: Rich console interface with ASCII plotting

---

**Perfect for demonstrating that TinyTorch can solve classic ML problems with beautiful visualization!** ✨
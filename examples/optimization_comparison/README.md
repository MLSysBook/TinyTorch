# Optimization Algorithm Comparison

Compare SGD, Momentum, and Adam optimizers to see how different algorithms navigate the loss landscape!

## What This Demonstrates

- **Different optimization strategies** and their trade-offs
- **Convergence speed** comparison between optimizers
- **Why Adam is popular** for deep learning
- **YOUR implementations** of all major optimizers

## Running the Comparison

```bash
python compare.py
```

Expected output:
```
⚡ Optimizer Comparison with TinyTorch
======================================================================

🏃 Training with different optimizers...
------------------------------------------------------------

Training with SGD:
  Initial loss: 4.2315
  Final loss:   0.0234
  Improvement:  99.4%

Training with Momentum:
  Initial loss: 4.2315
  Final loss:   0.0156
  Improvement:  99.6%

Training with Adam:
  Initial loss: 4.2315
  Final loss:   0.0098
  Improvement:  99.8%

📊 Loss Curves (lower is better):
------------------------------------------------------------
Epoch  0: SGD: 4.2315 ████████████████████  Momentum: 4.2315 ████████████████████  Adam: 4.2315 ████████████████████
Epoch  5: SGD: 1.5234 ███████  Momentum: 0.8976 ████  Adam: 0.2134 █
Epoch 10: SGD: 0.6789 ███  Momentum: 0.2345 █  Adam: 0.0567
Epoch 15: SGD: 0.3456 █  Momentum: 0.0876   Adam: 0.0234
...

🏆 Best optimizer: Adam (lowest final loss)
```

## Optimizers Compared

### SGD (Stochastic Gradient Descent)
```python
w = w - learning_rate * gradient
```
- Simple and reliable
- Can be slow to converge
- Fixed learning rate

### Momentum
```python
velocity = momentum * velocity - learning_rate * gradient
w = w + velocity
```
- Accelerates in consistent directions
- Dampens oscillations
- Helps escape shallow local minima

### Adam (Adaptive Moment Estimation)
```python
m = β₁ * m + (1 - β₁) * gradient        # First moment
v = β₂ * v + (1 - β₂) * gradient²       # Second moment
w = w - learning_rate * m / (√v + ε)
```
- Adaptive learning rates per parameter
- Combines momentum with RMSprop
- Often fastest convergence

## Key Insights

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **SGD** | Simple, stable | Slow convergence | Final fine-tuning |
| **Momentum** | Faster than SGD | Requires tuning | General training |
| **Adam** | Fast, adaptive | Can overfit | Most deep learning |

## Mathematical Foundation

Your TinyTorch implements:
- First-order optimization (gradient-based)
- Second-order moment estimation (Adam)
- Momentum accumulation
- Adaptive learning rates

## Requirements

- Module 10 (Optimizers) completed
- TinyTorch package exported

## Next Steps

Try experimenting with:
- Different learning rates
- Various momentum values
- Complex loss landscapes
- Your own optimization algorithms!
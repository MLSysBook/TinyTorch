# XOR Network Example

The classic XOR problem that launched the deep learning revolution!

## What This Demonstrates

- **Multi-layer networks** can solve non-linear problems
- **Hidden layers** transform the input space  
- **Backpropagation** finds the right weights
- **Your TinyTorch framework** works like PyTorch!

## The XOR Problem

XOR (exclusive OR) outputs 1 when inputs differ, 0 when they're the same:

```
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1  
1 XOR 1 = 0
```

Single neurons can't solve this - but 2 layers can!

## Running the Example

```bash
python train.py
```

Expected output:
```
Training XOR Network...
----------------------------------------
Epoch    0 | Loss: 0.2500 | Accuracy: 50.0%
Epoch  100 | Loss: 0.1234 | Accuracy: 75.0%
Epoch  200 | Loss: 0.0456 | Accuracy: 100.0%
...
Final Accuracy: 100.0%
🎉 SUCCESS! XOR problem solved!
```

## Architecture

```
Input Layer (2 neurons)
    ↓
Hidden Layer (4 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

## Key Insight

The hidden layer transforms XOR from "not linearly separable" to "linearly separable" - this is the power of deep learning!

## Requirements

- Module 05 (Dense Networks) completed
- TinyTorch package exported
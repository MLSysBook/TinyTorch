# MNIST Examples - Modern API

This directory contains MNIST digit classification examples using TinyTorch's modern, PyTorch-like API.

## Examples

### `train_mlp_modern_api.py`
**Simple Multi-Layer Perceptron for MNIST classification**

- **Architecture**: 784 → 128 → 64 → 10 (fully connected)
- **Features**: Automatic parameter registration, clean forward pass, modern optimizers
- **Educational Focus**: Neural network fundamentals with professional patterns

**Key Learning Objectives:**
- Understanding multi-layer perceptrons
- Modern API patterns (nn.Module, nn.Linear, F.relu)
- Automatic parameter collection for optimizers
- Clean training loop implementation

## Modern API Benefits

Students learn neural network fundamentals while using industry-standard patterns:

```python
# Clean, professional model definition
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)  # Auto-registered!
        self.hidden2 = nn.Linear(128, 64)   # Auto-registered!
        self.output = nn.Linear(64, 10)     # Auto-registered!
    
    def forward(self, x):
        x = F.flatten(x, start_dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)

# Automatic parameter collection
model = SimpleMLP()
optimizer = optim.Adam(model.parameters())  # All parameters automatically collected!
```

## Running the Examples

```bash
# From TinyTorch root directory
python examples/mnist/train_mlp_modern_api.py
```

## Educational Philosophy

These examples demonstrate that **clean APIs enhance learning** rather than obscure it:

1. **Students still implement core algorithms** - gradients, backpropagation, optimizers
2. **Professional patterns** - industry-standard model definition and training
3. **Reduced boilerplate** - focus on concepts, not parameter management
4. **Scalable practices** - patterns that work from toy examples to production models

The goal is teaching ML systems engineering through building, not just studying algorithms.
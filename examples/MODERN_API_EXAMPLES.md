# TinyTorch Modern API Examples

This directory contains examples showcasing TinyTorch's new PyTorch-compatible API introduced in the framework simplification.

## 🎯 Design Philosophy

**Students implement core algorithms while using professional interfaces.**

The modern API demonstrates that clean interfaces don't reduce educational value - they enhance it by letting students focus on the algorithms that matter rather than framework boilerplate.

## 📚 Example Files

### Core Comparisons

| Modern API File | Original File | Focus |
|----------------|---------------|-------|
| `cifar10/train_cnn_modern_api.py` | `cifar10/train_working_cnn.py` | CNN training with clean imports |
| `xornet/train_xor_modern_api.py` | `xornet/train_xor_network.py` | Simple MLP with auto parameter collection |

### Key API Improvements

#### ✅ Clean Imports
```python
# Modern API
import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim

# vs Old API
from tinytorch.core.layers import Dense
from tinytorch.core.spatial import MultiChannelConv2D
sys.path.insert(0, 'modules/source/06_spatial')
from spatial_dev import flatten, MaxPool2D
```

#### ✅ Automatic Parameter Registration
```python
# Modern API
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))  # Auto-registered!
        self.fc1 = nn.Linear(800, 10)          # Auto-registered!

optimizer = optim.Adam(model.parameters())  # Auto-collected!

# vs Old API
# Manual parameter collection and weight management...
```

#### ✅ Functional Interface
```python
# Modern API
def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.flatten(x)
    return self.fc1(x)

# vs Old API
# Manual activation and shape management...
```

## 🏗️ What Students Still Implement

Despite the clean API, students still build **all the core algorithms**:

- **Conv2d**: Multi-channel convolution with backprop (Module 06)
- **Linear**: Matrix multiplication + bias (Module 04)  
- **ReLU**: Nonlinear activation (Module 03)
- **Adam/SGD**: Optimization algorithms (Module 10)
- **Autograd**: Automatic differentiation (Module 09)

## 🎓 Educational Value

### Before: Fighting Framework Complexity
- Import path management
- Manual parameter collection
- Weight initialization boilerplate
- Shape management overhead

### After: Focus on Algorithms
- **Core Implementation**: Students implement convolution mathematics
- **Professional API**: Clean PyTorch-compatible interface
- **Immediate Productivity**: Write networks that look like production code
- **Systems Understanding**: Learn how frameworks provide abstractions

## 🚀 Running Examples

```bash
# Test the modern CNN example
cd examples/cifar10
python train_cnn_modern_api.py

# Test the modern XOR example  
cd examples/xornet
python train_xor_modern_api.py
```

## 📊 Results

Both modern examples demonstrate:
- **Identical functionality** to original versions
- **Dramatically simplified code** (50-70% reduction in boilerplate)
- **Professional development patterns** from day one
- **Full educational value** with algorithm implementation

## 💡 Key Insight

**Clean APIs enhance learning by removing cognitive load from framework mechanics and focusing attention on the algorithms that actually matter.**

Students learn:
1. **How to implement** ML algorithms (core educational goal)
2. **How to use** professional ML frameworks (career preparation)
3. **Why frameworks exist** (systems thinking)

This is the future of ML education: **implementation understanding** + **professional practices**.
# TinyTorch Examples - Modern API

**Professional ML Applications with Clean, PyTorch-like Interfaces**

These examples demonstrate TinyTorch's modern API that mirrors industry-standard PyTorch patterns. Students learn fundamental ML concepts while using professional development practices.

## 🎯 Modern API Philosophy

**Clean APIs enhance learning rather than obscure it:**
- Students still implement core algorithms (gradients, backpropagation, optimizers)
- Professional patterns prepare students for industry
- Reduced boilerplate lets students focus on concepts
- Scalable practices work from toys to production

## 📁 Available Examples

### 1. **mnist/** - Multi-Layer Perceptron Fundamentals
**Neural Network Basics with Modern Patterns**

- `train_mlp_modern_api.py` - Clean MLP implementation for digit classification
- Demonstrates automatic parameter registration and collection
- Shows modern training loop patterns with optimizers

**Key Learning**: Neural network fundamentals with professional interfaces

### 2. **xornet/** - Nonlinear Learning
**Proves Neural Networks Can Learn Complex Functions**

- `train_xor_modern_api.py` - Clean XOR solution using modern API
- Demonstrates PyTorch-like model definition and training
- Shows API comparison between old and new patterns

**Key Learning**: Nonlinear function approximation with clean code

### 3. **cifar10/** - Computer Vision
**Real-World Image Classification**

- `train_cnn_modern_api.py` - CNN training with modern patterns
- Full CIFAR-10 dataset loading and preprocessing
- Professional model definition and training loops

**Key Learning**: Convolutional networks and real data handling

## 🚀 Modern API Patterns Demonstrated

### Clean Model Definition
```python
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
```

### Automatic Parameter Collection
```python
model = SimpleMLP()
optimizer = optim.Adam(model.parameters())  # All parameters automatically collected!
```

### Professional Training Loop
```python
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 🏃 Running the Examples

```bash
# From TinyTorch root directory

# MNIST MLP - Quick demo with synthetic data
python examples/mnist/train_mlp_modern_api.py

# XOR Network - Seconds to solve, shows API comparison
python examples/xornet/train_xor_modern_api.py

# CIFAR-10 CNN - Real image classification (downloads data)
python examples/cifar10/train_cnn_modern_api.py
```

## 📊 Expected Results

- **MNIST MLP**: Learns synthetic data patterns quickly
- **XOR Network**: 100% accuracy on XOR problem (given sufficient training)
- **CIFAR-10 CNN**: 60%+ accuracy on real image classification

## 🎓 Educational Value

These examples prove that **modern APIs enhance educational outcomes**:

1. **Faster Learning**: Students spend time on concepts, not boilerplate
2. **Industry Preparation**: Patterns transfer directly to PyTorch/TensorFlow
3. **Scalable Practices**: Same patterns work for research and production
4. **Professional Development**: Real-world software engineering practices

## 🔧 API Features Showcased

- **Automatic Parameter Registration**: Models collect their own parameters
- **Functional Interface**: F.relu, F.flatten for common operations
- **Module System**: Hierarchical model construction
- **Modern Optimizers**: Adam, SGD with automatic parameter collection
- **Clean Training Loops**: Professional patterns for model training

## 💡 For Students

You've built a framework with **industry-standard interfaces** that can:
- **Learn any function** (XOR, MNIST patterns)
- **Process real data** (CIFAR-10 images)
- **Scale to complex models** (CNNs, future transformers)

This is exactly how professional ML engineers work!
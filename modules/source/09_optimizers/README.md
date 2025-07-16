# 🔥 Module: Optimizers

## 📊 Module Info
- **Difficulty**: ⭐⭐⭐⭐ Expert
- **Time Estimate**: 6-8 hours
- **Prerequisites**: Tensor, Autograd modules
- **Next Steps**: Training, MLOps modules

Build intelligent optimization algorithms that enable effective neural network training. This module implements the learning algorithms that power modern AI—from basic gradient descent to advanced adaptive methods that make training large-scale models possible.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Master gradient-based optimization theory**: Understand how gradients guide parameter updates and the mathematical foundations of learning
- **Implement core optimization algorithms**: Build SGD, momentum, and Adam optimizers from mathematical first principles
- **Design learning rate strategies**: Create scheduling systems that balance convergence speed with training stability
- **Apply optimization in practice**: Use optimizers effectively in complete training workflows with real neural networks
- **Analyze optimization dynamics**: Compare algorithm behavior, convergence patterns, and performance characteristics

## 🧠 Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement gradient descent, SGD with momentum, Adam optimizer, and learning rate scheduling from mathematical foundations
2. **Use**: Apply optimization algorithms to train neural networks and solve real optimization problems
3. **Optimize**: Analyze convergence behavior, compare algorithm performance, and tune hyperparameters for optimal training

## 📚 What You'll Build

### Core Optimization Algorithms
```python
# Gradient descent foundation
def gradient_descent_step(parameter, learning_rate):
    parameter.data = parameter.data - learning_rate * parameter.grad.data

# SGD with momentum for accelerated convergence
sgd = SGD(parameters=[w1, w2, bias], learning_rate=0.01, momentum=0.9)
sgd.zero_grad()  # Clear previous gradients
loss.backward()  # Compute new gradients
sgd.step()       # Update parameters

# Adam optimizer with adaptive learning rates
adam = Adam(parameters=[w1, w2, bias], learning_rate=0.001, beta1=0.9, beta2=0.999)
adam.zero_grad()
loss.backward()
adam.step()      # Adaptive updates per parameter
```

### Learning Rate Scheduling Systems
```python
# Strategic learning rate adjustment
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with scheduling
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.inputs), batch.targets)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Adjust learning rate each epoch
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

### Complete Training Integration
```python
# Modern training workflow
model = Sequential([Dense(784, 128), ReLU(), Dense(128, 10)])
optimizer = Adam(model.parameters(), learning_rate=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Training loop with optimization
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        # Forward pass
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)
        
        # Optimization step
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
    
    scheduler.step()  # Adjust learning rate
```

### Optimization Algorithm Implementations
- **Gradient Descent**: Basic parameter update rule using gradients
- **SGD with Momentum**: Velocity accumulation for smoother convergence
- **Adam Optimizer**: Adaptive learning rates with bias correction
- **Learning Rate Scheduling**: Strategic adjustment during training

## 🚀 Getting Started

### Prerequisites
Ensure you understand the mathematical foundations:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify prerequisite modules
tito test --module tensor
tito test --module autograd
```

### Development Workflow
1. **Open the development file**: `modules/source/09_optimizers/optimizers_dev.py`
2. **Implement gradient descent**: Start with basic parameter update mechanics
3. **Build SGD with momentum**: Add velocity accumulation for acceleration
4. **Create Adam optimizer**: Implement adaptive learning rates with moment estimation
5. **Add learning rate scheduling**: Build strategic learning rate adjustment systems
6. **Export and verify**: `tito export --module optimizers && tito test --module optimizers`

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify optimization algorithm correctness:

```bash
# TinyTorch CLI (recommended)
tito test --module optimizers

# Direct pytest execution
python -m pytest tests/ -k optimizers -v
```

### Test Coverage Areas
- ✅ **Algorithm Implementation**: Verify SGD, momentum, and Adam compute correct parameter updates
- ✅ **Mathematical Correctness**: Test against analytical solutions for convex optimization
- ✅ **State Management**: Ensure proper momentum and moment estimation tracking
- ✅ **Learning Rate Scheduling**: Verify step decay and scheduling functionality
- ✅ **Training Integration**: Test optimizers in complete neural network training workflows

### Inline Testing & Convergence Analysis
The module includes comprehensive mathematical validation and convergence visualization:
```python
# Example inline test output
🔬 Unit Test: SGD with momentum...
✅ Parameter updates follow momentum equations
✅ Velocity accumulation works correctly
✅ Convergence achieved on test function
📈 Progress: SGD with Momentum ✓

# Optimization analysis
🔬 Unit Test: Adam optimizer...
✅ First moment estimation (m_t) computed correctly
✅ Second moment estimation (v_t) computed correctly  
✅ Bias correction applied properly
✅ Adaptive learning rates working
📈 Progress: Adam Optimizer ✓
```

### Manual Testing Examples
```python
from optimizers_dev import SGD, Adam, StepLR
from autograd_dev import Variable

# Test SGD on simple quadratic function
x = Variable(10.0, requires_grad=True)
sgd = SGD([x], learning_rate=0.1, momentum=0.9)

for step in range(100):
    sgd.zero_grad()
    loss = x**2  # Minimize f(x) = x²
    loss.backward()
    sgd.step()
    if step % 10 == 0:
        print(f"Step {step}: x = {x.data:.4f}, loss = {loss.data:.4f}")

# Test Adam convergence
x = Variable([2.0, -3.0], requires_grad=True)
adam = Adam([x], learning_rate=0.01)

for step in range(50):
    adam.zero_grad()
    loss = (x[0]**2 + x[1]**2).sum()  # Minimize ||x||²
    loss.backward()
    adam.step()
    if step % 10 == 0:
        print(f"Step {step}: x = {x.data}, loss = {loss.data:.6f}")
```

## 🎯 Key Concepts

### Real-World Applications
- **Large Language Models**: GPT, BERT training relies on Adam optimization for stable convergence
- **Computer Vision**: ResNet, Vision Transformer training uses SGD with momentum for best final performance
- **Recommendation Systems**: Online learning systems use adaptive optimizers for continuous model updates
- **Reinforcement Learning**: Policy gradient methods depend on careful optimizer choice and learning rate tuning

### Mathematical Foundations
- **Gradient Descent**: θ_{t+1} = θ_t - α∇L(θ_t) where α is learning rate and ∇L is loss gradient
- **Momentum**: v_{t+1} = βv_t + ∇L(θ_t), θ_{t+1} = θ_t - αv_{t+1} for accelerated convergence
- **Adam**: Combines momentum with adaptive learning rates using first and second moment estimates
- **Learning Rate Scheduling**: Strategic decay schedules balance exploration and exploitation

### Optimization Theory
- **Convex Optimization**: Guarantees global minimum for convex loss functions
- **Non-convex Optimization**: Neural networks have complex loss landscapes with local minima
- **Convergence Analysis**: Understanding when and why optimization algorithms reach good solutions
- **Hyperparameter Sensitivity**: Learning rate is often the most critical hyperparameter

### Performance Characteristics
- **SGD**: Memory efficient, works well with large batches, good final performance
- **Adam**: Fast initial convergence, works with small batches, requires more memory
- **Learning Rate Schedules**: Often crucial for achieving best performance
- **Algorithm Selection**: Problem-dependent choice based on data, model, and computational constraints

## 🎉 Ready to Build?

You're about to implement the algorithms that power all of modern AI! From the neural networks that recognize your voice to the language models that write code, they all depend on the optimization algorithms you're building.

Understanding these algorithms from first principles—implementing momentum physics and adaptive learning rates yourself—will give you deep insight into why some training works and some doesn't. Take your time with the mathematics, test thoroughly, and enjoy building the intelligence behind intelligent systems!

```{grid} 3
:gutter: 3
:margin: 2

{grid-item-card} 🚀 Launch Builder
:link: https://mybinder.org/v2/gh/VJProductions/TinyTorch/main?filepath=modules/source/09_optimizers/optimizers_dev.py
:class-title: text-center
:class-body: text-center

Interactive development environment

{grid-item-card} 📓 Open in Colab  
:link: https://colab.research.google.com/github/VJProductions/TinyTorch/blob/main/modules/source/09_optimizers/optimizers_dev.ipynb
:class-title: text-center
:class-body: text-center

Google Colab notebook

{grid-item-card} 👀 View Source
:link: https://github.com/VJProductions/TinyTorch/blob/main/modules/source/09_optimizers/optimizers_dev.py  
:class-title: text-center
:class-body: text-center

Browse the code on GitHub
```

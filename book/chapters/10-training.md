---
title: "Training"
description: "Neural network training loops, loss functions, and metrics"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# 🏋️ Module 9: Training - Complete Neural Network Training Pipeline
---
**Course Navigation:** [Home](../intro.html) → [Module 10: 10 Training](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐⭐⭐⭐ | <strong>Time:</strong> 6-8 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐⭐⭐⭐⭐ Expert
- **Time Estimate**: 8-10 hours
- **Prerequisites**: Tensor, Activations, Layers, Networks, DataLoader, Autograd, Optimizers modules
- **Next Steps**: Compression, Kernels, Benchmarking, MLOps modules

**Build the complete training pipeline that brings all TinyTorch components together**

## 🎯 Learning Objectives

After completing this module, you will:
- Understand loss functions and how they guide neural network training
- Implement essential loss functions: MSE, CrossEntropy, and BinaryCrossEntropy
- Build evaluation metrics for classification and regression tasks
- Create a complete training loop that orchestrates the entire training process
- Master training workflows with validation, logging, and progress tracking

## 🧠 Build → Use → Optimize

This module follows the TinyTorch pedagogical framework:

1. **Build**: Loss functions, metrics, and training orchestration components
2. **Use**: Train complete neural networks on real datasets
3. **Optimize**: Analyze training dynamics and improve performance

## 📚 What You'll Build

### **Loss Functions**
```python
# Regression loss
mse = MeanSquaredError()
loss = mse(predictions, targets)

# Multi-class classification loss
ce = CrossEntropyLoss()
loss = ce(logits, class_indices)

# Binary classification loss
bce = BinaryCrossEntropyLoss()
loss = bce(logits, binary_labels)
```

### **Evaluation Metrics**
```python
# Classification accuracy
accuracy = Accuracy()
acc = accuracy(predictions, true_labels)  # Returns 0.0 to 1.0

# Regression metrics
mae = MeanAbsoluteError()
error = mae(predictions, targets)
```

### **Complete Training Pipeline**
```python
# Set up training components
model = Sequential([
    Dense(784, 128), ReLU(),
    Dense(128, 64), ReLU(),
    Dense(64, 10), Softmax()
])

optimizer = Adam(model.parameters, learning_rate=0.001)
loss_fn = CrossEntropyLoss()
metrics = [Accuracy()]

# Create trainer
trainer = Trainer(model, optimizer, loss_fn, metrics)

# Train the model
history = trainer.fit(
    train_dataloader, 
    val_dataloader, 
    epochs=10,
    verbose=True
)
```

### **Training with Real Data**
```python
# Load dataset
from tinytorch.core.dataloader import SimpleDataset, DataLoader

# Create dataset
train_dataset = SimpleDataset(size=1000, num_features=784, num_classes=10)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train on real data
history = trainer.fit(train_loader, epochs=50)

# Analyze training
print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"Final training accuracy: {history['train_accuracy'][-1]:.4f}")
```

## 🚀 Getting Started

### Prerequisites
- Complete Modules 1-8: Setup through Optimizers ✅
- Understand backpropagation and gradient descent
- Familiar with classification and regression tasks

### Quick Start
```bash
# Navigate to the training module
cd modules/source/09_training

# Open the development notebook
jupyter lab training_dev.py

# Or use the TinyTorch CLI
tito module info training
tito module test training
```

## 📖 Core Concepts

### **Loss Functions: The Training Signal**
Loss functions measure how far our predictions are from the true values:

- **MSE**: For regression tasks, penalizes large errors heavily
- **CrossEntropy**: For classification, works with softmax outputs
- **BinaryCrossEntropy**: For binary classification, works with sigmoid outputs

### **Metrics: Human-Interpretable Performance**
Metrics provide understandable measures of model performance:

- **Accuracy**: Fraction of correct predictions
- **Precision**: Of positive predictions, how many were correct?
- **Recall**: Of actual positives, how many were found?

### **Training Loop: Orchestrating Learning**
The training loop coordinates all components:

1. **Forward Pass**: Model makes predictions
2. **Loss Computation**: Measure prediction quality
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Improve model weights
5. **Validation**: Monitor generalization performance

### **Training Dynamics**
Understanding how training behaves:

- **Overfitting**: Model memorizes training data
- **Underfitting**: Model too simple to learn patterns
- **Convergence**: Loss stops decreasing
- **Validation**: Monitoring generalization

## 🔬 Advanced Features

### **Training Monitoring**
```python
# Track training progress
history = trainer.fit(train_loader, val_loader, epochs=100)

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### **Custom Metrics**
```python
# Create custom metrics
class F1Score:
    def __call__(self, y_pred, y_true):
        # Implement F1 score calculation
        pass

# Use in training
trainer = Trainer(model, optimizer, loss_fn, metrics=[Accuracy(), F1Score()])
```

### **Training Strategies**
```python
# Learning rate scheduling
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            return self.counter >= self.patience
```

## 🛠️ Real-World Applications

### **Computer Vision**
```python
# Image classification pipeline
model = Sequential([
    Conv2D((3, 3)), ReLU(),
    flatten,
    Dense(128), ReLU(),
    Dense(10), Softmax()
])

trainer = Trainer(model, Adam(model.parameters), CrossEntropyLoss(), [Accuracy()])
history = trainer.fit(cifar10_loader, epochs=50)
```

### **Natural Language Processing**
```python
# Text classification
model = Sequential([
    Dense(vocab_size, 128), ReLU(),
    Dense(128, 64), ReLU(),
    Dense(64, num_classes), Softmax()
])

trainer = Trainer(model, SGD(model.parameters), CrossEntropyLoss(), [Accuracy()])
history = trainer.fit(text_loader, epochs=20)
```

### **Regression Tasks**
```python
# House price prediction
model = Sequential([
    Dense(features, 64), ReLU(),
    Dense(64, 32), ReLU(),
    Dense(32, 1)  # Single output for regression
])

trainer = Trainer(model, Adam(model.parameters), MeanSquaredError(), [])
history = trainer.fit(housing_loader, epochs=100)
```

## 📈 Performance Optimization

### **Batch Size Selection**
- **Small batches**: More updates, noisier gradients
- **Large batches**: Fewer updates, smoother gradients
- **Sweet spot**: Usually 32-256 depending on dataset

### **Learning Rate Tuning**
- **Too high**: Training diverges or oscillates
- **Too low**: Training is slow or gets stuck
- **Adaptive methods**: Adam often works well out of the box

### **Regularization**
- **Dropout**: Randomly disable neurons during training
- **Weight decay**: L2 regularization on parameters
- **Early stopping**: Stop when validation performance plateaus

## 🎯 Module Completion

### **What You've Built**
✅ **Complete loss function library**: MSE, CrossEntropy, BinaryCrossEntropy  
✅ **Evaluation metrics**: Accuracy and extensible metric framework  
✅ **Training orchestration**: Full-featured Trainer class  
✅ **Real-world pipeline**: Train models on actual datasets  
✅ **Monitoring tools**: Track training progress and performance  

### **Skills Developed**
✅ **Training loop design**: Coordinate all training components  
✅ **Loss function implementation**: Measure prediction quality  
✅ **Metric computation**: Evaluate model performance  
✅ **Training dynamics**: Understand convergence and optimization  
✅ **Production workflows**: Build scalable training pipelines  

### **Next Steps**
1. **Export your training module**: `tito export training`
2. **Train a complete model**: Use all TinyTorch components together
3. **Explore advanced topics**: Regularization, scheduling, ensembles
4. **Build production pipelines**: Scale training to larger datasets

**Ready for the final stretch?** Your training module completes the core TinyTorch framework. Next up: compression, kernels, and MLOps! 🚀 

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/09_optimizers.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/11_compression.html" title="next page">Next Module →</a>
</div>

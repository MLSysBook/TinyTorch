---
title: "Training"
description: "Complete training loops with validation, checkpointing, and metrics"
module_number: 7
tier: "foundation"
difficulty: "intermediate"
time_estimate: "4-5 hours"
prerequisites: ["01-06"]
next_module: "08. DataLoader"
learning_objectives:
  - "Understand training loops as orchestrated sequences of forward pass, loss, backward pass, and optimization"
  - "Implement complete training workflows with validation and progress tracking"
  - "Build checkpointing systems for model saving and recovery"
  - "Recognize training dynamics: overfitting, convergence, and learning curves"
  - "Analyze training efficiency and debugging strategies for failed training"
---

# 07. Training

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 07 of 20
</div>

**Build end-to-end training loops that tie all components together.**

Difficulty: Intermediate | Time: 4-5 hours | Prerequisites: Modules 01-06

---

## What You'll Build

Training loops orchestrate all components (models, losses, optimizers, data) into a cohesive system that improves models through iterative learning.

By the end of this module, you'll have implemented:

- **Training Loop** - Iterate over data, compute loss, backpropagate, update parameters
- **Validation** - Evaluate model performance on held-out data
- **Checkpointing** - Save and load model state for recovery
- **Metrics Tracking** - Monitor loss, accuracy, and learning curves

### Example Usage

```python
from tinytorch.training import Trainer
from tinytorch.nn import MLP
from tinytorch.optim import Adam

# Setup
model = MLP([784, 128, 10])
optimizer = Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=CrossEntropyLoss()
)

# Train for 10 epochs
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    save_best=True
)

# Training automatically:
# - Iterates over batches
# - Computes forward pass and loss
# - Runs backward pass
# - Updates parameters
# - Tracks metrics
# - Saves checkpoints
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement training loops with proper gradient management, validation evaluation, and checkpointing logic.

### 2. Use
Train neural networks on real datasets, observing convergence behavior and model improvement over epochs.

### 3. Understand
Grasp training dynamics (overfitting, underfitting), debugging strategies (gradient magnitudes, loss curves), and best practices (early stopping, learning rate schedules).

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize training as an iterative optimization process that balances computational efficiency with model improvement

2. **Core Implementation**: Build complete training loops with batching, validation, checkpointing, and metric tracking

3. **Pattern Recognition**: Understand the train/val split pattern, epoch/batch iteration structure, and checkpoint saving strategies

4. **Framework Connection**: See how your Trainer mirrors PyTorch's training scripts and TensorFlow's `model.fit()`

5. **Performance Trade-offs**: Analyze batch size impact (larger = faster but more memory), validation frequency (more frequent = better monitoring but slower), and checkpoint storage

---

## Why This Matters

### Production Context

Training loops are where ML engineering meets reality:

- **Computer Vision**: Train ResNets for days on ImageNet (1.2M images, 1000 classes)
- **NLP**: Pre-train BERT for weeks on massive text corpora
- **Recommendation**: Train embeddings on billions of user-item interactions
- **Robotics**: Train RL policies over millions of simulation episodes

Efficient, robust training loops are critical infrastructure. A bug can waste days of GPU time.

### Systems Reality Check

**Performance Note**: Training is I/O bound (data loading) or compute bound (forward/backward). Profiling reveals bottlenecks. GPU utilization below 80% often indicates data loading issues.

**Memory Note**: Batch size is constrained by GPU memory. A 1GB model with batch size 32 might require 8GB GPU memory (model + activations + gradients + optimizer state).

---

## Implementation Guide

### Prerequisites Check

```bash
tito test 01 02 03 04 05 06
```

### Development Workflow

```bash
cd modules/source/07_training/
jupyter lab training_dev.py
```

### Step-by-Step Build

#### Step 1: Basic Training Loop

Core training iteration:

```python
def train_epoch(model, dataloader, optimizer, loss_fn):
    """Train for one epoch"""
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        predictions = model.forward(batch_x)
        loss = loss_fn.forward(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.data
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
```

**Pattern**: Forward → Loss → Backward → Update. This is the heartbeat of training.

#### Step 2: Validation Loop

Evaluate without gradients:

```python
def validate(model, dataloader, loss_fn):
    """Evaluate model on validation data"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        # Forward pass only (no backward)
        predictions = model.forward(batch_x)
        loss = loss_fn.forward(predictions, batch_y)
        
        # Compute accuracy
        pred_labels = np.argmax(predictions.data, axis=1)
        correct += np.sum(pred_labels == batch_y.data)
        total += len(batch_y.data)
        
        total_loss += loss.data
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
```

**Key difference**: No backward pass, no parameter updates. Validation measures generalization to unseen data.

#### Step 3: Complete Trainer

Orchestrate training:

```python
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def fit(self, train_loader, val_loader, epochs=10, save_best=True):
        """Complete training workflow"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"train_loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - "
                  f"val_acc: {val_acc:.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pkl')
        
        return self.history
```

**Design pattern**: Separate train/validate logic, track history, save checkpoints. Production-grade training.

#### Step 4: Checkpointing

Save and restore model state:

```python
def save_checkpoint(self, filepath):
    """Save model parameters"""
    state = {
        'parameters': [p.data for p in self.model.parameters()],
        'optimizer_state': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(self, filepath):
    """Restore model parameters"""
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    for p, data in zip(self.model.parameters(), state['parameters']):
        p.data = data
```

**Why checkpointing matters**: Training can crash (OOM, power loss). Checkpoints enable recovery. Also used for model deployment.

---

## Testing Your Implementation

### Inline Tests

```python
# Test training step
model = Sequential([Linear(10, 5), ReLU(), Linear(5, 2)])
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

# Create fake batch
batch_x = Tensor(np.random.randn(8, 10))
batch_y = Tensor(np.random.randint(0, 2, 8))

# Training step
predictions = model.forward(batch_x)
loss_before = loss_fn.forward(predictions, batch_y).data

loss = loss_fn.forward(predictions, batch_y)
loss.backward()
optimizer.step()

# Loss should decrease
predictions_after = model.forward(batch_x)
loss_after = loss_fn.forward(predictions_after, batch_y).data
assert loss_after < loss_before
print("✓ Training step working")
```

### Module Export & Validation

```bash
tito export 07
tito test 07
```

**Expected output**:
```
✓ All tests passed! [15/15]
✓ Module 07 complete!
```

---

## Where This Code Lives

Training ties everything together:

```python
# Complete TinyTorch training pipeline
from tinytorch.training import Trainer
from tinytorch.nn import MLP
from tinytorch.optim import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.data import DataLoader

# Everything you built comes together here:
model = MLP([784, 128, 10])  # Module 03: Layers
optimizer = Adam(model.parameters())  # Module 06: Optimizers
loss_fn = CrossEntropyLoss()  # Module 04: Losses
train_loader = DataLoader(...)  # Module 08: DataLoader

trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(train_loader, val_loader, epochs=10)
```

**Package structure**:
```
tinytorch/
├── training/
│   ├── trainer.py  ← YOUR training loop
├── optim/
│   ├── optimizers.py
├── core/
│   ├── losses.py
```

---

## Systems Thinking Questions

1. **Batch Size Trade-off**: Larger batches are more efficient (better GPU utilization) but use more memory. How would you choose batch size for a given GPU? What about distributed training across multiple GPUs?

2. **Validation Frequency**: Validating every epoch is expensive for large datasets. When would you validate less frequently (every N epochs)? What information do you lose?

3. **Overfitting Detection**: Training loss decreases but validation loss increases. What does this mean? How would you address it (regularization, dropout, early stopping)?

4. **Learning Rate Scheduling**: Why do practitioners decay learning rate during training? When should you reduce it (fixed schedule vs validation plateau)?

5. **Checkpoint Strategy**: Saving every epoch uses disk space. Save only best model? Last N epochs? What if validation loss is noisy?

---

## Real-World Connections

### Industry Training Workflows

- **ImageNet Classification**: Train for 90 epochs, reduce LR at epochs 30/60/90
- **BERT Pre-training**: Train for 1M steps, checkpoint every 10K steps, ~1 week on 64 TPUs
- **GPT-3**: Train for 300B tokens, checkpoint frequently due to long training time
- **Recommendation Systems**: Online training - update models continuously as new data arrives

### Production Challenges

- **GPU OOM**: Batch size too large, reduce or use gradient accumulation
- **Loss Spikes**: Learning rate too high or bad batch, reduce LR
- **Slow Convergence**: Learning rate too low or poor initialization
- **NaN Loss**: Exploding gradients, use gradient clipping

---

## Foundation Tier Complete!

**Congratulations!** You've built the entire mathematical engine of machine learning. You now have:

- ✅ Tensors and operations
- ✅ Activation functions
- ✅ Neural network layers
- ✅ Loss functions
- ✅ Automatic differentiation
- ✅ Optimizers
- ✅ Complete training loops

### Unlock Your First Milestone

You can now run the **1957: Rosenblatt's Perceptron** milestone:

```bash
python milestones/01_1957_perceptron/perceptron_digits.py
```

This uses YOUR implementations to recreate the first trainable neural network!

---

## What's Next?

**You're ready for Intelligence Tier!** Now you'll build systems that process real data—vision and language.

**Module 08: DataLoader** - Build efficient data pipelines for loading and preprocessing datasets

[Continue to Module 08: DataLoader →](08-dataloader.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Training API Reference](../appendices/api-reference.html#training)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)

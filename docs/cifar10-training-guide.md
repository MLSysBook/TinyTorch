# 🎯 CIFAR-10 Training Guide: Achieving 55%+ Accuracy

## Overview
This guide walks you through training a CNN on CIFAR-10 using your TinyTorch implementation to achieve solid 55%+ accuracy with your own framework built from scratch.

## Prerequisites
Complete these modules first:
- ✅ Module 08: DataLoader (for CIFAR-10 loading)
- ✅ Module 07: Training (for model checkpointing)
- ✅ Module 09: Convolutional Networks (for CNN layers)
- ✅ Module 06: Optimizers (for Adam optimizer)

## Step 1: Load CIFAR-10 Data

```python
from tinytorch.core.dataloader import CIFAR10Dataset, DataLoader

# Download CIFAR-10 (one-time, ~170MB)
dataset = CIFAR10Dataset(download=True, flatten=False)
print(f"✅ Training samples: {len(dataset.train_data)}")
print(f"✅ Test samples: {len(dataset.test_data)}")

# Create data loaders
train_loader = DataLoader(
    dataset.train_data, 
    dataset.train_labels, 
    batch_size=32, 
    shuffle=True
)

test_loader = DataLoader(
    dataset.test_data,
    dataset.test_labels,
    batch_size=32,
    shuffle=False
)
```

## Step 2: Build Your CNN Architecture

### Option A: Simple CNN (Good for initial testing)
```python
from tinytorch.core.networks import Sequential
from tinytorch.core.layers import Dense
from tinytorch.core.spatial import Conv2D, MaxPool2D, Flatten
from tinytorch.core.activations import ReLU

model = Sequential([
    # First conv block
    Conv2D(3, 32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    
    # Second conv block  
    Conv2D(32, 64, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    
    # Flatten and classify
    Flatten(),
    Dense(64 * 8 * 8, 128),
    ReLU(),
    Dense(128, 10)
])
```

### Option B: Deeper CNN (Better accuracy)
```python
model = Sequential([
    # Block 1
    Conv2D(3, 64, kernel_size=3, padding=1),
    ReLU(),
    Conv2D(64, 64, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    
    # Block 2
    Conv2D(64, 128, kernel_size=3, padding=1),
    ReLU(),
    Conv2D(128, 128, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    
    # Classifier
    Flatten(),
    Dense(128 * 8 * 8, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10)
])
```

## Step 3: Configure Training

```python
from tinytorch.core.training import Trainer, CrossEntropyLoss, Accuracy
from tinytorch.core.optimizers import Adam

# Setup training components
loss_fn = CrossEntropyLoss()
optimizer = Adam(lr=0.001)
metrics = [Accuracy()]

# Create trainer
trainer = Trainer(model, loss_fn, optimizer, metrics)
```

## Step 4: Train with Checkpointing

```python
# Train with automatic model saving
history = trainer.fit(
    train_loader,
    val_dataloader=test_loader,
    epochs=30,
    save_best=True,                    # Save best model
    checkpoint_path='best_cifar10.pkl', # Where to save
    early_stopping_patience=5,          # Stop if no improvement
    verbose=True                        # Show progress
)

print(f"🎉 Best validation accuracy: {max(history['val_accuracy']):.2%}")
print("🎯 Target: 55%+ accuracy - proving your framework works!")
```

## Step 5: Evaluate Performance

```python
from tinytorch.core.training import evaluate_model, plot_training_history

# Load best model
trainer.load_checkpoint('best_cifar10.pkl')

# Comprehensive evaluation
results = evaluate_model(model, test_loader)
print(f"\n📊 Test Results:")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Per-class accuracy:")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']
for i, class_name in enumerate(classes):
    class_acc = results['per_class_accuracy'][i]
    print(f"  {class_name}: {class_acc:.2%}")

# Visualize training curves
plot_training_history(history)
```

## Step 6: Analyze Confusion Matrix

```python
from tinytorch.core.training import compute_confusion_matrix
import numpy as np

# Get predictions for entire test set
all_preds = []
all_labels = []
for batch_x, batch_y in test_loader:
    preds = model(batch_x).data.argmax(axis=1)
    all_preds.extend(preds)
    all_labels.extend(batch_y.data)

# Compute confusion matrix
cm = compute_confusion_matrix(np.array(all_preds), np.array(all_labels))

# Analyze common mistakes
print("\n🔍 Common Confusions:")
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 100:  # More than 100 mistakes
            print(f"{classes[i]} confused as {classes[j]}: {cm[i, j]} times")
```

## Training Tips for Optimal Performance

### 1. Data Preprocessing
```python
# Normalize data for better convergence
from tinytorch.core.dataloader import Normalizer

normalizer = Normalizer()
normalizer.fit(dataset.train_data)
train_data_normalized = normalizer.transform(dataset.train_data)
test_data_normalized = normalizer.transform(dataset.test_data)
```

### 2. Learning Rate Scheduling
```python
# Reduce learning rate when stuck
for epoch in range(epochs):
    if epoch == 20:
        optimizer.lr *= 0.1  # Reduce by 10x
    trainer.train_epoch(train_loader)
```

### 3. Data Augmentation (Simple)
```python
# Random horizontal flips for training
def augment_batch(batch_x, batch_y):
    # Randomly flip half the images horizontally
    flip_mask = np.random.random(len(batch_x)) > 0.5
    batch_x[flip_mask] = batch_x[flip_mask][:, :, :, ::-1]
    return batch_x, batch_y
```

### 4. Monitor Training Progress
```python
# Check if model is learning
if epoch % 5 == 0:
    train_acc = evaluate_model(model, train_loader)['accuracy']
    test_acc = evaluate_model(model, test_loader)['accuracy']
    gap = train_acc - test_acc
    
    if gap > 0.15:
        print("⚠️ Overfitting detected! Consider:")
        print("  - Adding dropout layers")
        print("  - Reducing model complexity")
        print("  - Increasing batch size")
    elif train_acc < 0.6:
        print("⚠️ Underfitting! Consider:")
        print("  - Increasing model capacity")
        print("  - Checking learning rate")
        print("  - Training longer")
```

## Expected Results Timeline

- **After 5 epochs**: ~30-40% accuracy (model learning basic patterns)
- **After 10 epochs**: ~45-50% accuracy (recognizing shapes)
- **After 20 epochs**: ~50-55% accuracy (good feature extraction)
- **After 30 epochs**: ~55%+ accuracy (solid performance achieved! 🎉)

## Troubleshooting Common Issues

### Issue: Accuracy stuck at ~10%
**Solution**: Check loss is decreasing. If not, reduce learning rate.

### Issue: Loss is NaN
**Solution**: Learning rate too high. Start with 0.0001 instead.

### Issue: Accuracy oscillating wildly
**Solution**: Batch size too small. Try 64 or 128.

### Issue: Training very slow
**Solution**: Ensure you're using vectorized operations, not loops.

### Issue: Memory errors
**Solution**: Reduce batch size or model size.

## Celebrating Success! 🎉

Once you achieve 55%+ accuracy:

1. **Save your model**: This is a real achievement!
```python
trainer.save_checkpoint('my_75_percent_model.pkl')
```

2. **Document your architecture**: What worked?
```python
print(model.summary())  # Your architecture
print(f"Parameters: {model.count_parameters()}")
print(f"Best epoch: {np.argmax(history['val_accuracy'])}")
```

3. **Share your results**: You built this from scratch!
```python
print(f"🏆 CIFAR-10 Test Accuracy: {results['accuracy']:.2%}")
print("✅ Solid Performance Achieved!")
print("🎯 Built entirely with TinyTorch - no PyTorch/TensorFlow!")
```

## Next Challenges

After achieving 55%+:
- 🚀 Push for 60%+ with better architectures and hyperparameters
- 🎨 Implement data augmentation for improved generalization  
- ⚡ Optimize training speed with better kernels
- 🔬 Analyze what your CNN learned with visualizations
- 🏆 Try other datasets (Fashion-MNIST, etc.)

Remember: You built every component from scratch - from tensors to convolutions to optimizers. This 55%+ accuracy represents deep understanding of ML systems, not just API usage!
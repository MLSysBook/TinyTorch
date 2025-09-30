# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 07: Training - Complete Learning Loops

Welcome to Module 07! You're about to build the complete training infrastructure that brings neural networks to life through end-to-end learning.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors, activations, layers, losses, gradients, and optimizers
**You'll Build**: Complete training loops with checkpointing, scheduling, and gradient management
**You'll Enable**: Full model training pipeline for the MLP milestone

**Connection Map**:
```
Optimizers (Module 06) → Training (Module 07) → DataLoader (Module 08)
(parameter updates)     (complete loops)      (efficient batching)
```

## Learning Objectives
By the end of this module, you will:
1. Implement a complete Trainer class with train/eval modes
2. Build learning rate scheduling and gradient clipping
3. Create checkpointing for model persistence
4. Test training loops with immediate validation
5. Understand gradient accumulation patterns

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/07_training/training_dev.py
**Building Side:** Code exports to tinytorch.core.training

```python
# Final package structure:
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm  # This module
from tinytorch.core.tensor import Tensor  # Foundation (Module 01)
from tinytorch.core.optimizers import SGD, AdamW  # Parameter updates (Module 06)
from tinytorch.core.losses import CrossEntropyLoss  # Error measurement (Module 04)
```

**Why this matters:**
- **Learning:** Complete training system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's training infrastructure with all training components together
- **Consistency:** All training operations and scheduling functionality in core.training
- **Integration:** Works seamlessly with optimizers and losses for complete learning pipelines
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "locked": false, "solution": false}
#| default_exp core.training

import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

# %% [markdown]
"""
## 🏗️ Part 1: Introduction - What is Training?

Training is where the magic happens - it's the process that transforms a randomly initialized neural network into an intelligent system that can solve problems. Think of training as teaching: you show the model examples, it makes predictions, you measure how wrong it is, and then you adjust its parameters to do better next time.

The training process follows a consistent pattern across all machine learning:

1. **Forward Pass**: Input flows through the model to produce predictions
2. **Loss Calculation**: Compare predictions to true answers
3. **Backward Pass**: Compute gradients showing how to improve
4. **Parameter Update**: Adjust model weights using an optimizer
5. **Repeat**: Continue until the model learns the pattern

But production training systems need much more than this basic loop. They need learning rate scheduling (starting fast, slowing down), gradient clipping (preventing exploding gradients), checkpointing (saving progress), and evaluation modes (testing without learning).

**What we're building today:**
- A complete `Trainer` class that orchestrates the entire learning process
- Learning rate scheduling that adapts during training
- Gradient clipping that prevents training instability
- Checkpointing system for saving and resuming training
- Train/eval modes for proper model behavior
"""

# %% [markdown]
"""
## 📐 Part 2: Foundations - Mathematical Background

### Training Loop Mathematics

The core training loop implements gradient descent with sophisticated improvements:

**Basic Update Rule:**
```
θ(t+1) = θ(t) - η ∇L(θ(t))
```
Where θ are parameters, η is learning rate, and ∇L is the loss gradient.

**Learning Rate Scheduling:**
For cosine annealing over T epochs:
```
η(t) = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
```

**Gradient Clipping:**
When ||∇L|| > max_norm, rescale:
```
∇L ← ∇L * max_norm / ||∇L||
```

**Gradient Accumulation:**
For effective batch size B_eff = accumulation_steps * B_actual:
```
∇L_accumulated = (1/accumulation_steps) * Σ ∇L_batch_i
```

### Train vs Eval Modes

Many layers behave differently during training vs inference:
- **Dropout**: Active during training, disabled during evaluation
- **BatchNorm**: Updates statistics during training, uses fixed statistics during evaluation
- **Gradient computation**: Enabled during training, disabled during evaluation for efficiency

This mode switching is crucial for proper model behavior and performance.
"""

# %% [markdown]
"""
## 🏗️ Part 3: Implementation - Building Training Infrastructure

Now let's implement the complete training system. We'll build each component step by step: learning rate scheduling, gradient utilities, and finally the complete Trainer class.

Each component will follow the pattern: **Explanation → Implementation → Test** so you understand what you're building before you build it.
"""

# %% [markdown]
"""
### Learning Rate Scheduling - Adaptive Training Speed

Learning rate scheduling is like adjusting your driving speed based on road conditions. You start fast on the highway (high learning rate for quick progress), then slow down in neighborhoods (low learning rate for fine-tuning).

#### Why Cosine Scheduling Works

Cosine annealing follows a smooth curve that provides:
- **Aggressive learning initially** - Fast convergence when far from optimum
- **Gradual slowdown** - Stable convergence as you approach the solution
- **Smooth transitions** - No sudden learning rate drops that shock the model

#### The Mathematics

Cosine annealing uses the cosine function to smoothly transition from max_lr to min_lr:

```
Learning Rate Schedule:

max_lr ┌─\
       │   \
       │     \
       │       \
       │         \
min_lr └───────────\────────
       0    25    50   75  100 epochs

Formula: lr = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / total_epochs)) / 2
```

This creates a natural learning curve that adapts training speed to the optimization landscape.
"""

# %% nbgrader={"grade": false, "grade_id": "scheduler", "locked": false, "solution": true}
class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr, decreases following a cosine curve to min_lr over T epochs.
    This provides aggressive learning initially, then fine-tuning at the end.

    TODO: Implement cosine annealing schedule

    APPROACH:
    1. Store max_lr, min_lr, and total_epochs
    2. In get_lr(), compute cosine factor: (1 + cos(π * epoch / total_epochs)) / 2
    3. Interpolate: min_lr + (max_lr - min_lr) * cosine_factor

    EXAMPLE:
    >>> schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
    >>> print(schedule.get_lr(0))    # Start: 0.1
    >>> print(schedule.get_lr(50))   # Middle: ~0.055
    >>> print(schedule.get_lr(100))  # End: 0.01

    HINT: Use np.cos() and np.pi for the cosine calculation
    """
    ### BEGIN SOLUTION
    def __init__(self, max_lr: float = 0.1, min_lr: float = 0.01, total_epochs: int = 100):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: CosineSchedule
This test validates our learning rate scheduling implementation.
**What we're testing**: Cosine annealing produces correct learning rates
**Why it matters**: Proper scheduling often makes the difference between convergence and failure
**Expected**: Smooth decrease from max_lr to min_lr following cosine curve
"""

# %% nbgrader={"grade": true, "grade_id": "test_scheduler", "locked": true, "points": 10}
def test_unit_cosine_schedule():
    """🔬 Test CosineSchedule implementation."""
    print("🔬 Unit Test: CosineSchedule...")

    # Test basic schedule
    schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)

    # Test start, middle, and end
    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    print(f"Learning rate at epoch 0: {lr_start:.4f}")
    print(f"Learning rate at epoch 50: {lr_middle:.4f}")
    print(f"Learning rate at epoch 100: {lr_end:.4f}")

    # Validate behavior
    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end - 0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, f"Middle LR should be between min and max, got {lr_middle}"

    # Test monotonic decrease in first half
    lr_quarter = schedule.get_lr(25)
    assert lr_quarter > lr_middle, "LR should decrease monotonically in first half"

    print("✅ CosineSchedule works correctly!")

# test_unit_cosine_schedule()  # Moved to main guard

# %% [markdown]
"""
### Gradient Clipping - Preventing Training Explosions

Gradient clipping is like having a speed governor on your car - it prevents dangerous situations where gradients become so large they destroy training progress.

#### The Problem: Exploding Gradients

During training, gradients can sometimes become extremely large, causing:
- **Parameter updates that are too big** - Model jumps far from the optimal solution
- **Numerical instability** - Values become NaN or infinite
- **Training collapse** - Model performance suddenly degrades

#### The Solution: Global Norm Clipping

Instead of clipping each gradient individually, we compute the global norm across all parameters and scale uniformly:

```
Gradient Clipping Process:

1. Compute Global Norm:
   total_norm = √(sum of all gradient squares)

2. Check if Clipping Needed:
   if total_norm > max_norm:
       clip_coefficient = max_norm / total_norm

3. Scale All Gradients:
   for each gradient:
       gradient *= clip_coefficient

Visualization:
Original Gradients:  [100, 200, 50] → norm = 230
With max_norm=1.0:   [0.43, 0.87, 0.22] → norm = 1.0
```

This preserves the relative magnitudes while preventing explosion.
"""

# %% nbgrader={"grade": false, "grade_id": "gradient_clipping", "locked": false, "solution": true}
def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm to prevent exploding gradients.

    This is crucial for training stability, especially with RNNs and deep networks.
    Instead of clipping each gradient individually, we compute the global norm
    across all parameters and scale uniformly if needed.

    TODO: Implement gradient clipping by global norm

    APPROACH:
    1. Compute total norm: sqrt(sum of squared gradients across all parameters)
    2. If total_norm > max_norm, compute clip_coef = max_norm / total_norm
    3. Scale all gradients by clip_coef: grad *= clip_coef
    4. Return the original norm for monitoring

    EXAMPLE:
    >>> params = [Tensor([1, 2, 3], requires_grad=True)]
    >>> params[0].grad = Tensor([10, 20, 30])  # Large gradients
    >>> original_norm = clip_grad_norm(params, max_norm=1.0)
    >>> print(f"Clipped norm: {np.linalg.norm(params[0].grad.data):.2f}")  # Should be ≤ 1.0

    HINTS:
    - Use np.linalg.norm() to compute norms
    - Only clip if total_norm > max_norm
    - Modify gradients in-place for efficiency
    """
    ### BEGIN SOLUTION
    if not parameters:
        return 0.0

    # Collect all gradients and compute global norm
    total_norm = 0.0
    for param in parameters:
        if hasattr(param, 'grad') and param.grad is not None:
            total_norm += np.sum(param.grad.data ** 2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad.data *= clip_coef

    return float(total_norm)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Gradient Clipping
This test validates our gradient clipping implementation.
**What we're testing**: Global norm clipping properly rescales large gradients
**Why it matters**: Prevents exploding gradients that can destroy training
**Expected**: Gradients scaled down when norm exceeds threshold
"""

# %% nbgrader={"grade": true, "grade_id": "test_clipping", "locked": true, "points": 10}
def test_unit_clip_grad_norm():
    """🔬 Test clip_grad_norm implementation."""
    print("🔬 Unit Test: Gradient Clipping...")

    # Create mock parameters with gradients (simulating Tensor.grad)
    class MockParam:
        def __init__(self, grad_data):
            self.grad = type('grad', (), {'data': np.array(grad_data)})()

    # Test case 1: Large gradients that need clipping
    params = [
        MockParam([3.0, 4.0]),  # norm = 5.0
        MockParam([6.0, 8.0])   # norm = 10.0
    ]
    # Total norm = sqrt(5² + 10²) = sqrt(125) ≈ 11.18

    original_norm = clip_grad_norm(params, max_norm=1.0)

    # Check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"

    # Check gradients were clipped
    new_norm = 0.0
    for param in params:
        new_norm += np.sum(param.grad.data ** 2)
    new_norm = np.sqrt(new_norm)

    print(f"Original norm: {original_norm:.2f}")
    print(f"Clipped norm: {new_norm:.2f}")

    assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1.0, got {new_norm}"

    # Test case 2: Small gradients that don't need clipping
    small_params = [MockParam([0.1, 0.2])]
    original_small = clip_grad_norm(small_params, max_norm=1.0)

    assert original_small < 1.0, "Small gradients shouldn't be clipped"

    print("✅ Gradient clipping works correctly!")

# test_unit_clip_grad_norm()  # Moved to main guard

# %% [markdown]
"""
### The Trainer Class - Orchestrating Complete Training

The Trainer class is like a conductor orchestrating a symphony - it coordinates all the components (model, optimizer, loss function, scheduler) to create beautiful music (successful training).

#### Training Loop Architecture

The training loop follows a consistent pattern across all machine learning:

```
Training Loop Structure:

for epoch in range(num_epochs):
    ┌─────────────────── TRAINING PHASE ───────────────────┐
    │                                                       │
    │  for batch in dataloader:                            │
    │      ┌─── Forward Pass ───┐                          │
    │      │ 1. input → model   │                          │
    │      │ 2. predictions     │                          │
    │      └───────────────────┘                          │
    │               ↓                                      │
    │      ┌─── Loss Computation ───┐                     │
    │      │ 3. loss = loss_fn()    │                     │
    │      └───────────────────────┘                     │
    │               ↓                                      │
    │      ┌─── Backward Pass ───┐                       │
    │      │ 4. loss.backward()  │                       │
    │      │ 5. gradients        │                       │
    │      └────────────────────┘                       │
    │               ↓                                      │
    │      ┌─── Parameter Update ───┐                    │
    │      │ 6. optimizer.step()    │                    │
    │      │ 7. zero gradients      │                    │
    │      └───────────────────────┘                    │
    └───────────────────────────────────────────────────┘
             ↓
    ┌─── Learning Rate Update ───┐
    │ 8. scheduler.step()         │
    └────────────────────────────┘
```

#### Key Features

- **Train/Eval Modes**: Different behavior during training vs evaluation
- **Gradient Accumulation**: Effective larger batch sizes with limited memory
- **Checkpointing**: Save/resume training state for long experiments
- **Progress Tracking**: Monitor loss, learning rate, and other metrics
"""

# %% nbgrader={"grade": false, "grade_id": "trainer_class", "locked": false, "solution": true}
class Trainer:
    """
    Complete training orchestrator for neural networks.

    Handles the full training lifecycle: forward pass, loss computation,
    backward pass, optimization, scheduling, checkpointing, and evaluation.

    This is the central class that brings together all the components
    you've built in previous modules.

    TODO: Implement complete Trainer class

    APPROACH:
    1. Store model, optimizer, loss function, and optional scheduler
    2. train_epoch(): Loop through data, compute loss, update parameters
    3. evaluate(): Similar loop but without gradient updates
    4. save/load_checkpoint(): Persist training state for resumption

    DESIGN PATTERNS:
    - Context managers for train/eval modes
    - Gradient accumulation for effective large batch sizes
    - Progress tracking for monitoring
    - Flexible scheduling integration
    """
    ### BEGIN SOLUTION
    def __init__(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
        """
        Initialize trainer with model and training components.

        Args:
            model: Neural network to train
            optimizer: Parameter update strategy (SGD, Adam, etc.)
            loss_fn: Loss function (CrossEntropy, MSE, etc.)
            scheduler: Optional learning rate scheduler
            grad_clip_norm: Optional gradient clipping threshold
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        # Training state
        self.epoch = 0
        self.step = 0
        self.training_mode = True

        # History tracking
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': []
        }

    def train_epoch(self, dataloader, accumulation_steps=1):
        """
        Train for one epoch through the dataset.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches
            accumulation_steps: Number of batches to accumulate before update

        Returns:
            Average loss for the epoch
        """
        self.model.training = True
        self.training_mode = True

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            # Scale loss for accumulation
            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += scaled_loss

            # Backward pass
            if hasattr(loss, 'backward'):
                loss.backward()

            # Update parameters every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_norm is not None:
                    params = []
                    if hasattr(self.model, 'parameters'):
                        params = self.model.parameters()
                    clip_grad_norm(params, self.grad_clip_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1

        # Handle remaining accumulated gradients
        if accumulated_loss > 0:
            if self.grad_clip_norm is not None:
                params = []
                if hasattr(self.model, 'parameters'):
                    params = self.model.parameters()
                clip_grad_norm(params, self.grad_clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        # Update scheduler
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            # Update optimizer learning rate
            if hasattr(self.optimizer, 'lr'):
                self.optimizer.lr = current_lr
            self.history['learning_rates'].append(current_lr)

        self.epoch += 1
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluate model on dataset without updating parameters.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches

        Returns:
            Average loss and accuracy
        """
        self.model.training = False
        self.training_mode = False

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            # Forward pass only
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            total_loss += loss.data

            # Calculate accuracy (for classification)
            if hasattr(outputs, 'data') and hasattr(targets, 'data'):
                if len(outputs.data.shape) > 1:  # Multi-class
                    predictions = np.argmax(outputs.data, axis=1)
                    if len(targets.data.shape) == 1:  # Integer targets
                        correct += np.sum(predictions == targets.data)
                    else:  # One-hot targets
                        correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                    total += len(predictions)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)

        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """
        Save complete training state for resumption.

        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'scheduler_state': self._get_scheduler_state(),
            'history': self.history,
            'training_mode': self.training_mode
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        """
        Load training state from checkpoint.

        Args:
            path: File path to load checkpoint from
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training_mode = checkpoint['training_mode']

        # Restore states (simplified for educational purposes)
        if 'model_state' in checkpoint:
            self._set_model_state(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self._set_optimizer_state(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self._set_scheduler_state(checkpoint['scheduler_state'])

    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        if hasattr(self.model, 'parameters'):
            return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}
        return {}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                if i in state:
                    param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        if hasattr(self.optimizer, 'lr'):
            state['lr'] = self.optimizer.lr
        if hasattr(self.optimizer, 'momentum_buffers'):
            state['momentum_buffers'] = self.optimizer.momentum_buffers.copy()
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if 'lr' in state and hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = state['lr']
        if 'momentum_buffers' in state and hasattr(self.optimizer, 'momentum_buffers'):
            self.optimizer.momentum_buffers = state['momentum_buffers']

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing."""
        if self.scheduler is None:
            return None
        return {
            'max_lr': getattr(self.scheduler, 'max_lr', None),
            'min_lr': getattr(self.scheduler, 'min_lr', None),
            'total_epochs': getattr(self.scheduler, 'total_epochs', None)
        }

    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Trainer Class
This test validates our complete training system.
**What we're testing**: Trainer orchestrates training loop correctly
**Why it matters**: This is the backbone that enables all neural network training
**Expected**: Training reduces loss, evaluation works, checkpointing preserves state
"""

# %% nbgrader={"grade": true, "grade_id": "test_trainer", "locked": true, "points": 15}
def test_unit_trainer():
    """🔬 Test Trainer implementation."""
    print("🔬 Unit Test: Trainer...")

    # Create mock components for testing
    class MockModel:
        def __init__(self):
            self.training = True
            self.weight = type('param', (), {'data': np.array([1.0, 2.0]), 'grad': None})()

        def forward(self, x):
            # Simple linear operation
            result = type('output', (), {'data': np.dot(x.data, self.weight.data)})()
            return result

        def parameters(self):
            return [self.weight]

    class MockOptimizer:
        def __init__(self):
            self.lr = 0.01

        def step(self):
            pass  # Simplified

        def zero_grad(self):
            pass  # Simplified

    class MockLoss:
        def forward(self, outputs, targets):
            # Simple MSE
            diff = outputs.data - targets.data
            loss_value = np.mean(diff ** 2)
            result = type('loss', (), {'data': loss_value})()
            result.backward = lambda: None  # Simplified
            return result

    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)

    # Create trainer
    model = MockModel()
    optimizer = MockOptimizer()
    loss_fn = MockLoss()
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Test training
    print("Testing training epoch...")
    mock_dataloader = [
        (MockTensor([1.0, 0.5]), MockTensor([2.0])),
        (MockTensor([0.5, 1.0]), MockTensor([1.5]))
    ]

    loss = trainer.train_epoch(mock_dataloader)
    assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
    assert trainer.epoch == 1, f"Expected epoch 1, got {trainer.epoch}"

    # Test evaluation
    print("Testing evaluation...")
    eval_loss, accuracy = trainer.evaluate(mock_dataloader)
    assert isinstance(eval_loss, float), f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy, float), f"Expected float accuracy, got {type(accuracy)}"

    # Test checkpointing
    print("Testing checkpointing...")
    checkpoint_path = "/tmp/test_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Modify trainer state
    original_epoch = trainer.epoch
    trainer.epoch = 999

    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    assert trainer.epoch == original_epoch, f"Checkpoint didn't restore epoch correctly"

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✅ Trainer works correctly! Final loss: {loss:.4f}")

# test_unit_trainer()  # Moved to main guard

# %% [markdown]
"""
## 🔧 Part 4: Integration - Bringing Training Together

Now let's create a complete training example that demonstrates how all the components work together. This integration shows the full power of our training infrastructure.
"""

# %% nbgrader={"grade": false, "grade_id": "training_integration", "locked": false, "solution": true}
def demonstrate_complete_training():
    """
    Demonstrate complete training pipeline with all components.

    This shows how Trainer, CosineSchedule, and gradient clipping work together
    to create a robust training system that could handle real neural networks.
    """
    print("🏗️ Complete Training Pipeline Demonstration")
    print("=" * 50)

    # Create mock neural network components
    class SimpleModel:
        def __init__(self, input_size=2, hidden_size=4, output_size=1):
            self.training = True
            # Initialize weights (simplified)
            self.w1 = type('param', (), {
                'data': np.random.randn(input_size, hidden_size) * 0.1,
                'grad': None
            })()
            self.w2 = type('param', (), {
                'data': np.random.randn(hidden_size, output_size) * 0.1,
                'grad': None
            })()

        def forward(self, x):
            # Simple 2-layer network
            h = np.maximum(0, np.dot(x.data, self.w1.data))  # ReLU
            output = np.dot(h, self.w2.data)
            result = type('output', (), {'data': output})()
            return result

        def parameters(self):
            return [self.w1, self.w2]

    class MockSGD:
        def __init__(self, params, lr=0.01):
            self.params = params
            self.lr = lr

        def step(self):
            # Simplified parameter update
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data

        def zero_grad(self):
            for param in self.params:
                param.grad = None

    class MSELoss:
        def forward(self, outputs, targets):
            diff = outputs.data - targets.data
            loss_value = np.mean(diff ** 2)
            result = type('loss', (), {'data': loss_value})()

            # Simplified backward pass
            def backward():
                grad_output = 2 * diff / len(diff)
                # Set gradients (simplified)
                outputs.grad = type('grad', (), {'data': grad_output})()

            result.backward = backward
            return result

    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data, dtype=float)

    # 1. Create model and training components
    print("1. Setting up training components...")
    model = SimpleModel(input_size=2, hidden_size=8, output_size=1)
    optimizer = MockSGD(model.parameters(), lr=0.1)
    loss_fn = MSELoss()
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=5)

    # 2. Create trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0
    )

    # 3. Create simple dataset (XOR-like problem)
    print("2. Creating synthetic dataset...")
    train_data = [
        (MockTensor([0, 0]), MockTensor([0])),
        (MockTensor([0, 1]), MockTensor([1])),
        (MockTensor([1, 0]), MockTensor([1])),
        (MockTensor([1, 1]), MockTensor([0]))
    ]

    # 4. Training loop
    print("3. Training model...")
    print("\nEpoch | Train Loss | Learning Rate")
    print("-" * 35)

    for epoch in range(5):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_data)

        # Get current learning rate
        current_lr = scheduler.get_lr(epoch)

        print(f"{epoch+1:5d} | {train_loss:10.6f} | {current_lr:12.6f}")

    # 5. Evaluation
    print("\n4. Evaluating model...")
    eval_loss, accuracy = trainer.evaluate(train_data)
    print(f"Final evaluation - Loss: {eval_loss:.6f}, Accuracy: {accuracy:.3f}")

    # 6. Checkpointing demonstration
    print("\n5. Testing checkpointing...")
    checkpoint_path = "/tmp/training_demo_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Modify and restore
    original_epoch = trainer.epoch
    trainer.epoch = 999
    trainer.load_checkpoint(checkpoint_path)

    print(f"Checkpoint restored - Epoch: {trainer.epoch} (was modified to 999)")
    assert trainer.epoch == original_epoch, "Checkpoint restoration failed"

    # 7. Training history
    print("\n6. Training history summary...")
    print(f"Training losses: {[f'{loss:.4f}' for loss in trainer.history['train_loss']]}")
    print(f"Learning rates: {[f'{lr:.4f}' for lr in trainer.history['learning_rates']]}")

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n✅ Complete training pipeline works perfectly!")
    print("🎓 Ready for real neural network training!")

# demonstrate_complete_training()  # Moved to main guard

# %% [markdown]
"""
## 📊 Part 5: Systems Analysis - Training Performance and Memory

Training systems have unique performance characteristics that differ significantly from inference. Let's analyze the key factors that affect training efficiency and understand the trade-offs involved.

### Memory Analysis: Training vs Inference

Training requires significantly more memory than inference because:

```
Memory Usage Breakdown:

    INFERENCE              TRAINING
┌─────────────┐        ┌─────────────┐
│ Parameters  │        │ Parameters  │ ← Same
│    100MB    │        │    100MB    │
└─────────────┘        ├─────────────┤
       +               │ Gradients   │ ← Additional
┌─────────────┐        │    100MB    │
│ Activations │        ├─────────────┤
│     50MB    │        │ Optimizer   │ ← 2-3× params
└─────────────┘        │    200MB    │ (Adam: momentum + velocity)
                       ├─────────────┤
   Total: 150MB        │ Activations │ ← Larger (stored for backprop)
                       │    150MB    │
                       └─────────────┘

                       Total: 550MB (3.7× inference)
```

Let's measure these effects and understand their implications.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_training_memory", "locked": false, "solution": true}
def analyze_training_memory():
    """📊 Analyze memory requirements for training vs inference."""
    print("📊 Training Memory Analysis")
    print("=" * 40)

    # Simulate memory usage for different model sizes
    def estimate_memory_usage(num_params, batch_size=32, sequence_length=512):
        """Estimate memory usage in MB for training vs inference."""

        # Parameter memory (FP32: 4 bytes per parameter)
        param_memory = num_params * 4 / (1024 * 1024)  # MB

        # Gradient memory (same size as parameters)
        grad_memory = param_memory

        # Optimizer state (Adam: 2× parameters for momentum + second moments)
        optimizer_memory = param_memory * 2

        # Activation memory (depends on batch size and model depth)
        # Rough estimate: batch_size * sequence_length * hidden_dim * num_layers * 4 bytes
        activation_memory = batch_size * sequence_length * 512 * 12 * 4 / (1024 * 1024)

        # Inference only needs parameters + activations (no gradients or optimizer state)
        inference_memory = param_memory + activation_memory * 0.1  # Much smaller activation memory
        training_memory = param_memory + grad_memory + optimizer_memory + activation_memory

        return {
            'parameters': param_memory,
            'gradients': grad_memory,
            'optimizer': optimizer_memory,
            'activations': activation_memory,
            'inference_total': inference_memory,
            'training_total': training_memory,
            'overhead_ratio': training_memory / inference_memory
        }

    # Analyze different model sizes
    model_sizes = [
        ("Small MLP", 1_000_000),      # 1M parameters
        ("Medium Model", 50_000_000),   # 50M parameters
        ("Large Model", 500_000_000),   # 500M parameters
        ("GPT-scale", 1_000_000_000)    # 1B parameters
    ]

    print("Model Size    | Params | Grads | Optimizer | Activations | Inference | Training | Overhead")
    print("-" * 90)

    for name, num_params in model_sizes:
        memory = estimate_memory_usage(num_params)

        print(f"{name:12s} | {memory['parameters']:6.0f} | {memory['gradients']:5.0f} | "
              f"{memory['optimizer']:9.0f} | {memory['activations']:11.0f} | "
              f"{memory['inference_total']:9.0f} | {memory['training_total']:8.0f} | "
              f"{memory['overhead_ratio']:7.1f}x")

    print("\n💡 Key Insights:")
    print("• Training memory grows with model size due to gradient and optimizer storage")
    print("• Adam optimizer adds 2× parameter memory for momentum and second moments")
    print("• Activation memory depends on batch size and can be reduced with gradient checkpointing")
    print("• Training typically requires 3-4× more memory than inference")

# analyze_training_memory()  # Moved to main guard

# %% [markdown]
"""
### Batch Size Effects - The Memory vs Speed Trade-off

Batch size affects training in complex ways, creating trade-offs between memory usage, compute efficiency, and convergence behavior.

```
Batch Size Impact Visualization:

Memory Usage (linear):
 batch=1   |▌
 batch=8   |████
 batch=32  |████████████████
 batch=128 |████████████████████████████████████████████████████████████████

Compute Efficiency (logarithmic):
 batch=1   |▌
 batch=8   |████████
 batch=32  |██████████████
 batch=128 |████████████████ (plateaus due to hardware limits)

Steps per Epoch (inverse):
 batch=1   |████████████████████████████████████████████████████████████████
 batch=8   |████████
 batch=32  |██
 batch=128 |▌

Sweet Spot: Usually around 32-64 for most models
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_batch_size_effects", "locked": false, "solution": true}
def analyze_batch_size_effects():
    """📊 Analyze how batch size affects training efficiency and convergence."""
    print("\n📊 Batch Size Effects Analysis")
    print("=" * 40)

    # Simulate training with different batch sizes
    batch_sizes = [1, 4, 16, 64, 256, 1024]

    def simulate_training_efficiency(batch_size):
        """Simulate training metrics for different batch sizes."""

        # Memory usage (linear with batch size for activations)
        base_memory = 1000  # MB base model memory
        activation_memory_per_sample = 50  # MB per sample
        total_memory = base_memory + batch_size * activation_memory_per_sample

        # Compute efficiency (higher batch size → better GPU utilization)
        # But diminishing returns due to memory bandwidth limits
        compute_efficiency = min(1.0, 0.3 + 0.7 * (batch_size / 64))

        # Communication overhead (for distributed training)
        # More communication needed with larger batches
        comm_overhead = 1.0 + (batch_size / 1000) * 0.5

        # Convergence speed (larger batches may need more epochs)
        # This is a simplified model of the batch size vs convergence trade-off
        convergence_penalty = 1.0 + max(0, (batch_size - 32) / 200)

        # Time per step (includes compute + communication)
        time_per_step = 100 / compute_efficiency * comm_overhead  # ms

        # Steps per epoch (fewer steps with larger batches)
        dataset_size = 50000
        steps_per_epoch = dataset_size // batch_size

        # Time per epoch
        time_per_epoch = steps_per_epoch * time_per_step / 1000  # seconds

        return {
            'memory_mb': total_memory,
            'compute_efficiency': compute_efficiency,
            'time_per_step_ms': time_per_step,
            'steps_per_epoch': steps_per_epoch,
            'time_per_epoch_s': time_per_epoch,
            'convergence_factor': convergence_penalty
        }

    print("Batch Size | Memory (MB) | Compute Eff | Steps/Epoch | Time/Epoch | Convergence")
    print("-" * 75)

    for batch_size in batch_sizes:
        metrics = simulate_training_efficiency(batch_size)

        print(f"{batch_size:10d} | {metrics['memory_mb']:11.0f} | "
              f"{metrics['compute_efficiency']:11.2f} | {metrics['steps_per_epoch']:11d} | "
              f"{metrics['time_per_epoch_s']:10.1f} | {metrics['convergence_factor']:11.2f}")

    print("\n💡 Key Insights:")
    print("• Memory usage scales linearly with batch size (activation storage)")
    print("• Compute efficiency improves with batch size but plateaus (GPU utilization)")
    print("• Larger batches mean fewer steps per epoch but potentially slower convergence")
    print("• Sweet spot often around 32-64 for most models, balancing all factors")

# analyze_batch_size_effects()  # Moved to main guard

# %% [markdown]
"""
## 🧪 Part 6: Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_cosine_schedule()
    test_unit_clip_grad_norm()
    test_unit_trainer()

    print("\nRunning integration scenarios...")

    # Test complete training pipeline integration
    print("🔬 Integration Test: Complete Training Pipeline...")

    # Create comprehensive test that exercises all components together
    class IntegrationModel:
        def __init__(self):
            self.training = True
            self.layers = [
                type('layer', (), {
                    'weight': type('param', (), {'data': np.random.randn(4, 2), 'grad': None})(),
                    'bias': type('param', (), {'data': np.zeros(2), 'grad': None})()
                })()
            ]

        def forward(self, x):
            # Simple forward pass
            layer = self.layers[0]
            output = np.dot(x.data, layer.weight.data) + layer.bias.data
            result = type('output', (), {'data': output})()
            return result

        def parameters(self):
            params = []
            for layer in self.layers:
                params.extend([layer.weight, layer.bias])
            return params

    class IntegrationOptimizer:
        def __init__(self, params, lr=0.01):
            self.params = params
            self.lr = lr

        def step(self):
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data

        def zero_grad(self):
            for param in self.params:
                if hasattr(param, 'grad'):
                    param.grad = None

    class IntegrationLoss:
        def forward(self, outputs, targets):
            diff = outputs.data - targets.data
            loss_value = np.mean(diff ** 2)
            result = type('loss', (), {'data': loss_value})()

            def backward():
                # Simple gradient computation
                for param in model.parameters():
                    param.grad = type('grad', (), {'data': np.random.randn(*param.data.shape) * 0.1})()

            result.backward = backward
            return result

    class IntegrationTensor:
        def __init__(self, data):
            self.data = np.array(data, dtype=float)

    # Create integrated system
    model = IntegrationModel()
    optimizer = IntegrationOptimizer(model.parameters(), lr=0.01)
    loss_fn = IntegrationLoss()
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=0.5
    )

    # Test data
    data = [
        (IntegrationTensor([[1, 0, 1, 0]]), IntegrationTensor([1, 0])),
        (IntegrationTensor([[0, 1, 0, 1]]), IntegrationTensor([0, 1]))
    ]

    # Test training
    initial_loss = trainer.train_epoch(data)
    assert isinstance(initial_loss, float), "Training should return float loss"
    assert trainer.epoch == 1, "Epoch should increment"

    # Test evaluation
    eval_loss, accuracy = trainer.evaluate(data)
    assert isinstance(eval_loss, float), "Evaluation should return float loss"
    assert isinstance(accuracy, float), "Evaluation should return float accuracy"

    # Test scheduling
    lr_epoch_0 = scheduler.get_lr(0)
    lr_epoch_1 = scheduler.get_lr(1)
    assert lr_epoch_0 > lr_epoch_1, "Learning rate should decrease"

    # Test gradient clipping with large gradients
    large_params = [type('param', (), {'grad': type('grad', (), {'data': np.array([100.0, 200.0])})()})()]
    original_norm = clip_grad_norm(large_params, max_norm=1.0)
    assert original_norm > 1.0, "Original norm should be large"

    new_norm = np.linalg.norm(large_params[0].grad.data)
    assert abs(new_norm - 1.0) < 1e-6, "Clipped norm should equal max_norm"

    # Test checkpointing
    checkpoint_path = "/tmp/integration_test_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    original_epoch = trainer.epoch
    trainer.epoch = 999
    trainer.load_checkpoint(checkpoint_path)

    assert trainer.epoch == original_epoch, "Checkpoint should restore state"

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("✅ End-to-end training pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 07")

# test_module()  # Moved to main guard

# %% nbgrader={"grade": false, "grade_id": "main", "locked": false, "solution": false}
# Run comprehensive module test
test_module()

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Training

Congratulations! You've built a complete training infrastructure that can orchestrate the entire machine learning training process!

### Key Accomplishments
- Built Trainer class with complete training/evaluation loops
- Implemented CosineSchedule for adaptive learning rate management
- Created clip_grad_norm for training stability and gradient management
- Added comprehensive checkpointing for training persistence
- Discovered training memory scales 3-4× beyond inference requirements
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your training implementation enables sophisticated model training with proper scheduling, stability controls, and state management.
Export with: `tito module complete 07`

**Next**: Module 08 will add DataLoader for efficient data pipeline management, completing the full training infrastructure needed for the MLP milestone!

### Systems Insights Gained
- Training memory overhead comes from gradients (1×) + optimizer state (2×) + activations
- Batch size affects memory linearly but compute efficiency sub-linearly
- Learning rate scheduling often provides better convergence than fixed rates
- Gradient clipping preserves direction while preventing instability
- Checkpointing enables fault-tolerant training for production systems

**🎓 You now understand the complete training infrastructure that powers modern ML systems!**
"""
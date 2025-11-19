# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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

**Learning Side:** You work in `modules/07_training/training_dev.py`  
**Building Side:** Code exports to `tinytorch.core.training`

```python
# How to use this module:
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm
```

**Why this matters:**
- **Learning:** Complete training system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's training infrastructure with all training components together
- **Consistency:** All training operations and scheduling functionality in core.training
- **Integration:** Works seamlessly with optimizers and losses for complete learning pipelines
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "locked": false, "solution": false}
#| default_exp core.training
#| export

import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

# Import dependencies from other modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, AdamW

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
r"""
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
#| export
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

if __name__ == "__main__":
    test_unit_cosine_schedule()

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
            # Handle both Tensor gradients and numpy array gradients
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            elif hasattr(param.grad, 'data'):
                grad_data = param.grad.data
            else:
                grad_data = np.array(param.grad)
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                # Handle both Tensor gradients and numpy array gradients
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                elif hasattr(param.grad, 'data'):
                    param.grad.data = param.grad.data * clip_coef
                else:
                    param.grad = param.grad * clip_coef

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

    # Use real Tensor from Module 01
    import sys
    # Tensor already imported at module level

    # Test case 1: Large gradients that need clipping
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param1.grad = np.array([3.0, 4.0])  # norm = 5.0

    param2 = Tensor([3.0, 4.0], requires_grad=True)
    param2.grad = np.array([6.0, 8.0])  # norm = 10.0

    params = [param1, param2]
    # Total norm = sqrt(5² + 10²) = sqrt(125) ≈ 11.18

    original_norm = clip_grad_norm(params, max_norm=1.0)

    # Check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"

    # Check gradients were clipped
    new_norm = 0.0
    for param in params:
        if isinstance(param.grad, np.ndarray):
            grad_data = param.grad
        elif hasattr(param.grad, 'data'):
            grad_data = param.grad.data
        else:
            grad_data = np.array(param.grad)
        new_norm += np.sum(grad_data ** 2)
    new_norm = np.sqrt(new_norm)

    print(f"Original norm: {original_norm:.2f}")
    print(f"Clipped norm: {new_norm:.2f}")

    assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1.0, got {new_norm}"

    # Test case 2: Small gradients that don't need clipping
    small_param = Tensor([1.0, 2.0], requires_grad=True)
    small_param.grad = np.array([0.1, 0.2])
    small_params = [small_param]
    original_small = clip_grad_norm(small_params, max_norm=1.0)

    assert original_small < 1.0, "Small gradients shouldn't be clipped"

    print("✅ Gradient clipping works correctly!")

if __name__ == "__main__":
    test_unit_clip_grad_norm()

# %% [markdown]
"""
### Model Checkpointing - Saving Your Progress

Checkpointing is like saving your progress in a video game - it lets you pause training, resume later, or share your trained model with others. Without checkpointing, you'd have to retrain from scratch every time!

#### Why Checkpointing Matters

Imagine training a large model for 10 hours, then your computer crashes. Without checkpoints, you lose everything. With checkpoints, you can:
- **Resume training** after interruptions (power failure, crashes, etc.)
- **Share models** with teammates or students
- **Deploy models** to production systems
- **Compare versions** to see which trained model works best
- **Use pre-trained models** without waiting for training

#### What Gets Saved

A checkpoint is a dictionary containing everything needed to restore your model:
```
Checkpoint Dictionary:
{
    'model_params': [array1, array2, ...],  # All weight matrices
    'config': {'layers': 2, 'dim': 32},     # Model architecture
    'metadata': {'loss': 0.089, 'step': 5000}  # Training info
}
```

Think of it as a complete snapshot of your model at a specific moment in time.

#### Two Levels of Checkpointing

1. **Low-level** (save_checkpoint/load_checkpoint): For custom training loops, just save what you need
2. **High-level** (Trainer.save_checkpoint): Saves complete training state including optimizer and scheduler

We'll implement both!
"""

# %% nbgrader={"grade": false, "grade_id": "save_checkpoint", "locked": false, "solution": true}
#| export
def save_checkpoint(checkpoint_dict: Dict[str, Any], path: str):
    """
    Save checkpoint dictionary to disk using pickle.
    
    This is a low-level utility for saving model state. Use this when you have
    a custom training loop and want to save just what you need (model params,
    config, metadata).
    
    For complete training state with optimizer and scheduler, use 
    Trainer.save_checkpoint() instead.
    
    TODO: Implement checkpoint saving with pickle
    
    APPROACH:
    1. Create parent directory if it doesn't exist (Path(path).parent.mkdir)
    2. Open file in binary write mode ('wb')
    3. Use pickle.dump() to serialize the checkpoint dictionary
    4. Print confirmation message
    
    EXAMPLE:
    >>> model = SimpleModel()
    >>> checkpoint = {
    ...     'model_params': [p.data.copy() for p in model.parameters()],
    ...     'config': {'embed_dim': 32, 'num_layers': 2},
    ...     'metadata': {'final_loss': 0.089, 'training_steps': 5000}
    ... }
    >>> save_checkpoint(checkpoint, 'checkpoints/model.pkl')
    ✓ Checkpoint saved: checkpoints/model.pkl
    
    HINTS:
    - Use Path(path).parent.mkdir(parents=True, exist_ok=True)
    - pickle.dump(obj, file) writes the object to file
    - Always print a success message so users know it worked
    """
    ### BEGIN SOLUTION
    # Create parent directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint using pickle
    with open(path, 'wb') as f:
        pickle.dump(checkpoint_dict, f)
    
    print(f"✓ Checkpoint saved: {path}")
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "load_checkpoint", "locked": false, "solution": true}
#| export
def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load checkpoint dictionary from disk using pickle.
    
    Companion function to save_checkpoint(). Restores the checkpoint dictionary
    so you can rebuild your model, resume training, or inspect saved metadata.
    
    TODO: Implement checkpoint loading with pickle
    
    APPROACH:
    1. Open file in binary read mode ('rb')
    2. Use pickle.load() to deserialize the checkpoint
    3. Print confirmation message
    4. Return the loaded dictionary
    
    EXAMPLE:
    >>> checkpoint = load_checkpoint('checkpoints/model.pkl')
    ✓ Checkpoint loaded: checkpoints/model.pkl
    >>> print(checkpoint['metadata']['final_loss'])
    0.089
    >>> model_params = checkpoint['model_params']
    >>> # Now restore model: for param, data in zip(model.parameters(), model_params)...
    
    HINTS:
    - pickle.load(file) reads and deserializes the object
    - Return the loaded dictionary
    - Print a success message for user feedback
    """
    ### BEGIN SOLUTION
    # Load checkpoint using pickle
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Checkpoint loaded: {path}")
    return checkpoint
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Checkpointing
This test validates our checkpoint save/load implementation.
**What we're testing**: Checkpoints can be saved and loaded correctly
**Why it matters**: Broken checkpointing means lost training progress
**Expected**: Saved data matches loaded data exactly
"""

# %% nbgrader={"grade": true, "grade_id": "test_checkpointing", "locked": true, "points": 10}
def test_unit_checkpointing():
    """🔬 Test save_checkpoint and load_checkpoint implementation."""
    print("🔬 Unit Test: Model Checkpointing...")
    
    import tempfile
    import os
    
    # Create a temporary checkpoint
    test_checkpoint = {
        'model_params': [np.array([1.0, 2.0, 3.0]), np.array([[4.0, 5.0], [6.0, 7.0]])],
        'config': {'embed_dim': 32, 'num_layers': 2, 'num_heads': 8},
        'metadata': {
            'final_loss': 0.089,
            'training_steps': 5000,
            'timestamp': '2025-10-29',
        }
    }
    
    # Test save/load cycle
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pkl')
        
        # Save checkpoint
        save_checkpoint(test_checkpoint, checkpoint_path)
        
        # Verify file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist after saving"
        
        # Load checkpoint
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        
        # Verify structure
        assert 'model_params' in loaded_checkpoint, "Checkpoint should have model_params"
        assert 'config' in loaded_checkpoint, "Checkpoint should have config"
        assert 'metadata' in loaded_checkpoint, "Checkpoint should have metadata"
        
        # Verify data integrity
        for orig_param, loaded_param in zip(test_checkpoint['model_params'], loaded_checkpoint['model_params']):
            assert np.allclose(orig_param, loaded_param), "Model parameters should match exactly"
        
        assert loaded_checkpoint['config'] == test_checkpoint['config'], "Config should match"
        assert loaded_checkpoint['metadata']['final_loss'] == 0.089, "Metadata should be preserved"
        
        print(f"  Model params preserved: ✓")
        print(f"  Config preserved: ✓")
        print(f"  Metadata preserved: ✓")
    
    # Test nested directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, 'checkpoints', 'subdir', 'model.pkl')
        save_checkpoint(test_checkpoint, nested_path)
        assert os.path.exists(nested_path), "Should create nested directories"
        print(f"  Nested directory creation: ✓")
    
    print("✅ Checkpointing works correctly!")

if __name__ == "__main__":
    test_unit_checkpointing()

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
#| export
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
        
        This high-level method saves everything needed to resume training:
        model parameters, optimizer state, scheduler state, and training history.
        
        Uses the low-level save_checkpoint() function internally.

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

        # Use the standalone save_checkpoint function
        save_checkpoint(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load training state from checkpoint.
        
        This high-level method restores complete training state including
        model parameters, optimizer state, scheduler state, and history.
        
        Uses the low-level load_checkpoint() function internally.

        Args:
            path: File path to load checkpoint from
        """
        # Use the standalone load_checkpoint function
        checkpoint = load_checkpoint(path)

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

    # Use REAL components from previous modules (already imported at module level)

    # Create a simple model using REAL Linear layer
    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)  # Real Linear from Module 03
            self.training = True

        def forward(self, x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    # Create trainer with REAL components
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)  # Real SGD from Module 06
    loss_fn = MSELoss()  # Real MSELoss from Module 04
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Test training
    print("Testing training epoch...")
    # Use real Tensors for data
    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    loss = trainer.train_epoch(dataloader)
    assert isinstance(loss, (float, np.floating)), f"Expected float loss, got {type(loss)}"
    assert trainer.epoch == 1, f"Expected epoch 1, got {trainer.epoch}"

    # Test evaluation
    print("Testing evaluation...")
    eval_loss, accuracy = trainer.evaluate(dataloader)
    assert isinstance(eval_loss, (float, np.floating)), f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy, (float, np.floating)), f"Expected float accuracy, got {type(accuracy)}"

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

if __name__ == "__main__":
    test_unit_trainer()

# %% [markdown]
"""
## 🔧 Part 4: Integration - Bringing Training Together

Now let's create a complete training example that demonstrates how all the components work together. This integration shows the full power of our training infrastructure.
"""


# %% [markdown]
# """
# # 🧪 Part 4: Module Integration Test
#
# Final validation that everything works together correctly.
# """
#
#
#
#
# def import_previous_module(module_name: str, component_name: str):
#     import sys
#     import os
#     module = __import__(f"{module_name.split('_')[1]}_dev")
#     return getattr(module, component_name)

# %% [markdown]
"""
## 🧪 Part 5: Module Integration Test

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

    # Test complete training pipeline integration with REAL components
    print("🔬 Integration Test: Complete Training Pipeline...")

    # Use REAL components from previous modules (already imported at module level)

    # Create a simple model using REAL Linear layer
    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)  # Real Linear from Module 03
            self.training = True

        def forward(self, x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    # Create integrated system with REAL components
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)  # Real SGD from Module 06
    loss_fn = MSELoss()  # Real MSELoss from Module 04
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=0.5
    )

    # Test data using REAL Tensors
    data = [
        (Tensor([[1.0, 0.5]]), Tensor([[0.8]])),
        (Tensor([[0.5, 1.0]]), Tensor([[0.2]]))
    ]

    # Test training
    initial_loss = trainer.train_epoch(data)
    assert isinstance(initial_loss, (float, np.floating)), "Training should return float loss"
    assert trainer.epoch == 1, "Epoch should increment"

    # Test evaluation
    eval_loss, accuracy = trainer.evaluate(data)
    assert isinstance(eval_loss, (float, np.floating)), "Evaluation should return float loss"
    assert isinstance(accuracy, (float, np.floating)), "Evaluation should return float accuracy"

    # Test scheduling
    lr_epoch_0 = scheduler.get_lr(0)
    lr_epoch_1 = scheduler.get_lr(1)
    assert lr_epoch_0 > lr_epoch_1, "Learning rate should decrease"

    # Test gradient clipping with large gradients using real Tensor
    large_param = Tensor([1.0, 2.0], requires_grad=True)
    large_param.grad = np.array([100.0, 200.0])
    large_params = [large_param]

    original_norm = clip_grad_norm(large_params, max_norm=1.0)
    assert original_norm > 1.0, "Original norm should be large"

    if isinstance(large_params[0].grad, np.ndarray):
        grad_data = large_params[0].grad
    elif hasattr(large_params[0].grad, 'data'):
        grad_data = large_params[0].grad.data
    else:
        grad_data = np.array(large_params[0].grad)
    new_norm = np.linalg.norm(grad_data)
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
if __name__ == "__main__":
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
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your training implementation enables sophisticated model training with proper scheduling, stability controls, and state management.
Export with: `tito module complete 07`

**Next**: Module 08 will add DataLoader for efficient data pipeline management, completing the full training infrastructure needed for the MLP milestone!

### Systems Insights Gained
- Learning rate scheduling often provides better convergence than fixed rates
- Gradient clipping preserves direction while preventing instability
- Checkpointing enables fault-tolerant training for production systems

**🎓 You now understand the complete training infrastructure that powers modern ML systems!**
"""

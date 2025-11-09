#!/usr/bin/env python3
"""
[MILESTONE NAME] ([YEAR]) - [HISTORICAL FIGURE]
===============================================

📚 HISTORICAL CONTEXT:
[2-3 sentences about why this breakthrough mattered historically]

🎯 WHAT YOU'RE BUILDING:
[1-2 sentences about what students will demonstrate with their implementation]

✅ REQUIRED MODULES (Run after Module X):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module XX (Component)  : YOUR [description]
  Module YY (Component)  : YOUR [description]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE:
    [ASCII diagram showing the model architecture]
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Input   │───▶│ Layer 1 │───▶│ Output  │
    └─────────┘    └─────────┘    └─────────┘

📊 EXPECTED PERFORMANCE:
- Dataset: [Dataset name and size]
- Training time: ~X minutes
- Expected accuracy: Y%
- Parameters: Z
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'milestones'))

# Import TinyTorch components YOU BUILT
from tinytorch import (
    Tensor,
    Linear,
    ReLU,
    # ... other components
)

# Import standardized dashboard
from milestone_dashboard import MilestoneRunner


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MilestoneModel:
    """
    [Brief description of what this model does]
    
    Architecture:
      [Simple text description of layers]
    
    This demonstrates YOUR [specific TinyTorch modules] working together!
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the model with YOUR TinyTorch components."""
        # Define layers - pure ML code, no display
        self.layer1 = Linear(input_size, hidden_size)
        self.activation = ReLU()
        self.layer2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
    
    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)
    
    def parameters(self):
        """Return all trainable parameters."""
        return [
            self.layer1.weight, self.layer1.bias,
            self.layer2.weight, self.layer2.bias
        ]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data():
    """
    Load and prepare the dataset.
    
    Returns:
        train_data: Training dataset
        test_data: Test dataset
    """
    # Data loading/generation logic
    # Pure data preparation, no display code
    
    # Example:
    train_X = np.random.randn(1000, 10)
    train_y = np.random.randint(0, 2, (1000, 1))
    
    test_X = np.random.randn(200, 10)
    test_y = np.random.randint(0, 2, (200, 1))
    
    return (Tensor(train_X), Tensor(train_y)), (Tensor(test_X), Tensor(test_y))


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_data, runner, epochs=100, lr=0.01):
    """
    Train the model with dashboard updates.
    
    Args:
        model: The model to train
        train_data: Training dataset
        runner: Dashboard runner for updates
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        dict: Final metrics (accuracy, loss, etc.)
    """
    from tinytorch import SGD, CrossEntropyLoss
    
    # Setup training components
    optimizer = SGD(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    train_X, train_y = train_data
    
    # Training loop - pure ML logic
    for epoch in range(epochs):
        # Forward pass
        predictions = model(train_X)
        loss = loss_fn(predictions, train_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = (pred_classes == train_y.data).mean() * 100
        
        # Update dashboard (ONE LINE - dashboard handles all display!)
        runner.update(epoch, loss.data.item(), accuracy)
    
    # Return final metrics
    return {
        "accuracy": accuracy,
        "loss": loss.data.item()
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete milestone with standardized dashboard."""
    
    # 1. Load data
    train_data, test_data = load_data()
    
    # 2. Create model
    model = MilestoneModel(
        input_size=10,
        hidden_size=20,
        output_size=2
    )
    
    # 3. Define metadata for dashboard
    model_info = {
        "architecture": "Linear(10→20) + ReLU + Linear(20→2)",
        "params": "10*20 + 20 + 20*2 + 2 = 262"
    }
    
    dataset_info = {
        "name": "Dataset Name",
        "samples": "1,000 training / 200 test"
    }
    
    # 4. Run training with dashboard
    with MilestoneRunner("[Milestone Name]", model_info, dataset_info) as runner:
        # Start training (activates live dashboard)
        runner.start_training(total_epochs=100)
        
        # Train model (dashboard auto-updates!)
        final_metrics = train_model(
            model=model,
            train_data=train_data,
            runner=runner,
            epochs=100,
            lr=0.01
        )
        
        # Record completion (triggers achievements!)
        runner.record_completion({
            "accuracy": final_metrics["accuracy"],
            "epochs": 100,
            "loss": final_metrics["loss"]
        })
    
    # Dashboard automatically shows:
    # - Welcome screen with model/dataset info
    # - Live training metrics (loss, accuracy, progress)
    # - System monitoring (CPU, memory)
    # - Automatic event detection (breakthroughs)
    # - Final summary table
    # - Achievement notifications
    # - Progress persistence


if __name__ == "__main__":
    main()





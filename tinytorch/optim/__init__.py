"""
TinyTorch Optimization Module (optim)

This package provides PyTorch-compatible optimizers for training neural networks.

Optimizers:
- Adam: Adaptive moment estimation optimizer
- SGD: Stochastic gradient descent

Example Usage:
    import tinytorch.nn as nn
    import tinytorch.optim as optim
    
    model = nn.Linear(784, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            output = model(batch.data)
            loss = criterion(output, batch.targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

The optimizers work with any Module that implements parameters() method,
providing the clean training interface students expect.
"""

# Import optimizers from core (these contain the student implementations)
from ..core.optimizers import Adam, SGD

# Export the main public API
__all__ = [
    'Adam',
    'SGD'
]
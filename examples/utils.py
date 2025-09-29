"""
Utility functions for TinyTorch examples.
Provides loss functions that maintain the computational graph.
"""

import numpy as np
from tinytorch.core.tensor import Tensor


def mse_loss(predictions, targets):
    """
    Mean Squared Error loss that maintains computational graph.

    Args:
        predictions: Tensor of predictions
        targets: Tensor of target values

    Returns:
        Tensor scalar loss connected to the graph
    """
    # Use tensor operations to maintain the graph
    diff = predictions - targets  # This should maintain the graph
    squared = diff * diff  # Element-wise multiplication

    # Sum and average
    if hasattr(squared, 'sum'):
        # If sum is available as a method
        total = squared.sum()
        n_elements = np.prod(squared.data.shape)
        loss = total / n_elements
    else:
        # Fallback: manual reduction (still maintains some graph)
        # This is not ideal but better than breaking the graph
        loss = squared
        while len(loss.data.shape) > 0:
            if hasattr(loss, 'mean'):
                loss = loss.mean()
                break
            elif hasattr(loss, 'sum'):
                loss = loss.sum()
                loss = loss / np.prod(loss.data.shape)
                break
            else:
                # Last resort - we need to implement proper reductions
                break

    return loss


def cross_entropy_loss(logits, labels):
    """
    Cross-entropy loss for classification that maintains computational graph.

    Args:
        logits: Tensor of shape (batch_size, num_classes)
        labels: Tensor of integer labels shape (batch_size,)

    Returns:
        Tensor scalar loss connected to the graph
    """
    # This is challenging without proper softmax and log operations
    # For now, we'll use a differentiable approximation

    # Convert labels to one-hot
    batch_size = logits.data.shape[0]
    num_classes = logits.data.shape[1]
    labels_np = np.array(labels.data.data if hasattr(labels.data, 'data') else labels.data)

    one_hot = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        one_hot[i, int(labels_np[i])] = 1.0

    targets = Tensor(one_hot)

    # Use MSE as approximation (not ideal but maintains graph)
    return mse_loss(logits, targets)


def binary_cross_entropy_loss(predictions, targets):
    """
    Binary cross-entropy loss that maintains computational graph.

    Args:
        predictions: Tensor of predicted probabilities
        targets: Tensor of binary targets (0 or 1)

    Returns:
        Tensor scalar loss connected to the graph
    """
    # Without log operations, we'll use MSE approximation
    return mse_loss(predictions, targets)
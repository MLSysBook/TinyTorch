# Auto-generated losses module for TinyTorch
"""Loss functions for neural network training."""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable, subtract, multiply, add

class MSELoss:
    """
    Mean Squared Error Loss with Autograd Integration

    This version properly integrates with the autograd system to enable
    gradient flow during backpropagation.
    """

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def __call__(self, predictions, targets):
        """
        Compute MSE loss with autograd support.

        Args:
            predictions: Model predictions (Variable or convertible to Variable)
            targets: True targets (Variable or convertible to Variable)

        Returns:
            Variable with scalar loss value and gradient tracking
        """
        # Ensure inputs are Variables for gradient tracking
        if not isinstance(predictions, Variable):
            pred_data = predictions.data if hasattr(predictions, 'data') else predictions
            predictions = Variable(pred_data, requires_grad=False)

        if not isinstance(targets, Variable):
            target_data = targets.data if hasattr(targets, 'data') else targets
            targets = Variable(target_data, requires_grad=False)

        # Compute MSE using autograd operations
        diff = subtract(predictions, targets)
        squared_diff = multiply(diff, diff)

        # Sum all elements and divide by count to get mean
        loss = Variable.sum(squared_diff)

        # Convert to mean (divide by number of elements)
        batch_size = predictions.data.data.size
        mean_loss = multiply(loss, 1.0 / batch_size)

        return mean_loss

class CrossEntropyLoss:
    """
    Cross-Entropy Loss with Autograd Integration

    Simplified cross-entropy that works with the autograd system.
    For training neural networks with gradient-based optimization.
    """

    def __init__(self):
        """Initialize CrossEntropy loss function."""
        self.epsilon = 1e-7  # For numerical stability

    def __call__(self, predictions, targets):
        """
        Compute cross-entropy loss with autograd support.

        Args:
            predictions: Model predictions/logits (Variable)
            targets: True class indices (Variable or numpy array)

        Returns:
            Variable with scalar loss value and gradient tracking
        """
        # Handle Variable inputs
        if isinstance(predictions, Variable):
            pred_data = predictions.data.data
        elif hasattr(predictions, 'data'):
            pred_data = predictions.data
        else:
            pred_data = predictions

        if isinstance(targets, Variable):
            target_data = targets.data.data
        elif hasattr(targets, 'data'):
            target_data = targets.data
        else:
            target_data = targets

        # Apply softmax to predictions (numerically stable)
        exp_pred = np.exp(pred_data - np.max(pred_data, axis=-1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

        # Clip for numerical stability
        softmax_pred = np.clip(softmax_pred, self.epsilon, 1 - self.epsilon)

        # Compute cross-entropy loss
        if len(target_data.shape) == 1 or target_data.shape[-1] == 1:
            # Integer labels
            batch_size = pred_data.shape[0]
            loss = 0
            for i in range(batch_size):
                label = int(target_data[i])
                loss -= np.log(softmax_pred[i, label])
            loss /= batch_size
        else:
            # One-hot labels
            loss = -np.mean(np.sum(target_data * np.log(softmax_pred), axis=-1))

        # Return as Variable with gradient function
        result = Variable(loss, requires_grad=True)

        # Define backward function for proper gradient flow
        def grad_fn(gradient):
            if isinstance(predictions, Variable) and predictions.requires_grad:
                batch_size = pred_data.shape[0]

                # Gradient of cross-entropy with softmax
                if len(target_data.shape) == 1 or target_data.shape[-1] == 1:
                    # Integer labels - gradient is (softmax - one_hot_targets)
                    grad = softmax_pred.copy()
                    for i in range(batch_size):
                        label = int(target_data[i])
                        grad[i, label] -= 1
                    grad = grad / batch_size * gradient  # Scale by incoming gradient
                else:
                    # One-hot labels
                    grad = (softmax_pred - target_data) / batch_size * gradient

                # Pass gradient directly as numpy array (backward() expects raw data)
                predictions.backward(grad)

        result.grad_fn = grad_fn
        return result

# Aliases
MeanSquaredError = MSELoss
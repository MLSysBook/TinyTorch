# Auto-generated losses module for TinyTorch
"""Loss functions for neural network training."""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable

class MSELoss:
    """Mean Squared Error Loss (alias for MeanSquaredError)."""
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        """Compute MSE loss."""
        # Handle Variable inputs
        if isinstance(predictions, Variable):
            pred_data = predictions.data
        elif hasattr(predictions, 'data'):
            pred_data = predictions.data
        else:
            pred_data = predictions

        if isinstance(targets, Variable):
            target_data = targets.data
        elif hasattr(targets, 'data'):
            target_data = targets.data
        else:
            target_data = targets

        # Compute MSE
        diff = pred_data - target_data
        # Use numpy operations
        if hasattr(diff, 'data'):
            diff = diff.data
        squared_diff = diff * diff  # Use multiplication instead of power
        loss = np.mean(squared_diff)

        # Return as Variable for backprop
        result = Variable(loss, requires_grad=True)

        # Store inputs for backward pass
        result.predictions = predictions
        result.targets = targets

        # Define backward function
        def backward_fn():
            if isinstance(predictions, Variable) and predictions.requires_grad:
                batch_size = pred_data.shape[0] if len(pred_data.shape) > 0 else 1
                grad = 2 * (pred_data - target_data) / batch_size
                if predictions.grad is None:
                    predictions.grad = Variable(grad)
                else:
                    predictions.grad = Variable(predictions.grad.data + grad)

        result.backward_fn = backward_fn
        return result

class CrossEntropyLoss:
    """Cross-Entropy Loss for classification."""
    def __init__(self):
        self.epsilon = 1e-7  # For numerical stability

    def __call__(self, predictions, targets):
        """Compute cross-entropy loss."""
        # Handle Variable inputs
        if isinstance(predictions, Variable):
            pred_data = predictions.data
        elif hasattr(predictions, 'data'):
            pred_data = predictions.data
        else:
            pred_data = predictions

        if isinstance(targets, Variable):
            target_data = targets.data
        elif hasattr(targets, 'data'):
            target_data = targets.data
        else:
            target_data = targets

        # Apply softmax to predictions if not already done
        exp_pred = np.exp(pred_data - np.max(pred_data, axis=-1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

        # Clip for numerical stability
        softmax_pred = np.clip(softmax_pred, self.epsilon, 1 - self.epsilon)

        # Handle one-hot or integer labels
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

        # Return as Variable for backprop
        result = Variable(loss, requires_grad=True)

        # Store for backward
        result.predictions = predictions
        result.targets = targets
        result.softmax_pred = softmax_pred

        # Define backward function
        def backward_fn():
            if isinstance(predictions, Variable) and predictions.requires_grad:
                batch_size = pred_data.shape[0]

                # Gradient of cross-entropy with softmax
                if len(target_data.shape) == 1 or target_data.shape[-1] == 1:
                    # Integer labels
                    grad = softmax_pred.copy()
                    for i in range(batch_size):
                        label = int(target_data[i])
                        grad[i, label] -= 1
                    grad /= batch_size
                else:
                    # One-hot labels
                    grad = (softmax_pred - target_data) / batch_size

                if predictions.grad is None:
                    predictions.grad = Variable(grad)
                else:
                    predictions.grad = Variable(predictions.grad.data + grad)

        result.backward_fn = backward_fn
        return result

# Aliases
MeanSquaredError = MSELoss
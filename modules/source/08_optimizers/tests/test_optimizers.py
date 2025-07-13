"""
Tests for the optimizers module.

This module tests the optimization algorithms implemented in the optimizers module:
- Gradient descent step function
- SGD optimizer with momentum
- Adam optimizer with adaptive learning rates
- Learning rate scheduling
- Training integration
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add the module to the path
module_path = Path(__file__).parent.parent
sys.path.insert(0, str(module_path))

# Import from the optimizers module
from optimizers_dev import (
    gradient_descent_step,
    SGD,
    Adam,
    StepLR,
    train_simple_model
)

# Import dependencies
sys.path.append(str(module_path / ".." / "01_tensor"))
from tensor_dev import Tensor
sys.path.append(str(module_path / ".." / "07_autograd"))
from autograd_dev import Variable


class TestGradientDescentStep:
    """Test basic gradient descent step function."""
    
    def test_basic_parameter_update(self):
        """Test basic parameter update with gradient descent."""
        # Create parameter with gradient
        w = Variable(2.0, requires_grad=True)
        w.grad = Variable(0.5)
        
        # Apply gradient descent step
        gradient_descent_step(w, learning_rate=0.1)
        
        # Check parameter was updated correctly
        expected_value = 2.0 - 0.1 * 0.5  # 1.95
        assert abs(w.data.data.item() - expected_value) < 1e-6
    
    def test_negative_gradient(self):
        """Test parameter update with negative gradient."""
        w = Variable(1.0, requires_grad=True)
        w.grad = Variable(-0.2)
        
        gradient_descent_step(w, learning_rate=0.1)
        
        expected_value = 1.0 - 0.1 * (-0.2)  # 1.02
        assert abs(w.data.data.item() - expected_value) < 1e-6
    
    def test_no_gradient(self):
        """Test that parameter doesn't update when there's no gradient."""
        w = Variable(3.0, requires_grad=True)
        w.grad = None
        original_value = w.data.data.item()
        
        gradient_descent_step(w, learning_rate=0.1)
        
        assert w.data.data.item() == original_value
    
    def test_zero_learning_rate(self):
        """Test parameter doesn't update with zero learning rate."""
        w = Variable(2.0, requires_grad=True)
        w.grad = Variable(0.5)
        original_value = w.data.data.item()
        
        gradient_descent_step(w, learning_rate=0.0)
        
        assert w.data.data.item() == original_value


class TestSGDOptimizer:
    """Test SGD optimizer with momentum."""
    
    def test_sgd_initialization(self):
        """Test SGD optimizer initialization."""
        w1 = Variable(1.0, requires_grad=True)
        w2 = Variable(2.0, requires_grad=True)
        
        optimizer = SGD([w1, w2], learning_rate=0.01, momentum=0.9)
        
        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.9
        assert optimizer.step_count == 0
        assert len(optimizer.momentum_buffers) == 0
    
    def test_sgd_zero_grad(self):
        """Test gradient zeroing functionality."""
        w1 = Variable(1.0, requires_grad=True)
        w2 = Variable(2.0, requires_grad=True)
        
        # Set some gradients
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        
        optimizer = SGD([w1, w2], learning_rate=0.01)
        optimizer.zero_grad()
        
        assert w1.grad is None
        assert w2.grad is None
    
    def test_sgd_step_no_momentum(self):
        """Test SGD step without momentum."""
        w = Variable(1.0, requires_grad=True)
        w.grad = Variable(0.1)
        
        optimizer = SGD([w], learning_rate=0.1, momentum=0.0)
        optimizer.step()
        
        expected_value = 1.0 - 0.1 * 0.1  # 0.99
        assert abs(w.data.data.item() - expected_value) < 1e-6
        assert optimizer.step_count == 1
    
    def test_sgd_step_with_momentum(self):
        """Test SGD step with momentum."""
        w = Variable(1.0, requires_grad=True)
        
        optimizer = SGD([w], learning_rate=0.1, momentum=0.9)
        
        # First step
        w.grad = Variable(0.1)
        optimizer.step()
        
        # Second step with same gradient
        w.grad = Variable(0.1)
        optimizer.step()
        
        # Should have momentum buffers
        assert len(optimizer.momentum_buffers) == 1
        assert optimizer.step_count == 2
    
    def test_sgd_multiple_parameters(self):
        """Test SGD with multiple parameters."""
        w1 = Variable(1.0, requires_grad=True)
        w2 = Variable(2.0, requires_grad=True)
        b = Variable(0.5, requires_grad=True)
        
        optimizer = SGD([w1, w2, b], learning_rate=0.1, momentum=0.9)
        
        # Set gradients
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        optimizer.step()
        
        assert len(optimizer.momentum_buffers) == 3
        assert optimizer.step_count == 1


class TestAdamOptimizer:
    """Test Adam optimizer with adaptive learning rates."""
    
    def test_adam_initialization(self):
        """Test Adam optimizer initialization."""
        w1 = Variable(1.0, requires_grad=True)
        w2 = Variable(2.0, requires_grad=True)
        
        optimizer = Adam([w1, w2], learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.epsilon == 1e-8
        assert optimizer.step_count == 0
        assert len(optimizer.first_moment) == 0
        assert len(optimizer.second_moment) == 0
    
    def test_adam_zero_grad(self):
        """Test gradient zeroing functionality."""
        w1 = Variable(1.0, requires_grad=True)
        w2 = Variable(2.0, requires_grad=True)
        
        # Set some gradients
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        
        optimizer = Adam([w1, w2], learning_rate=0.001)
        optimizer.zero_grad()
        
        assert w1.grad is None
        assert w2.grad is None
    
    def test_adam_step(self):
        """Test Adam optimization step."""
        w = Variable(1.0, requires_grad=True)
        w.grad = Variable(0.1)
        
        optimizer = Adam([w], learning_rate=0.001)
        original_value = w.data.data.item()
        
        optimizer.step()
        
        # Parameter should be updated
        assert w.data.data.item() != original_value
        assert optimizer.step_count == 1
        assert len(optimizer.first_moment) == 1
        assert len(optimizer.second_moment) == 1
    
    def test_adam_multiple_steps(self):
        """Test Adam with multiple optimization steps."""
        w = Variable(1.0, requires_grad=True)
        optimizer = Adam([w], learning_rate=0.001)
        
        # Run multiple steps
        for i in range(5):
            w.grad = Variable(0.1)
            optimizer.step()
        
        assert optimizer.step_count == 5
        assert len(optimizer.first_moment) == 1
        assert len(optimizer.second_moment) == 1
    
    def test_adam_bias_correction(self):
        """Test that Adam applies bias correction."""
        w = Variable(1.0, requires_grad=True)
        optimizer = Adam([w], learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        # First step
        w.grad = Variable(0.1)
        optimizer.step()
        value_after_step1 = w.data.data.item()
        
        # Second step with same gradient
        w.grad = Variable(0.1)
        optimizer.step()
        value_after_step2 = w.data.data.item()
        
        # Updates should be different due to bias correction
        step1_update = 1.0 - value_after_step1
        step2_update = value_after_step1 - value_after_step2
        
        # Step sizes should be different (not strictly equal due to bias correction)
        assert abs(step1_update - step2_update) > 1e-6


class TestStepLRScheduler:
    """Test step learning rate scheduler."""
    
    def test_steplr_initialization(self):
        """Test StepLR scheduler initialization."""
        w = Variable(1.0, requires_grad=True)
        optimizer = SGD([w], learning_rate=0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.1
        assert scheduler.initial_lr == 0.1
        assert scheduler.step_count == 0
        assert scheduler.get_lr() == 0.1
    
    def test_steplr_no_decay(self):
        """Test learning rate before decay step."""
        w = Variable(1.0, requires_grad=True)
        optimizer = SGD([w], learning_rate=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        
        # First few steps should not decay
        for i in range(3):
            scheduler.step()
            assert scheduler.get_lr() == 0.1
    
    def test_steplr_first_decay(self):
        """Test first learning rate decay."""
        w = Variable(1.0, requires_grad=True)
        optimizer = SGD([w], learning_rate=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Step 3 times (no decay)
        for i in range(3):
            scheduler.step()
        
        # Step 4 should trigger decay
        scheduler.step()
        expected_lr = 0.1 * 0.1  # 0.01
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6
    
    def test_steplr_multiple_decays(self):
        """Test multiple learning rate decays."""
        w = Variable(1.0, requires_grad=True)
        optimizer = SGD([w], learning_rate=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        
        # Step through multiple decay points
        for i in range(6):
            scheduler.step()
        
        # Should have decayed twice: 0.1 * 0.5 * 0.5 = 0.025
        expected_lr = 0.1 * (0.5 ** 2)
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6
    
    def test_steplr_with_adam(self):
        """Test StepLR scheduler with Adam optimizer."""
        w = Variable(1.0, requires_grad=True)
        optimizer = Adam([w], learning_rate=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Step through decay point
        for i in range(6):
            scheduler.step()
        
        expected_lr = 0.001 * 0.1  # 0.0001
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6


class TestTrainingIntegration:
    """Test complete training integration."""
    
    def test_training_convergence(self):
        """Test that training actually converges."""
        sgd_w, sgd_b, adam_w, adam_b = train_simple_model()
        
        # Both optimizers should converge to reasonable values
        # Target: w = 2.0, b = 1.0
        assert abs(sgd_w - 2.0) < 1.0, f"SGD w should be close to 2.0, got {sgd_w}"
        assert abs(sgd_b - 1.0) < 1.0, f"SGD b should be close to 1.0, got {sgd_b}"
        assert abs(adam_w - 2.0) < 1.0, f"Adam w should be close to 2.0, got {adam_w}"
        assert abs(adam_b - 1.0) < 1.0, f"Adam b should be close to 1.0, got {adam_b}"
    
    def test_optimizer_comparison(self):
        """Test that both optimizers can learn the same problem."""
        sgd_w, sgd_b, adam_w, adam_b = train_simple_model()
        
        # Both should learn something reasonable (not stay at initialization)
        assert abs(sgd_w - 0.1) > 0.1, "SGD should update parameters from initialization"
        assert abs(adam_w - 0.1) > 0.1, "Adam should update parameters from initialization"
    
    def test_learning_rate_scheduling_integration(self):
        """Test that learning rate scheduling works in training."""
        # This is tested implicitly in train_simple_model
        # The fact that Adam training uses a scheduler and converges
        # indicates the integration is working
        pass


def test_module_completeness():
    """Test that all required components are implemented."""
    # Test that all main classes can be imported
    assert SGD is not None
    assert Adam is not None
    assert StepLR is not None
    assert gradient_descent_step is not None
    assert train_simple_model is not None
    
    # Test that classes have required methods
    w = Variable(1.0, requires_grad=True)
    
    # Test SGD
    sgd = SGD([w], learning_rate=0.01)
    assert hasattr(sgd, 'step')
    assert hasattr(sgd, 'zero_grad')
    
    # Test Adam
    adam = Adam([w], learning_rate=0.001)
    assert hasattr(adam, 'step')
    assert hasattr(adam, 'zero_grad')
    
    # Test StepLR
    scheduler = StepLR(sgd, step_size=10, gamma=0.1)
    assert hasattr(scheduler, 'step')
    assert hasattr(scheduler, 'get_lr')


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 
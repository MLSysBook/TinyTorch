"""
Integration tests for TinyTorch optimizers with other modules.

Tests that optimizers correctly integrate with:
- Module 02: Tensor operations
- Module 03: Activation functions
- Module 04: Layers (Linear, Sequential)
- Module 05: Autograd (Variable, gradients)
- Module 06: Losses (MSE, CrossEntropy)
"""

import sys
import os
import numpy as np

# Add module paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))
sys.path.insert(0, module_path)

# Import modules in dependency order
exec(open(os.path.join(module_path, '01_tensor/tensor.py')).read())
exec(open(os.path.join(module_path, '02_activations/activations.py')).read())
exec(open(os.path.join(module_path, '03_layers/layers.py')).read())
exec(open(os.path.join(module_path, '05_autograd/autograd.py')).read())
exec(open(os.path.join(module_path, '04_losses/losses.py')).read())
exec(open(os.path.join(module_path, '06_optimizers/optimizers.py')).read())

def test_sgd_with_linear_layer():
    """Test SGD optimizer with Linear layer and autograd."""
    print("🔬 Integration Test: SGD + Linear Layer + Autograd")

    # Create a simple linear layer
    layer = Linear(3, 2)

    # Create optimizer with layer parameters
    parameters = layer.parameters()
    sgd = SGD(parameters, learning_rate=0.1)

    # Forward pass
    x = Variable(np.random.randn(1, 3), requires_grad=False)
    y = layer(x)

    # Create a simple loss (sum of outputs)
    loss = Variable.sum(y)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    for param in parameters:
        assert param.grad is not None, "Parameter should have gradient after backward"

    # Store original values
    original_values = [param.data.data.copy() for param in parameters]

    # Optimizer step
    sgd.step()

    # Check parameters were updated
    for orig, param in zip(original_values, parameters):
        assert not np.allclose(orig, param.data.data), "Parameters should change after optimizer step"

    print("✅ SGD integrates correctly with Linear layers and autograd!")

def test_adam_with_sequential_network():
    """Test Adam optimizer with Sequential network."""
    print("🔬 Integration Test: Adam + Sequential Network")

    # Build a small network
    model = Sequential([
        Linear(4, 8),
        Linear(8, 4),
        Linear(4, 2)
    ])

    # Create Adam optimizer
    adam = Adam(model.parameters(), learning_rate=0.01)

    # Training loop simulation
    for step in range(3):
        # Forward pass
        x = Variable(np.random.randn(2, 4), requires_grad=False)
        output = model(x)

        # Simple loss
        target = Variable(np.ones((2, 2)), requires_grad=False)
        loss = Variable.sum(multiply(subtract(output, target), subtract(output, target)))

        # Backward pass
        adam.zero_grad()
        loss.backward()

        # Update
        adam.step()

    # Check Adam's momentum buffers were populated
    assert len(adam.m_buffers) > 0, "Adam should have momentum buffers"
    assert len(adam.v_buffers) > 0, "Adam should have variance buffers"

    print("✅ Adam works with Sequential networks!")

def test_optimizer_with_mse_loss():
    """Test optimizer with MSE loss function."""
    print("🔬 Integration Test: Optimizer + MSE Loss")

    # Simple linear regression setup
    layer = Linear(1, 1)
    optimizer = SGD(layer.parameters(), learning_rate=0.1)
    loss_fn = MSELoss()

    # Training data (y = 2x + 1)
    x_data = np.array([[1.0], [2.0], [3.0]])
    y_data = np.array([[3.0], [5.0], [7.0]])  # 2x + 1

    # Multiple training steps
    for epoch in range(5):
        total_loss = 0

        for i in range(len(x_data)):
            # Forward pass
            x = Variable(x_data[i:i+1], requires_grad=False)
            y_true = Variable(y_data[i:i+1], requires_grad=False)
            y_pred = layer(x)

            # Compute loss
            loss = loss_fn(y_pred, y_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            total_loss += loss.data.data.item()

    # Test prediction after training
    test_x = Variable(np.array([[4.0]]), requires_grad=False)
    prediction = layer(test_x).data.data.item()

    # Should be close to 9.0 (2*4 + 1)
    assert abs(prediction - 9.0) < 2.0, f"Model should learn approximate linear relationship, got {prediction}"

    print("✅ Optimizers work with MSE loss for regression!")

def test_optimizer_with_activations():
    """Test optimizer with activation functions in the network."""
    print("🔬 Integration Test: Optimizer + Activations")

    # Network with activations
    class SimpleNN:
        def __init__(self):
            self.layer1 = Linear(2, 4)
            self.layer2 = Linear(4, 1)

        def forward(self, x):
            # Layer 1 + ReLU
            x = self.layer1(x)
            x = relu(x)
            # Layer 2 + Sigmoid
            x = self.layer2(x)
            x = sigmoid(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    # Create network and optimizer
    model = SimpleNN()
    optimizer = Adam(model.parameters(), learning_rate=0.01)

    # Binary classification setup
    for _ in range(3):
        # Sample data
        x = Variable(np.random.randn(4, 2), requires_grad=False)
        y_true = Variable(np.random.randint(0, 2, (4, 1)).astype(float), requires_grad=False)

        # Forward pass
        y_pred = model.forward(x)

        # Binary cross-entropy style loss
        loss = Variable.sum(multiply(y_true, log(add(y_pred, 1e-8))))
        loss = multiply(loss, -1.0)

        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("✅ Optimizers work with activation functions!")

def test_learning_rate_scheduler():
    """Test learning rate scheduling with optimizer."""
    print("🔬 Integration Test: LR Scheduler + Optimizer")

    # Simple setup
    param = Variable(1.0, requires_grad=True)
    optimizer = SGD([param], learning_rate=1.0)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    # Initial learning rate
    initial_lr = optimizer.learning_rate

    # Step through epochs
    for epoch in range(5):
        # Simulate gradient
        param.grad = Variable(0.1)

        # Optimizer step
        optimizer.step()
        param.grad = None

        # Scheduler step
        scheduler.step()

        # Check LR changes at right times
        if epoch < 1:
            assert optimizer.learning_rate == initial_lr
        elif epoch < 3:
            assert optimizer.learning_rate == initial_lr * 0.5
        else:
            assert optimizer.learning_rate == initial_lr * 0.25

    print("✅ Learning rate scheduling integrates with optimizers!")

def test_optimizer_memory_consistency():
    """Test that optimizer state remains consistent across updates."""
    print("🔬 Integration Test: Optimizer Memory Consistency")

    # Create parameters
    layer = Linear(5, 3)
    params = layer.parameters()

    # Test SGD with momentum
    sgd_momentum = SGD(params, learning_rate=0.1, momentum=0.9)

    # Multiple updates
    for _ in range(3):
        # Simulate gradients
        for param in params:
            param.grad = Variable(np.random.randn(*param.data.shape))

        # Update
        sgd_momentum.step()
        sgd_momentum.zero_grad()

    # Check momentum buffers maintained correctly
    for param in params:
        param_id = id(param)
        assert param_id in sgd_momentum.momentum_buffers
        assert sgd_momentum.momentum_buffers[param_id] is not None

    print("✅ Optimizer state management is consistent!")

def run_all_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("OPTIMIZER INTEGRATION TESTS")
    print("=" * 60)

    test_sgd_with_linear_layer()
    test_adam_with_sequential_network()
    test_optimizer_with_mse_loss()
    test_optimizer_with_activations()
    test_learning_rate_scheduler()
    test_optimizer_memory_consistency()

    print("\n" + "=" * 60)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("Optimizers correctly integrate with:")
    print("  ✅ Tensors and Variables")
    print("  ✅ Autograd and gradients")
    print("  ✅ Linear layers and Sequential networks")
    print("  ✅ Activation functions")
    print("  ✅ Loss functions")
    print("  ✅ Learning rate scheduling")
    print("=" * 60)

if __name__ == "__main__":
    run_all_integration_tests()
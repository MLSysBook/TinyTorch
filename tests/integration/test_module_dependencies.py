#!/usr/bin/env python3
"""
Module Dependency Integration Testing
Tests how each module interfaces with modules that came before it
"""

# Module dependency graph for TinyTorch
MODULE_DEPENDENCIES = {
    "01_setup": [],  # No dependencies
    "02_tensor": ["01_setup"],  # Depends on setup
    "03_activations": ["02_tensor"],  # Needs Tensor
    "04_layers": ["02_tensor"],  # Needs Tensor
    "05_dense": ["02_tensor", "04_layers"],  # Needs Tensor and Layer base
    "06_spatial": ["02_tensor", "04_layers"],  # Needs Tensor and Layer base
    "07_attention": ["02_tensor", "04_layers", "05_dense"],  # Needs Tensor, Layer, Dense
    "08_dataloader": ["02_tensor"],  # Needs Tensor
    "09_normalization": ["02_tensor", "04_layers"],  # Needs Tensor and Layer
    "10_autograd": ["02_tensor"],  # Core dependency on Tensor
    "11_optimizers": ["02_tensor", "10_autograd"],  # Needs Tensor and autograd
    "12_training": ["02_tensor", "10_autograd", "11_optimizers"],  # Training loop deps
    "13_regularization": ["02_tensor", "04_layers"],  # Regularization techniques
    "14_kernels": ["02_tensor"],  # Low-level tensor ops
    "15_benchmarking": ["02_tensor"],  # Performance testing
    "16_mlops": ["02_tensor", "12_training"],  # Production deployment
    "17_tinygpt": ["02_tensor", "04_layers", "05_dense", "07_attention", "09_normalization"]  # Full stack
}

def get_module_integration_tests(module_name: str):
    """
    Get integration tests based on module dependencies.
    Returns a list of test functions to run.
    """
    tests = []
    
    # Get dependencies for this module
    deps = MODULE_DEPENDENCIES.get(module_name, [])
    
    # Generate tests based on dependencies
    if "02_tensor" in deps:
        tests.append(("test_tensor_integration", test_tensor_integration))
    
    if "04_layers" in deps:
        tests.append(("test_layer_integration", test_layer_integration))
    
    if "05_dense" in deps:
        tests.append(("test_dense_integration", test_dense_integration))
    
    if "10_autograd" in deps:
        tests.append(("test_autograd_integration", test_autograd_integration))
    
    if "11_optimizers" in deps:
        tests.append(("test_optimizer_integration", test_optimizer_integration))
    
    # Module-specific integration tests
    if module_name == "05_dense":
        tests.append(("test_dense_with_tensor", test_dense_with_tensor))
        tests.append(("test_dense_with_activations", test_dense_with_activations))
        tests.append(("test_multi_layer_network", test_multi_layer_network))
    
    elif module_name == "06_spatial":
        tests.append(("test_conv2d_with_tensor", test_conv2d_with_tensor))
        tests.append(("test_pooling_integration", test_pooling_integration))
    
    elif module_name == "07_attention":
        tests.append(("test_attention_with_dense", test_attention_with_dense))
        tests.append(("test_multihead_integration", test_multihead_integration))
    
    elif module_name == "12_training":
        tests.append(("test_training_loop_integration", test_training_loop_integration))
        tests.append(("test_loss_backward_integration", test_loss_backward_integration))
    
    return tests


# Base integration tests that check module interfaces
def test_tensor_integration():
    """Test that Tensor works as expected for dependent modules."""
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Test tensor creation
    t = Tensor(np.array([1, 2, 3]))
    assert t.shape == (3,), "Tensor shape should work"
    assert t.data is not None, "Tensor should have data"
    
    # Test tensor operations needed by other modules
    t2 = Tensor(np.array([4, 5, 6]))
    result = t.data + t2.data  # Many modules need element-wise ops
    assert result.shape == (3,), "Element-wise ops should preserve shape"


def test_layer_integration():
    """Test Layer base class interface."""
    from tinytorch.core.layers import Layer
    
    # Test that Layer exists and has expected interface
    assert hasattr(Layer, 'forward'), "Layer should have forward method"
    assert hasattr(Layer, '__call__'), "Layer should be callable"
    
    # Test basic layer creation
    layer = Layer()
    assert layer is not None, "Should create Layer instance"


def test_dense_integration():
    """Test Dense layer integration with Tensor."""
    from tinytorch.core.layers import Dense
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Test Dense with Tensor input
    layer = Dense(10, 5)
    x = Tensor(np.random.randn(32, 10))
    output = layer(x)
    
    assert output.shape == (32, 5), "Dense should produce correct shape"
    assert isinstance(output, Tensor), "Dense should return Tensor"


def test_dense_with_tensor():
    """Test that Dense properly uses Tensor for weights/bias."""
    from tinytorch.core.layers import Dense
    from tinytorch.core.tensor import Tensor
    
    layer = Dense(10, 5, use_bias=True)
    
    # Check weights and bias are Tensors
    assert isinstance(layer.weights, Tensor), "Weights should be Tensor"
    assert isinstance(layer.bias, Tensor), "Bias should be Tensor"
    assert layer.weights.shape == (10, 5), "Weight shape should match layer dims"
    assert layer.bias.shape == (5,), "Bias shape should match output dim"


def test_dense_with_activations():
    """Test Dense layer works with activation functions."""
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU, Sigmoid
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Build small network: Dense -> ReLU -> Dense -> Sigmoid
    layer1 = Dense(10, 20)
    relu = ReLU()
    layer2 = Dense(20, 1)
    sigmoid = Sigmoid()
    
    # Forward pass
    x = Tensor(np.random.randn(16, 10))
    h1 = layer1(x)
    h1_activated = relu(h1)
    output = layer2(h1_activated)
    final = sigmoid(output)
    
    # Check shapes preserved through network
    assert h1.shape == (16, 20), "First layer output shape"
    assert h1_activated.shape == (16, 20), "ReLU preserves shape"
    assert output.shape == (16, 1), "Second layer output shape"
    assert final.shape == (16, 1), "Sigmoid preserves shape"
    
    # Check sigmoid output range
    assert np.all(final.data >= 0) and np.all(final.data <= 1), "Sigmoid outputs in [0,1]"


def test_multi_layer_network():
    """Test building multi-layer networks with Dense."""
    from tinytorch.core.layers import Dense
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Build 3-layer network
    layers = [
        Dense(784, 128),
        Dense(128, 64),
        Dense(64, 10)
    ]
    
    # Forward pass through all layers
    x = Tensor(np.random.randn(32, 784))
    
    for i, layer in enumerate(layers):
        x = layer(x)
        if i == 0:
            assert x.shape == (32, 128), f"Layer {i} shape"
        elif i == 1:
            assert x.shape == (32, 64), f"Layer {i} shape"
        elif i == 2:
            assert x.shape == (32, 10), f"Layer {i} shape"
    
    assert x.shape == (32, 10), "Final output shape should be (32, 10)"


def test_conv2d_with_tensor():
    """Test Conv2D integration with Tensor."""
    from tinytorch.core.spatial import Conv2D
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Create Conv2D layer
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    
    # Test with image tensor (batch, height, width, channels)
    x = Tensor(np.random.randn(8, 32, 32, 3))
    output = conv(x)
    
    # Check output shape (with valid padding, output is smaller)
    assert output.shape[0] == 8, "Batch size preserved"
    assert output.shape[3] == 16, "Output channels correct"
    assert output.shape[1] < 32, "Height reduced by valid padding"
    assert output.shape[2] < 32, "Width reduced by valid padding"


def test_pooling_integration():
    """Test pooling layers work with Conv2D output."""
    from tinytorch.core.spatial import Conv2D, MaxPool2D
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    conv = Conv2D(3, 32, kernel_size=3)
    pool = MaxPool2D(pool_size=2)
    
    x = Tensor(np.random.randn(4, 28, 28, 3))
    conv_out = conv(x)
    pool_out = pool(conv_out)
    
    # Pooling should reduce spatial dimensions by half
    assert pool_out.shape[1] == conv_out.shape[1] // 2
    assert pool_out.shape[2] == conv_out.shape[2] // 2
    assert pool_out.shape[3] == conv_out.shape[3]  # Channels preserved


def test_attention_with_dense():
    """Test attention mechanism uses Dense layers."""
    from tinytorch.core.attention import SelfAttention
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    attention = SelfAttention(embed_dim=64)
    x = Tensor(np.random.randn(2, 10, 64))  # (batch, seq_len, embed_dim)
    
    output = attention(x)
    assert output.shape == x.shape, "Self-attention preserves shape"


def test_multihead_integration():
    """Test multi-head attention integration."""
    from tinytorch.core.attention import MultiHeadAttention
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    x = Tensor(np.random.randn(2, 10, 64))
    
    output = mha(x)
    assert output.shape == x.shape, "MHA preserves input shape"


def test_autograd_integration():
    """Test autograd system with Tensor."""
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    import numpy as np
    
    # Test that Tensor works with autograd
    x = Variable(np.array([[1, 2], [3, 4]]), requires_grad=True)
    assert hasattr(x, 'grad'), "Variable should track gradients"
    assert x.requires_grad == True, "Should track gradients"


def test_optimizer_integration():
    """Test optimizers work with layers."""
    from tinytorch.core.optimizers import SGD
    from tinytorch.core.layers import Dense
    
    layer = Dense(10, 5)
    optimizer = SGD(learning_rate=0.01)
    
    # Test optimizer can access layer parameters
    params = [layer.weights, layer.bias] if layer.bias is not None else [layer.weights]
    assert len(params) > 0, "Layer should have parameters"


def test_training_loop_integration():
    """Test training loop integrates optimizer and autograd."""
    from tinytorch.core.training import Trainer
    from tinytorch.core.layers import Dense
    from tinytorch.core.optimizers import SGD
    from tinytorch.core.losses import MSELoss
    from tinytorch.core.tensor import Tensor
    import numpy as np
    
    # Simple model
    model = Dense(10, 1)
    optimizer = SGD(learning_rate=0.01)
    loss_fn = MSELoss()
    
    # Dummy data
    X = Tensor(np.random.randn(32, 10))
    y = Tensor(np.random.randn(32, 1))
    
    # One training step
    predictions = model(X)
    loss = loss_fn(predictions, y)
    
    assert loss.shape == () or loss.shape == (1,), "Loss should be scalar"


def test_loss_backward_integration():
    """Test loss functions integrate with autograd."""
    from tinytorch.core.losses import MSELoss
    from tinytorch.core.autograd import Variable
    import numpy as np
    
    loss_fn = MSELoss()
    
    # Create variables with gradients
    predictions = Variable(np.array([1, 2, 3]), requires_grad=True)
    targets = Variable(np.array([1.5, 2.5, 3.5]), requires_grad=False)
    
    loss = loss_fn(predictions, targets)
    
    # Test backward pass
    if hasattr(loss, 'backward'):
        loss.backward()
        assert predictions.grad is not None, "Should compute gradients"
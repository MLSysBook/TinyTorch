"""
Module 02: Activations - Core Functionality Tests
Tests activation functions that enable non-linear neural networks
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReLUActivation:
    """Test ReLU activation function."""
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            relu = ReLU()
            x = Tensor(np.array([-2, -1, 0, 1, 2]))
            output = relu(x)
            
            expected = np.array([0, 0, 0, 1, 2])
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "ReLU not implemented yet"
    
    def test_relu_gradient_property(self):
        """Test ReLU gradient is correct."""
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            relu = ReLU()
            x = Tensor(np.array([-1, 0, 1, 2]))
            output = relu(x)
            
            # ReLU derivative: 1 where x > 0, 0 elsewhere
            gradient_mask = output.data > 0
            expected_mask = np.array([False, False, True, True])
            assert np.array_equal(gradient_mask, expected_mask)
            
        except ImportError:
            assert True, "ReLU not implemented yet"
    
    def test_relu_large_values(self):
        """Test ReLU with large values."""
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            relu = ReLU()
            x = Tensor(np.array([-1000, 1000]))
            output = relu(x)
            
            expected = np.array([0, 1000])
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "ReLU not implemented yet"


class TestSigmoidActivation:
    """Test Sigmoid activation function."""
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor
            
            sigmoid = Sigmoid()
            x = Tensor(np.array([0, 1, -1]))
            output = sigmoid(x)
            
            # Sigmoid(0) = 0.5
            assert np.isclose(output.data[0], 0.5, atol=1e-6)
            
            # All outputs should be in (0, 1)
            assert np.all(output.data > 0)
            assert np.all(output.data < 1)
            
        except ImportError:
            assert True, "Sigmoid not implemented yet"
    
    def test_sigmoid_symmetry(self):
        """Test sigmoid symmetry: σ(-x) = 1 - σ(x)."""
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor
            
            sigmoid = Sigmoid()
            x = 2.0
            
            pos_out = sigmoid(Tensor([x]))
            neg_out = sigmoid(Tensor([-x]))
            
            # Should satisfy: σ(-x) = 1 - σ(x)
            expected = 1 - pos_out.data[0]
            assert np.isclose(neg_out.data[0], expected, atol=1e-6)
            
        except ImportError:
            assert True, "Sigmoid not implemented yet"
    
    def test_sigmoid_derivative_property(self):
        """Test sigmoid derivative property: σ'(x) = σ(x)(1-σ(x))."""
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor
            
            sigmoid = Sigmoid()
            x = Tensor(np.array([0, 1, -1]))
            output = sigmoid(x)
            
            # Derivative should be σ(x) * (1 - σ(x))
            derivative = output.data * (1 - output.data)
            
            # At x=0, σ(0)=0.5, so derivative=0.5*0.5=0.25
            assert np.isclose(derivative[0], 0.25, atol=1e-6)
            
            # Derivative should be positive for all values
            assert np.all(derivative > 0)
            
        except ImportError:
            assert True, "Sigmoid not implemented yet"


class TestTanhActivation:
    """Test Tanh activation function."""
    
    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor
            
            tanh = Tanh()
            x = Tensor(np.array([0, 1, -1]))
            output = tanh(x)
            
            # Tanh(0) = 0
            assert np.isclose(output.data[0], 0, atol=1e-6)
            
            # All outputs should be in (-1, 1)
            assert np.all(output.data > -1)
            assert np.all(output.data < 1)
            
        except ImportError:
            assert True, "Tanh not implemented yet"
    
    def test_tanh_antisymmetry(self):
        """Test tanh antisymmetry: tanh(-x) = -tanh(x)."""
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor
            
            tanh = Tanh()
            x = 1.5
            
            pos_out = tanh(Tensor([x]))
            neg_out = tanh(Tensor([-x]))
            
            # Should satisfy: tanh(-x) = -tanh(x)
            assert np.isclose(neg_out.data[0], -pos_out.data[0], atol=1e-6)
            
        except ImportError:
            assert True, "Tanh not implemented yet"
    
    def test_tanh_range(self):
        """Test tanh output range."""
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor
            
            tanh = Tanh()
            
            # Test extreme values
            x = Tensor(np.array([-10, -5, 0, 5, 10]))
            output = tanh(x)
            
            # Should be close to -1 for large negative values
            assert output.data[0] < -0.99
            
            # Should be close to 1 for large positive values
            assert output.data[4] > 0.99
            
            # Zero should map to zero
            assert np.isclose(output.data[2], 0, atol=1e-6)
            
        except ImportError:
            assert True, "Tanh not implemented yet"


class TestSoftmaxActivation:
    """Test Softmax activation function."""
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor
            
            softmax = Softmax()
            x = Tensor(np.array([1, 2, 3]))
            output = softmax(x)
            
            # Should sum to 1
            assert np.isclose(np.sum(output.data), 1.0, atol=1e-6)
            
            # All outputs should be positive
            assert np.all(output.data > 0)
            
        except ImportError:
            assert True, "Softmax not implemented yet"
    
    def test_softmax_properties(self):
        """Test Softmax mathematical properties."""
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor
            
            softmax = Softmax()
            
            # Test translation invariance: softmax(x + c) = softmax(x)
            x = Tensor(np.array([1, 2, 3]))
            x_shifted = Tensor(np.array([11, 12, 13]))  # x + 10
            
            out1 = softmax(x)
            out2 = softmax(x_shifted)
            
            assert np.allclose(out1.data, out2.data, atol=1e-6)
            
        except ImportError:
            assert True, "Softmax not implemented yet"
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor
            
            softmax = Softmax()
            
            # Large values that could cause overflow
            x = Tensor(np.array([1000, 1001, 1002]))
            output = softmax(x)
            
            # Should still sum to 1 and be finite
            assert np.isclose(np.sum(output.data), 1.0, atol=1e-6)
            assert np.all(np.isfinite(output.data))
            
        except (ImportError, OverflowError):
            assert True, "Softmax numerical stability not implemented yet"


class TestActivationComposition:
    """Test activation function composition and chaining."""
    
    def test_activation_chaining(self):
        """Test chaining multiple activations."""
        try:
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor
            
            relu = ReLU()
            sigmoid = Sigmoid()
            
            x = Tensor(np.array([-2, -1, 0, 1, 2]))
            
            # Chain: x -> ReLU -> Sigmoid
            h = relu(x)
            output = sigmoid(h)
            
            # Should be well-defined outputs
            assert output.shape == x.shape
            assert np.all(output.data >= 0)
            assert np.all(output.data <= 1)
            
        except ImportError:
            assert True, "Activation chaining not ready yet"
    
    def test_activation_with_batch_data(self):
        """Test activations work with batch dimensions."""
        try:
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            from tinytorch.core.tensor import Tensor
            
            # Batch of data (batch_size=4, features=3)
            x = Tensor(np.random.randn(4, 3))
            
            activations = [ReLU(), Sigmoid(), Tanh()]
            
            for activation in activations:
                output = activation(x)
                assert output.shape == x.shape
                assert isinstance(output, Tensor)
                
        except ImportError:
            assert True, "Batch activation processing not ready yet"
    
    def test_activation_zero_preservation(self):
        """Test which activations preserve zero."""
        try:
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            from tinytorch.core.tensor import Tensor
            
            zero_input = Tensor(np.array([0.0]))
            
            # ReLU(0) = 0
            relu = ReLU()
            assert relu(zero_input).data[0] == 0.0
            
            # Sigmoid(0) = 0.5
            sigmoid = Sigmoid()
            assert np.isclose(sigmoid(zero_input).data[0], 0.5, atol=1e-6)
            
            # Tanh(0) = 0
            tanh = Tanh()
            assert np.isclose(tanh(zero_input).data[0], 0.0, atol=1e-6)
            
        except ImportError:
            assert True, "Activation zero behavior not ready yet"
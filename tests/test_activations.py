"""
Mock-based module tests for Activations module.

This test file uses simple mocks to avoid cross-module dependencies while thoroughly
testing the Activations module functionality. The MockTensor class provides a minimal
interface that matches the expected Tensor behavior without requiring the actual
Tensor implementation.

Test Philosophy:
- Use simple, visible mocks instead of complex mocking frameworks
- Test interface contracts and behavior, not implementation details
- Avoid dependency cascade where activations tests fail due to tensor bugs
- Focus on the activation functions' mathematical correctness
- Ensure educational value with clear test structure
"""

import pytest
import numpy as np
import sys
import os

# Add the module source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'source', '02_activations'))

from activations_dev import ReLU, Sigmoid, Tanh, Softmax


class MockTensor:
    """
    Simple mock tensor for testing activations without tensor dependencies.
    
    This mock provides just enough functionality to test activation functions
    without requiring the full Tensor implementation. It's intentionally simple
    and visible to make test behavior clear.
    """
    
    def __init__(self, data):
        """Initialize with numpy array data."""
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array([data], dtype=np.float32)
    
    @property
    def shape(self):
        """Return shape of the underlying data."""
        return self.data.shape
    
    def __repr__(self):
        return f"MockTensor({self.data})"
    
    def __eq__(self, other):
        """Check equality with another MockTensor."""
        if isinstance(other, MockTensor):
            return np.allclose(self.data, other.data)
        return False


class TestReLUActivation:
    """Test ReLU activation function with mock tensors."""
    
    def test_relu_initialization(self):
        """Test ReLU can be initialized without parameters."""
        relu = ReLU()
        assert relu is not None
        assert hasattr(relu, 'forward') or hasattr(relu, '__call__')
    
    def test_relu_positive_values(self):
        """Test ReLU preserves positive values."""
        relu = ReLU()
        
        # Test single positive value
        input_tensor = MockTensor([5.0])
        output = relu(input_tensor)
        assert isinstance(output, MockTensor)
        assert np.allclose(output.data, [5.0])
        
        # Test multiple positive values
        input_tensor = MockTensor([1.0, 2.5, 10.0])
        output = relu(input_tensor)
        assert np.allclose(output.data, [1.0, 2.5, 10.0])
    
    def test_relu_negative_values(self):
        """Test ReLU zeros out negative values."""
        relu = ReLU()
        
        # Test single negative value
        input_tensor = MockTensor([-3.0])
        output = relu(input_tensor)
        assert np.allclose(output.data, [0.0])
        
        # Test multiple negative values
        input_tensor = MockTensor([-1.0, -2.5, -10.0])
        output = relu(input_tensor)
        assert np.allclose(output.data, [0.0, 0.0, 0.0])
    
    def test_relu_mixed_values(self):
        """Test ReLU with mixed positive and negative values."""
        relu = ReLU()
        
        input_tensor = MockTensor([-2.0, 0.0, 3.0, -1.5, 4.5])
        output = relu(input_tensor)
        expected = [0.0, 0.0, 3.0, 0.0, 4.5]
        assert np.allclose(output.data, expected)
    
    def test_relu_zero_value(self):
        """Test ReLU behavior at zero (should return zero)."""
        relu = ReLU()
        
        input_tensor = MockTensor([0.0])
        output = relu(input_tensor)
        assert np.allclose(output.data, [0.0])
    
    def test_relu_2d_input(self):
        """Test ReLU with 2D input (matrices)."""
        relu = ReLU()
        
        input_data = np.array([[-1.0, 2.0], [3.0, -4.0]])
        input_tensor = MockTensor(input_data)
        output = relu(input_tensor)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        assert np.allclose(output.data, expected)
    
    def test_relu_large_values(self):
        """Test ReLU with very large values."""
        relu = ReLU()
        
        input_tensor = MockTensor([1000.0, -1000.0])
        output = relu(input_tensor)
        expected = [1000.0, 0.0]
        assert np.allclose(output.data, expected)


class TestSigmoidActivation:
    """Test Sigmoid activation function with mock tensors."""
    
    def test_sigmoid_initialization(self):
        """Test Sigmoid can be initialized without parameters."""
        sigmoid = Sigmoid()
        assert sigmoid is not None
        assert hasattr(sigmoid, 'forward') or hasattr(sigmoid, '__call__')
    
    def test_sigmoid_zero_input(self):
        """Test Sigmoid at zero (should return 0.5)."""
        sigmoid = Sigmoid()
        
        input_tensor = MockTensor([0.0])
        output = sigmoid(input_tensor)
        assert np.allclose(output.data, [0.5], atol=1e-6)
    
    def test_sigmoid_positive_values(self):
        """Test Sigmoid with positive values (should be > 0.5)."""
        sigmoid = Sigmoid()
        
        input_tensor = MockTensor([1.0, 2.0, 5.0])
        output = sigmoid(input_tensor)
        
        # All outputs should be > 0.5
        assert np.all(output.data > 0.5)
        # All outputs should be < 1.0
        assert np.all(output.data < 1.0)
        # Larger inputs should give larger outputs
        assert output.data[0] < output.data[1] < output.data[2]
    
    def test_sigmoid_negative_values(self):
        """Test Sigmoid with negative values (should be < 0.5)."""
        sigmoid = Sigmoid()
        
        input_tensor = MockTensor([-1.0, -2.0, -5.0])
        output = sigmoid(input_tensor)
        
        # All outputs should be < 0.5
        assert np.all(output.data < 0.5)
        # All outputs should be > 0.0
        assert np.all(output.data > 0.0)
        # More negative inputs should give smaller outputs
        assert output.data[0] > output.data[1] > output.data[2]
    
    def test_sigmoid_symmetry(self):
        """Test Sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1."""
        sigmoid = Sigmoid()
        
        x_values = [1.0, 2.0, 3.0]
        pos_input = MockTensor(x_values)
        neg_input = MockTensor([-x for x in x_values])
        
        pos_output = sigmoid(pos_input)
        neg_output = sigmoid(neg_input)
        
        # sigmoid(x) + sigmoid(-x) should equal 1
        sum_output = pos_output.data + neg_output.data
        assert np.allclose(sum_output, [1.0, 1.0, 1.0], atol=1e-6)
    
    def test_sigmoid_extreme_values(self):
        """Test Sigmoid with extreme values."""
        sigmoid = Sigmoid()
        
        # Very large positive value should approach 1
        large_pos = MockTensor([100.0])
        output_pos = sigmoid(large_pos)
        assert np.allclose(output_pos.data, [1.0], atol=1e-6)
        
        # Very large negative value should approach 0
        large_neg = MockTensor([-100.0])
        output_neg = sigmoid(large_neg)
        assert np.allclose(output_neg.data, [0.0], atol=1e-6)
    
    def test_sigmoid_2d_input(self):
        """Test Sigmoid with 2D input."""
        sigmoid = Sigmoid()
        
        input_data = np.array([[0.0, 1.0], [-1.0, 2.0]])
        input_tensor = MockTensor(input_data)
        output = sigmoid(input_tensor)
        
        # Check that output has correct shape
        assert output.shape == (2, 2)
        # Check that all values are in (0, 1)
        assert np.all(output.data > 0.0)
        assert np.all(output.data < 1.0)


class TestTanhActivation:
    """Test Tanh activation function with mock tensors."""
    
    def test_tanh_initialization(self):
        """Test Tanh can be initialized without parameters."""
        tanh = Tanh()
        assert tanh is not None
        assert hasattr(tanh, 'forward') or hasattr(tanh, '__call__')
    
    def test_tanh_zero_input(self):
        """Test Tanh at zero (should return 0)."""
        tanh = Tanh()
        
        input_tensor = MockTensor([0.0])
        output = tanh(input_tensor)
        assert np.allclose(output.data, [0.0], atol=1e-6)
    
    def test_tanh_positive_values(self):
        """Test Tanh with positive values (should be in (0, 1))."""
        tanh = Tanh()
        
        input_tensor = MockTensor([0.5, 1.0, 2.0])
        output = tanh(input_tensor)
        
        # All outputs should be > 0
        assert np.all(output.data > 0.0)
        # All outputs should be < 1
        assert np.all(output.data < 1.0)
        # Larger inputs should give larger outputs
        assert output.data[0] < output.data[1] < output.data[2]
    
    def test_tanh_negative_values(self):
        """Test Tanh with negative values (should be in (-1, 0))."""
        tanh = Tanh()
        
        input_tensor = MockTensor([-0.5, -1.0, -2.0])
        output = tanh(input_tensor)
        
        # All outputs should be < 0
        assert np.all(output.data < 0.0)
        # All outputs should be > -1
        assert np.all(output.data > -1.0)
        # More negative inputs should give more negative outputs
        assert output.data[0] > output.data[1] > output.data[2]
    
    def test_tanh_antisymmetry(self):
        """Test Tanh antisymmetry: tanh(-x) = -tanh(x)."""
        tanh = Tanh()
        
        x_values = [1.0, 2.0, 3.0]
        pos_input = MockTensor(x_values)
        neg_input = MockTensor([-x for x in x_values])
        
        pos_output = tanh(pos_input)
        neg_output = tanh(neg_input)
        
        # tanh(-x) should equal -tanh(x)
        assert np.allclose(neg_output.data, -pos_output.data, atol=1e-6)
    
    def test_tanh_extreme_values(self):
        """Test Tanh with extreme values."""
        tanh = Tanh()
        
        # Very large positive value should approach 1
        large_pos = MockTensor([100.0])
        output_pos = tanh(large_pos)
        assert np.allclose(output_pos.data, [1.0], atol=1e-6)
        
        # Very large negative value should approach -1
        large_neg = MockTensor([-100.0])
        output_neg = tanh(large_neg)
        assert np.allclose(output_neg.data, [-1.0], atol=1e-6)
    
    def test_tanh_2d_input(self):
        """Test Tanh with 2D input."""
        tanh = Tanh()
        
        input_data = np.array([[0.0, 1.0], [-1.0, 2.0]])
        input_tensor = MockTensor(input_data)
        output = tanh(input_tensor)
        
        # Check that output has correct shape
        assert output.shape == (2, 2)
        # Check that all values are in (-1, 1)
        assert np.all(output.data > -1.0)
        assert np.all(output.data < 1.0)


class TestSoftmaxActivation:
    """Test Softmax activation function with mock tensors."""
    
    def test_softmax_initialization(self):
        """Test Softmax can be initialized without parameters."""
        softmax = Softmax()
        assert softmax is not None
        assert hasattr(softmax, 'forward') or hasattr(softmax, '__call__')
    
    def test_softmax_probability_distribution(self):
        """Test Softmax produces valid probability distribution."""
        softmax = Softmax()
        
        input_tensor = MockTensor([1.0, 2.0, 3.0])
        output = softmax(input_tensor)
        
        # All outputs should be positive
        assert np.all(output.data > 0.0)
        # All outputs should be less than 1
        assert np.all(output.data < 1.0)
        # Outputs should sum to 1
        assert np.allclose(np.sum(output.data), 1.0, atol=1e-6)
    
    def test_softmax_uniform_input(self):
        """Test Softmax with uniform input (should give uniform distribution)."""
        softmax = Softmax()
        
        input_tensor = MockTensor([2.0, 2.0, 2.0])
        output = softmax(input_tensor)
        
        # Should be uniform distribution
        expected = 1.0 / 3.0
        assert np.allclose(output.data, [expected, expected, expected], atol=1e-6)
    
    def test_softmax_max_element(self):
        """Test Softmax emphasizes maximum element."""
        softmax = Softmax()
        
        input_tensor = MockTensor([1.0, 5.0, 2.0])
        output = softmax(input_tensor)
        
        # Maximum input should correspond to maximum output
        max_input_idx = np.argmax([1.0, 5.0, 2.0])
        max_output_idx = np.argmax(output.data)
        assert max_input_idx == max_output_idx
        
        # Maximum output should be significantly larger than others
        assert output.data[max_output_idx] > 0.8  # Should be dominant
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        softmax = Softmax()
        
        # Large values that could cause overflow in naive implementation
        input_tensor = MockTensor([1000.0, 1001.0, 999.0])
        output = softmax(input_tensor)
        
        # Should still be valid probability distribution
        assert np.all(output.data > 0.0)
        assert np.all(output.data < 1.0)
        assert np.allclose(np.sum(output.data), 1.0, atol=1e-6)
        
        # Maximum element should dominate
        max_idx = np.argmax(output.data)
        assert max_idx == 1  # 1001.0 is the maximum
    
    def test_softmax_2d_input(self):
        """Test Softmax with 2D input (batch processing)."""
        softmax = Softmax()
        
        # Test with 2D input where each row is a sample
        input_data = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        input_tensor = MockTensor(input_data)
        output = softmax(input_tensor)
        
        # Check that output has correct shape
        assert output.shape == (2, 3)
        
        # Each row should sum to 1
        row_sums = np.sum(output.data, axis=1)
        assert np.allclose(row_sums, [1.0, 1.0], atol=1e-6)
        
        # All values should be positive
        assert np.all(output.data > 0.0)
    
    def test_softmax_single_element(self):
        """Test Softmax with single element (should return 1.0)."""
        softmax = Softmax()
        
        input_tensor = MockTensor([5.0])
        output = softmax(input_tensor)
        
        # Single element should have probability 1.0
        assert np.allclose(output.data, [1.0], atol=1e-6)


class TestActivationIntegration:
    """Test integration between different activation functions."""
    
    def test_activation_chaining(self):
        """Test chaining different activation functions."""
        # Create activations
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        # Test chaining: input -> ReLU -> Sigmoid
        input_tensor = MockTensor([-1.0, 0.0, 1.0, 2.0])
        
        # Apply ReLU first
        relu_output = relu(input_tensor)
        expected_relu = [0.0, 0.0, 1.0, 2.0]
        assert np.allclose(relu_output.data, expected_relu)
        
        # Apply Sigmoid to ReLU output
        sigmoid_output = sigmoid(relu_output)
        
        # Should be valid sigmoid values
        assert np.all(sigmoid_output.data > 0.0)
        assert np.all(sigmoid_output.data < 1.0)
        
        # First two should be 0.5 (sigmoid of 0)
        assert np.allclose(sigmoid_output.data[:2], [0.5, 0.5], atol=1e-6)
    
    def test_activation_consistency(self):
        """Test that activations are consistent across calls."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        input_tensor = MockTensor([1.0, 2.0, 3.0])
        
        for activation in activations:
            # Apply activation twice
            output1 = activation(input_tensor)
            output2 = activation(input_tensor)
            
            # Should get identical results
            assert np.allclose(output1.data, output2.data, atol=1e-10)
    
    def test_activation_shapes(self):
        """Test that all activations preserve input shapes."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        test_shapes = [
            [5],           # 1D
            [3, 4],        # 2D
            [2, 3, 4],     # 3D
        ]
        
        for shape in test_shapes:
            input_data = np.random.randn(*shape)
            input_tensor = MockTensor(input_data)
            
            for activation in activations:
                output = activation(input_tensor)
                assert output.shape == input_tensor.shape, f"Shape mismatch for {activation.__class__.__name__}"


class TestActivationEdgeCases:
    """Test edge cases and error conditions for activation functions."""
    
    def test_empty_input(self):
        """Test activations with empty input."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        empty_tensor = MockTensor([])
        
        for activation in activations:
            output = activation(empty_tensor)
            assert output.shape == (0,), f"Empty input failed for {activation.__class__.__name__}"
    
    def test_very_small_values(self):
        """Test activations with very small values."""
        activations = [ReLU(), Sigmoid(), Tanh()]  # Exclude Softmax for this test
        
        small_tensor = MockTensor([1e-10, -1e-10, 1e-15])
        
        for activation in activations:
            output = activation(small_tensor)
            # Should not crash and should produce finite values
            assert np.all(np.isfinite(output.data)), f"Small values failed for {activation.__class__.__name__}"
    
    def test_inf_and_nan_handling(self):
        """Test activation behavior with inf and nan values."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        # Test with inf values
        inf_tensor = MockTensor([np.inf, -np.inf, 0.0])
        
        for activation in activations:
            try:
                output = activation(inf_tensor)
                # Should either handle gracefully or raise appropriate exception
                # At minimum, should not crash the program
                assert output is not None
            except (ValueError, RuntimeWarning):
                # Acceptable to raise warnings/errors for inf values
                pass


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 
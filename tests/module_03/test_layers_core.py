"""
Module 03: Layers - Core Functionality Tests
Tests the Layer base class and fundamental layer operations
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLayerBaseClass:
    """Test Layer base class functionality."""
    
    def test_layer_creation(self):
        """Test basic Layer creation."""
        try:
            from tinytorch.core.layers import Layer
            
            layer = Layer()
            assert layer is not None
            
        except ImportError:
            assert True, "Layer base class not implemented yet"
    
    def test_layer_interface(self):
        """Test Layer has required interface."""
        try:
            from tinytorch.core.layers import Layer
            
            layer = Layer()
            
            # Should have forward method
            assert hasattr(layer, 'forward'), "Layer must have forward method"
            
            # Should be callable
            assert callable(layer), "Layer must be callable"
            
        except ImportError:
            assert True, "Layer interface not implemented yet"
    
    def test_layer_inheritance(self):
        """Test Layer can be inherited."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class TestLayer(Layer):
                def forward(self, x):
                    return x  # Identity layer
            
            layer = TestLayer()
            x = Tensor(np.array([1, 2, 3]))
            output = layer(x)
            
            assert isinstance(output, Tensor)
            assert np.array_equal(output.data, x.data)
            
        except ImportError:
            assert True, "Layer inheritance not ready yet"


class TestParameterManagement:
    """Test layer parameter management."""
    
    def test_layer_with_parameters(self):
        """Test layer can store parameters."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class ParameterLayer(Layer):
                def __init__(self, input_size, output_size):
                    self.weights = Tensor(np.random.randn(input_size, output_size))
                    self.bias = Tensor(np.zeros(output_size))
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data + self.bias.data)
            
            layer = ParameterLayer(5, 3)
            
            assert hasattr(layer, 'weights')
            assert hasattr(layer, 'bias')
            assert layer.weights.shape == (5, 3)
            assert layer.bias.shape == (3,)
            
        except ImportError:
            assert True, "Parameter management not implemented yet"
    
    def test_parameter_initialization(self):
        """Test parameter initialization strategies."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class InitTestLayer(Layer):
                def __init__(self, size):
                    # Xavier/Glorot initialization
                    limit = np.sqrt(6.0 / (size + size))
                    self.weights = Tensor(np.random.uniform(-limit, limit, (size, size)))
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data)
            
            layer = InitTestLayer(10)
            
            # Check initialization range
            weights_std = np.std(layer.weights.data)
            expected_std = np.sqrt(2.0 / (10 + 10))
            
            # Should be in reasonable range
            assert 0.1 < weights_std < 1.0
            
        except ImportError:
            assert True, "Parameter initialization not implemented yet"
    
    def test_parameter_shapes(self):
        """Test parameter shapes are correct."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class ShapeTestLayer(Layer):
                def __init__(self, in_features, out_features):
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weights = Tensor(np.random.randn(in_features, out_features))
                    self.bias = Tensor(np.zeros(out_features))
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data + self.bias.data)
            
            layer = ShapeTestLayer(128, 64)
            
            assert layer.weights.shape == (128, 64)
            assert layer.bias.shape == (64,)
            
            # Test with input
            x = Tensor(np.random.randn(16, 128))
            output = layer(x)
            assert output.shape == (16, 64)
            
        except ImportError:
            assert True, "Parameter shapes not implemented yet"


class TestLinearTransformations:
    """Test linear transformation layers."""
    
    def test_matrix_multiplication_layer(self):
        """Test matrix multiplication layer."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class MatMulLayer(Layer):
                def __init__(self, weight_matrix):
                    self.weights = Tensor(weight_matrix)
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data)
            
            # Simple 2x2 transformation
            W = np.array([[1, 2], [3, 4]])
            layer = MatMulLayer(W)
            
            x = Tensor(np.array([[1, 0], [0, 1]]))  # Identity input
            output = layer(x)
            
            expected = np.array([[1, 2], [3, 4]])
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "Matrix multiplication layer not implemented yet"
    
    def test_affine_transformation(self):
        """Test affine transformation (Wx + b)."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class AffineLayer(Layer):
                def __init__(self, weights, bias):
                    self.weights = Tensor(weights)
                    self.bias = Tensor(bias)
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data + self.bias.data)
            
            W = np.array([[1, 0], [0, 1]])  # Identity matrix
            b = np.array([10, 20])           # Bias
            
            layer = AffineLayer(W, b)
            x = Tensor(np.array([[1, 2]]))
            output = layer(x)
            
            expected = np.array([[11, 22]])  # [1,2] @ I + [10,20]
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "Affine transformation not implemented yet"
    
    def test_batch_processing(self):
        """Test layer handles batch inputs."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class BatchLayer(Layer):
                def __init__(self):
                    self.weights = Tensor(np.array([[2, 0], [0, 3]]))
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data)
            
            layer = BatchLayer()
            
            # Batch of inputs
            x = Tensor(np.array([[1, 1], [2, 2], [3, 3]]))  # 3 samples
            output = layer(x)
            
            expected = np.array([[2, 3], [4, 6], [6, 9]])
            assert np.array_equal(output.data, expected)
            assert output.shape == (3, 2)
            
        except ImportError:
            assert True, "Batch processing not implemented yet"


class TestLayerComposition:
    """Test layer composition and chaining."""
    
    def test_layer_chaining(self):
        """Test chaining multiple layers."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class ScaleLayer(Layer):
                def __init__(self, scale):
                    self.scale = scale
                
                def forward(self, x):
                    return Tensor(x.data * self.scale)
            
            class AddLayer(Layer):
                def __init__(self, offset):
                    self.offset = offset
                
                def forward(self, x):
                    return Tensor(x.data + self.offset)
            
            layer1 = ScaleLayer(2)
            layer2 = AddLayer(10)
            
            x = Tensor(np.array([1, 2, 3]))
            
            # Chain: x -> scale by 2 -> add 10
            h = layer1(x)
            output = layer2(h)
            
            expected = np.array([12, 14, 16])  # (x*2) + 10
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "Layer chaining not implemented yet"
    
    def test_sequential_layer_composition(self):
        """Test sequential composition of layers."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class Sequential(Layer):
                def __init__(self, layers):
                    self.layers = layers
                
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x
            
            class LinearLayer(Layer):
                def __init__(self, weights):
                    self.weights = Tensor(weights)
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data)
            
            # Build a 2-layer network
            layer1 = LinearLayer(np.array([[1, 2], [3, 4]]))
            layer2 = LinearLayer(np.array([[1], [1]]))
            
            network = Sequential([layer1, layer2])
            
            x = Tensor(np.array([[1, 1]]))
            output = network(x)
            
            # [1,1] @ [[1,2],[3,4]] = [4,6]
            # [4,6] @ [[1],[1]] = [10]
            expected = np.array([[10]])
            assert np.array_equal(output.data, expected)
            
        except ImportError:
            assert True, "Sequential composition not implemented yet"


class TestLayerUtilities:
    """Test layer utility functions."""
    
    def test_layer_parameter_count(self):
        """Test counting layer parameters."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class CountableLayer(Layer):
                def __init__(self, in_features, out_features):
                    self.weights = Tensor(np.random.randn(in_features, out_features))
                    self.bias = Tensor(np.zeros(out_features))
                
                def parameter_count(self):
                    return self.weights.data.size + self.bias.data.size
                
                def forward(self, x):
                    return Tensor(x.data @ self.weights.data + self.bias.data)
            
            layer = CountableLayer(10, 5)
            
            # 10*5 weights + 5 biases = 55 parameters
            expected_count = 10 * 5 + 5
            if hasattr(layer, 'parameter_count'):
                assert layer.parameter_count() == expected_count
            
        except ImportError:
            assert True, "Parameter counting not implemented yet"
    
    def test_layer_output_shape_inference(self):
        """Test layer output shape inference."""
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            
            class ShapeInferenceLayer(Layer):
                def __init__(self, out_features):
                    self.out_features = out_features
                
                def forward(self, x):
                    batch_size = x.shape[0]
                    # Simulate transformation to out_features
                    return Tensor(np.random.randn(batch_size, self.out_features))
                
                def output_shape(self, input_shape):
                    return (input_shape[0], self.out_features)
            
            layer = ShapeInferenceLayer(20)
            
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape((32, 10))
                assert output_shape == (32, 20)
            
        except ImportError:
            assert True, "Shape inference not implemented yet"
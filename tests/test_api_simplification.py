#!/usr/bin/env python3
"""
Test Suite for TinyTorch API Simplification

This test suite validates all the new PyTorch-compatible API features
introduced in the API simplification project.

Test Hierarchy:
1. Unit Tests: Individual components (Parameter, Module, Linear, Conv2d)
2. Integration Tests: Components working together 
3. End-to-End Tests: Complete workflows (model creation, training setup)
"""

import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestParameterFunction(unittest.TestCase):
    """Unit tests for the Parameter() helper function."""
    
    def setUp(self):
        from tinytorch.core.tensor import Parameter, Tensor
        self.Parameter = Parameter
        self.Tensor = Tensor
    
    def test_parameter_creation(self):
        """Test basic Parameter creation."""
        param = self.Parameter([[1.0, 2.0], [3.0, 4.0]])
        
        self.assertTrue(param.requires_grad, "Parameter should have requires_grad=True")
        self.assertEqual(param.shape, (2, 2), "Parameter should preserve shape")
        self.assertEqual(param.dtype, np.float32, "Parameter should default to float32")
    
    def test_parameter_vs_tensor(self):
        """Test Parameter vs Tensor differences."""
        data = [[1.0, 2.0]]
        
        tensor = self.Tensor(data)
        param = self.Parameter(data)
        
        self.assertFalse(tensor.requires_grad, "Tensor should default requires_grad=False")
        self.assertTrue(param.requires_grad, "Parameter should have requires_grad=True")
    
    def test_parameter_with_dtype(self):
        """Test Parameter with explicit dtype."""
        param = self.Parameter([1, 2, 3], dtype='int32')
        self.assertEqual(param.dtype, np.int32, "Parameter should respect dtype")
        self.assertTrue(param.requires_grad, "Parameter should still have requires_grad=True")


class TestModuleBaseClass(unittest.TestCase):
    """Unit tests for the Module base class."""
    
    def setUp(self):
        import tinytorch.nn as nn
        from tinytorch.core.tensor import Parameter
        self.nn = nn
        self.Parameter = Parameter
    
    def test_module_creation(self):
        """Test basic Module creation."""
        module = self.nn.Module()
        
        self.assertEqual(len(module._parameters), 0, "New module should have no parameters")
        self.assertEqual(len(module._modules), 0, "New module should have no submodules")
        self.assertEqual(len(list(module.parameters())), 0, "parameters() should return empty list")
    
    def test_parameter_registration(self):
        """Test automatic parameter registration."""
        Parameter = self.Parameter
        
        class TestModule(self.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([[1.0, 2.0]])
                self.bias = Parameter([0.5])
                self.non_param = "not a parameter"
        
        module = TestModule()
        params = list(module.parameters())
        
        self.assertEqual(len(params), 2, "Should register 2 parameters")
        self.assertTrue(all(p.requires_grad for p in params), "All parameters should require gradients")
    
    def test_submodule_registration(self):
        """Test automatic submodule registration."""
        Parameter = self.Parameter
        
        class SubModule(self.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([[1.0]])
        
        class MainModule(self.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()
                self.weight = Parameter([[2.0]])
        
        module = MainModule()
        params = list(module.parameters())
        
        self.assertEqual(len(params), 3, "Should collect parameters from submodules: 2 + 1 = 3")
        self.assertEqual(len(module._modules), 2, "Should register 2 submodules")
    
    def test_callable_interface(self):
        """Test that modules are callable via __call__."""
        class TestModule(self.nn.Module):
            def forward(self, x):
                return x + 1
        
        module = TestModule()
        result = module(5)  # Should call forward(5)
        
        self.assertEqual(result, 6, "Module should be callable and delegate to forward()")


class TestLinearLayer(unittest.TestCase):
    """Unit tests for the Linear layer (renamed from Dense)."""
    
    def setUp(self):
        import tinytorch.nn as nn
        from tinytorch.core.tensor import Tensor
        self.nn = nn
        self.Tensor = Tensor
    
    def test_linear_creation(self):
        """Test Linear layer creation."""
        linear = self.nn.Linear(5, 3)
        
        self.assertEqual(linear.input_size, 5, "Should store input size")
        self.assertEqual(linear.output_size, 3, "Should store output size")
        self.assertEqual(linear.weights.shape, (5, 3), "Weights should have correct shape")
        self.assertEqual(linear.bias.shape, (3,), "Bias should have correct shape")
        self.assertTrue(linear.weights.requires_grad, "Weights should require gradients")
        self.assertTrue(linear.bias.requires_grad, "Bias should require gradients")
    
    def test_linear_no_bias(self):
        """Test Linear layer without bias."""
        linear = self.nn.Linear(5, 3, use_bias=False)
        
        self.assertIsNone(linear.bias, "Should have no bias when use_bias=False")
        self.assertEqual(len(list(linear.parameters())), 1, "Should only have weight parameter")
    
    def test_linear_forward(self):
        """Test Linear layer forward pass."""
        linear = self.nn.Linear(3, 2)
        x = self.Tensor([[1.0, 2.0, 3.0]])
        
        output = linear(x)
        
        self.assertEqual(output.shape, (1, 2), "Output should have correct shape")
        self.assertIsInstance(output, self.Tensor, "Output should be a Tensor")
    
    def test_linear_parameter_collection(self):
        """Test that Linear parameters are properly collected."""
        linear = self.nn.Linear(4, 2)
        params = list(linear.parameters())
        
        self.assertEqual(len(params), 2, "Should have 2 parameters (weight + bias)")
        shapes = [p.shape for p in params]
        self.assertIn((4, 2), shapes, "Should include weight shape")
        self.assertIn((2,), shapes, "Should include bias shape")


class TestConv2dLayer(unittest.TestCase):
    """Unit tests for the Conv2d layer (renamed from MultiChannelConv2D)."""
    
    def setUp(self):
        import tinytorch.nn as nn
        from tinytorch.core.tensor import Tensor
        self.nn = nn
        self.Tensor = Tensor
    
    def test_conv2d_creation(self):
        """Test Conv2d layer creation."""
        conv = self.nn.Conv2d(3, 16, (3, 3))
        
        self.assertEqual(conv.in_channels, 3, "Should store input channels")
        self.assertEqual(conv.out_channels, 16, "Should store output channels")
        self.assertEqual(conv.kernel_size, (3, 3), "Should store kernel size")
        self.assertEqual(conv.weight.shape, (16, 3, 3, 3), "Weight should have correct shape")
        self.assertEqual(conv.bias.shape, (16,), "Bias should have correct shape")
    
    def test_conv2d_parameter_collection(self):
        """Test that Conv2d parameters are properly collected."""
        conv = self.nn.Conv2d(3, 8, (3, 3))
        params = list(conv.parameters())
        
        self.assertEqual(len(params), 2, "Should have 2 parameters (weight + bias)")
        weight_params = [p for p in params if len(p.shape) == 4]
        bias_params = [p for p in params if len(p.shape) == 1]
        
        self.assertEqual(len(weight_params), 1, "Should have 1 weight parameter")
        self.assertEqual(len(bias_params), 1, "Should have 1 bias parameter")


class TestFunctionalInterface(unittest.TestCase):
    """Unit tests for the functional interface (F.relu, F.flatten, etc.)."""
    
    def setUp(self):
        import tinytorch.nn.functional as F
        from tinytorch.core.tensor import Tensor
        self.F = F
        self.Tensor = Tensor
    
    def test_relu_function(self):
        """Test F.relu function."""
        x = self.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        output = self.F.relu(x)
        
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(output.data, expected, "ReLU should zero negative values")
    
    def test_flatten_function(self):
        """Test F.flatten function."""
        x = self.Tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])  # Shape: (1, 2, 2, 2)
        output = self.F.flatten(x)
        
        self.assertEqual(output.shape, (1, 8), "Flatten should preserve batch dimension")
        expected = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(output.data, expected, "Flatten should preserve order")
    
    def test_flatten_start_dim(self):
        """Test F.flatten with custom start_dim."""
        x = self.Tensor([[[1, 2], [3, 4]]])  # Shape: (1, 2, 2) = 4 elements
        output = self.F.flatten(x, start_dim=0)  # Flatten everything
        
        self.assertEqual(output.shape, (4,), "Should flatten from dimension 0")
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(output.data, expected, "Should flatten all dimensions")


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for optimizers with the new API."""
    
    def setUp(self):
        import tinytorch.nn as nn
        import tinytorch.optim as optim
        self.nn = nn
        self.optim = optim
    
    def test_adam_with_model_parameters(self):
        """Test Adam optimizer with model.parameters()."""
        model = self.nn.Linear(5, 3)
        optimizer = self.optim.Adam(model.parameters(), learning_rate=0.001)
        
        # Check that optimizer received the parameters
        self.assertEqual(len(optimizer.parameters), 2, "Adam should receive 2 parameters")
        
        # Check parameter types
        for param in optimizer.parameters:
            self.assertTrue(param.requires_grad, "All optimizer parameters should require gradients")
    
    def test_sgd_with_model_parameters(self):
        """Test SGD optimizer with model.parameters()."""
        nn = self.nn
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)
        
        model = SimpleNet()
        optimizer = self.optim.SGD(model.parameters(), learning_rate=0.01)
        
        # Should collect parameters from both layers
        self.assertEqual(len(optimizer.parameters), 4, "SGD should receive 4 parameters (2 layers × 2 params)")


class TestBackwardCompatibility(unittest.TestCase):
    """Integration tests for backward compatibility."""
    
    def test_dense_alias_works(self):
        """Test that Dense alias still works."""
        from tinytorch.core.layers import Dense
        
        dense = Dense(5, 3)
        self.assertEqual(dense.input_size, 5, "Dense alias should work")
        self.assertEqual(dense.output_size, 3, "Dense alias should work")
    
    def test_multichannel_conv2d_alias_works(self):
        """Test that MultiChannelConv2D alias still works."""
        from tinytorch.core.spatial import MultiChannelConv2D
        
        conv = MultiChannelConv2D(3, 16, (3, 3))
        self.assertEqual(conv.in_channels, 3, "MultiChannelConv2D alias should work")
        self.assertEqual(conv.out_channels, 16, "MultiChannelConv2D alias should work")


class TestCompleteModelWorkflow(unittest.TestCase):
    """End-to-end integration tests for complete model workflows."""
    
    def setUp(self):
        import tinytorch.nn as nn
        import tinytorch.nn.functional as F
        import tinytorch.optim as optim
        from tinytorch.core.tensor import Tensor
        
        self.nn = nn
        self.F = F
        self.optim = optim
        self.Tensor = Tensor
    
    def test_complete_mlp_workflow(self):
        """Test complete MLP creation and setup."""
        nn = self.nn
        F = self.F
        
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)
        
        # Create model
        model = MLP()
        
        # Test parameter collection
        params = list(model.parameters())
        self.assertEqual(len(params), 4, "Should collect all parameters")
        
        # Test optimizer creation
        optimizer = self.optim.Adam(model.parameters(), learning_rate=0.001)
        self.assertIsNotNone(optimizer, "Should create optimizer")
        
        # Test forward pass
        x = self.Tensor([[1.0, 2.0, 3.0, 4.0]])
        output = model(x)
        self.assertEqual(output.shape, (1, 2), "Should produce correct output shape")
    
    def test_complete_cnn_workflow(self):
        """Test complete CNN creation and setup."""
        nn = self.nn
        F = self.F
        
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, (3, 3))
                self.fc1 = nn.Linear(128, 10)  # Simplified size
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.flatten(x)
                return self.fc1(x)
        
        # Create model
        model = CNN()
        
        # Test parameter collection
        params = list(model.parameters())
        self.assertEqual(len(params), 4, "Should collect conv + fc parameters")
        
        # Test optimizer creation  
        optimizer = self.optim.Adam(model.parameters(), learning_rate=0.001)
        self.assertIsNotNone(optimizer, "Should create optimizer")
    
    def test_pytorch_like_syntax(self):
        """Test that the syntax matches PyTorch patterns."""
        # This test verifies the API feels like PyTorch
        import tinytorch.nn as nn
        import tinytorch.nn.functional as F
        import tinytorch.optim as optim
        
        # Should be able to write PyTorch-like code
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
        
        # Test that this workflow doesn't crash
        x = self.Tensor([[1.0] * 10])
        output = model(x)
        
        # Should be able to chain operations
        output = F.relu(output)
        
        self.assertIsNotNone(output, "PyTorch-like workflow should work")


if __name__ == '__main__':
    # Run the test suite
    print("🧪 TinyTorch API Simplification Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestParameterFunction,
        TestModuleBaseClass, 
        TestLinearLayer,
        TestConv2dLayer,
        TestFunctionalInterface,
        TestOptimizerIntegration,
        TestBackwardCompatibility,
        TestCompleteModelWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All API simplification tests passed!")
        print("🎉 PyTorch-compatible API is working correctly!")
    else:
        print("\n❌ Some tests failed!")
        print("🔧 API needs fixes before deployment.")
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
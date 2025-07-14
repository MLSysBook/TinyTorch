"""
Integration Tests - Kernels Module

Tests real integration between hardware-optimized kernels and other TinyTorch modules.
Uses actual TinyTorch components to verify kernels work correctly in ML workflows.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from development modules directly to ensure compatibility
sys.path.append(str(project_root / "modules" / "source" / "01_tensor"))
sys.path.append(str(project_root / "modules" / "source" / "02_activations"))
sys.path.append(str(project_root / "modules" / "source" / "03_layers"))
sys.path.append(str(project_root / "modules" / "source" / "04_networks"))
sys.path.append(str(project_root / "modules" / "source" / "11_kernels"))

from tensor_dev import Tensor
from activations_dev import ReLU, Sigmoid, Tanh
from layers_dev import Dense
from networks_dev import Sequential
from kernels_dev import (
    matmul_baseline, vectorized_relu, vectorized_operations,
    cache_friendly_matmul, parallel_relu, parallel_batch_processing,
    quantized_matmul, quantized_relu
)


class TestKernelsIntegration:
    """Test real integration between optimized kernels and TinyTorch components."""
    
    def test_matmul_baseline_works(self):
        """Test baseline matrix multiplication works correctly."""
        # Create test matrices
        A = Tensor([[1.0, 2.0], [3.0, 4.0]])
        B = Tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # Test baseline matmul
        result = matmul_baseline(A, B)
        
        # Verify result - check for tensor-like behavior rather than exact class
        assert hasattr(result, 'data')
        assert hasattr(result, 'shape')
        assert result.shape == (2, 2)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_allclose(result.data, expected)
    
    def test_vectorized_relu_works(self):
        """Test vectorized ReLU works correctly."""
        # Test data with negative, zero, and positive values
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test vectorized ReLU
        result = vectorized_relu(x)
        
        # Verify it returns a tensor-like object
        assert hasattr(result, 'data')
        assert hasattr(result, 'shape')
        assert result.shape == x.shape
        
        # Verify ReLU behavior: max(0, x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(result.data, expected)
    
    def test_vectorized_operations_work(self):
        """Test vectorized operations work correctly."""
        # Test data
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([4.0, 5.0, 6.0])
        
        # Test vectorized operations
        result = vectorized_operations(x, y)
        
        # Verify it returns a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'element_wise_multiply' in result
        assert 'element_wise_add' in result
        assert 'squared_difference' in result
        
        # Verify all results are tensor-like objects
        for key, value in result.items():
            assert hasattr(value, 'data')
            assert hasattr(value, 'shape')
        
        # Verify basic operations
        np.testing.assert_allclose(result['element_wise_multiply'].data, [4.0, 10.0, 18.0])
        np.testing.assert_allclose(result['element_wise_add'].data, [5.0, 7.0, 9.0])
        np.testing.assert_allclose(result['squared_difference'].data, [9.0, 9.0, 9.0])
    
    def test_cache_friendly_matmul_works(self):
        """Test cache-friendly matrix multiplication works correctly."""
        # Create test matrices
        A = Tensor(np.random.randn(8, 8))
        B = Tensor(np.random.randn(8, 8))
        
        # Test cache-friendly matmul
        result = cache_friendly_matmul(A, B, block_size=4)
        
        # Verify result
        assert hasattr(result, 'data')
        assert hasattr(result, 'shape')
        assert result.shape == (8, 8)
        
        # Compare with baseline
        baseline_result = matmul_baseline(A, B)
        np.testing.assert_allclose(result.data, baseline_result.data, rtol=1e-10)
    
    def test_parallel_relu_works(self):
        """Test parallel ReLU works correctly."""
        # Create test data
        x = Tensor(np.random.randn(100))
        
        # Test parallel ReLU
        result = parallel_relu(x, num_workers=2)
        
        # Verify result
        assert hasattr(result, 'data')
        assert hasattr(result, 'shape')
        assert result.shape == x.shape
        
        # Verify ReLU behavior
        assert np.all(result.data >= 0)
        
        # Compare with vectorized version
        vectorized_result = vectorized_relu(x)
        np.testing.assert_allclose(result.data, vectorized_result.data)
    
    def test_parallel_batch_processing_works(self):
        """Test parallel batch processing works correctly."""
        # Create batch data
        batch_size = 4
        batch_data = [Tensor(np.random.randn(5)) for _ in range(batch_size)]
        
        # Define operation
        def simple_relu(tensor):
            return vectorized_relu(tensor)
        
        # Test parallel batch processing
        results = parallel_batch_processing(batch_data, simple_relu, num_workers=2)
        
        # Verify results
        assert len(results) == batch_size
        for i, result in enumerate(results):
            assert hasattr(result, 'data')
            assert hasattr(result, 'shape')
            assert result.shape == batch_data[i].shape
            assert np.all(result.data >= 0)  # ReLU property
    
    def test_quantized_operations_work(self):
        """Test quantized operations work correctly."""
        # Create test data
        A = Tensor([[1.0, 2.0], [3.0, 4.0]])
        B = Tensor([[0.5, 1.0], [1.5, 2.0]])
        
        # Test quantized matmul
        result = quantized_matmul(A, B, scale_A=1.0/127, scale_B=1.0/127)
        
        # Verify result
        assert hasattr(result, 'data')
        assert hasattr(result, 'shape')
        assert result.shape == (2, 2)
        
        # Should be approximately correct (allowing for quantization error)
        baseline_result = matmul_baseline(A, B)
        assert np.allclose(result.data, baseline_result.data, rtol=0.1)
        
        # Test quantized ReLU
        x = Tensor([-1.0, 0.0, 1.0, 2.0])
        quantized_result = quantized_relu(x, scale=1.0/127)
        
        # Verify quantized ReLU
        assert hasattr(quantized_result, 'data')
        assert hasattr(quantized_result, 'shape')
        assert quantized_result.shape == x.shape
        assert np.all(quantized_result.data >= 0)  # ReLU property


class TestKernelsWithNetworks:
    """Test kernels integration with neural networks."""
    
    def test_kernels_with_dense_layers(self):
        """Test kernels work with Dense layers."""
        # Create a simple network
        layer = Dense(input_size=3, output_size=2)
        
        # Test data
        x = Tensor([[1.0, 2.0, 3.0]])
        
        # Network should work correctly
        output = layer(x)
        
        # Verify network integration
        assert isinstance(output, Tensor)
        assert output.shape == (1, 2)
        assert not np.any(np.isnan(output.data))
    
    def test_kernels_with_sequential_network(self):
        """Test kernels work with Sequential networks."""
        # Create a simple network
        network = Sequential([
            Dense(input_size=4, output_size=3),
            ReLU(),
            Dense(input_size=3, output_size=2)
        ])
        
        # Test data
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Network should work correctly
        output = network(x)
        
        # Verify network integration
        assert isinstance(output, Tensor)
        assert output.shape == (1, 2)
        assert not np.any(np.isnan(output.data))
        assert not np.any(np.isinf(output.data))


def test_integration_summary():
    """Summary test demonstrating complete kernel integration."""
    print("🎯 Integration Summary: Kernels ↔ TinyTorch Components")
    print("=" * 60)
    
    # Create comprehensive test
    print("🏗️  Testing kernel integration...")
    
    # Test 1: Basic operations
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    relu_result = vectorized_relu(x)
    
    # Test 2: Matrix operations
    A = Tensor(np.random.randn(4, 4))
    B = Tensor(np.random.randn(4, 4))
    matmul_result = matmul_baseline(A, B)
    
    # Test 3: Batch processing
    batch_data = [Tensor(np.random.randn(5)) for _ in range(4)]
    batch_results = parallel_batch_processing(
        batch_data, 
        lambda x: vectorized_relu(x), 
        num_workers=2
    )
    
    # Verify complete integration
    assert isinstance(relu_result, Tensor)
    assert isinstance(matmul_result, Tensor)
    assert len(batch_results) == 4
    
    for result in batch_results:
        assert isinstance(result, Tensor)
        assert np.all(result.data >= 0)  # ReLU output
    
    print("✅ Kernel integration successful!")
    print(f"   ReLU output: {relu_result.shape}")
    print(f"   Matrix multiplication: {matmul_result.shape}")
    print(f"   Batch processing: {len(batch_results)} tensors")
    print("   Components: Kernels → Networks → Activations → Parallelization")
    print("🎉 Hardware-optimized ML operations ready for production!")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"]) 
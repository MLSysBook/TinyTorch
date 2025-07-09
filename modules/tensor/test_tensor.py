#!/usr/bin/env python3
"""
Automated tests for the Tensor module.

This file contains comprehensive tests to verify that students have
correctly implemented the Tensor class with all required functionality.
"""

import sys
import os
import numpy as np
import pytest

# Add the parent directory to the path so we can import from tinytorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from tinytorch.core.tensor import Tensor
    TENSOR_AVAILABLE = True
except ImportError:
    TENSOR_AVAILABLE = False
    print("⚠️  Tensor class not found. Make sure to implement it first!")

class TestTensorBasics:
    """Test basic Tensor functionality."""
    
    def test_tensor_creation_scalar(self):
        """Test creating a tensor from a scalar."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        # Test scalar creation
        t = Tensor(5)
        assert t.shape == (1,)
        assert t.size == 1
        assert t.dtype == np.int64 or t.dtype == np.int32
        assert t.data[0] == 5
    
    def test_tensor_creation_list(self):
        """Test creating a tensor from a list."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        # Test list creation
        data = [1, 2, 3, 4]
        t = Tensor(data)
        assert t.shape == (4,)
        assert t.size == 4
        assert np.array_equal(t.data, np.array(data))
    
    def test_tensor_creation_matrix(self):
        """Test creating a tensor from a 2D list."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        # Test matrix creation
        data = [[1, 2], [3, 4]]
        t = Tensor(data)
        assert t.shape == (2, 2)
        assert t.size == 4
        assert np.array_equal(t.data, np.array(data))
    
    def test_tensor_creation_numpy(self):
        """Test creating a tensor from a numpy array."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        # Test numpy array creation
        data = np.array([[1, 2], [3, 4]])
        t = Tensor(data)
        assert t.shape == (2, 2)
        assert t.size == 4
        assert np.array_equal(t.data, data)
    
    def test_tensor_dtype(self):
        """Test tensor data type handling."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        # Test float dtype
        t = Tensor([1, 2, 3], dtype=np.float32)
        assert t.dtype == np.float32
        
        # Test int dtype
        t = Tensor([1.5, 2.5], dtype=np.int32)
        assert t.dtype == np.int32

class TestTensorOperations:
    """Test tensor arithmetic operations."""
    
    def test_addition_tensor_tensor(self):
        """Test adding two tensors."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        
        expected = np.array([5, 7, 9])
        assert np.array_equal(c.data, expected)
    
    def test_addition_tensor_scalar(self):
        """Test adding a tensor and a scalar."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        c = a + 5
        
        expected = np.array([6, 7, 8])
        assert np.array_equal(c.data, expected)
    
    def test_subtraction_tensor_tensor(self):
        """Test subtracting two tensors."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])
        c = a - b
        
        expected = np.array([4, 4, 4])
        assert np.array_equal(c.data, expected)
    
    def test_subtraction_tensor_scalar(self):
        """Test subtracting a scalar from a tensor."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([5, 6, 7])
        c = a - 2
        
        expected = np.array([3, 4, 5])
        assert np.array_equal(c.data, expected)
    
    def test_multiplication_tensor_tensor(self):
        """Test multiplying two tensors."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        
        expected = np.array([4, 10, 18])
        assert np.array_equal(c.data, expected)
    
    def test_multiplication_tensor_scalar(self):
        """Test multiplying a tensor by a scalar."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        c = a * 3
        
        expected = np.array([3, 6, 9])
        assert np.array_equal(c.data, expected)
    
    def test_division_tensor_tensor(self):
        """Test dividing two tensors."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])
        c = a / b
        
        expected = np.array([5, 5, 6])
        assert np.array_equal(c.data, expected)
    
    def test_division_tensor_scalar(self):
        """Test dividing a tensor by a scalar."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([10, 20, 30])
        c = a / 2
        
        expected = np.array([5, 10, 15])
        assert np.array_equal(c.data, expected)
    
    def test_equality(self):
        """Test tensor equality."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        b = Tensor([1, 2, 3])
        c = Tensor([1, 2, 4])
        
        assert a == b
        assert a != c

class TestTensorUtilities:
    """Test tensor utility methods."""
    
    def test_reshape(self):
        """Test tensor reshaping."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        reshaped = t.reshape(3, 2)
        
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        assert np.array_equal(reshaped.data, expected)
        assert reshaped.shape == (3, 2)
    
    def test_transpose(self):
        """Test tensor transposition."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        transposed = t.transpose()
        
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert np.array_equal(transposed.data, expected)
        assert transposed.shape == (3, 2)
    
    def test_sum_no_axis(self):
        """Test tensor sum without axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum()
        
        expected = np.array(21)  # 1+2+3+4+5+6
        assert np.array_equal(result.data, expected)
    
    def test_sum_with_axis(self):
        """Test tensor sum with axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=1)
        
        expected = np.array([6, 15])  # [1+2+3, 4+5+6]
        assert np.array_equal(result.data, expected)
    
    def test_mean_no_axis(self):
        """Test tensor mean without axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean()
        
        expected = np.array(3.5)  # (1+2+3+4+5+6)/6
        assert np.allclose(result.data, expected)
    
    def test_mean_with_axis(self):
        """Test tensor mean with axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean(axis=0)
        
        expected = np.array([2.5, 3.5, 4.5])  # [(1+4)/2, (2+5)/2, (3+6)/2]
        assert np.allclose(result.data, expected)
    
    def test_max_no_axis(self):
        """Test tensor max without axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.max()
        
        expected = np.array(6)
        assert np.array_equal(result.data, expected)
    
    def test_min_no_axis(self):
        """Test tensor min without axis."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.min()
        
        expected = np.array(1)
        assert np.array_equal(result.data, expected)
    
    def test_flatten(self):
        """Test tensor flattening."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        flattened = t.flatten()
        
        expected = np.array([1, 2, 3, 4, 5, 6])
        assert np.array_equal(flattened.data, expected)
        assert flattened.shape == (6,)

class TestTensorErrors:
    """Test tensor error handling."""
    
    def test_invalid_data_type(self):
        """Test error handling for invalid data types."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        with pytest.raises(ValueError):
            Tensor("invalid")
    
    def test_invalid_operation(self):
        """Test error handling for invalid operations."""
        if not TENSOR_AVAILABLE:
            pytest.skip("Tensor class not implemented")
        
        a = Tensor([1, 2, 3])
        
        with pytest.raises(TypeError):
            a + "invalid"

def run_tests():
    """Run all tensor tests and return results."""
    print("🧪 Running Tensor Tests...")
    print("=" * 50)
    
    if not TENSOR_AVAILABLE:
        print("❌ Tensor class not found!")
        print("💡 Make sure to implement the Tensor class in tinytorch/core/tensor.py")
        return False
    
    # Run pytest
    import pytest
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("\n✅ All tensor tests passed!")
        return True
    else:
        print("\n❌ Some tensor tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 
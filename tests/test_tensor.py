"""
Tests for the Tensor module.

This file contains comprehensive tests for the Tensor class implementation.
Run with: python bin/tito.py test --module tensor
"""

import pytest
import numpy as np
from tinytorch.core.tensor import Tensor


class TestTensorCreation:
    """Test tensor creation and basic properties."""
    
    def test_scalar_tensor(self):
        """Test creating scalar tensors."""
        t = Tensor(5.0)
        assert t.shape == ()
        assert t.size == 1
        assert float(t.data) == 5.0
    
    def test_1d_tensor(self):
        """Test creating 1D tensors."""
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert t.size == 3
        np.testing.assert_array_equal(t.data, [1, 2, 3])
    
    def test_2d_tensor(self):
        """Test creating 2D tensors."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.size == 4
        np.testing.assert_array_equal(t.data, [[1, 2], [3, 4]])
    
    def test_tensor_from_numpy(self):
        """Test creating tensors from NumPy arrays."""
        arr = np.array([1, 2, 3])
        t = Tensor(arr)
        assert t.shape == (3,)
        np.testing.assert_array_equal(t.data, arr)


class TestTensorProperties:
    """Test tensor properties."""
    
    def test_shape_property(self):
        """Test the shape property."""
        t1 = Tensor([1, 2, 3])
        assert t1.shape == (3,)
        
        t2 = Tensor([[1, 2], [3, 4]])
        assert t2.shape == (2, 2)
    
    def test_size_property(self):
        """Test the size property."""
        t1 = Tensor([1, 2, 3])
        assert t1.size == 3
        
        t2 = Tensor([[1, 2], [3, 4]])
        assert t2.size == 4
    
    def test_dtype_property(self):
        """Test the dtype property."""
        t_int = Tensor([1, 2, 3])
        t_float = Tensor([1.0, 2.0, 3.0])
        
        # NumPy automatically infers types
        assert np.issubdtype(t_int.dtype, np.integer)
        assert np.issubdtype(t_float.dtype, np.floating)


class TestTensorArithmetic:
    """Test tensor arithmetic operations."""
    
    def test_tensor_addition(self):
        """Test tensor + tensor addition."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [5, 7, 9])
    
    def test_scalar_addition(self):
        """Test tensor + scalar addition."""
        a = Tensor([1, 2, 3])
        result = a + 10
        
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [11, 12, 13])
    
    def test_tensor_multiplication(self):
        """Test tensor * tensor multiplication."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a * b
        
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [4, 10, 18])
    
    def test_scalar_multiplication(self):
        """Test tensor * scalar multiplication."""
        a = Tensor([1, 2, 3])
        result = a * 2
        
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [2, 4, 6])
    
    def test_2d_operations(self):
        """Test operations on 2D tensors."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a + b
        
        assert isinstance(result, Tensor)
        expected = [[6, 8], [10, 12]]
        np.testing.assert_array_equal(result.data, expected)


class TestTensorUtils:
    """Test tensor utility methods (if implemented)."""
    
    def test_sum_exists(self):
        """Test that sum method exists (may not be implemented yet)."""
        t = Tensor([1, 2, 3])
        if hasattr(t, 'sum'):
            result = t.sum()
            assert result == 6 or (hasattr(result, 'data') and result.data == 6)
    
    def test_mean_exists(self):
        """Test that mean method exists (may not be implemented yet).""" 
        t = Tensor([1, 2, 3])
        if hasattr(t, 'mean'):
            result = t.mean()
            expected = 2.0
            assert abs(result - expected) < 1e-6 or (hasattr(result, 'data') and abs(result.data - expected) < 1e-6)
    
    def test_reshape_exists(self):
        """Test that reshape method exists (may not be implemented yet)."""
        t = Tensor([[1, 2], [3, 4]])
        if hasattr(t, 'reshape'):
            result = t.reshape(1, 4)
            assert result.shape == (1, 4)
    
    def test_transpose_exists(self):
        """Test that transpose method exists (may not be implemented yet)."""
        t = Tensor([[1, 2], [3, 4]])
        if hasattr(t, 'transpose'):
            result = t.transpose()
            expected = [[1, 3], [2, 4]]
            np.testing.assert_array_equal(result.data, expected)


class TestTensorIntegration:
    """Test tensor integration with the package."""
    
    def test_tensor_import(self):
        """Test that Tensor can be imported from the correct location."""
        from tinytorch.core.tensor import Tensor
        assert Tensor is not None
    
    def test_tensor_representation(self):
        """Test tensor string representation."""
        t = Tensor([1, 2, 3])
        repr_str = repr(t)
        assert "Tensor" in repr_str
        assert "shape" in repr_str or "Shape" in repr_str


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"]) 
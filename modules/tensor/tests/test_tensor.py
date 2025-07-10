"""
Tests for TinyTorch Tensor module.

Tests the core tensor functionality including creation, arithmetic operations,
utility methods, and edge cases.
"""

import sys
import os
import pytest
import numpy as np

# Add the parent directory to path to import tensor_dev
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from the module's development file
from tensor_dev import Tensor

class TestTensorCreation:
    """Test tensor creation from different data types."""
    
    def test_scalar_creation(self):
        """Test creating tensors from scalars."""
        # Float scalar
        t1 = Tensor(5.0)
        assert t1.shape == ()
        assert t1.size == 1
        assert t1.item() == 5.0
        
        # Integer scalar  
        t2 = Tensor(42)
        assert t2.shape == ()
        assert t2.size == 1
        assert t2.item() == 42.0  # Should convert to float32
    
    def test_vector_creation(self):
        """Test creating 1D tensors."""
        t = Tensor([1, 2, 3, 4])
        assert t.shape == (4,)
        assert t.size == 4
        assert t.dtype == np.int32  # Integer list defaults to int32
        np.testing.assert_array_equal(t.numpy(), [1, 2, 3, 4])
    
    def test_matrix_creation(self):
        """Test creating 2D tensors.""" 
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.size == 4
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        np.testing.assert_array_equal(t.numpy(), expected)
    
    def test_numpy_array_creation(self):
        """Test creating tensors from numpy arrays."""
        arr = np.array([1, 2, 3], dtype='int32')
        t = Tensor(arr)
        assert t.shape == (3,)
        assert t.dtype in ['int32', 'float32']  # May convert
    
    def test_dtype_specification(self):
        """Test explicit dtype specification."""
        t = Tensor([1, 2, 3], dtype='int32')
        assert t.dtype == np.int32
    
    def test_invalid_data_type(self):
        """Test error handling for invalid data types."""
        with pytest.raises(TypeError):
            Tensor("invalid")
        with pytest.raises(TypeError):
            Tensor({"dict": "invalid"})

class TestTensorProperties:
    """Test tensor properties and methods."""
    
    def test_shape_property(self):
        """Test shape property for different dimensions."""
        assert Tensor(5).shape == ()
        assert Tensor([1, 2, 3]).shape == (3,)
        assert Tensor([[1, 2], [3, 4]]).shape == (2, 2)
        assert Tensor([[[1]]]).shape == (1, 1, 1)
    
    def test_size_property(self):
        """Test size property."""
        assert Tensor(5).size == 1
        assert Tensor([1, 2, 3]).size == 3
        assert Tensor([[1, 2], [3, 4]]).size == 4
        assert Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).size == 8
    
    def test_dtype_property(self):
        """Test dtype property."""
        t1 = Tensor(5.0)
        assert t1.dtype == np.float32
        
        t2 = Tensor([1, 2, 3], dtype='int32')
        assert t2.dtype == np.int32
    
    def test_repr(self):
        """Test string representation."""
        t = Tensor([1, 2, 3])
        repr_str = repr(t)
        assert 'Tensor' in repr_str
        assert 'shape=' in repr_str
        assert 'dtype=' in repr_str

class TestArithmeticOperations:
    """Test tensor arithmetic operations."""
    
    def test_tensor_addition(self):
        """Test tensor + tensor addition."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        expected = [5.0, 7.0, 9.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_scalar_addition(self):
        """Test tensor + scalar addition."""
        a = Tensor([1, 2, 3])
        result = a + 10
        expected = [11.0, 12.0, 13.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_reverse_addition(self):
        """Test scalar + tensor addition."""
        a = Tensor([1, 2, 3])
        result = 10 + a
        expected = [11.0, 12.0, 13.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_tensor_subtraction(self):
        """Test tensor - tensor subtraction."""
        a = Tensor([5, 7, 9])
        b = Tensor([1, 2, 3])
        result = a - b
        expected = [4.0, 5.0, 6.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_scalar_subtraction(self):
        """Test tensor - scalar subtraction."""
        a = Tensor([10, 20, 30])
        result = a - 5
        expected = [5.0, 15.0, 25.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_tensor_multiplication(self):
        """Test tensor * tensor multiplication."""
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        result = a * b
        expected = [10.0, 18.0, 28.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_scalar_multiplication(self):
        """Test tensor * scalar multiplication."""
        a = Tensor([1, 2, 3])
        result = a * 3
        expected = [3.0, 6.0, 9.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_reverse_multiplication(self):
        """Test scalar * tensor multiplication."""
        a = Tensor([1, 2, 3])
        result = 3 * a
        expected = [3.0, 6.0, 9.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_tensor_division(self):
        """Test tensor / tensor division."""
        a = Tensor([6, 8, 10])
        b = Tensor([2, 4, 5])
        result = a / b
        expected = [3.0, 2.0, 2.0]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_scalar_division(self):
        """Test tensor / scalar division."""
        a = Tensor([6, 8, 10])
        result = a / 2
        expected = [3.0, 4.0, 5.0]
        np.testing.assert_array_equal(result.numpy(), expected)

class TestUtilityMethods:
    """Test tensor utility methods."""
    
    def test_reshape(self):
        """Test tensor reshaping."""
        t = Tensor([[1, 2], [3, 4]])
        reshaped = t.reshape(4)
        assert reshaped.shape == (4,)
        expected = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(reshaped.numpy(), expected)
        
        # Reshape to 2D
        reshaped2 = t.reshape(1, 4)
        assert reshaped2.shape == (1, 4)
    
    def test_transpose(self):
        """Test tensor transpose."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        transposed = t.transpose()
        assert transposed.shape == (3, 2)
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        np.testing.assert_array_equal(transposed.numpy(), expected)
    
    def test_sum_all(self):
        """Test summing all elements."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.sum()
        assert result.item() == 10.0
    
    def test_sum_axis(self):
        """Test summing along specific axes."""
        t = Tensor([[1, 2], [3, 4]])
        
        # Sum along axis 0 (columns)
        sum0 = t.sum(axis=0)
        expected0 = [4.0, 6.0]
        np.testing.assert_array_equal(sum0.numpy(), expected0)
        
        # Sum along axis 1 (rows)
        sum1 = t.sum(axis=1)
        expected1 = [3.0, 7.0]
        np.testing.assert_array_equal(sum1.numpy(), expected1)
    
    def test_mean(self):
        """Test mean calculation."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.mean()
        assert result.item() == 2.5
    
    def test_max(self):
        """Test maximum value."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.max()
        assert result.item() == 4.0
    
    def test_min(self):
        """Test minimum value."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.min()
        assert result.item() == 1.0
    
    def test_item_scalar(self):
        """Test converting single-element tensor to scalar."""
        t = Tensor(42.0)
        assert t.item() == 42.0
        
        # Single element tensor from computation
        t2 = Tensor([5]).sum()
        assert t2.item() == 5.0
    
    def test_item_error(self):
        """Test item() error for multi-element tensors."""
        t = Tensor([1, 2, 3])
        with pytest.raises(ValueError):
            t.item()
    
    def test_numpy_conversion(self):
        """Test converting tensor to numpy array."""
        t = Tensor([[1, 2], [3, 4]])
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        expected = [[1.0, 2.0], [3.0, 4.0]]
        np.testing.assert_array_equal(arr, expected)

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_list(self):
        """Test creating tensor from empty list."""
        t = Tensor([])
        assert t.shape == (0,)
        assert t.size == 0
    
    def test_mixed_operations(self):
        """Test combining different operations."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 2], [2, 2]])
        
        # Complex expression
        result = (a + b) * 2 - 1
        expected = [[5.0, 7.0], [9.0, 11.0]]
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_chained_operations(self):
        """Test chaining multiple operations."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=1).mean()
        # Row sums: [6, 15], mean: 10.5
        assert result.item() == 10.5

def run_tensor_tests():
    """Run all tensor tests and return results."""
    print("🧪 Running Tensor module tests...")
    
    # Run tests using pytest
    test_results = pytest.main([
        __file__, 
        "-v",
        "--tb=short"
    ])
    
    return test_results == 0

if __name__ == "__main__":
    success = run_tensor_tests()
    if success:
        print("✅ All tensor tests passed!")
    else:
        print("❌ Some tensor tests failed!")
        sys.exit(1) 
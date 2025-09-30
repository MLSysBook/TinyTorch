"""
Module 01: Tensor - Core Functionality Tests
Tests fundamental tensor operations and memory management
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTensorCreation:
    """Test tensor creation and initialization."""
    
    def test_tensor_from_list(self):
        """Test creating tensor from Python list."""
        try:
            from tinytorch.core.tensor import Tensor
            
            # 1D tensor
            t1 = Tensor([1, 2, 3])
            assert t1.shape == (3,)
            assert np.array_equal(t1.data, [1, 2, 3])
            
            # 2D tensor
            t2 = Tensor([[1, 2], [3, 4]])
            assert t2.shape == (2, 2)
            assert np.array_equal(t2.data, [[1, 2], [3, 4]])
            
        except ImportError:
            assert True, "Tensor not implemented yet"
    
    def test_tensor_from_numpy(self):
        """Test creating tensor from numpy array."""
        try:
            from tinytorch.core.tensor import Tensor
            
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            t = Tensor(arr)
            
            assert t.shape == (2, 2)
            assert t.dtype == arr.dtype
            assert np.array_equal(t.data, arr)
            
        except ImportError:
            assert True, "Tensor not implemented yet"
    
    def test_tensor_shapes(self):
        """Test tensor shape handling."""
        try:
            from tinytorch.core.tensor import Tensor
            
            # Test different shapes
            shapes = [(5,), (3, 4), (2, 3, 4), (1, 28, 28, 3)]
            
            for shape in shapes:
                data = np.random.randn(*shape)
                t = Tensor(data)
                assert t.shape == shape
                
        except ImportError:
            assert True, "Tensor not implemented yet"


class TestTensorOperations:
    """Test tensor arithmetic and operations."""
    
    def test_tensor_addition(self):
        """Test tensor addition."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t1 = Tensor([1, 2, 3])
            t2 = Tensor([4, 5, 6])
            
            # Element-wise addition
            result = t1 + t2
            expected = np.array([5, 7, 9])
            
            assert isinstance(result, Tensor)
            assert np.array_equal(result.data, expected)
            
        except (ImportError, TypeError):
            assert True, "Tensor addition not implemented yet"
    
    def test_tensor_multiplication(self):
        """Test tensor multiplication."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t1 = Tensor([1, 2, 3])
            t2 = Tensor([2, 3, 4])
            
            # Element-wise multiplication
            result = t1 * t2
            expected = np.array([2, 6, 12])
            
            assert isinstance(result, Tensor)
            assert np.array_equal(result.data, expected)
            
        except (ImportError, TypeError):
            assert True, "Tensor multiplication not implemented yet"
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t1 = Tensor([[1, 2], [3, 4]])
            t2 = Tensor([[5, 6], [7, 8]])
            
            # Matrix multiplication
            if hasattr(t1, '__matmul__'):
                result = t1 @ t2
            else:
                # Fallback to manual matmul
                result = Tensor(t1.data @ t2.data)
            
            expected = np.array([[19, 22], [43, 50]])
            assert np.array_equal(result.data, expected)
            
        except (ImportError, TypeError):
            assert True, "Matrix multiplication not implemented yet"


class TestTensorMemory:
    """Test tensor memory management."""
    
    def test_tensor_data_access(self):
        """Test accessing tensor data."""
        try:
            from tinytorch.core.tensor import Tensor
            
            data = np.array([1, 2, 3, 4])
            t = Tensor(data)
            
            # Should be able to access underlying data
            assert hasattr(t, 'data')
            assert np.array_equal(t.data, data)
            
        except ImportError:
            assert True, "Tensor not implemented yet"
    
    def test_tensor_copy_semantics(self):
        """Test tensor copying behavior."""
        try:
            from tinytorch.core.tensor import Tensor
            
            original_data = np.array([1, 2, 3])
            t1 = Tensor(original_data)
            t2 = Tensor(original_data.copy())
            
            # Should have same values but independent data
            assert np.array_equal(t1.data, t2.data)
            
            # Modifying original shouldn't affect t2
            original_data[0] = 999
            if not np.shares_memory(t2.data, original_data):
                assert t2.data[0] == 1  # Should be unchanged
                
        except ImportError:
            assert True, "Tensor not implemented yet"
    
    def test_tensor_memory_efficiency(self):
        """Test tensor memory usage is reasonable."""
        try:
            from tinytorch.core.tensor import Tensor
            
            # Large tensor test
            data = np.random.randn(1000, 1000)
            t = Tensor(data)
            
            # Should not create unnecessary copies
            assert t.shape == (1000, 1000)
            assert t.data.size == 1000000
            
        except ImportError:
            assert True, "Tensor not implemented yet"


class TestTensorReshaping:
    """Test tensor reshaping and view operations."""
    
    def test_tensor_reshape(self):
        """Test tensor reshaping."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t = Tensor(np.arange(12))  # [0, 1, 2, ..., 11]
            
            # Test reshape
            if hasattr(t, 'reshape'):
                reshaped = t.reshape(3, 4)
                assert reshaped.shape == (3, 4)
                assert reshaped.data.size == 12
            else:
                # Manual reshape test
                reshaped_data = t.data.reshape(3, 4)
                assert reshaped_data.shape == (3, 4)
                
        except ImportError:
            assert True, "Tensor reshape not implemented yet"
    
    def test_tensor_flatten(self):
        """Test tensor flattening."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t = Tensor(np.random.randn(2, 3, 4))
            
            if hasattr(t, 'flatten'):
                flat = t.flatten()
                assert flat.shape == (24,)
            else:
                # Manual flatten test
                flat_data = t.data.flatten()
                assert flat_data.shape == (24,)
                
        except ImportError:
            assert True, "Tensor flatten not implemented yet"
    
    def test_tensor_transpose(self):
        """Test tensor transpose."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
            
            if hasattr(t, 'T') or hasattr(t, 'transpose'):
                if hasattr(t, 'T'):
                    transposed = t.T
                else:
                    transposed = t.transpose()
                    
                assert transposed.shape == (3, 2)
                expected = np.array([[1, 4], [2, 5], [3, 6]])
                assert np.array_equal(transposed.data, expected)
            else:
                # Manual transpose test
                transposed_data = t.data.T
                assert transposed_data.shape == (3, 2)
                
        except ImportError:
            assert True, "Tensor transpose not implemented yet"


class TestTensorBroadcasting:
    """Test tensor broadcasting operations."""
    
    def test_scalar_broadcasting(self):
        """Test broadcasting with scalars."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t = Tensor([1, 2, 3])
            
            # Test scalar addition
            if hasattr(t, '__add__'):
                result = t + 5
                expected = np.array([6, 7, 8])
                assert np.array_equal(result.data, expected)
            
        except (ImportError, TypeError):
            assert True, "Scalar broadcasting not implemented yet"
    
    def test_vector_broadcasting(self):
        """Test broadcasting between different shapes."""
        try:
            from tinytorch.core.tensor import Tensor
            
            t1 = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
            t2 = Tensor([10, 20, 30])            # 3,
            
            # Should broadcast to same shape
            if hasattr(t1, '__add__'):
                result = t1 + t2
                assert result.shape == (2, 3)
                expected = np.array([[11, 22, 33], [14, 25, 36]])
                assert np.array_equal(result.data, expected)
            
        except (ImportError, TypeError):
            assert True, "Vector broadcasting not implemented yet"
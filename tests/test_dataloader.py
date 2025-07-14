"""
Mock-based module tests for DataLoader module.

This test file uses simple mocks to avoid cross-module dependencies while thoroughly
testing the DataLoader module functionality. The MockTensor class provides a minimal
interface that matches expected behavior without requiring actual implementations.

Test Philosophy:
- Use simple, visible mocks instead of complex mocking frameworks
- Test interface contracts and behavior, not implementation details
- Avoid dependency cascade where dataloader tests fail due to tensor bugs
- Focus on Dataset interface, DataLoader functionality, and data pipeline patterns
- Ensure educational value with clear test structure
"""

import pytest
import numpy as np
import sys
import os

# Add the module source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'source', '06_dataloader'))

from dataloader_dev import Dataset, DataLoader, SimpleDataset


class MockTensor:
    """
    Simple mock tensor for testing dataloader operations without tensor dependencies.
    
    This mock provides just enough functionality to test data loading operations
    without requiring the full Tensor implementation.
    """
    
    def __init__(self, data):
        """Initialize with numpy array data."""
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
    
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


class MockDataset(Dataset):
    """
    Simple mock dataset for testing without cross-module dependencies.
    
    This mock implements the Dataset interface with predictable, testable behavior.
    """
    
    def __init__(self, size=10, num_features=3, num_classes=2):
        """Initialize mock dataset with synthetic data."""
        self.size = size
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Generate predictable synthetic data
        np.random.seed(42)  # For reproducible tests
        self.data = np.random.randn(size, num_features).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=size)
    
    def __getitem__(self, index):
        """Get item by index."""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for dataset of size {self.size}")
        
        data = MockTensor(self.data[index])
        label = MockTensor(self.labels[index])
        return data, label
    
    def __len__(self):
        """Get dataset size."""
        return self.size
    
    def get_num_classes(self):
        """Get number of classes."""
        return self.num_classes


class TestDatasetInterface:
    """Test Dataset abstract base class and interface."""
    
    def test_dataset_abstract_methods(self):
        """Test that Dataset abstract methods raise NotImplementedError."""
        dataset = Dataset()
        
        with pytest.raises(NotImplementedError):
            dataset[0]
        
        with pytest.raises(NotImplementedError):
            len(dataset)
        
        with pytest.raises(NotImplementedError):
            dataset.get_num_classes()
    
    def test_dataset_get_sample_shape(self):
        """Test Dataset get_sample_shape method."""
        mock_dataset = MockDataset(size=5, num_features=4, num_classes=3)
        
        sample_shape = mock_dataset.get_sample_shape()
        assert sample_shape == (4,)  # Should match num_features
    
    def test_mock_dataset_basic_functionality(self):
        """Test MockDataset basic functionality."""
        dataset = MockDataset(size=10, num_features=5, num_classes=3)
        
        # Test length
        assert len(dataset) == 10
        
        # Test get_num_classes
        assert dataset.get_num_classes() == 3
        
        # Test item access
        data, label = dataset[0]
        assert isinstance(data, MockTensor)
        assert isinstance(label, MockTensor)
        assert data.shape == (5,)
        assert label.shape == ()
    
    def test_mock_dataset_index_bounds(self):
        """Test MockDataset index bounds checking."""
        dataset = MockDataset(size=5)
        
        # Valid indices should work
        for i in range(5):
            data, label = dataset[i]
            assert isinstance(data, MockTensor)
            assert isinstance(label, MockTensor)
        
        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            dataset[5]
        
        with pytest.raises(IndexError):
            dataset[-1]  # Negative indices not supported
    
    def test_mock_dataset_consistency(self):
        """Test MockDataset produces consistent results."""
        dataset = MockDataset(size=5, num_features=3, num_classes=2)
        
        # Multiple accesses should return same data
        data1, label1 = dataset[0]
        data2, label2 = dataset[0]
        
        assert np.allclose(data1.data, data2.data)
        assert np.allclose(label1.data, label2.data)
    
    def test_mock_dataset_different_configurations(self):
        """Test MockDataset with different configurations."""
        configs = [
            (5, 2, 2),     # Small dataset
            (100, 10, 5),  # Medium dataset
            (1000, 50, 10) # Large dataset
        ]
        
        for size, num_features, num_classes in configs:
            dataset = MockDataset(size=size, num_features=num_features, num_classes=num_classes)
            
            assert len(dataset) == size
            assert dataset.get_num_classes() == num_classes
            
            data, label = dataset[0]
            assert data.shape == (num_features,)


class TestDataLoaderBasic:
    """Test DataLoader basic functionality."""
    
    def test_dataloader_initialization(self):
        """Test DataLoader initialization."""
        dataset = MockDataset(size=10)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        assert dataloader.dataset is dataset
        assert dataloader.batch_size == 4
        assert dataloader.shuffle == True
    
    def test_dataloader_default_parameters(self):
        """Test DataLoader with default parameters."""
        dataset = MockDataset(size=10)
        dataloader = DataLoader(dataset)
        
        assert dataloader.batch_size == 32  # Default batch size
        assert dataloader.shuffle == True   # Default shuffle
    
    def test_dataloader_length_calculation(self):
        """Test DataLoader length calculation (number of batches)."""
        dataset = MockDataset(size=10)
        
        # Test different batch sizes
        test_cases = [
            (10, 2, 5),    # 10 samples, batch size 2 -> 5 batches
            (10, 3, 4),    # 10 samples, batch size 3 -> 4 batches (ceiling division)
            (10, 5, 2),    # 10 samples, batch size 5 -> 2 batches
            (10, 10, 1),   # 10 samples, batch size 10 -> 1 batch
            (10, 15, 1),   # 10 samples, batch size 15 -> 1 batch
        ]
        
        for dataset_size, batch_size, expected_batches in test_cases:
            dataset = MockDataset(size=dataset_size)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            assert len(dataloader) == expected_batches
    
    def test_dataloader_iteration_basic(self):
        """Test basic DataLoader iteration."""
        dataset = MockDataset(size=8, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        batches = list(dataloader)
        
        # Should have 3 batches: [3, 3, 2] samples
        assert len(batches) == 3
        
        # Check batch shapes
        batch_data, batch_labels = batches[0]
        assert batch_data.shape == (3, 3)  # 3 samples, 3 features each
        assert batch_labels.shape == (3,)  # 3 labels
        
        # Check last batch (partial)
        batch_data, batch_labels = batches[2]
        assert batch_data.shape == (2, 3)  # 2 samples, 3 features each
        assert batch_labels.shape == (2,)  # 2 labels
    
    def test_dataloader_iteration_complete(self):
        """Test that DataLoader iteration covers all samples."""
        dataset = MockDataset(size=10, num_features=4, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        total_samples = 0
        all_batch_data = []
        all_batch_labels = []
        
        for batch_data, batch_labels in dataloader:
            batch_size = batch_data.shape[0]
            total_samples += batch_size
            
            # Collect all data
            all_batch_data.append(batch_data.data)
            all_batch_labels.append(batch_labels.data)
        
        # Should process all samples
        assert total_samples == 10
        
        # Should have 4 batches: [3, 3, 3, 1]
        assert len(all_batch_data) == 4
        assert all_batch_data[0].shape == (3, 4)
        assert all_batch_data[1].shape == (3, 4)
        assert all_batch_data[2].shape == (3, 4)
        assert all_batch_data[3].shape == (1, 4)


class TestDataLoaderShuffling:
    """Test DataLoader shuffling functionality."""
    
    def test_dataloader_no_shuffle(self):
        """Test DataLoader without shuffling."""
        dataset = MockDataset(size=6, num_features=2, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get first batch
        batch_data, batch_labels = next(iter(dataloader))
        
        # Should be samples 0 and 1
        expected_data_0, expected_label_0 = dataset[0]
        expected_data_1, expected_label_1 = dataset[1]
        
        assert np.allclose(batch_data.data[0], expected_data_0.data)
        assert np.allclose(batch_data.data[1], expected_data_1.data)
    
    def test_dataloader_with_shuffle(self):
        """Test DataLoader with shuffling."""
        dataset = MockDataset(size=10, num_features=3, num_classes=2)
        
        # Create two dataloaders with different shuffle states
        dataloader1 = DataLoader(dataset, batch_size=5, shuffle=True)
        dataloader2 = DataLoader(dataset, batch_size=5, shuffle=True)
        
        # Get first batches
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))
        
        # Should have same shapes
        assert batch1[0].shape == batch2[0].shape
        assert batch1[1].shape == batch2[1].shape
        
        # Note: Due to randomness, batches might be different
        # This is a basic test that shuffling doesn't break functionality
    
    def test_dataloader_shuffle_consistency(self):
        """Test that DataLoader shuffling is consistent within an epoch."""
        dataset = MockDataset(size=8, num_features=2, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Collect all data from one epoch
        epoch_data = []
        epoch_labels = []
        
        for batch_data, batch_labels in dataloader:
            epoch_data.append(batch_data.data)
            epoch_labels.append(batch_labels.data)
        
        # Should have processed all samples
        total_samples = sum(data.shape[0] for data in epoch_data)
        assert total_samples == 8
        
        # All data should be accounted for
        assert len(epoch_data) == 2  # 8 samples / 4 batch_size = 2 batches


class TestDataLoaderEdgeCases:
    """Test DataLoader edge cases and error conditions."""
    
    def test_dataloader_empty_dataset(self):
        """Test DataLoader with empty dataset."""
        dataset = MockDataset(size=0)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Should have 0 batches
        assert len(dataloader) == 0
        
        # Iteration should produce no batches
        batches = list(dataloader)
        assert len(batches) == 0
    
    def test_dataloader_single_sample(self):
        """Test DataLoader with single sample."""
        dataset = MockDataset(size=1, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Should have 1 batch
        assert len(dataloader) == 1
        
        # Single batch should contain the one sample
        batch_data, batch_labels = next(iter(dataloader))
        assert batch_data.shape == (1, 3)
        assert batch_labels.shape == (1,)
    
    def test_dataloader_batch_size_larger_than_dataset(self):
        """Test DataLoader with batch size larger than dataset."""
        dataset = MockDataset(size=5, num_features=4, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=10)
        
        # Should have 1 batch
        assert len(dataloader) == 1
        
        # Batch should contain all samples
        batch_data, batch_labels = next(iter(dataloader))
        assert batch_data.shape == (5, 4)
        assert batch_labels.shape == (5,)
    
    def test_dataloader_batch_size_one(self):
        """Test DataLoader with batch size of 1."""
        dataset = MockDataset(size=5, num_features=2, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Should have 5 batches
        assert len(dataloader) == 5
        
        # Each batch should have 1 sample
        for batch_data, batch_labels in dataloader:
            assert batch_data.shape == (1, 2)
            assert batch_labels.shape == (1,)
    
    def test_dataloader_multiple_epochs(self):
        """Test DataLoader across multiple epochs."""
        dataset = MockDataset(size=6, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test 3 epochs
        for epoch in range(3):
            epoch_samples = 0
            batch_count = 0
            
            for batch_data, batch_labels in dataloader:
                batch_count += 1
                epoch_samples += batch_data.shape[0]
            
            # Each epoch should process all samples
            assert epoch_samples == 6
            assert batch_count == 3  # 6 samples / 2 batch_size = 3 batches


class TestDataLoaderIntegration:
    """Test DataLoader integration with different dataset types."""
    
    def test_dataloader_with_simple_dataset(self):
        """Test DataLoader with SimpleDataset."""
        # Note: This test assumes SimpleDataset exists and works
        try:
            dataset = SimpleDataset(size=20, num_features=5, num_classes=3)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Test basic functionality
            assert len(dataloader) == 5  # 20 / 4 = 5 batches
            
            # Test iteration
            total_samples = 0
            for batch_data, batch_labels in dataloader:
                total_samples += batch_data.shape[0]
                assert batch_data.shape[1] == 5  # num_features
            
            assert total_samples == 20
            
        except (ImportError, NameError):
            # SimpleDataset might not be available in all test environments
            pytest.skip("SimpleDataset not available")
    
    def test_dataloader_with_custom_dataset(self):
        """Test DataLoader with custom dataset implementation."""
        class CustomDataset(Dataset):
            def __init__(self):
                self.data = [(i, i % 2) for i in range(10)]
            
            def __getitem__(self, index):
                value, label = self.data[index]
                return MockTensor([value]), MockTensor([label])
            
            def __len__(self):
                return len(self.data)
            
            def get_num_classes(self):
                return 2
        
        dataset = CustomDataset()
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        # Test that it works with custom dataset
        batches = list(dataloader)
        assert len(batches) == 4  # 10 / 3 = 4 batches (ceiling division)
        
        # Check first batch
        batch_data, batch_labels = batches[0]
        assert batch_data.shape == (3, 1)
        assert batch_labels.shape == (3, 1)
    
    def test_dataloader_different_data_types(self):
        """Test DataLoader with different data types."""
        class MultiTypeDataset(Dataset):
            def __init__(self):
                self.samples = [
                    (np.array([1.0, 2.0]), 0),
                    (np.array([3.0, 4.0]), 1),
                    (np.array([5.0, 6.0]), 0),
                    (np.array([7.0, 8.0]), 1),
                ]
            
            def __getitem__(self, index):
                data, label = self.samples[index]
                return MockTensor(data), MockTensor(label)
            
            def __len__(self):
                return len(self.samples)
            
            def get_num_classes(self):
                return 2
        
        dataset = MultiTypeDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test batching different data types
        batch_data, batch_labels = next(iter(dataloader))
        assert batch_data.shape == (2, 2)
        assert batch_labels.shape == (2,)


class TestDataLoaderPerformance:
    """Test DataLoader performance characteristics."""
    
    def test_dataloader_memory_efficiency(self):
        """Test DataLoader memory efficiency with large datasets."""
        # Create relatively large dataset
        dataset = MockDataset(size=1000, num_features=50, num_classes=10)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Should be able to iterate without memory issues
        batch_count = 0
        for batch_data, batch_labels in dataloader:
            batch_count += 1
            assert batch_data.shape[1] == 50
            assert batch_labels.shape[0] <= 32
            
            # Only process first few batches for performance
            if batch_count >= 5:
                break
        
        assert batch_count == 5
    
    def test_dataloader_iteration_speed(self):
        """Test DataLoader iteration speed."""
        dataset = MockDataset(size=100, num_features=10, num_classes=5)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        # Should be able to iterate quickly
        import time
        start_time = time.time()
        
        total_samples = 0
        for batch_data, batch_labels in dataloader:
            total_samples += batch_data.shape[0]
        
        end_time = time.time()
        
        # Should process all samples
        assert total_samples == 100
        
        # Should complete reasonably quickly (less than 1 second)
        assert end_time - start_time < 1.0
    
    def test_dataloader_scalability(self):
        """Test DataLoader scalability with different sizes."""
        sizes = [10, 100, 1000]
        batch_sizes = [1, 8, 32]
        
        for size in sizes:
            for batch_size in batch_sizes:
                dataset = MockDataset(size=size, num_features=5, num_classes=3)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Should handle different scales
                expected_batches = (size + batch_size - 1) // batch_size
                assert len(dataloader) == expected_batches
                
                # Should iterate correctly
                total_samples = 0
                for batch_data, batch_labels in dataloader:
                    total_samples += batch_data.shape[0]
                
                assert total_samples == size


class TestDataLoaderRobustness:
    """Test DataLoader robustness and error handling."""
    
    def test_dataloader_with_invalid_batch_size(self):
        """Test DataLoader with invalid batch sizes."""
        dataset = MockDataset(size=10)
        
        # Zero batch size should raise error
        with pytest.raises((ValueError, AssertionError)):
            DataLoader(dataset, batch_size=0)
        
        # Negative batch size should raise error
        with pytest.raises((ValueError, AssertionError)):
            DataLoader(dataset, batch_size=-1)
    
    def test_dataloader_with_none_dataset(self):
        """Test DataLoader with None dataset."""
        with pytest.raises((TypeError, AttributeError)):
            DataLoader(None, batch_size=4)
    
    def test_dataloader_iteration_consistency(self):
        """Test DataLoader iteration consistency."""
        dataset = MockDataset(size=12, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        # Multiple iterations should be consistent
        batches1 = list(dataloader)
        batches2 = list(dataloader)
        
        assert len(batches1) == len(batches2)
        
        # Without shuffle, should be identical
        for (batch1_data, batch1_labels), (batch2_data, batch2_labels) in zip(batches1, batches2):
            assert np.allclose(batch1_data.data, batch2_data.data)
            assert np.allclose(batch1_labels.data, batch2_labels.data)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 
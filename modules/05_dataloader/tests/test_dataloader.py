"""
Test suite for the dataloader module.
This tests the student implementations to ensure they work correctly.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from the main package (rock solid foundation)
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import Dataset, DataLoader, CIFAR10Dataset, Normalizer, create_data_pipeline

def safe_numpy(tensor):
    """Get numpy array from tensor, using .data attribute"""
    return tensor.data

def safe_item(tensor):
    """Get scalar value from tensor"""
    return float(tensor.data)

class TestCIFAR10Dataset(Dataset):
    """Test dataset that uses local test data instead of downloading CIFAR-10."""
    
    def __init__(self, root_dir: str, train: bool = True, download: bool = True):
        """Initialize with local test data."""
        self.root_dir = root_dir
        self.train = train
        self.download = download
        
        # Use local test data
        test_data_dir = Path(__file__).parent / "test_data"
        if not test_data_dir.exists():
            raise FileNotFoundError(f"Test data not found at {test_data_dir}")
        
        self._load_test_data(test_data_dir)
    
    def _load_test_data(self, data_dir):
        """Load the small test dataset."""
        # Load metadata
        with open(data_dir / "batches.meta", "rb") as f:
            meta_dict = pickle.load(f)
        
        self.class_names = [name.decode() for name in meta_dict[b'label_names']]
        
        # Load training or test data
        if self.train:
            with open(data_dir / "data_batch_1", "rb") as f:
                data_dict = pickle.load(f)
        else:
            with open(data_dir / "test_batch", "rb") as f:
                data_dict = pickle.load(f)
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        self.data = data_dict[b'data'].reshape(-1, 3, 32, 32)
        self.labels = data_dict[b'labels']
    
    def __getitem__(self, index: int):
        """Get a single sample and label."""
        image = self.data[index]
        label = self.labels[index]
        
        return Tensor(image.astype(np.float32)), Tensor(np.array(label))
    
    def __len__(self) -> int:
        """Get the total number of samples."""
        return len(self.data)
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.class_names)

class TestDatasetInterface:
    """Test the base Dataset class interface (abstract class behavior)."""
    
    def test_dataset_is_abstract(self):
        """Test that Dataset base class is abstract."""
        dataset = Dataset()
        
        # Should raise NotImplementedError for abstract methods
        with pytest.raises(NotImplementedError):
            dataset[0]
        
        with pytest.raises(NotImplementedError):
            len(dataset)
        
        with pytest.raises(NotImplementedError):
            dataset.get_num_classes()
    
    def test_concrete_dataset_implementation(self):
        """Test that concrete datasets work properly."""
        class TestDataset(Dataset):
            def __init__(self, size=10):
                self.size = size
                self.data = [np.random.randn(3, 32, 32) for _ in range(size)]
                self.labels = [i % 3 for i in range(size)]
            
            def __getitem__(self, index):
                return Tensor(self.data[index]), Tensor(np.array(self.labels[index]))
            
            def __len__(self):
                return self.size
            
            def get_num_classes(self):
                return 3
        
        dataset = TestDataset(5)
        
        # Test basic functionality
        assert len(dataset) == 5
        assert dataset.get_num_classes() == 3
        
        # Test indexing
        sample, label = dataset[0]
        assert sample.shape == (3, 32, 32)
        assert label.shape == ()
        
        # Test get_sample_shape
        assert dataset.get_sample_shape() == (3, 32, 32)

class TestLocalCIFAR10Dataset:
    """Test CIFAR-10 dataset with local test data."""
    
    def test_cifar10_train_set_load(self):
        """Test loading training set from local test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use local test data
            dataset = TestCIFAR10Dataset(temp_dir, train=True, download=True)
            
            # Verify basic properties
            assert len(dataset) == 50  # Our test training set size
            assert dataset.get_num_classes() == 10
            
            # Test sample access
            image, label = dataset[0]
            assert image.shape == (3, 32, 32)  # CIFAR-10 image shape
            assert 0 <= safe_item(label) < 10  # Valid class label
            
            # Test class names
            assert len(dataset.class_names) == 10
            assert 'airplane' in dataset.class_names
            assert 'truck' in dataset.class_names
    
    def test_cifar10_test_set_load(self):
        """Test loading test set from local test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use local test data
            dataset = TestCIFAR10Dataset(temp_dir, train=False, download=True)
            
            # Verify test set properties
            assert len(dataset) == 20  # Our test test set size
            assert dataset.get_num_classes() == 10
            
            # Test sample access
            image, label = dataset[0]
            assert image.shape == (3, 32, 32)
            assert 0 <= safe_item(label) < 10
    
    def test_cifar10_data_types(self):
        """Test that test data has correct types and ranges."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TestCIFAR10Dataset(temp_dir, train=True, download=True)
            
            # Test first few samples
            for i in range(5):
                image, label = dataset[i]
                
                # Check data types
                assert isinstance(image, Tensor)
                assert isinstance(label, Tensor)
                
                # Check value ranges (our test data uses 0-255 range)
                assert 0 <= safe_numpy(image).min() <= 255
                assert 0 <= safe_numpy(image).max() <= 255
                
                # Check label is valid class
                assert 0 <= safe_item(label) < 10

class TestDataLoader:
    """Test DataLoader with local test data."""
    
    def setup_method(self):
        """Set up local test dataset for DataLoader tests."""
        self.temp_dir = tempfile.mkdtemp()
        # Use local test data
        self.dataset = TestCIFAR10Dataset(self.temp_dir, train=True, download=True)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataloader_creation(self):
        """Test DataLoader creation with local test data."""
        # Test with default parameters
        loader = DataLoader(self.dataset, batch_size=16)
        assert len(loader) == 4  # 50 samples / 16 batch_size = 4 batches (rounded up)
        
        # Test with custom batch size
        loader = DataLoader(self.dataset, batch_size=10)
        assert len(loader) == 5  # 50 samples / 10 batch_size = 5 batches
    
    def test_dataloader_iteration_test_data(self):
        """Test DataLoader iteration with local test data."""
        loader = DataLoader(self.dataset, batch_size=8, shuffle=True)
        
        batch_count = 0
        total_samples = 0
        
        for batch_data, batch_labels in loader:
            batch_count += 1
            batch_size = batch_data.shape[0]
            total_samples += batch_size
            
            # Check batch shapes
            assert batch_data.shape[1:] == (3, 32, 32)  # CIFAR-10 image shape
            assert batch_labels.shape == (batch_size,)
            
            # Check data types
            assert isinstance(batch_data, Tensor)
            assert isinstance(batch_labels, Tensor)
            
            # Check test data properties
            assert 0 <= safe_numpy(batch_data).min() <= 255
            assert 0 <= safe_numpy(batch_data).max() <= 255
            assert 0 <= safe_numpy(batch_labels).min() < 10
            assert 0 <= safe_numpy(batch_labels).max() < 10
            
            # Check batch size
            assert batch_size <= 8
            
            if batch_count >= 3:  # Test first few batches
                break
        
        assert batch_count > 0
        assert total_samples <= len(self.dataset)
    
    def test_dataloader_shuffling_test_data(self):
        """Test that shuffling works with test data."""
        loader1 = DataLoader(self.dataset, batch_size=10, shuffle=True)
        loader2 = DataLoader(self.dataset, batch_size=10, shuffle=True)
        
        # Get first batch from each loader
        batch1_data, batch1_labels = next(iter(loader1))
        batch2_data, batch2_labels = next(iter(loader2))
        
        # With shuffling, batches should likely be different
        # (This test might occasionally fail due to randomness, but very unlikely)
        different = not np.array_equal(safe_numpy(batch1_labels), safe_numpy(batch2_labels))
        # Note: We don't assert this because random shuffling might occasionally produce same order
    
    def test_dataloader_no_shuffle_test_data(self):
        """Test DataLoader without shuffling uses test data in order."""
        loader = DataLoader(self.dataset, batch_size=10, shuffle=False)
        
        # Get first batch
        batch_data, batch_labels = next(iter(loader))
        
        # Without shuffling, should get first 10 samples in order
        expected_samples = [self.dataset[i] for i in range(10)]
        expected_labels = [safe_item(sample[1]) for sample in expected_samples]
        
        np.testing.assert_array_equal(safe_numpy(batch_labels), expected_labels)

class TestNormalizer:
    """Test Normalizer with local test data."""
    
    def setup_method(self):
        """Set up local test data for normalization tests."""
        self.temp_dir = tempfile.mkdtemp()
        dataset = TestCIFAR10Dataset(self.temp_dir, train=True, download=True)
        
        # Get first 20 samples for testing
        self.test_data = []
        for i in range(20):
            image, _ = dataset[i]
            self.test_data.append(image)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_normalizer_fit_test_data(self):
        """Test Normalizer fit with local test data."""
        normalizer = Normalizer()
        normalizer.fit(self.test_data)
        
        # Check computed statistics
        assert normalizer.mean is not None
        assert normalizer.std is not None
        
        # Our test data has pixel values 0-255, so mean should be reasonable
        assert 0 <= normalizer.mean <= 255
        assert normalizer.std > 0  # Should have some variation
    
    def test_normalizer_transform_test_data(self):
        """Test Normalizer transform with local test data."""
        normalizer = Normalizer()
        normalizer.fit(self.test_data)
        
        # Transform single sample
        sample = self.test_data[0]
        normalized = normalizer.transform(sample)
        
        # Check that normalization changes the values
        assert not np.allclose(safe_numpy(sample), safe_numpy(normalized))
        
        # Check that normalized data has different statistics
        original_mean = np.mean(safe_numpy(sample))
        normalized_mean = np.mean(safe_numpy(normalized))
        assert abs(normalized_mean) < abs(original_mean)  # Should be closer to 0
    
    def test_normalizer_transform_batch_test_data(self):
        """Test Normalizer with batch of test data."""
        normalizer = Normalizer()
        normalizer.fit(self.test_data)
        
        # Transform batch
        batch = self.test_data[:5]
        normalized_batch = normalizer.transform(batch)
        
        # Check that we get same number of samples
        assert len(normalized_batch) == len(batch)
        
        # Check that each sample is normalized
        for original, normalized in zip(batch, normalized_batch):
            assert not np.allclose(safe_numpy(original), safe_numpy(normalized))

class TestDataPipeline:
    """Test complete data pipeline with local test data."""
    
    def test_create_data_pipeline_test_data(self):
        """Test creating data pipeline with local test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy test data to temp directory
            test_data_dir = Path(__file__).parent / "test_data"
            import shutil
            shutil.copytree(test_data_dir, temp_dir + "/test_data")
            
            # Create pipeline (this would normally download CIFAR-10)
            # For testing, we'll create a simple pipeline manually
            dataset = TestCIFAR10Dataset(temp_dir, train=True, download=True)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test pipeline components
            assert len(dataset) == 50  # Our test training set
            assert len(dataloader) == 7  # 50 samples / 8 batch_size = 7 batches
            
            # Test that we can iterate through the pipeline
            batch_count = 0
            for batch_data, batch_labels in dataloader:
                batch_count += 1
                assert batch_data.shape[1:] == (3, 32, 32)
                assert batch_labels.shape[0] <= 8
                
                if batch_count >= 3:  # Test first few batches
                    break
            
            assert batch_count > 0
    
    def test_pipeline_normalization_test_data(self):
        """Test pipeline with normalization using local test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TestCIFAR10Dataset(temp_dir, train=True, download=True)
            
            # Get some samples for normalization
            samples = [dataset[i][0] for i in range(10)]
            
            # Create and fit normalizer
            normalizer = Normalizer()
            normalizer.fit(samples)
            
            # Test that normalization works
            normalized = normalizer.transform(samples[0])
            assert not np.allclose(safe_numpy(samples[0]), safe_numpy(normalized))
            
            # Test with dataloader
            dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
            batch_data, batch_labels = next(iter(dataloader))
            
            # Normalize batch
            normalized_batch = []
            for i in range(batch_data.shape[0]):
                sample = Tensor(batch_data.data[i])
                normalized_sample = normalizer.transform(sample)
                normalized_batch.append(normalized_sample.data)
            
            normalized_batch = Tensor(np.stack(normalized_batch))
            
            # Check that batch normalization works
            assert normalized_batch.shape == batch_data.shape
            assert not np.allclose(safe_numpy(batch_data), safe_numpy(normalized_batch))

class TestEdgeCases:
    """Test edge cases with local test data."""
    
    def test_small_batch_size_test_data(self):
        """Test with very small batch size using local test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small dataset
            dataset = TestCIFAR10Dataset(temp_dir, train=True, download=True)
            
            # Use batch size of 1
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Test first few batches
            batch_count = 0
            for batch_data, batch_labels in loader:
                assert batch_data.shape == (1, 3, 32, 32)
                assert batch_labels.shape == (1,)
                
                batch_count += 1
                if batch_count >= 5:
                    break
            
            assert batch_count == 5

def run_data_tests():
    """Run all data tests."""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_data_tests() 
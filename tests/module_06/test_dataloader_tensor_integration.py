"""
Integration Tests - DataLoader and Tensor

Tests real integration between DataLoader and Tensor modules.
Uses actual TinyTorch components to verify they work together correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import DataLoader, Dataset, SimpleDataset
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Dense


class TestDataLoaderTensorIntegration:
    """Test integration between DataLoader and Tensor components."""
    
    def test_simple_dataset_produces_tensors(self):
        """Test SimpleDataset produces real Tensor objects."""
        # Create SimpleDataset
        dataset = SimpleDataset(size=10, num_features=3, num_classes=2)
        
        # Get a sample
        data, label = dataset[0]
        
        # Verify outputs are tensors
        assert isinstance(data, Tensor), "Data should be a Tensor"
        assert isinstance(label, Tensor), "Label should be a Tensor"
        
        # Verify tensor properties
        assert data.shape == (3,), f"Expected data shape (3,), got {data.shape}"
        assert label.shape == (), f"Expected label shape (), got {label.shape}"
        assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
        assert label.dtype == np.int32, f"Expected int32, got {label.dtype}"
    
    def test_dataloader_produces_tensor_batches(self):
        """Test DataLoader produces batches of real Tensor objects."""
        # Create dataset and dataloader
        dataset = SimpleDataset(size=20, num_features=4, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        # Get first batch
        batch_data, batch_labels = next(iter(dataloader))
        
        # Verify batch outputs are tensors
        assert isinstance(batch_data, Tensor), "Batch data should be a Tensor"
        assert isinstance(batch_labels, Tensor), "Batch labels should be a Tensor"
        
        # Verify batch shapes
        assert batch_data.shape == (5, 4), f"Expected batch data shape (5, 4), got {batch_data.shape}"
        assert batch_labels.shape == (5,), f"Expected batch labels shape (5,), got {batch_labels.shape}"
        
        # Verify data types
        assert batch_data.dtype == np.float32, f"Expected float32, got {batch_data.dtype}"
        assert batch_labels.dtype == np.int32, f"Expected int32, got {batch_labels.dtype}"
    
    def test_dataloader_tensor_compatibility_with_activations(self):
        """Test DataLoader tensors work with activation functions."""
        # Create dataset and dataloader
        dataset = SimpleDataset(size=10, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get batch
        batch_data, batch_labels = next(iter(dataloader))
        
        # Apply activation function
        relu = ReLU()
        activated_data = relu(batch_data)
        
        # Verify result is tensor
        assert isinstance(activated_data, Tensor), "Activated data should be a Tensor"
        assert activated_data.shape == batch_data.shape, "Shape should be preserved"
        
        # Verify ReLU applied correctly (non-negative values)
        assert np.all(activated_data.data >= 0), "ReLU should produce non-negative values"
    
    def test_dataloader_tensor_compatibility_with_layers(self):
        """Test DataLoader tensors work with neural network layers."""
        # Create dataset and dataloader
        dataset = SimpleDataset(size=10, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get batch
        batch_data, batch_labels = next(iter(dataloader))
        
        # Apply dense layer
        dense = Dense(input_size=3, output_size=2)
        output = dense(batch_data)
        
        # Verify result is tensor
        assert isinstance(output, Tensor), "Layer output should be a Tensor"
        assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
        assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
    
    def test_dataloader_full_pipeline_integration(self):
        """Test DataLoader tensors in complete ML pipeline."""
        # Create dataset and dataloader
        dataset = SimpleDataset(size=12, num_features=4, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False)
        
        # Get batch
        batch_data, batch_labels = next(iter(dataloader))
        
        # Apply full pipeline: Dense → ReLU → Dense
        dense1 = Dense(input_size=4, output_size=8)
        relu = ReLU()
        dense2 = Dense(input_size=8, output_size=3)
        
        # Forward pass
        hidden = dense1(batch_data)
        activated = relu(hidden)
        output = dense2(activated)
        
        # Verify all outputs are tensors
        assert isinstance(hidden, Tensor), "Hidden layer should be Tensor"
        assert isinstance(activated, Tensor), "Activated layer should be Tensor"
        assert isinstance(output, Tensor), "Output layer should be Tensor"
        
        # Verify shapes through pipeline
        assert hidden.shape == (6, 8), f"Hidden shape should be (6, 8), got {hidden.shape}"
        assert activated.shape == (6, 8), f"Activated shape should be (6, 8), got {activated.shape}"
        assert output.shape == (6, 3), f"Output shape should be (6, 3), got {output.shape}"


class TestDataLoaderTensorBatching:
    """Test DataLoader batching with tensor integration."""
    
    def test_different_batch_sizes(self):
        """Test DataLoader with different batch sizes produces correct tensors."""
        dataset = SimpleDataset(size=20, num_features=3, num_classes=2)
        
        batch_sizes = [1, 4, 8, 10]
        for batch_size in batch_sizes:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            batch_data, batch_labels = next(iter(dataloader))
            
            # Verify tensor shapes
            assert batch_data.shape == (batch_size, 3), f"Data shape should be ({batch_size}, 3), got {batch_data.shape}"
            assert batch_labels.shape == (batch_size,), f"Label shape should be ({batch_size},), got {batch_labels.shape}"
            
            # Verify tensor types
            assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
            assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"
    
    def test_shuffling_preserves_tensor_integrity(self):
        """Test that shuffling preserves tensor data integrity."""
        dataset = SimpleDataset(size=10, num_features=2, num_classes=2)
        
        # Create two dataloaders with different shuffle settings
        dataloader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)
        dataloader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
        
        # Get batches
        batch_no_shuffle = next(iter(dataloader_no_shuffle))
        batch_shuffle = next(iter(dataloader_shuffle))
        
        # Both should produce valid tensors
        for batch_data, batch_labels in [batch_no_shuffle, batch_shuffle]:
            assert isinstance(batch_data, Tensor), "Data should be Tensor"
            assert isinstance(batch_labels, Tensor), "Labels should be Tensor"
            assert batch_data.shape == (5, 2), f"Expected shape (5, 2), got {batch_data.shape}"
            assert batch_labels.shape == (5,), f"Expected shape (5,), got {batch_labels.shape}"
    
    def test_iteration_produces_consistent_tensors(self):
        """Test that iterating through DataLoader produces consistent tensors."""
        dataset = SimpleDataset(size=12, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch_count = 0
        for batch_data, batch_labels in dataloader:
            batch_count += 1
            
            # Verify each batch produces valid tensors
            assert isinstance(batch_data, Tensor), f"Batch {batch_count} data should be Tensor"
            assert isinstance(batch_labels, Tensor), f"Batch {batch_count} labels should be Tensor"
            
            # Verify shapes (last batch might be smaller)
            assert batch_data.shape[1] == 3, f"Feature dim should be 3, got {batch_data.shape[1]}"
            assert batch_data.shape[0] == batch_labels.shape[0], "Batch and label sizes should match"
            
            # Verify data types
            assert batch_data.dtype == np.float32, "Data should be float32"
            assert batch_labels.dtype == np.int32, "Labels should be int32"
        
        # Should have processed all data
        assert batch_count == 3, f"Expected 3 batches, got {batch_count}"


class TestDataLoaderTensorDataTypes:
    """Test DataLoader tensor data type handling."""
    
    def test_float32_tensor_production(self):
        """Test DataLoader produces float32 tensors for data."""
        dataset = SimpleDataset(size=8, num_features=2, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch_data, batch_labels = next(iter(dataloader))
        
        # Verify data types
        assert batch_data.dtype == np.float32, f"Expected float32, got {batch_data.dtype}"
        assert isinstance(batch_data.data, np.ndarray), "Underlying data should be numpy array"
        assert batch_data.data.dtype == np.float32, "Underlying array should be float32"
    
    def test_int32_tensor_production(self):
        """Test DataLoader produces int32 tensors for labels."""
        dataset = SimpleDataset(size=8, num_features=2, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch_data, batch_labels = next(iter(dataloader))
        
        # Verify label types
        assert batch_labels.dtype == np.int32, f"Expected int32, got {batch_labels.dtype}"
        assert isinstance(batch_labels.data, np.ndarray), "Underlying labels should be numpy array"
        assert batch_labels.data.dtype == np.int32, "Underlying array should be int32"
    
    def test_tensor_data_ranges(self):
        """Test DataLoader produces tensors with reasonable data ranges."""
        dataset = SimpleDataset(size=10, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        batch_data, batch_labels = next(iter(dataloader))
        
        # Check data ranges
        assert np.all(np.isfinite(batch_data.data)), "Data should be finite"
        assert np.all(batch_labels.data >= 0), "Labels should be non-negative"
        assert np.all(batch_labels.data < 2), "Labels should be less than num_classes"


class TestDataLoaderTensorRealisticScenarios:
    """Test DataLoader with realistic tensor scenarios."""
    
    def test_training_loop_simulation(self):
        """Test DataLoader tensors in training loop simulation."""
        dataset = SimpleDataset(size=16, num_features=4, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Simulate training loop
        epoch_batches = 0
        for epoch in range(2):
            batch_count = 0
            for batch_data, batch_labels in dataloader:
                batch_count += 1
                
                # Simulate forward pass
                dense = Dense(input_size=4, output_size=2)
                output = dense(batch_data)
                
                # Verify tensor operations work
                assert isinstance(output, Tensor), "Forward pass should produce Tensor"
                assert output.shape == (8, 2), f"Expected shape (8, 2), got {output.shape}"
                
                # Simulate loss computation (simplified)
                loss = output.data.mean()  # Simple loss
                assert np.isfinite(loss), "Loss should be finite"
                
                epoch_batches += 1
            
            assert batch_count == 2, f"Expected 2 batches per epoch, got {batch_count}"
        
        assert epoch_batches == 4, f"Expected 4 total batches, got {epoch_batches}"
    
    def test_different_dataset_sizes(self):
        """Test DataLoader with different dataset sizes."""
        test_cases = [
            (5, 2),    # Small dataset
            (32, 8),   # Medium dataset
            (100, 16), # Large dataset
        ]
        
        for dataset_size, batch_size in test_cases:
            dataset = SimpleDataset(size=dataset_size, num_features=3, num_classes=2)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            total_samples = 0
            for batch_data, batch_labels in dataloader:
                # Verify tensor properties
                assert isinstance(batch_data, Tensor), "Data should be Tensor"
                assert isinstance(batch_labels, Tensor), "Labels should be Tensor"
                
                # Count samples
                total_samples += batch_data.shape[0]
                
                # Verify shapes
                assert batch_data.shape[1] == 3, "Feature dim should be 3"
                assert batch_data.shape[0] == batch_labels.shape[0], "Batch sizes should match"
            
            assert total_samples == dataset_size, f"Should process all {dataset_size} samples, got {total_samples}"
    
    def test_dataloader_with_complex_pipeline(self):
        """Test DataLoader integration with complex neural network pipeline."""
        dataset = SimpleDataset(size=20, num_features=5, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        # Create complex pipeline
        dense1 = Dense(input_size=5, output_size=16)
        relu1 = ReLU()
        dense2 = Dense(input_size=16, output_size=8)
        relu2 = ReLU()
        dense3 = Dense(input_size=8, output_size=3)
        
        # Process batches
        for batch_data, batch_labels in dataloader:
            # Forward pass through complex pipeline
            x = dense1(batch_data)
            x = relu1(x)
            x = dense2(x)
            x = relu2(x)
            output = dense3(x)
            
            # Verify final output
            assert isinstance(output, Tensor), "Final output should be Tensor"
            assert output.shape == (10, 3), f"Expected shape (10, 3), got {output.shape}"
            assert output.dtype == np.float32, "Output should be float32"
            assert np.all(np.isfinite(output.data)), "Output should be finite"
    
    def test_dataloader_memory_efficiency(self):
        """Test DataLoader memory efficiency with tensor operations."""
        dataset = SimpleDataset(size=50, num_features=10, num_classes=5)
        dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
        
        # Process batches and verify memory usage patterns
        processed_batches = []
        for batch_data, batch_labels in dataloader:
            # Store tensor info (not the actual tensors to avoid memory issues)
            batch_info = {
                'data_shape': batch_data.shape,
                'label_shape': batch_labels.shape,
                'data_type': batch_data.dtype,
                'label_type': batch_labels.dtype
            }
            processed_batches.append(batch_info)
            
            # Verify tensors are properly formed
            assert isinstance(batch_data, Tensor), "Data should be Tensor"
            assert isinstance(batch_labels, Tensor), "Labels should be Tensor"
        
        # Verify we processed expected number of batches
        assert len(processed_batches) == 2, f"Expected 2 batches, got {len(processed_batches)}"
        
        # Verify consistency across batches
        for i, batch_info in enumerate(processed_batches):
            assert batch_info['data_shape'][1] == 10, f"Batch {i} should have 10 features"
            assert batch_info['data_type'] == np.float32, f"Batch {i} data should be float32"
            assert batch_info['label_type'] == np.int32, f"Batch {i} labels should be int32"


class TestCustomDatasetIntegration:
    """Test custom dataset integration with tensor operations."""
    
    def test_custom_dataset_with_tensors(self):
        """Test custom dataset that produces tensors works with DataLoader."""
        
        class CustomTensorDataset(Dataset):
            def __init__(self, size: int):
                self.size = size
                self.data = [Tensor(np.random.rand(3).astype(np.float32)) for _ in range(size)]
                self.labels = [Tensor(np.random.randint(0, 2, dtype=np.int32)) for _ in range(size)]
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, index):
                return self.data[index], self.labels[index]
        
        # Create custom dataset and dataloader
        dataset = CustomTensorDataset(size=12)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test integration
        batch_data, batch_labels = next(iter(dataloader))
        
        # Verify tensor properties
        assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
        assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"
        assert batch_data.shape == (4, 3), f"Expected shape (4, 3), got {batch_data.shape}"
        assert batch_labels.shape == (4,), f"Expected shape (4,), got {batch_labels.shape}"
        
        # Test with neural network components
        dense = Dense(input_size=3, output_size=2)
        output = dense(batch_data)
        
        assert isinstance(output, Tensor), "Dense output should be Tensor"
        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}" 
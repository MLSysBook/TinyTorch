"""
Integration Tests - Training Module

Tests real integration between training components and other TinyTorch modules.
Uses actual TinyTorch components to verify training pipeline works correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential, create_mlp
from tinytorch.core.dataloader import DataLoader, SimpleDataset
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.core.training import MeanSquaredError, CrossEntropyLoss, BinaryCrossEntropyLoss, Accuracy, Trainer


class TestLossIntegration:
    """Test loss functions with real TinyTorch components."""
    
    def test_mse_with_real_tensors(self):
        """Test MSE loss works with real Tensor objects."""
        mse = MeanSquaredError()
        
        # Test with real tensors
        y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y_true = Tensor([[1.5, 2.5], [2.5, 3.5]])
        
        loss = mse(y_pred, y_true)
        
        # Verify integration
        assert isinstance(loss, Tensor)
        assert loss.data == 0.25  # All squared differences are 0.25
    
    def test_crossentropy_with_real_tensors(self):
        """Test CrossEntropy loss works with real Tensor objects."""
        ce = CrossEntropyLoss()
        
        # Test with real tensors
        y_pred = Tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        y_true = Tensor([0, 1])
        
        loss = ce(y_pred, y_true)
        
        # Verify integration
        assert isinstance(loss, Tensor)
        assert loss.data > 0.0  # Should be positive loss
    
    def test_binary_crossentropy_with_real_tensors(self):
        """Test BinaryCrossEntropy loss works with real Tensor objects."""
        bce = BinaryCrossEntropyLoss()
        
        # Test with real tensors
        y_pred = Tensor([[1.0], [0.0], [-1.0]])
        y_true = Tensor([[1.0], [1.0], [0.0]])
        
        loss = bce(y_pred, y_true)
        
        # Verify integration
        assert isinstance(loss, Tensor)
        assert loss.data > 0.0  # Should be positive loss
    
    def test_losses_with_network_outputs(self):
        """Test losses work with real network outputs."""
        # Create real network
        network = Sequential([
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
            Softmax()
        ])
        
        # Generate predictions
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predictions = network(x)
        
        # Test with CrossEntropy
        ce = CrossEntropyLoss()
        y_true = Tensor([0, 1])
        loss = ce(predictions, y_true)
        
        # Verify integration
        assert isinstance(loss, Tensor)
        assert loss.data > 0.0
        assert predictions.shape == (2, 2)  # Batch size 2, 2 classes


class TestMetricsIntegration:
    """Test metrics with real TinyTorch components."""
    
    def test_accuracy_with_real_tensors(self):
        """Test Accuracy metric works with real Tensor objects."""
        accuracy = Accuracy()
        
        # Test with real tensors
        y_pred = Tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        y_true = Tensor([0, 1, 0])
        
        acc = accuracy(y_pred, y_true)
        
        # Verify integration
        assert isinstance(acc, float)
        assert acc == 1.0  # All predictions correct
    
    def test_accuracy_with_network_outputs(self):
        """Test Accuracy metric works with real network outputs."""
        # Create real network
        network = Sequential([
            Dense(4, 8),
            ReLU(),
            Dense(8, 3),
            Softmax()
        ])
        
        # Generate predictions
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        predictions = network(x)
        
        # Test accuracy
        accuracy = Accuracy()
        y_true = Tensor([0, 1])
        acc = accuracy(predictions, y_true)
        
        # Verify integration
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert predictions.shape == (2, 3)  # Batch size 2, 3 classes


class TestTrainerIntegration:
    """Test Trainer with real TinyTorch components."""
    
    def test_trainer_with_real_components(self):
        """Test Trainer works with real network, optimizer, and loss."""
        # Create real components
        model = Sequential([
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
            Sigmoid()
        ])
        
        # Create real optimizer (simplified for testing)
        optimizer = SGD([], learning_rate=0.01)
        
        # Create real loss and metrics
        loss_fn = MeanSquaredError()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Verify integration
        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.loss_function is loss_fn
        assert len(trainer.metrics) == 1
        assert isinstance(trainer.metrics[0], Accuracy)
    
    def test_trainer_history_tracking(self):
        """Test Trainer properly tracks training history."""
        # Create real components
        model = Sequential([Dense(2, 1)])
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = MeanSquaredError()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Verify history structure
        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'train_accuracy' in trainer.history
        assert 'val_accuracy' in trainer.history
        assert 'epoch' in trainer.history
        
        # All should be empty initially
        assert len(trainer.history['train_loss']) == 0
        assert len(trainer.history['epoch']) == 0
    
    def test_trainer_with_mlp(self):
        """Test Trainer works with MLP networks."""
        # Create real MLP
        mlp = create_mlp(
            input_size=10,
            hidden_sizes=[8, 4],
            output_size=3,
            activation=ReLU,
            output_activation=Softmax
        )
        
        # Create trainer components
        optimizer = Adam([], learning_rate=0.001)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(mlp, optimizer, loss_fn, metrics)
        
        # Verify integration
        assert trainer.model is mlp
        assert isinstance(trainer.loss_function, CrossEntropyLoss)
        assert len(trainer.metrics) == 1


class TestTrainingPipelineIntegration:
    """Test complete training pipeline integration."""
    
    def test_classification_pipeline(self):
        """Test complete classification training pipeline."""
        # Create synthetic dataset using the real SimpleDataset from dataloader
        # SimpleDataset(size=100, num_features=4, num_classes=3) by default
        
        # Create real components
        model = Sequential([
            Dense(4, 8),  # 4 features input (SimpleDataset default)
            ReLU(),
            Dense(8, 3),  # 3 classes output (SimpleDataset default)
            Softmax()
        ])
        
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Test epoch training
        dataset = SimpleDataset()
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        train_metrics = trainer.train_epoch(dataloader)
        
        # Verify integration
        assert 'loss' in train_metrics
        assert 'accuracy' in train_metrics
        assert train_metrics['loss'] >= 0.0
        assert 0.0 <= train_metrics['accuracy'] <= 1.0
    
    def test_regression_pipeline(self):
        """Test complete regression training pipeline."""
        # Create synthetic dataset for regression
        dataset = SimpleDataset(size=50, num_features=4, num_classes=1)  # 1 class for regression
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create real components
        model = Sequential([
            Dense(4, 8),  # 4 features input
            ReLU(),
            Dense(8, 1)  # Single output for regression
        ])
        
        optimizer = Adam([], learning_rate=0.001)
        loss_fn = MeanSquaredError()
        metrics = []  # No accuracy for regression
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Test epoch training
        train_metrics = trainer.train_epoch(dataloader)
        
        # Verify integration
        assert 'loss' in train_metrics
        assert train_metrics['loss'] >= 0.0
        assert len(train_metrics) == 1  # Only loss, no accuracy
    
    def test_validation_integration(self):
        """Test validation works with real components."""
        # Create synthetic dataset for validation
        dataset = SimpleDataset(size=30, num_features=4, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # No shuffling for validation
        
        # Create real components
        model = Sequential([
            Dense(4, 8),  # 4 features input
            ReLU(),
            Dense(8, 3),  # 3 classes output
            Softmax()
        ])
        
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Test validation
        val_metrics = trainer.validate_epoch(dataloader)
        
        # Verify integration
        assert 'loss' in val_metrics
        assert 'accuracy' in val_metrics
        assert val_metrics['loss'] >= 0.0
        assert 0.0 <= val_metrics['accuracy'] <= 1.0


class TestMultiModuleIntegration:
    """Test training with multiple TinyTorch modules."""
    
    def test_training_with_cnn_components(self):
        """Test training pipeline works with CNN components."""
        try:
            # Try to import CNN components
            from tinytorch.core.cnn import Conv2D, flatten
        except ImportError:
            pytest.skip("CNN components not available")
        
        # Create CNN model
        model = Sequential([
            Conv2D((3, 3)),
            ReLU(),
            flatten,
            Dense(16, 10),
            Softmax()
        ])
        
        # Create training components
        optimizer = Adam([], learning_rate=0.001)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Verify CNN integration
        assert trainer.model is model
        assert len(trainer.model.layers) == 5  # Conv2D, ReLU, flatten, Dense, Softmax
    
    def test_training_with_autograd_integration(self):
        """Test training pipeline works with autograd components."""
        # Create model with Variables (autograd)
        model = Sequential([
            Dense(3, 4),
            ReLU(),
            Dense(4, 2)
        ])
        
        # Create optimizer with real parameters
        optimizer = SGD([], learning_rate=0.01)
        
        # Create loss and metrics
        loss_fn = MeanSquaredError()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Verify autograd integration
        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert hasattr(trainer.optimizer, 'zero_grad')
        assert hasattr(trainer.optimizer, 'step')
    
    def test_end_to_end_training_integration(self):
        """Test complete end-to-end training with all components."""
        # Create synthetic dataset
        class EndToEndDataset:
            def __init__(self):
                # Create more realistic dataset
                np.random.seed(42)  # For reproducibility
                self.data = []
                for i in range(10):
                    x = Tensor(np.random.randn(4).tolist())
                    y = Tensor([i % 2])  # Binary classification
                    self.data.append((x, y))
            
            def __iter__(self):
                return iter(self.data)
        
        # Create complete pipeline
        model = create_mlp(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=2,
            activation=ReLU,
            output_activation=Softmax
        )
        
        optimizer = Adam([], learning_rate=0.001)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        
        # Create trainer
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Test complete training
        dataset = EndToEndDataset()
        train_metrics = trainer.train_epoch(dataset)
        val_metrics = trainer.validate_epoch(dataset)
        
        # Verify end-to-end integration
        assert 'loss' in train_metrics
        assert 'accuracy' in train_metrics
        assert 'loss' in val_metrics
        assert 'accuracy' in val_metrics
        
        # Verify reasonable values
        assert train_metrics['loss'] >= 0.0
        assert 0.0 <= train_metrics['accuracy'] <= 1.0
        assert val_metrics['loss'] >= 0.0
        assert 0.0 <= val_metrics['accuracy'] <= 1.0


class TestErrorHandlingIntegration:
    """Test error handling in training integration."""
    
    def test_incompatible_loss_and_model(self):
        """Test error handling for incompatible loss and model."""
        # Create model with wrong output size
        model = Sequential([Dense(2, 5)])  # 5 outputs
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn, [])
        
        # This should work (trainer creation)
        assert trainer.model is model
        
        # Error would come during actual training with wrong-sized targets
        # But trainer creation should succeed
    
    def test_metric_compatibility(self):
        """Test metrics work with different model outputs."""
        # Create binary classification model
        model = Sequential([
            Dense(2, 1),
            Sigmoid()
        ])
        
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = BinaryCrossEntropyLoss()
        metrics = [Accuracy()]
        
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Verify binary classification setup
        assert trainer.model is model
        assert isinstance(trainer.loss_function, BinaryCrossEntropyLoss)
        assert len(trainer.metrics) == 1
        assert isinstance(trainer.metrics[0], Accuracy) 
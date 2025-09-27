#!/usr/bin/env python3
"""
Training Validation for Enhanced TinyTorch Framework
====================================================

Comprehensive validation that TinyTorch can successfully train neural networks
and demonstrate clear learning signals (loss decreasing, accuracy improving).

This script validates:
1. Simple MLP training on XOR problem with SGD and Adam
2. CNN training on synthetic image classification  
3. Complete training pipeline with gradient flow validation
4. Enhanced feature integration during training
5. Integration under load with larger models
6. Memory management and performance characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import time
import tracemalloc

# Import TinyTorch modules
import tinytorch.core.tensor as T
import tinytorch.core.layers as layers
import tinytorch.core.activations as activations
import tinytorch.core.optimizers as optimizers
import tinytorch.core.training as training
from tinytorch.core.networks import Sequential

# Set random seed for reproducibility
np.random.seed(42)

class ValidationSuite:
    """Comprehensive training validation suite for TinyTorch."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
        
    def log_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Log validation test results."""
        self.results[test_name] = {
            'success': success,
            'details': details or {}
        }
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if details:
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def measure_memory(self, operation_name: str):
        """Context manager for measuring memory usage."""
        class MemoryMeasurer:
            def __init__(self, suite, name):
                self.suite = suite
                self.name = name
                
            def __enter__(self):
                tracemalloc.start()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.suite.memory_usage[self.name] = {
                    'current_mb': current / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024
                }
                
        return MemoryMeasurer(self, operation_name)

def create_xor_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create XOR dataset for MLP training validation."""
    # Generate random inputs
    X = np.random.randint(0, 2, (n_samples, 2)).astype(np.float32)
    # XOR logic: output is 1 if inputs are different
    y = (X[:, 0] != X[:, 1]).astype(np.float32).reshape(-1, 1)
    return X, y

def create_synthetic_images(n_samples: int = 1000, size: int = 8, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic image classification dataset."""
    # Create simple patterns for each class
    images = []
    labels = []
    
    for i in range(n_samples):
        class_id = i % n_classes
        
        # Create base noise
        img = np.random.randn(1, size, size) * 0.1
        
        if class_id == 0:  # Horizontal stripes
            for j in range(0, size, 2):
                img[0, j, :] += 1.0
        elif class_id == 1:  # Vertical stripes  
            for j in range(0, size, 2):
                img[0, :, j] += 1.0
        else:  # Diagonal pattern
            for j in range(size):
                img[0, j, j] += 1.0
                if j < size - 1:
                    img[0, j, j + 1] += 0.5
                    
        images.append(img)
        labels.append(class_id)
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y

def validate_mlp_training(suite: ValidationSuite):
    """Validate MLP training on XOR problem with both SGD and Adam."""
    print("\n🔬 Validating MLP Training on XOR Problem...")
    
    # Create XOR dataset
    X_train, y_train = create_xor_dataset(800)
    X_test, y_test = create_xor_dataset(200)
    
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Test with SGD
    print("\n📊 Testing with SGD Optimizer...")
    with suite.measure_memory("mlp_sgd_training"):
        mlp_sgd = Sequential([
            layers.Linear(2, 8),
            activations.ReLU(),
            layers.Linear(8, 4), 
            activations.ReLU(),
            layers.Linear(4, 1),
            activations.Sigmoid()
        ])
        
        optimizer_sgd = optimizers.SGD(mlp_sgd.parameters(), learning_rate=1.0)
        
        # Training loop
        losses_sgd = []
        for epoch in range(20):
            epoch_loss = 0.0
            for i in range(0, len(X_train), 32):  # Mini-batches
                batch_X = X_train[i:i+32]
                batch_y = y_train[i:i+32]
                
                # Forward pass
                predictions = mlp_sgd.forward(T.Tensor(batch_X))
                mse_loss = training.MeanSquaredError()
                loss = mse_loss(predictions, T.Tensor(batch_y))
                
                # Backward pass
                optimizer_sgd.zero_grad()
                loss.backward()
                optimizer_sgd.step()
                
                if hasattr(loss.data, 'data'):
                    epoch_loss += loss.data.data
                else:
                    epoch_loss += loss.data
                
            avg_loss = epoch_loss / (len(X_train) // 32)
            losses_sgd.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Check learning signal
    initial_loss_sgd = losses_sgd[0]
    final_loss_sgd = losses_sgd[-1]
    learning_signal_sgd = (initial_loss_sgd - final_loss_sgd) / initial_loss_sgd
    
    suite.log_result("MLP_SGD_Training", learning_signal_sgd > 0.3, {
        "initial_loss": f"{initial_loss_sgd:.4f}",
        "final_loss": f"{final_loss_sgd:.4f}", 
        "improvement": f"{learning_signal_sgd:.1%}"
    })
    
    # Test with Adam
    print("\n📊 Testing with Adam Optimizer...")
    with suite.measure_memory("mlp_adam_training"):
        mlp_adam = Sequential([
            layers.Linear(2, 8),
            activations.ReLU(),
            layers.Linear(8, 4),
            activations.ReLU(), 
            layers.Linear(4, 1),
            activations.Sigmoid()
        ])
        
        optimizer_adam = optimizers.Adam(mlp_adam.parameters(), learning_rate=0.01)
        
        # Training loop
        losses_adam = []
        for epoch in range(15):  # Adam typically converges faster
            epoch_loss = 0.0
            for i in range(0, len(X_train), 32):
                batch_X = X_train[i:i+32]
                batch_y = y_train[i:i+32]
                
                predictions = mlp_adam.forward(T.Tensor(batch_X))
                mse_loss = training.MeanSquaredError()
                loss = mse_loss(predictions, T.Tensor(batch_y))
                
                optimizer_adam.zero_grad()
                loss.backward()
                optimizer_adam.step()
                
                if hasattr(loss.data, 'data'):
                    epoch_loss += loss.data.data
                else:
                    epoch_loss += loss.data
                
            avg_loss = epoch_loss / (len(X_train) // 32)
            losses_adam.append(avg_loss)
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Check Adam learning signal
    initial_loss_adam = losses_adam[0]
    final_loss_adam = losses_adam[-1]
    learning_signal_adam = (initial_loss_adam - final_loss_adam) / initial_loss_adam
    
    suite.log_result("MLP_Adam_Training", learning_signal_adam > 0.5, {
        "initial_loss": f"{initial_loss_adam:.4f}",
        "final_loss": f"{final_loss_adam:.4f}",
        "improvement": f"{learning_signal_adam:.1%}"
    })
    
    # Test accuracy on validation set
    test_predictions_sgd = mlp_sgd.forward(T.Tensor(X_test))
    test_predictions_adam = mlp_adam.forward(T.Tensor(X_test))
    
    # Convert to binary predictions
    if hasattr(test_predictions_sgd.data, 'data'):
        sgd_data = test_predictions_sgd.data.data
        adam_data = test_predictions_adam.data.data
    else:
        sgd_data = test_predictions_sgd.data
        adam_data = test_predictions_adam.data
        
    binary_preds_sgd = (sgd_data > 0.5).astype(int)
    binary_preds_adam = (adam_data > 0.5).astype(int)
    
    accuracy_sgd = np.mean(binary_preds_sgd.flatten() == y_test.flatten())
    accuracy_adam = np.mean(binary_preds_adam.flatten() == y_test.flatten())
    
    suite.log_result("MLP_Accuracy_Validation", accuracy_sgd > 0.7 and accuracy_adam > 0.8, {
        "SGD_accuracy": f"{accuracy_sgd:.1%}",
        "Adam_accuracy": f"{accuracy_adam:.1%}"
    })

def validate_cnn_training(suite: ValidationSuite):
    """Validate CNN training on synthetic image classification."""
    print("\n🔬 Validating CNN Training on Synthetic Images...")
    
    # Create synthetic image dataset
    X_train, y_train = create_synthetic_images(600, size=8, n_classes=3)
    X_test, y_test = create_synthetic_images(150, size=8, n_classes=3)
    
    print(f"Dataset: {X_train.shape[0]} training images ({X_train.shape})")
    print(f"Classes: {np.unique(y_train)} ({len(np.unique(y_train))} total)")
    
    with suite.measure_memory("cnn_training"):
        # Build CNN: Conv2d → ReLU → MaxPool → Flatten → Linear
        from tinytorch.core.spatial import Conv2d, MaxPool2D, flatten
        conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
        pool1 = MaxPool2D(pool_size=(2, 2))
        conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2)) 
        
        # Calculate linear layer input size:
        # Input: 8x8 → Conv(3x3): 6x6 → Pool(2x2): 3x3 → Conv(2x2): 2x2 → 16 channels
        # Final: 16 channels * 2 * 2 = 64 features
        linear = layers.Linear(64, 3)  # 3 classes
        
        optimizer = optimizers.Adam([
            *conv1.parameters(),
            *conv2.parameters(), 
            *linear.parameters()
        ], learning_rate=0.01)
        
        # Training loop
        losses = []
        accuracies = []
        
        for epoch in range(8):  # Small number for validation
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(0, len(X_train), 16):  # Small batches
                batch_X = X_train[i:i+16]
                batch_y = y_train[i:i+16]
                
                # Forward pass through CNN
                x = T.Tensor(batch_X)
                
                # Conv → ReLU → MaxPool
                x = conv1.forward(x)
                relu = activations.ReLU()
                x = relu.forward(x)
                x = pool1.forward(x)
                
                # Conv → ReLU  
                x = conv2.forward(x)
                x = relu.forward(x)
                
                # Flatten and classify
                x = flatten(x)
                logits = linear.forward(x)
                
                # Compute loss (simplified cross-entropy)
                targets_one_hot = np.eye(3)[batch_y]
                mse_loss = training.MeanSquaredError()
                loss = mse_loss(logits, T.Tensor(targets_one_hot))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if hasattr(loss.data, 'data'):
                    epoch_loss += loss.data.data
                else:
                    epoch_loss += loss.data
                
                # Compute accuracy
                predictions = np.argmax(logits.data, axis=1)
                correct_predictions += np.sum(predictions == batch_y)
                total_predictions += len(batch_y)
            
            avg_loss = epoch_loss / (len(X_train) // 16)
            accuracy = correct_predictions / total_predictions
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1%}")
    
    # Check learning signals
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_improvement = (initial_loss - final_loss) / initial_loss
    
    initial_accuracy = accuracies[0] 
    final_accuracy = accuracies[-1]
    accuracy_improvement = final_accuracy - initial_accuracy
    
    suite.log_result("CNN_Training", loss_improvement > 0.2 and final_accuracy > 0.8, {
        "loss_improvement": f"{loss_improvement:.1%}",
        "accuracy_improvement": f"{accuracy_improvement:.1%}",
        "final_accuracy": f"{final_accuracy:.1%}"
    })

def validate_training_pipeline(suite: ValidationSuite):
    """Validate complete training pipeline with gradient flow."""
    print("\n🔬 Validating Training Pipeline Components...")
    
    # Test gradient flow validation
    print("Testing gradient flow...")
    
    # Create simple model to test gradients
    model = Sequential([
        layers.Linear(3, 5),
        activations.ReLU(),
        layers.Linear(5, 1)
    ])
    
    # Test input
    x = T.Tensor(np.random.randn(4, 3))
    target = T.Tensor(np.random.randn(4, 1))
    
    # Forward pass
    output = model.forward(x)
    mse_loss = training.MeanSquaredError()
    loss = mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are non-zero
    gradients_exist = True
    gradients_nonzero = True
    
    for param in model.parameters():
        if param.grad is None:
            gradients_exist = False
        elif np.allclose(param.grad, 0):
            gradients_nonzero = False
    
    suite.log_result("Gradient_Flow", gradients_exist and gradients_nonzero, {
        "gradients_computed": gradients_exist,
        "gradients_nonzero": gradients_nonzero
    })
    
    # Test optimizer state updates
    print("Testing optimizer state updates...")
    
    optimizer = optimizers.Adam(model.parameters(), learning_rate=0.01)
    
    # Get initial parameter values
    initial_params = [param.data.copy() for param in model.parameters()]
    
    # Run optimization step
    optimizer.step()
    
    # Check parameter updates
    params_updated = any(
        not np.allclose(initial, param.data) 
        for initial, param in zip(initial_params, model.parameters())
    )
    
    suite.log_result("Optimizer_Updates", params_updated, {
        "parameters_changed": params_updated
    })
    
    # Test loss function behavior
    print("Testing loss functions...")
    
    # Test that loss decreases with better predictions
    good_pred = T.Tensor(target.data + np.random.randn(*target.shape) * 0.1)
    bad_pred = T.Tensor(target.data + np.random.randn(*target.shape) * 2.0)
    
    mse_loss = training.MeanSquaredError()
    good_loss = mse_loss(good_pred, target)
    bad_loss = mse_loss(bad_pred, target)
    
    # Extract scalar values for comparison
    if hasattr(good_loss.data, 'data'):
        good_loss_val = good_loss.data.data
        bad_loss_val = bad_loss.data.data
    else:
        good_loss_val = good_loss.data
        bad_loss_val = bad_loss.data
        
    loss_behavior_correct = bad_loss_val > good_loss_val
    
    suite.log_result("Loss_Function_Behavior", loss_behavior_correct, {
        "good_prediction_loss": f"{good_loss_val:.4f}",
        "bad_prediction_loss": f"{bad_loss_val:.4f}",
        "behaves_correctly": loss_behavior_correct
    })

def validate_enhanced_features(suite: ValidationSuite):
    """Test enhanced feature integration during training."""
    print("\n🔬 Validating Enhanced Features Integration...")
    
    # Test systems insights during training
    print("Testing systems insights...")
    
    # Create a model and test memory profiling
    model = Sequential([
        layers.Linear(100, 200),
        activations.ReLU(),
        layers.Linear(200, 100),
        activations.ReLU(),
        layers.Linear(100, 10)
    ])
    
    # Test that we can profile memory during forward pass
    with suite.measure_memory("enhanced_features_test"):
        x = T.Tensor(np.random.randn(64, 100))  # Larger batch for memory test
        output = model.forward(x)
        mse_loss = training.MeanSquaredError()
        loss = mse_loss(output, T.Tensor(np.random.randn(64, 10)))
        loss.backward()
    
    # Check systems insights work
    param_count = sum(np.prod(param.data.shape) for param in model.parameters())
    memory_per_param = 4  # bytes for float32
    estimated_memory_mb = (param_count * memory_per_param) / (1024 * 1024)
    
    suite.log_result("Systems_Insights", True, {
        "parameter_count": param_count,
        "estimated_memory_mb": f"{estimated_memory_mb:.2f}",
        "memory_profiling_works": "enhanced_features_test" in suite.memory_usage
    })

def validate_integration_under_load(suite: ValidationSuite):
    """Test integration with larger models to ensure scaling works."""
    print("\n🔬 Validating Integration Under Load...")
    
    # Create larger model to test scaling
    print("Testing larger model performance...")
    
    with suite.measure_memory("large_model_test"):
        # Larger MLP
        large_model = Sequential([
            layers.Linear(512, 1024),
            activations.ReLU(),
            layers.Linear(1024, 512),
            activations.ReLU(),
            layers.Linear(512, 256),
            activations.ReLU(),
            layers.Linear(256, 10)
        ])
        
        optimizer = optimizers.Adam(large_model.parameters(), learning_rate=0.001)
        
        # Test with larger batches
        batch_size = 128
        x = T.Tensor(np.random.randn(batch_size, 512))
        target = T.Tensor(np.random.randn(batch_size, 10))
        
        start_time = time.time()
        
        # Run a few training steps
        total_loss = 0.0
        mse_loss = training.MeanSquaredError()
        for step in range(5):
            output = large_model.forward(x)
            loss = mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if hasattr(loss.data, 'data'):
                total_loss += loss.data.data
            else:
                total_loss += loss.data
        
        end_time = time.time()
        training_time = end_time - start_time
        
    # Check performance is reasonable
    avg_loss = total_loss / 5
    performance_ok = training_time < 10.0  # Should complete in under 10 seconds
    
    suite.log_result("Large_Model_Performance", performance_ok, {
        "training_time_seconds": f"{training_time:.2f}",
        "average_loss": f"{avg_loss:.4f}",
        "performance_acceptable": performance_ok
    })
    
    # Test for memory leaks (simple check)
    print("Testing for memory consistency...")
    
    memory_before = suite.memory_usage.get("large_model_test", {}).get("peak_mb", 0)
    
    # Run another round
    with suite.measure_memory("memory_leak_test"):
        mse_loss = training.MeanSquaredError()
        for step in range(3):
            output = large_model.forward(x)
            loss = mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    memory_after = suite.memory_usage.get("memory_leak_test", {}).get("peak_mb", 0)
    
    # Memory should not grow significantly (allowing for some variation)
    memory_growth = memory_after - memory_before
    no_major_leak = memory_growth < memory_before * 0.5  # Less than 50% growth
    
    suite.log_result("Memory_Consistency", no_major_leak, {
        "memory_before_mb": f"{memory_before:.2f}",
        "memory_after_mb": f"{memory_after:.2f}",
        "memory_growth_mb": f"{memory_growth:.2f}",
        "no_major_leak": no_major_leak
    })

def print_validation_report(suite: ValidationSuite):
    """Print comprehensive validation report."""
    print("\n" + "="*60)
    print("🎯 TINYTORCH TRAINING VALIDATION REPORT")
    print("="*60)
    
    total_tests = len(suite.results)
    passed_tests = sum(1 for result in suite.results.values() if result['success'])
    
    print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in suite.results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"\n{status} {test_name}")
        if result['details']:
            for key, value in result['details'].items():
                print(f"    {key}: {value}")
    
    # Memory usage summary
    if suite.memory_usage:
        print(f"\n💾 MEMORY USAGE SUMMARY:")
        for operation, memory in suite.memory_usage.items():
            print(f"  {operation}:")
            print(f"    Peak: {memory['peak_mb']:.2f} MB")
            print(f"    Current: {memory['current_mb']:.2f} MB")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if passed_tests == total_tests:
        print("✅ EXCELLENT: All training validations passed!")
        print("   TinyTorch framework is ready for educational use.")
        print("   Students can successfully train neural networks and see learning signals.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  GOOD: Most validations passed with minor issues.")
        print("   Framework is functional but may need some refinements.")
    else:
        print("❌ NEEDS WORK: Multiple validation failures detected.")
        print("   Framework requires fixes before educational deployment.")
    
    print("\n" + "="*60)

def main():
    """Run complete training validation suite."""
    print("🚀 Starting TinyTorch Training Validation Suite...")
    print("   This validates that the enhanced framework can train neural networks")
    print("   and demonstrate clear learning signals (loss decreasing, accuracy improving).")
    
    suite = ValidationSuite()
    
    try:
        # Run all validation tests
        validate_mlp_training(suite)
        validate_cnn_training(suite)
        validate_training_pipeline(suite)
        validate_enhanced_features(suite)
        validate_integration_under_load(suite)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during validation: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        suite.log_result("Critical_Error", False, {"error": str(e)})
    
    finally:
        # Always print report
        print_validation_report(suite)

if __name__ == "__main__":
    main()
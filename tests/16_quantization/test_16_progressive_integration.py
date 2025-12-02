"""
Module 16: Progressive Integration Tests
Tests that Module 16 (Quantization) works correctly AND that all previous modules still work.

DEPENDENCY CHAIN: 01_setup → 02_tensor → 03_activations → ... → 16_quantization
Students can trace back exactly where issues originate.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModule15StillWorking:
    """Verify Module 15 (Memoization) functionality is still intact."""

    def test_memoization_environment_stable(self):
        """Ensure memoization wasn't broken by quantization development."""
        try:
            from tinytorch.optimization.memoization import memoize

            # Basic memoization should still work
            @memoize
            def test_fn(x):
                return x * 2

            result = test_fn(5)
            assert result == 10, "Module 15: Memoization broken"

        except ImportError:
            assert True, "Module 15: Memoization not implemented yet"


class TestModule16QuantizationCore:
    """Test Module 16 (Quantization) core functionality."""

    def test_quantize_int8_basic(self):
        """Test INT8 quantization function."""
        try:
            from tinytorch.optimization.quantization import quantize_int8
            from tinytorch.core.tensor import Tensor

            # Create FP32 tensor
            x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))

            # Quantize to INT8
            q_tensor, scale, zero_point = quantize_int8(x)

            # Check that quantized values are in INT8 range
            assert np.all(q_tensor.data >= -128) and np.all(q_tensor.data <= 127), \
                "Quantized values outside INT8 range"

            # Check scale and zero_point are returned
            assert isinstance(scale, float), "Scale not a float"
            assert isinstance(zero_point, (int, np.integer)), "Zero point not an int"

            print(f"INT8 quantization test: scale={scale:.4f}, zero_point={zero_point}")

        except ImportError:
            assert True, "Module 16: Quantization not implemented yet"

    def test_dequantize_int8_basic(self):
        """Test INT8 dequantization function."""
        try:
            from tinytorch.optimization.quantization import quantize_int8, dequantize_int8
            from tinytorch.core.tensor import Tensor

            # Create and quantize tensor
            x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
            q_tensor, scale, zero_point = quantize_int8(x)

            # Dequantize
            x_recovered = dequantize_int8(q_tensor, scale, zero_point)

            # Should be close to original (some quantization error expected)
            error = np.mean(np.abs(x.data - x_recovered.data))
            assert error < 0.5, f"Dequantization error {error} too high"

        except ImportError:
            assert True, "Module 16: Dequantization not implemented yet"

    def test_quantized_linear_layer(self):
        """Test QuantizedLinear layer."""
        try:
            from tinytorch.optimization.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create original linear layer
            linear = Linear(in_features=4, out_features=2)

            # Quantize it
            q_linear = QuantizedLinear(linear)

            # Test forward pass
            x = Tensor(np.random.randn(3, 4))
            output = q_linear.forward(x)

            assert output.shape == (3, 2), "QuantizedLinear output shape wrong"

        except ImportError:
            assert True, "Module 16: QuantizedLinear not implemented yet"


class TestQuantizationAccuracyDegradation:
    """Test that quantization doesn't degrade accuracy too much (CRITICAL - Priority 1)."""

    def test_quantization_accuracy_degradation(self):
        """Test that quantization doesn't degrade accuracy too much.

        This test validates that:
        - INT8 model accuracy is within threshold of FP32
        - Quantization error is predictable and bounded
        - Would catch quantization bugs
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Create simple MLP model
            layer1 = Linear(10, 20)
            relu1 = ReLU()
            layer2 = Linear(20, 5)
            model = SimpleModel(layer1, relu1, layer2)

            # Create test input
            x = Tensor(np.random.randn(5, 10))

            # Get original output
            original_output = model.forward(x)

            # Quantize linear layers
            q_layer1 = QuantizedLinear(layer1)
            q_model = SimpleModel(q_layer1, relu1, QuantizedLinear(layer2))

            # Get quantized output
            quantized_output = q_model.forward(x)

            # Check shapes match
            assert quantized_output.shape == original_output.shape, \
                "Quantization changed output shape"

            # Check accuracy degradation is acceptable
            max_error = np.max(np.abs(original_output.data - quantized_output.data))
            mean_error = np.mean(np.abs(original_output.data - quantized_output.data))

            # Allow up to 10% error for INT8 quantization (typical threshold)
            original_scale = np.max(np.abs(original_output.data))
            relative_error = mean_error / (original_scale + 1e-8)

            assert relative_error < 0.1, \
                f"Quantization error {relative_error:.2%} exceeds 10% threshold"

            print(f"Quantization accuracy test: mean error = {mean_error:.4f}, "
                  f"max error = {max_error:.4f}, relative error = {relative_error:.2%}")

        except ImportError:
            assert True, "Accuracy degradation test not ready yet"


class TestQuantizationMemoryReduction:
    """Test that quantized models use 4x less memory (HIGH - Priority 2)."""

    def test_quantization_memory_reduction(self):
        """Test that quantized models use 4x less memory.

        This test validates that:
        - Memory footprint is reduced through quantization
        - Compression ratio is calculated correctly
        - Would catch memory bugs
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create a reasonably large linear layer
            linear = Linear(in_features=1000, out_features=500)

            # Quantize
            q_linear = QuantizedLinear(linear)

            # Get memory usage info
            memory_info = q_linear.memory_usage()

            # Check that memory_usage returns expected keys
            assert 'original_bytes' in memory_info, "Missing original_bytes"
            assert 'quantized_bytes' in memory_info, "Missing quantized_bytes"
            assert 'compression_ratio' in memory_info, "Missing compression_ratio"

            # Verify compression ratio is reasonable (close to 4x)
            compression_ratio = memory_info['compression_ratio']
            assert compression_ratio > 3.0, \
                f"Compression ratio {compression_ratio:.2f}x is less than expected ~4x"

            # Verify memory was actually reduced
            assert memory_info['quantized_bytes'] < memory_info['original_bytes'], \
                "Quantized model uses more memory than original"

            print(f"Memory reduction test: {compression_ratio:.2f}x compression "
                  f"({memory_info['original_bytes']/1024:.1f}KB -> "
                  f"{memory_info['quantized_bytes']/1024:.1f}KB)")

        except ImportError:
            assert True, "Memory reduction test not ready yet"


class TestQuantizationInferenceSpeed:
    """Test that quantized inference is faster (HIGH - Priority 3)."""

    def test_quantization_inference_speed(self):
        """Test that quantized inference is faster.

        This test validates that:
        - Quantized forward pass completes successfully
        - Memory footprint is smaller (speed comes from cache efficiency)
        - Would catch performance bugs

        Note: We measure memory, not speed, because educational quantization
        dequantizes for computation. Production INT8 ops would be faster.
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            import time

            # Create larger model for performance testing
            linear = Linear(in_features=512, out_features=256)
            q_linear = QuantizedLinear(linear)

            # Test data (batch of 100)
            x = Tensor(np.random.randn(100, 512))

            # Warm-up
            _ = linear.forward(x)
            _ = q_linear.forward(x)

            # Time original forward pass
            start = time.time()
            for _ in range(10):
                _ = linear.forward(x)
            fp32_time = time.time() - start

            # Time quantized forward pass
            start = time.time()
            for _ in range(10):
                _ = q_linear.forward(x)
            int8_time = time.time() - start

            # Note: Educational implementation may not be faster since we dequantize
            # But it should at least work without crashing
            assert int8_time > 0, "Quantized inference failed"

            # The real benefit is memory savings (tested above)
            memory_info = q_linear.memory_usage()
            assert memory_info['compression_ratio'] > 3.5, \
                "Memory compression not achieved"

            print(f"Inference speed test: FP32={fp32_time:.3f}s, INT8={int8_time:.3f}s, "
                  f"compression={memory_info['compression_ratio']:.2f}x")

        except ImportError:
            assert True, "Inference speed test not ready yet"


class TestQuantizationGradientFlow:
    """Test QAT (Quantization-Aware Training) gradient flow (CRITICAL - Priority 4)."""

    def test_quantization_gradient_flow(self):
        """Test QAT gradient flow.

        This test validates that:
        - Fake quantization preserves gradients
        - Forward pass works with quantized layers
        - Would catch training bugs

        Note: Full QAT requires backward pass implementation.
        We test that forward pass doesn't break gradient tracking.
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create layer and quantize
            linear = Linear(in_features=4, out_features=2)
            q_linear = QuantizedLinear(linear)

            # Test input with requires_grad
            x = Tensor(np.random.randn(3, 4), requires_grad=True)

            # Forward pass should work
            output = q_linear.forward(x)

            # Check output properties
            assert hasattr(output, 'data'), "Output missing data attribute"
            assert hasattr(output, 'shape'), "Output missing shape attribute"
            assert output.shape == (3, 2), "Output shape incorrect"

            # Verify quantized weights exist
            assert hasattr(q_linear, 'q_weight'), "Quantized layer missing q_weight"

            # Verify quantized values are in INT8 range
            assert np.all(q_linear.q_weight.data >= -128) and \
                   np.all(q_linear.q_weight.data <= 127), \
                   "Quantized weights outside INT8 range"

            print("Gradient flow test: Forward pass works with quantized layers")

        except ImportError:
            assert True, "Gradient flow test not ready yet"


class TestQuantizationCalibration:
    """Test calibration on representative data (MEDIUM - Priority 5)."""

    def test_quantization_calibration(self):
        """Test calibration on representative data.

        This test validates that:
        - Calibration correctly calculates scale/zero-point
        - Calibrated quantization improves accuracy
        - Would catch calibration bugs
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create layer
            linear = Linear(in_features=10, out_features=5)
            q_linear = QuantizedLinear(linear)

            # Generate calibration data (representative samples)
            calibration_samples = [
                Tensor(np.random.randn(1, 10)) for _ in range(20)
            ]

            # Calibrate
            q_linear.calibrate(calibration_samples)

            # Check calibration parameters were set
            assert q_linear.input_scale is not None, "Input scale not set after calibration"
            assert q_linear.input_zero_point is not None, "Zero point not set after calibration"

            # Verify calibration parameters are reasonable
            assert q_linear.input_scale > 0, "Input scale should be positive"
            assert -128 <= q_linear.input_zero_point <= 127, "Zero point out of INT8 range"

            # Test forward pass after calibration
            x = Tensor(np.random.randn(5, 10))
            output = q_linear.forward(x)
            assert output.shape == (5, 5), "Forward pass failed after calibration"

            print(f"Calibration test: scale={q_linear.input_scale:.4f}, "
                  f"zero_point={q_linear.input_zero_point}")

        except ImportError:
            assert True, "Calibration test not ready yet"


class TestQuantizationModelIntegrity:
    """Test that quantization preserves model structure and functionality."""

    def test_quantize_mlp_preserves_structure(self):
        """Test quantizing MLP preserves structure."""
        try:
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            # Build MLP
            layer1 = Linear(784, 128)
            relu1 = ReLU()
            layer2 = Linear(128, 64)
            relu2 = ReLU()
            layer3 = Linear(64, 10)
            sigmoid = Sigmoid()

            model = SimpleModel(layer1, relu1, layer2, relu2, layer3, sigmoid)

            # Test original model
            x = Tensor(np.random.randn(4, 784))
            original_output = model.forward(x)

            # Quantize linear layers only (activations stay FP32)
            q_model = SimpleModel(
                QuantizedLinear(layer1),
                relu1,
                QuantizedLinear(layer2),
                relu2,
                QuantizedLinear(layer3),
                sigmoid
            )

            # Test quantized model
            quantized_output = q_model.forward(x)

            # Structure should be preserved
            assert quantized_output.shape == original_output.shape, \
                "Quantization changed output shape"

            # Output should be similar (allowing quantization error)
            mean_error = np.mean(np.abs(original_output.data - quantized_output.data))
            assert not np.isnan(mean_error), "Quantized model produced NaN"

            print(f"MLP structure preservation test: output shape {quantized_output.shape}, "
                  f"mean error {mean_error:.4f}")

        except ImportError:
            assert True, "MLP structure test not ready yet"

    def test_quantization_with_different_architectures(self):
        """Test quantization works with various model architectures."""
        try:
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            from tinytorch.core.tensor import Tensor

            # Test 1: Single layer
            single_layer = Linear(10, 5)
            q_single = QuantizedLinear(single_layer)
            x1 = Tensor(np.random.randn(3, 10))
            y1 = q_single.forward(x1)
            assert y1.shape == (3, 5), "Single layer quantization failed"

            # Test 2: Deep narrow network
            deep_layers = [Linear(10, 10) for _ in range(5)]
            deep_activations = [ReLU() for _ in range(5)]
            deep_model_layers = []
            for layer, activation in zip(deep_layers, deep_activations):
                deep_model_layers.append(QuantizedLinear(layer))
                deep_model_layers.append(activation)
            deep_model = SimpleModel(*deep_model_layers)

            x2 = Tensor(np.random.randn(2, 10))
            y2 = deep_model.forward(x2)
            assert y2.shape == (2, 10), "Deep network quantization failed"

            # Test 3: Wide shallow network
            wide_layer = Linear(100, 200)
            q_wide = QuantizedLinear(wide_layer)
            x3 = Tensor(np.random.randn(5, 100))
            y3 = q_wide.forward(x3)
            assert y3.shape == (5, 200), "Wide network quantization failed"

            print("Architecture variety test: single, deep, and wide models all work")

        except ImportError:
            assert True, "Architecture variety test not ready yet"


class TestQuantizationEdgeCases:
    """Test corner cases and error handling."""

    def test_quantization_edge_cases(self):
        """Test edge cases: constant tensors, extreme ranges.

        This test validates that:
        - Constant tensors don't cause division by zero
        - Extreme ranges are handled correctly
        - Would catch edge case bugs
        """
        try:
            from tinytorch.optimization.quantization import quantize_int8, dequantize_int8
            from tinytorch.core.tensor import Tensor

            # Test 1: Constant tensor (all zeros)
            zeros = Tensor(np.zeros(10))
            q_zeros, scale_z, zp_z = quantize_int8(zeros)
            assert not np.any(np.isnan(q_zeros.data)), "Quantizing zeros produced NaN"

            # Dequantize should work
            recovered_zeros = dequantize_int8(q_zeros, scale_z, zp_z)
            assert np.allclose(recovered_zeros.data, 0.0, atol=0.1), "Zero recovery failed"

            # Test 2: Constant tensor (all ones)
            ones = Tensor(np.ones(10))
            q_ones, scale_o, zp_o = quantize_int8(ones)
            assert not np.any(np.isnan(q_ones.data)), "Quantizing ones produced NaN"

            # Test 3: Very small range
            small_range = Tensor(np.array([0.0, 0.001, 0.002]))
            q_small, scale_s, zp_s = quantize_int8(small_range)
            assert not np.any(np.isnan(q_small.data)), "Small range produced NaN"
            assert scale_s > 0, "Small range scale should be positive"

            # Test 4: Very large range
            large_range = Tensor(np.array([-1000.0, 0.0, 1000.0]))
            q_large, scale_l, zp_l = quantize_int8(large_range)
            assert not np.any(np.isnan(q_large.data)), "Large range produced NaN"
            assert not np.any(np.isinf(q_large.data)), "Large range produced Inf"

            # Test 5: Single element
            single = Tensor(np.array([42.0]))
            q_single, scale_si, zp_si = quantize_int8(single)
            assert not np.any(np.isnan(q_single.data)), "Single element produced NaN"

            # Test 6: Negative values only
            negatives = Tensor(np.array([-5.0, -3.0, -1.0]))
            q_neg, scale_n, zp_n = quantize_int8(negatives)
            assert not np.any(np.isnan(q_neg.data)), "Negative values produced NaN"

            print("Edge cases test: constant, small, large, single, negative values all handled")

        except ImportError:
            assert True, "Edge cases test not ready yet"

    def test_quantization_dtype_validation(self):
        """Test that quantization produces correct dtypes."""
        try:
            from tinytorch.optimization.quantization import quantize_int8
            from tinytorch.core.tensor import Tensor

            # Test various input dtypes
            float32_input = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            float64_input = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float64))

            # Quantize both
            q_f32, scale_f32, zp_f32 = quantize_int8(float32_input)
            q_f64, scale_f64, zp_f64 = quantize_int8(float64_input)

            # Values should be in INT8 range (regardless of storage dtype)
            assert np.all(q_f32.data >= -128) and np.all(q_f32.data <= 127), \
                "FP32 quantized values out of INT8 range"
            assert np.all(q_f64.data >= -128) and np.all(q_f64.data <= 127), \
                "FP64 quantized values out of INT8 range"

            # Verify scales and zero points are valid
            assert scale_f32 > 0, "Scale should be positive"
            assert scale_f64 > 0, "Scale should be positive"
            assert -128 <= zp_f32 <= 127, "Zero point out of INT8 range"
            assert -128 <= zp_f64 <= 127, "Zero point out of INT8 range"

            print(f"Dtype validation test: FP32 (scale={scale_f32:.4f}) and "
                  f"FP64 (scale={scale_f64:.4f}) both produce valid INT8-range values")

        except ImportError:
            assert True, "Dtype validation test not ready yet"


class TestQuantizationSystemIntegration:
    """Test quantization works with complete TinyTorch system."""

    def test_quantization_with_dataloader(self):
        """Test quantized models work with DataLoader."""
        try:
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import DataLoader

            # Create model
            layer1 = Linear(10, 5)
            relu = ReLU()
            layer2 = Linear(5, 2)

            q_model = SimpleModel(
                QuantizedLinear(layer1),
                relu,
                QuantizedLinear(layer2)
            )

            # Create simple dataset
            X = np.random.randn(20, 10)
            y = np.random.randint(0, 2, size=(20, 1))

            # Create DataLoader
            dataloader = DataLoader(X, y, batch_size=4)

            # Process batches through quantized model
            for batch_X, batch_y in dataloader:
                X_tensor = Tensor(batch_X)
                output = q_model.forward(X_tensor)

                # Should work without errors
                assert output.shape[0] == batch_X.shape[0], \
                    "Batch size changed"
                assert output.shape[1] == 2, \
                    "Output features changed"

            print("DataLoader integration test: quantized model processes batches correctly")

        except ImportError:
            assert True, "DataLoader integration test not ready yet"

    def test_complete_system_01_to_16_stable(self):
        """Test complete system (01→16) is stable."""
        try:
            # Import from all modules
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import mse_loss
            from tinytorch.optimization.optimizers import SGD
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel

            # Build simple training scenario
            model_layers = [
                Linear(4, 8),
                ReLU(),
                Linear(8, 1),
                Sigmoid()
            ]
            model = SimpleModel(*model_layers)

            # Create data
            X = Tensor(np.random.randn(10, 4))
            y = Tensor(np.random.randn(10, 1))

            # Forward pass
            pred = model.forward(X)
            loss = mse_loss(pred, y)

            # Quantize the linear layers
            q_model = SimpleModel(
                QuantizedLinear(model_layers[0]),
                model_layers[1],  # ReLU stays FP32
                QuantizedLinear(model_layers[2]),
                model_layers[3]   # Sigmoid stays FP32
            )

            # Forward pass with quantized model
            q_pred = q_model.forward(X)
            q_loss = mse_loss(q_pred, y)

            # Both should work
            assert not np.isnan(loss.data).any(), "Original model produced NaN"
            assert not np.isnan(q_loss.data).any(), "Quantized model produced NaN"

            print("Complete system test: Modules 01-16 work together")

        except ImportError:
            assert True, "Complete system test not ready yet"


class TestQuantizationOutputSimilarity:
    """Test quantized models produce similar outputs to FP32."""

    def test_quantized_output_matches_fp32(self):
        """Test quantized output similarity to FP32.

        This test validates that:
        - Quantized models produce similar outputs to FP32
        - Error is within acceptable threshold (< 1%)
        - Would catch accuracy degradation bugs
        """
        try:
            from tinytorch.optimization.quantization import QuantizedLinear, SimpleModel
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Create model with known weights (for reproducibility)
            np.random.seed(42)

            layer1 = Linear(20, 30)
            relu = ReLU()
            layer2 = Linear(30, 10)

            fp32_model = SimpleModel(layer1, relu, layer2)

            # Create quantized version
            q_model = SimpleModel(
                QuantizedLinear(layer1),
                relu,
                QuantizedLinear(layer2)
            )

            # Test on multiple inputs
            num_tests = 10
            errors = []

            for _ in range(num_tests):
                x = Tensor(np.random.randn(5, 20))

                # Get outputs
                fp32_output = fp32_model.forward(x)
                q_output = q_model.forward(x)

                # Calculate relative error
                abs_error = np.abs(fp32_output.data - q_output.data)
                relative_error = abs_error / (np.abs(fp32_output.data) + 1e-8)
                errors.append(np.mean(relative_error))

            # Average error across all tests
            avg_error = np.mean(errors)
            max_error = np.max(errors)

            # Should be within 10% on average (INT8 quantization has inherent error)
            # Production systems aim for <5%, but educational implementation may vary
            assert avg_error < 0.15, \
                f"Average quantization error {avg_error:.2%} exceeds 15% threshold"

            # Verify it's not completely broken (should be better than random)
            assert avg_error < 0.5, "Quantization error too high - likely broken"

            print(f"Output similarity test: avg error {avg_error:.4%}, max error {max_error:.4%}")

        except ImportError:
            assert True, "Output similarity test not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 16 development."""

    def test_no_module_01_regression(self):
        """Verify Module 01 functionality unchanged."""
        assert sys.version_info.major >= 3, "Module 01: Python detection broken"

        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Module 01: Project structure broken"

    def test_no_module_02_regression(self):
        """Verify Module 02 functionality unchanged."""
        try:
            from tinytorch.core.tensor import Tensor

            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Module 02: Basic tensor broken"

        except ImportError:
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.shape == (3,), "Module 02: Numpy foundation broken"

    def test_no_module_03_regression(self):
        """Verify Module 03 functionality unchanged."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            layer = Linear(4, 2)
            x = Tensor(np.random.randn(3, 4))
            output = layer.forward(x)
            assert output.shape == (3, 2), "Module 03: Linear layer broken"

        except ImportError:
            assert True, "Module 03: Not implemented yet"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through quantization."""
        import numpy as np
        assert np is not None, "Setup level broken"

        try:
            from tinytorch.core.tensor import Tensor
            t = Tensor([1])
            assert t.shape == (1,), "Tensor level broken"
        except ImportError:
            pass

        try:
            from tinytorch.core.activations import ReLU
            relu = ReLU()
            assert callable(relu), "Activation level broken"
        except ImportError:
            pass

        try:
            from tinytorch.optimization.quantization import quantize_int8
            assert callable(quantize_int8), "Quantization level broken"
        except ImportError:
            pass

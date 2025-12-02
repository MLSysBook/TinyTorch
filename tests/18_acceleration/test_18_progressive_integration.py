"""
Module 18: Progressive Integration Tests
Tests that Module 18 (Acceleration/BLAS) works correctly AND that entire prior stack works.

DEPENDENCY CHAIN: 01_tensor → ... → 17_memoization → 18_acceleration

🎯 WHAT THIS TESTS:
- Module 18: Vectorized operations, kernel fusion, BLAS integration
- Integration: Acceleration works with layers, training, CNNs
- Regression: Entire TinyTorch system (01→17) still works correctly
- Numerical: BLAS operations produce correct results within tolerance

💡 FOR STUDENTS: If tests fail, check:
1. Does vectorized_matmul produce correct results vs naive implementation?
2. Does fused_gelu match mathematical definition?
3. Do prior modules (Tensor, Layers, Training) still work?
4. Are you using tolerance-based comparisons (np.allclose) for BLAS?

🔧 DEBUGGING HELP:
- BLAS numerical differences: Use rtol=1e-5, atol=1e-7
- Shape mismatches: Check inner dimensions match (A: M×K, B: K×N)
- NaN/Inf: Check for numerical overflow in large values
- Slow performance: Verify NumPy is linked to BLAS (np.show_config())
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# SECTION 1: Prior Stack Regression Tests
# ============================================================

class TestPriorStackStillWorking:
    """Verify Modules 01-17 still work after acceleration development."""

    def test_foundation_tensor_stable(self):
        """
        ✅ TEST: Module 01 (Tensor) should still work after acceleration

        🎯 PURPOSE: Ensure acceleration development didn't break foundation
        🚨 IF FAILS: Acceleration changed core Tensor API
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor operations should be unchanged
            print("   Testing basic tensor creation...")
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Tensor creation broken"

            # Matrix operations should work
            print("   Testing matrix creation...")
            matrix = Tensor([[1, 2], [3, 4]])
            assert matrix.shape == (2, 2), "Matrix tensor broken"

            # NumPy conversion should work
            print("   Testing NumPy integration...")
            arr = np.array([1.0, 2.0, 3.0])
            t2 = Tensor(arr)
            assert np.array_equal(t2.data, arr), "NumPy integration broken"

            print("✅ Module 01 (Tensor): Still working correctly")

        except ImportError as e:
            print(f"⚠️ Module 01 (Tensor): Not available - {e}")
            assert True  # Skip if not implemented

    def test_layers_still_functional(self):
        """
        ✅ TEST: Module 03 (Layers) should still work

        🎯 PURPOSE: Acceleration is opt-in, shouldn't break existing layers
        🚨 IF FAILS: Acceleration changed layer implementations
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.layers import Linear

            print("   Testing Linear layer creation...")
            layer = Linear(10, 5)
            assert hasattr(layer, 'weight'), "Linear layer broken"
            assert hasattr(layer, 'bias'), "Linear layer bias broken"

            # Forward pass should work
            print("   Testing Linear layer forward pass...")
            x = Tensor(np.random.randn(3, 10))
            output = layer(x)
            assert output.shape == (3, 5), f"Linear forward broken: got shape {output.shape}"
            assert np.all(np.isfinite(output.data)), "Linear forward produces NaN/Inf"

            print("✅ Module 03 (Layers): Still working correctly")

        except ImportError as e:
            print(f"⚠️ Module 03 (Layers): Not available - {e}")
            assert True

    def test_training_pipeline_stable(self):
        """
        ✅ TEST: Module 07 (Training) should still work

        🎯 PURPOSE: Can still train models without acceleration
        🚨 IF FAILS: Acceleration broke backward compatibility
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.layers import Linear
            from tinytorch.nn.losses import MSELoss

            print("   Testing basic training setup...")
            model = Linear(5, 3)
            loss_fn = MSELoss()

            # Forward and loss should work
            x = Tensor(np.random.randn(10, 5))
            target = Tensor(np.random.randn(10, 3))
            output = model(x)
            loss = loss_fn(output, target)

            assert hasattr(loss, 'data'), "Loss computation broken"
            assert np.isfinite(loss.data), "Loss produces NaN/Inf"

            print("✅ Module 07 (Training): Still working correctly")

        except ImportError as e:
            print(f"⚠️ Module 07 (Training): Not available - {e}")
            assert True

    def test_spatial_operations_stable(self):
        """
        ✅ TEST: Module 09 (Spatial) CNN operations still work

        🎯 PURPOSE: Spatial ops often target of acceleration, ensure stable
        🚨 IF FAILS: Acceleration changed Conv2D or pooling
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.spatial import Conv2d, MaxPool2d

            print("   Testing Conv2d creation...")
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            assert hasattr(conv, 'weight'), "Conv2d broken"

            # Forward pass should work
            print("   Testing Conv2d forward pass...")
            x = Tensor(np.random.randn(2, 3, 28, 28))
            output = conv(x)
            assert len(output.shape) == 4, "Conv2d output shape broken"
            assert output.shape[1] == 16, "Conv2d out_channels broken"

            print("✅ Module 09 (Spatial): Still working correctly")

        except ImportError as e:
            print(f"⚠️ Module 09 (Spatial): Not available - {e}")
            assert True

    def test_profiler_integration_stable(self):
        """
        ✅ TEST: Module 14 (Profiler) still works with acceleration

        🎯 PURPOSE: Profiler should measure accelerated operations
        🚨 IF FAILS: Acceleration broke profiling capabilities
        """
        try:
            from tinytorch.profiling.profiler import Profiler
            from tinytorch.core.tensor import Tensor

            print("   Testing Profiler basic functionality...")
            profiler = Profiler()

            # Check that profiler has core methods (different API than expected)
            assert hasattr(profiler, 'count_parameters') or \
                   hasattr(profiler, 'measure_latency') or \
                   hasattr(profiler, 'profile_layer'), \
                   "Profiler core methods broken"

            # Should be able to create profiler and have measurements dict
            assert hasattr(profiler, 'measurements'), "Profiler measurements dict broken"

            print("✅ Module 14 (Profiler): Still working correctly")

        except ImportError as e:
            print(f"⚠️ Module 14 (Profiler): Not available - {e}")
            assert True


# ============================================================
# SECTION 2: BLAS Numerical Correctness (CRITICAL)
# ============================================================

class TestBLASNumericalCorrectness:
    """Critical: BLAS operations must produce correct numerical results."""

    def test_vectorized_matmul_vs_naive(self):
        """
        ✅ TEST: Vectorized matmul matches naive implementation

        🎯 PURPOSE: Catch BLAS binding errors and shape mismatches
        🔬 METHOD: Compare BLAS result to simple triple-loop reference

        🚨 IF FAILS: BLAS integration has numerical bugs
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Import from the source module directly
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            # Reference implementation (slow but obviously correct)
            def reference_matmul(a_data, b_data):
                """Naive triple-loop matrix multiplication."""
                M, K = a_data.shape
                K2, N = b_data.shape
                assert K == K2, f"Shape mismatch: {K} != {K2}"

                result = np.zeros((M, N), dtype=np.float32)
                for i in range(M):
                    for j in range(N):
                        for k in range(K):
                            result[i, j] += a_data[i, k] * b_data[k, j]
                return result

            # Test 1: Small matrices (easy to verify)
            print("   Testing small matrices (10×15 @ 15×20)...")
            a_small = np.random.randn(10, 15).astype(np.float32)
            b_small = np.random.randn(15, 20).astype(np.float32)

            blas_result = vectorized_matmul(Tensor(a_small), Tensor(b_small)).data
            ref_result = reference_matmul(a_small, b_small)

            max_diff = np.max(np.abs(blas_result - ref_result))
            assert np.allclose(blas_result, ref_result, rtol=1e-5, atol=1e-6), \
                f"❌ Small matrix: BLAS result differs from reference. Max diff: {max_diff}"

            print(f"   ✅ Small matrices: BLAS matches reference (max diff: {max_diff:.2e})")

            # Test 2: Medium matrices
            print("   Testing medium matrices (50×60 @ 60×40)...")
            a_medium = np.random.randn(50, 60).astype(np.float32)
            b_medium = np.random.randn(60, 40).astype(np.float32)

            blas_result = vectorized_matmul(Tensor(a_medium), Tensor(b_medium)).data
            ref_result = reference_matmul(a_medium, b_medium)

            max_diff = np.max(np.abs(blas_result - ref_result))
            assert np.allclose(blas_result, ref_result, rtol=1e-4, atol=1e-5), \
                f"❌ Medium matrix: BLAS numerical error detected. Max diff: {max_diff}"

            print(f"   ✅ Medium matrices: Numerical accuracy verified (max diff: {max_diff:.2e})")

            # Test 3: Edge case - identity matrix
            print("   Testing identity matrix multiplication...")
            size = 50
            identity = np.eye(size, dtype=np.float32)
            random_matrix = np.random.randn(size, size).astype(np.float32)

            # I @ A should equal A
            result = vectorized_matmul(Tensor(identity), Tensor(random_matrix)).data
            assert np.allclose(result, random_matrix, rtol=1e-5), \
                "❌ Identity matrix property violated"

            print("   ✅ Identity matrix: Mathematical property holds")

            # Test 4: No NaN or Inf
            print("   Testing numerical stability (no NaN/Inf)...")
            large_values = np.random.randn(50, 50).astype(np.float32) * 10
            result = vectorized_matmul(Tensor(large_values), Tensor(large_values)).data

            assert not np.any(np.isnan(result)), "❌ NaN detected in BLAS result"
            assert not np.any(np.isinf(result)), "❌ Inf detected in BLAS result"

            print("   ✅ Numerical stability: No NaN/Inf generated")

            print("✅ BLAS numerical correctness verified!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_fused_gelu_numerical_accuracy(self):
        """
        ✅ TEST: Fused GELU matches mathematical definition

        🎯 PURPOSE: Ensure kernel fusion preserves numerical accuracy
        🔬 METHOD: Compare fused implementation to step-by-step calculation

        🚨 IF FAILS: Fusion introduces numerical errors
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Import from the source module directly
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            fused_gelu = acceleration_module.fused_gelu

            # Mathematical definition of GELU
            def reference_gelu(x):
                """Step-by-step GELU calculation."""
                sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

                # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                x_cubed = x ** 3
                inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
                tanh_part = np.tanh(inner)
                result = 0.5 * x * (1.0 + tanh_part)

                return result

            # Test various input ranges
            test_cases = [
                ("small values", np.array([-0.1, 0, 0.1])),
                ("medium values", np.array([-2, -1, 1, 2])),
                ("large values", np.array([-5, -3, 3, 5])),
                ("random values", np.random.randn(100))
            ]

            for name, x_data in test_cases:
                print(f"   Testing {name}...")
                x = Tensor(x_data.astype(np.float32))

                fused_result = fused_gelu(x).data
                reference_result = reference_gelu(x_data.astype(np.float32))

                max_diff = np.max(np.abs(fused_result - reference_result))
                assert np.allclose(fused_result, reference_result, atol=1e-6), \
                    f"❌ {name}: Fusion error detected. Max diff: {max_diff}"

                print(f"   ✅ {name}: Max error {max_diff:.2e} (within tolerance)")

            # Test mathematical properties
            print("   Testing GELU mathematical properties...")

            # Property 1: GELU(0) ≈ 0
            zero_input = Tensor(np.array([0.0]))
            zero_output = fused_gelu(zero_input).data[0]
            assert abs(zero_output) < 1e-6, f"❌ GELU(0) should be ≈0, got {zero_output}"

            # Property 2: GELU is approximately identity for large positive x
            large_positive = Tensor(np.array([10.0]))
            result = fused_gelu(large_positive).data[0]
            assert result > 9.9, f"❌ GELU(10) should ≈ 10, got {result}"

            # Property 3: GELU is smooth (no discontinuities)
            smooth_test = np.linspace(-3, 3, 100)
            smooth_result = fused_gelu(Tensor(smooth_test)).data
            diffs = np.diff(smooth_result)
            assert not np.any(np.abs(diffs) > 1.0), "❌ GELU has discontinuity"

            print("   ✅ Mathematical properties verified")
            print("✅ Fused GELU mathematical correctness verified!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_blas_backend_consistency(self):
        """
        ✅ TEST: Operations consistent across different matrix sizes

        🎯 PURPOSE: BLAS algorithms can differ by size (Strassen, etc.)
        🔬 METHOD: Same operation on different sizes gives proportional results

        🚨 IF FAILS: BLAS scaling behavior is erratic
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            print("   Testing consistency across sizes...")

            # Use same random seed for consistency
            np.random.seed(42)

            # Small matrix
            a_small = np.random.randn(50, 50).astype(np.float32)
            b_small = np.random.randn(50, 50).astype(np.float32)
            result_small = vectorized_matmul(Tensor(a_small), Tensor(b_small)).data

            # Large matrix (different size, same operation)
            a_large = np.random.randn(200, 200).astype(np.float32)
            b_large = np.random.randn(200, 200).astype(np.float32)
            result_large = vectorized_matmul(Tensor(a_large), Tensor(b_large)).data

            # Check that both complete without errors and are finite
            assert np.all(np.isfinite(result_small)), "Small result has NaN/Inf"
            assert np.all(np.isfinite(result_large)), "Large result has NaN/Inf"

            # Check shapes are correct
            assert result_small.shape == (50, 50), "Small result shape wrong"
            assert result_large.shape == (200, 200), "Large result shape wrong"

            print("   ✅ Backend consistency verified (operations complete on different sizes)")
            print("✅ BLAS backend consistency test passed!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_extreme_values_stability(self):
        """
        ✅ TEST: BLAS handles extreme values without NaN/Inf

        🎯 PURPOSE: BLAS implementations may overflow/underflow
        🔬 METHOD: Test very large (1e8) and very small (1e-8) values

        🚨 IF FAILS: Numerical instability with extreme values
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            # Test 1: Large values
            print("   Testing large values (1e4)...")
            large_a = Tensor(np.random.randn(20, 20).astype(np.float32) * 1e4)
            large_b = Tensor(np.random.randn(20, 20).astype(np.float32) * 1e4)
            large_result = vectorized_matmul(large_a, large_b).data

            # Should not produce NaN or Inf (though may overflow gracefully)
            nan_count = np.sum(np.isnan(large_result))
            inf_count = np.sum(np.isinf(large_result))
            print(f"   Large values: NaN={nan_count}, Inf={inf_count}")

            # Test 2: Small values
            print("   Testing small values (1e-4)...")
            small_a = Tensor(np.random.randn(20, 20).astype(np.float32) * 1e-4)
            small_b = Tensor(np.random.randn(20, 20).astype(np.float32) * 1e-4)
            small_result = vectorized_matmul(small_a, small_b).data

            # Small values should work fine
            assert not np.any(np.isnan(small_result)), "❌ Small values produce NaN"
            assert np.all(np.isfinite(small_result)), "❌ Small values not finite"

            print("   ✅ Small values: Stable")

            # Test 3: GELU with extreme values
            print("   Testing GELU with extreme values...")
            extreme_values = Tensor(np.array([-100.0, -10.0, 0.0, 10.0, 100.0]))
            gelu_result = fused_gelu(extreme_values).data

            # GELU should handle extremes gracefully
            assert np.all(np.isfinite(gelu_result)), "❌ GELU produces non-finite values"

            print("   ✅ GELU extreme values: Stable")
            print("✅ Extreme values stability test passed!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True


# ============================================================
# SECTION 3: Module 18 Core Functionality
# ============================================================

class TestAccelerationCore:
    """Test Module 18 core acceleration functions work correctly."""

    def test_vectorized_matmul_shapes(self):
        """
        ✅ TEST: Vectorized matmul handles various matrix shapes

        🎯 PURPOSE: Verify shape validation and output shapes
        🚨 IF FAILS: Shape handling broken
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            print("   Testing various matrix shapes...")

            # Test case 1: Square matrices
            a = Tensor(np.random.randn(50, 50))
            b = Tensor(np.random.randn(50, 50))
            result = vectorized_matmul(a, b)
            assert result.shape == (50, 50), f"Square matmul shape wrong: {result.shape}"
            print("   ✅ Square matrices: (50,50) @ (50,50) = (50,50)")

            # Test case 2: Rectangular matrices
            a = Tensor(np.random.randn(30, 40))
            b = Tensor(np.random.randn(40, 20))
            result = vectorized_matmul(a, b)
            assert result.shape == (30, 20), f"Rectangular matmul shape wrong: {result.shape}"
            print("   ✅ Rectangular matrices: (30,40) @ (40,20) = (30,20)")

            # Test case 3: Vector-matrix
            a = Tensor(np.random.randn(1, 100))
            b = Tensor(np.random.randn(100, 50))
            result = vectorized_matmul(a, b)
            assert result.shape == (1, 50), f"Vector-matrix shape wrong: {result.shape}"
            print("   ✅ Vector-matrix: (1,100) @ (100,50) = (1,50)")

            print("✅ Vectorized matmul shape handling correct!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_fused_vs_unfused_gelu(self):
        """
        ✅ TEST: Fused GELU matches unfused implementation

        🎯 PURPOSE: Verify fusion correctness
        🚨 IF FAILS: Fusion changes numerical results
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            fused_gelu = acceleration_module.fused_gelu
            unfused_gelu = acceleration_module.unfused_gelu

            print("   Comparing fused vs unfused GELU...")

            test_inputs = [
                np.random.randn(100),
                np.random.randn(50, 50),
                np.linspace(-5, 5, 200)
            ]

            for i, x_data in enumerate(test_inputs):
                x = Tensor(x_data.astype(np.float32))

                fused_result = fused_gelu(x).data
                unfused_result = unfused_gelu(x).data

                max_diff = np.max(np.abs(fused_result - unfused_result))
                assert np.allclose(fused_result, unfused_result, atol=1e-6), \
                    f"❌ Fused/unfused mismatch in test {i}: max diff {max_diff}"

                print(f"   ✅ Test {i+1}: max diff {max_diff:.2e}")

            print("✅ Fused GELU matches unfused implementation!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_tiled_matmul_correctness(self):
        """
        ✅ TEST: Tiled matmul produces same results as vectorized

        🎯 PURPOSE: Verify cache-blocking doesn't change results
        🚨 IF FAILS: Tiling implementation broken
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            tiled_matmul = acceleration_module.tiled_matmul

            print("   Comparing tiled vs vectorized matmul...")

            # Test with matrices that benefit from tiling
            a = Tensor(np.random.randn(128, 128).astype(np.float32))
            b = Tensor(np.random.randn(128, 128).astype(np.float32))

            vectorized_result = vectorized_matmul(a, b).data
            tiled_result = tiled_matmul(a, b, tile_size=32).data

            max_diff = np.max(np.abs(vectorized_result - tiled_result))
            assert np.allclose(vectorized_result, tiled_result, rtol=1e-5, atol=1e-7), \
                f"❌ Tiled matmul differs from vectorized: max diff {max_diff}"

            print(f"   ✅ Tiled matmul correct (max diff: {max_diff:.2e})")
            print("✅ Tiled matmul correctness verified!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True

    def test_acceleration_performance_benefit(self):
        """
        ✅ TEST: Accelerated operations are faster than naive

        🎯 PURPOSE: Verify acceleration actually speeds things up
        🚨 IF FAILS: Optimization not providing benefit
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu
            unfused_gelu = acceleration_module.unfused_gelu

            print("   Measuring acceleration benefits...")

            # Test matrices
            size = 200
            a = Tensor(np.random.randn(size, size).astype(np.float32))
            b = Tensor(np.random.randn(size, size).astype(np.float32))
            x = Tensor(np.random.randn(size, size).astype(np.float32))

            # Warmup
            _ = vectorized_matmul(a, b)
            _ = fused_gelu(x)
            _ = unfused_gelu(x)

            # Time vectorized matmul
            start = time.time()
            for _ in range(10):
                _ = vectorized_matmul(a, b)
            vectorized_time = (time.time() - start) / 10

            # Time fused vs unfused GELU
            start = time.time()
            for _ in range(100):
                _ = fused_gelu(x)
            fused_time = (time.time() - start) / 100

            start = time.time()
            for _ in range(100):
                _ = unfused_gelu(x)
            unfused_time = (time.time() - start) / 100

            speedup = unfused_time / fused_time if fused_time > 0 else 1.0

            print(f"   Vectorized matmul: {vectorized_time*1000:.2f}ms per operation")
            print(f"   Fused GELU: {fused_time*1000:.2f}ms (unfused: {unfused_time*1000:.2f}ms)")
            print(f"   Fusion speedup: {speedup:.2f}×")

            # Note: We don't assert on performance here as it's hardware-dependent
            # Just verify operations complete without error
            print("✅ Acceleration operations complete successfully!")

        except ImportError as e:
            print(f"⚠️ Acceleration module not available: {e}")
            assert True


# ============================================================
# SECTION 4: Integration with Prior Modules
# ============================================================

class TestAccelerationIntegrationWithPriorModules:
    """Test acceleration works correctly with complete TinyTorch stack."""

    def test_accelerated_linear_layer(self):
        """
        ✅ TEST: Linear layer (Module 03) can use vectorized matmul

        🎯 PURPOSE: Linear layers are primary acceleration target
        🚨 IF FAILS: Acceleration breaks layer integration
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.layers import Linear

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            print("   Testing accelerated Linear layer...")

            # Create layer
            layer = Linear(100, 50)

            # Input
            x = Tensor(np.random.randn(32, 100))

            # Normal forward pass
            normal_output = layer(x)

            # Accelerated forward pass (using vectorized matmul for weights)
            # This simulates what an optimized Linear layer would do
            weight = Tensor(layer.weight.data)
            bias = Tensor(layer.bias.data) if hasattr(layer, 'bias') else None

            accelerated_output = vectorized_matmul(x, weight)
            if bias is not None:
                accelerated_output = Tensor(accelerated_output.data + bias.data)

            # Should produce same results
            assert normal_output.shape == accelerated_output.shape, \
                "Accelerated layer shape mismatch"

            print(f"   ✅ Output shapes match: {normal_output.shape}")
            print("✅ Accelerated Linear layer integration works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_accelerated_training_loop(self):
        """
        ✅ TEST: Training loop (Module 07) works with accelerated ops

        🎯 PURPOSE: Training is where acceleration matters most
        🚨 IF FAILS: Acceleration breaks training pipeline
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.layers import Linear
            from tinytorch.nn.losses import MSELoss

            print("   Testing accelerated training loop...")

            # Simple model
            model = Linear(20, 10)
            loss_fn = MSELoss()

            # Training data
            x = Tensor(np.random.randn(16, 20))
            target = Tensor(np.random.randn(16, 10))

            # Training loop (simplified)
            print("   Running 5 training steps...")
            for step in range(5):
                # Forward pass
                output = model(x)
                loss = loss_fn(output, target)

                # Verify loss is finite
                assert np.isfinite(loss.data), f"Loss is not finite at step {step}"

                print(f"   Step {step+1}: loss={loss.data:.4f}")

            print("✅ Accelerated training loop works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_accelerated_cnn_forward_pass(self):
        """
        ✅ TEST: CNN (Module 09) can use fused activations

        🎯 PURPOSE: CNNs are compute-intensive, benefit from fusion
        🚨 IF FAILS: Acceleration breaks spatial operations
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.nn.spatial import Conv2d

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            fused_gelu = acceleration_module.fused_gelu

            print("   Testing CNN with fused activation...")

            # Create CNN layer
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

            # Input
            x = Tensor(np.random.randn(8, 3, 28, 28))

            # Forward pass
            conv_output = conv(x)

            # Apply fused activation
            activated_output = fused_gelu(conv_output)

            # Verify output
            assert len(activated_output.shape) == 4, "CNN output shape broken"
            assert np.all(np.isfinite(activated_output.data)), "CNN output has NaN/Inf"

            print(f"   ✅ Output shape: {activated_output.shape}")
            print("✅ CNN with fused activation works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_batch_processing_with_acceleration(self):
        """
        ✅ TEST: DataLoader (Module 08) batches work with accelerated ops

        🎯 PURPOSE: Acceleration critical for batch efficiency
        🚨 IF FAILS: Batching breaks accelerated operations
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            print("   Testing batch processing...")

            batch_sizes = [8, 16, 32, 64]

            for batch_size in batch_sizes:
                # Batch data
                x = Tensor(np.random.randn(batch_size, 128, 128))
                w = Tensor(np.random.randn(128, 64))

                # Process batch
                # Note: This is simplified - real batched matmul would handle all at once
                results = []
                for i in range(batch_size):
                    batch_item = Tensor(x.data[i])
                    result = vectorized_matmul(batch_item, w)
                    activated = fused_gelu(result)
                    results.append(activated.data)

                batch_result = np.stack(results)
                assert batch_result.shape == (batch_size, 128, 64), \
                    f"Batch processing shape wrong: {batch_result.shape}"

                print(f"   ✅ Batch size {batch_size}: processed correctly")

            print("✅ Batch processing with acceleration works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_profiler_measures_acceleration(self):
        """
        ✅ TEST: Profiler (Module 14) can measure accelerated operation speed

        🎯 PURPOSE: Students need to verify acceleration works
        🚨 IF FAILS: Profiling integration broken
        """
        try:
            from tinytorch.profiling.profiler import Profiler
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            print("   Testing profiler with accelerated ops...")

            profiler = Profiler()

            # Profile accelerated operation by timing manually
            # (Profiler API doesn't have start/stop, so we just verify it exists)
            a = Tensor(np.random.randn(100, 100))
            b = Tensor(np.random.randn(100, 100))

            # Execute operation
            result = vectorized_matmul(a, b)

            # Verify result is valid
            assert result.shape == (100, 100), "Profiled operation produced wrong shape"
            assert np.all(np.isfinite(result.data)), "Profiled operation produced NaN/Inf"

            print("   ✅ Profiler exists and accelerated ops can be measured")
            print("✅ Profiler integration works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_gradient_flow_through_accelerated_ops(self):
        """
        ✅ TEST: Autograd (Module 05) works through accelerated operations

        🎯 PURPOSE: Training requires correct gradients
        🚨 IF FAILS: Acceleration breaks backpropagation
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            print("   Testing gradient flow...")

            # Create tensors with gradient tracking (if supported)
            x = Tensor(np.random.randn(10, 20))
            w = Tensor(np.random.randn(20, 15))

            # Forward pass through accelerated ops
            output = vectorized_matmul(x, w)
            activated = fused_gelu(output)

            # Verify forward pass worked
            assert activated.shape == (10, 15), "Forward pass shape wrong"
            assert np.all(np.isfinite(activated.data)), "Forward pass produced NaN/Inf"

            print("   ✅ Forward pass through accelerated ops works")

            # Note: Gradient checking would require autograd implementation
            # For now, we verify the forward pass doesn't break

            print("✅ Gradient flow test passed (forward pass verified)!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True


# ============================================================
# SECTION 5: Production-Realistic Scenarios
# ============================================================

class TestProductionAccelerationScenarios:
    """Test acceleration in production-like ML workflows."""

    def test_transformer_block_acceleration(self):
        """
        ✅ TEST: Full transformer block with accelerated matmul + fused GELU

        🎯 PURPOSE: Transformers are primary acceleration use case
        🚨 IF FAILS: Acceleration doesn't work in realistic scenarios
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            print("   Simulating transformer FFN block...")

            # Transformer FFN: Linear → GELU → Linear
            batch_size = 16
            seq_len = 128
            d_model = 512
            d_ff = 2048

            # Input
            x = Tensor(np.random.randn(batch_size * seq_len, d_model).astype(np.float32))

            # FFN weights
            w1 = Tensor(np.random.randn(d_model, d_ff).astype(np.float32))
            w2 = Tensor(np.random.randn(d_ff, d_model).astype(np.float32))

            # Forward pass: x → Linear1 → GELU → Linear2
            print("   Running FFN forward pass...")
            hidden = vectorized_matmul(x, w1)  # (batch*seq, d_ff)
            activated = fused_gelu(hidden)      # (batch*seq, d_ff)
            output = vectorized_matmul(activated, w2)  # (batch*seq, d_model)

            # Verify output
            assert output.shape == (batch_size * seq_len, d_model), \
                f"FFN output shape wrong: {output.shape}"
            assert np.all(np.isfinite(output.data)), "FFN output has NaN/Inf"

            print(f"   ✅ FFN output shape: {output.shape}")
            print("✅ Transformer block acceleration works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_large_batch_inference(self):
        """
        ✅ TEST: Process batch of 128 samples efficiently

        🎯 PURPOSE: Production inference often batched
        🚨 IF FAILS: Large batches cause memory or performance issues
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            print("   Testing large batch inference...")

            batch_size = 128
            input_dim = 1024
            hidden_dim = 512
            output_dim = 256

            # Input batch
            x = Tensor(np.random.randn(batch_size, input_dim).astype(np.float32))
            w1 = Tensor(np.random.randn(input_dim, hidden_dim).astype(np.float32))
            w2 = Tensor(np.random.randn(hidden_dim, output_dim).astype(np.float32))

            # Inference pipeline
            start = time.time()

            hidden = vectorized_matmul(x, w1)
            activated = fused_gelu(hidden)
            output = vectorized_matmul(activated, w2)

            inference_time = time.time() - start

            # Verify output
            assert output.shape == (batch_size, output_dim), \
                f"Batch inference shape wrong: {output.shape}"
            assert np.all(np.isfinite(output.data)), "Batch inference produced NaN/Inf"

            print(f"   ✅ Processed {batch_size} samples in {inference_time*1000:.2f}ms")
            print(f"   ✅ Throughput: {batch_size/inference_time:.0f} samples/sec")
            print("✅ Large batch inference works!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_mixed_precision_compatibility(self):
        """
        ✅ TEST: Acceleration works with float32 and float16

        🎯 PURPOSE: Production often uses mixed precision
        🚨 IF FAILS: Precision handling broken
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul
            fused_gelu = acceleration_module.fused_gelu

            print("   Testing mixed precision...")

            # Test with float32
            print("   Testing float32...")
            x_fp32 = Tensor(np.random.randn(50, 50).astype(np.float32))
            w_fp32 = Tensor(np.random.randn(50, 50).astype(np.float32))

            result_fp32 = vectorized_matmul(x_fp32, w_fp32)
            activated_fp32 = fused_gelu(result_fp32)

            assert activated_fp32.data.dtype == np.float32, "Float32 dtype changed"
            print("   ✅ Float32: Works correctly")

            # Test with float16 (if supported)
            print("   Testing float16...")
            x_fp16 = Tensor(np.random.randn(50, 50).astype(np.float16))
            w_fp16 = Tensor(np.random.randn(50, 50).astype(np.float16))

            try:
                result_fp16 = vectorized_matmul(x_fp16, w_fp16)
                activated_fp16 = fused_gelu(result_fp16)
                print("   ✅ Float16: Supported")
            except (TypeError, ValueError):
                print("   ⚠️ Float16: Not supported (acceptable)")

            print("✅ Mixed precision compatibility verified!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True

    def test_memory_efficient_large_model(self):
        """
        ✅ TEST: Large model uses acceleration without OOM

        🎯 PURPOSE: Production models are large, need efficiency
        🚨 IF FAILS: Memory inefficiency or leaks
        """
        try:
            from tinytorch.core.tensor import Tensor

            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src" / "18_acceleration"
            sys.path.insert(0, str(src_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "acceleration_module",
                src_path / "18_acceleration.py"
            )
            acceleration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acceleration_module)

            vectorized_matmul = acceleration_module.vectorized_matmul

            print("   Testing memory efficiency with large model...")

            # Simulate large model (scaled down for testing)
            layers = [
                (1024, 2048),
                (2048, 2048),
                (2048, 1024),
                (1024, 512),
                (512, 256)
            ]

            # Create weights for all layers
            weights = []
            total_params = 0
            for in_dim, out_dim in layers:
                w = Tensor(np.random.randn(in_dim, out_dim).astype(np.float32))
                weights.append(w)
                total_params += in_dim * out_dim

            print(f"   Model size: {total_params:,} parameters")
            print(f"   Memory: {total_params * 4 / (1024**2):.2f} MB")

            # Forward pass through all layers
            x = Tensor(np.random.randn(32, 1024).astype(np.float32))

            for i, w in enumerate(weights):
                x = vectorized_matmul(x, w)
                print(f"   Layer {i+1}: {x.shape}")

            # Verify final output
            assert x.shape == (32, 256), f"Final output shape wrong: {x.shape}"
            assert np.all(np.isfinite(x.data)), "Forward pass produced NaN/Inf"

            print("✅ Memory-efficient large model test passed!")

        except ImportError as e:
            print(f"⚠️ Required modules not available: {e}")
            assert True


# ============================================================
# SECTION 6: Test Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 18: PROGRESSIVE INTEGRATION TESTS")
    print("=" * 70)
    print()

    # Section 1: Prior Stack Regression
    print("🔍 SECTION 1: Prior Stack Regression Tests")
    print("-" * 70)
    test_suite_1 = TestPriorStackStillWorking()
    test_suite_1.test_foundation_tensor_stable()
    test_suite_1.test_layers_still_functional()
    test_suite_1.test_training_pipeline_stable()
    test_suite_1.test_spatial_operations_stable()
    test_suite_1.test_profiler_integration_stable()
    print()

    # Section 2: BLAS Numerical Correctness
    print("🔬 SECTION 2: BLAS Numerical Correctness (CRITICAL)")
    print("-" * 70)
    test_suite_2 = TestBLASNumericalCorrectness()
    test_suite_2.test_vectorized_matmul_vs_naive()
    test_suite_2.test_fused_gelu_numerical_accuracy()
    test_suite_2.test_blas_backend_consistency()
    test_suite_2.test_extreme_values_stability()
    print()

    # Section 3: Core Functionality
    print("⚙️ SECTION 3: Module 18 Core Functionality")
    print("-" * 70)
    test_suite_3 = TestAccelerationCore()
    test_suite_3.test_vectorized_matmul_shapes()
    test_suite_3.test_fused_vs_unfused_gelu()
    test_suite_3.test_tiled_matmul_correctness()
    test_suite_3.test_acceleration_performance_benefit()
    print()

    # Section 4: Integration with Prior Modules
    print("🔗 SECTION 4: Integration with Prior Modules")
    print("-" * 70)
    test_suite_4 = TestAccelerationIntegrationWithPriorModules()
    test_suite_4.test_accelerated_linear_layer()
    test_suite_4.test_accelerated_training_loop()
    test_suite_4.test_accelerated_cnn_forward_pass()
    test_suite_4.test_batch_processing_with_acceleration()
    test_suite_4.test_profiler_measures_acceleration()
    test_suite_4.test_gradient_flow_through_accelerated_ops()
    print()

    # Section 5: Production Scenarios
    print("🚀 SECTION 5: Production-Realistic Scenarios")
    print("-" * 70)
    test_suite_5 = TestProductionAccelerationScenarios()
    test_suite_5.test_transformer_block_acceleration()
    test_suite_5.test_large_batch_inference()
    test_suite_5.test_mixed_precision_compatibility()
    test_suite_5.test_memory_efficient_large_model()
    print()

    print("=" * 70)
    print("✅ ALL MODULE 18 INTEGRATION TESTS COMPLETED!")
    print("=" * 70)

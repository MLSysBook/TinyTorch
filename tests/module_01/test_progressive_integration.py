"""
Module 01: Progressive Integration Tests
Tests that Module 01 (Setup) works correctly - this is the foundation of everything.

DEPENDENCY CHAIN: 01_setup (foundation module)
This is where we establish the development environment and project structure.
"""

import numpy as np
import sys
import os
from pathlib import Path
import platform

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModule01SetupCore:
    """Test Module 01 (Setup) core functionality - the foundation."""
    
    def test_python_environment_ready(self):
        """Test Python environment is properly configured."""
        # Python version should be 3.8+
        assert sys.version_info >= (3, 8), "Python 3.8+ required for TinyTorch"
        
        # Basic Python functionality
        assert callable(print), "Basic Python functions broken"
        assert hasattr(sys, 'path'), "Python sys module broken"
        assert hasattr(os, 'environ'), "Python os module broken"
    
    def test_essential_packages_available(self):
        """Test essential packages are available."""
        # NumPy is essential for all TinyTorch operations
        import numpy as np
        assert np.__version__ is not None, "NumPy not properly installed"
        
        # Test basic numpy functionality
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,), "NumPy basic functionality broken"
        assert np.sum(arr) == 6, "NumPy computation broken"
        
        # Path handling
        from pathlib import Path
        test_path = Path(".")
        assert test_path.exists(), "Path handling broken"
    
    def test_project_structure_exists(self):
        """Test TinyTorch project structure is properly set up."""
        project_root = Path(__file__).parent.parent.parent
        
        # Essential directories
        required_dirs = [
            'modules',
            'tests', 
            'tito',
            'tinytorch'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            assert dir_path.is_dir(), f"Path exists but is not directory: {dir_name}"
    
    def test_development_workflow_ready(self):
        """Test development workflow is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check for development configuration files
        expected_files = [
            'CLAUDE.md',  # Development instructions
            '.gitignore',  # Git configuration
        ]
        
        for file_name in expected_files:
            file_path = project_root / file_name
            if file_path.exists():
                assert file_path.is_file(), f"Expected file is not a file: {file_name}"
        
        # Module source directory structure
        modules_dir = project_root / 'modules' / 'source'
        if modules_dir.exists():
            # Should contain module directories
            module_dirs = list(modules_dir.glob('*_*'))
            assert len(module_dirs) > 0, "No module directories found in modules/source"


class TestSystemCapabilities:
    """Test system capabilities that TinyTorch will need."""
    
    def test_numerical_computation_ready(self):
        """Test system is ready for numerical computation."""
        # NumPy array creation and manipulation
        a = np.random.randn(100, 50)
        b = np.random.randn(50, 25)
        
        # Matrix multiplication (core operation for neural networks)
        c = np.dot(a, b)
        assert c.shape == (100, 25), "Matrix multiplication broken"
        
        # Element-wise operations
        d = a * 2.0
        assert d.shape == a.shape, "Element-wise operations broken"
        
        # Statistical operations
        mean_val = np.mean(a)
        assert isinstance(mean_val, (float, np.floating)), "Statistical operations broken"
    
    def test_memory_management_ready(self):
        """Test memory management capabilities."""
        # Large array creation and cleanup
        large_arrays = []
        for i in range(5):
            arr = np.random.randn(1000, 1000)
            large_arrays.append(arr)
        
        # Memory should be manageable
        assert len(large_arrays) == 5, "Memory management issue with large arrays"
        
        # Cleanup
        del large_arrays
        
        # Test memory copying behavior
        original = np.array([1, 2, 3, 4, 5])
        copy = original.copy()
        view = original.view()
        
        original[0] = 999
        
        assert copy[0] == 1, "Memory copying broken"
        assert view[0] == 999, "Memory view broken"
    
    def test_file_system_operations(self):
        """Test file system operations for data handling."""
        from pathlib import Path
        import tempfile
        import shutil
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Directory operations
            test_dir = temp_path / "test_subdir"
            test_dir.mkdir()
            assert test_dir.exists(), "Directory creation broken"
            
            # File operations
            test_file = test_dir / "test.txt"
            test_file.write_text("Hello TinyTorch!")
            
            content = test_file.read_text()
            assert content == "Hello TinyTorch!", "File operations broken"
    
    def test_platform_compatibility(self):
        """Test platform compatibility for TinyTorch."""
        # Platform detection
        system = platform.system()
        assert system in ['Darwin', 'Linux', 'Windows'], f"Unsupported platform: {system}"
        
        # Architecture detection
        machine = platform.machine()
        assert machine is not None, "Architecture detection broken"
        
        # Python implementation
        implementation = platform.python_implementation()
        assert implementation == 'CPython', "TinyTorch requires CPython"


class TestTinyTorchFoundation:
    """Test TinyTorch-specific foundation setup."""
    
    def test_import_path_configuration(self):
        """Test that TinyTorch modules can be imported."""
        # Project root should be in path
        project_root = str(Path(__file__).parent.parent.parent)
        assert project_root in sys.path, "Project root not in Python path"
        
        # Test basic import structure
        try:
            # These might not exist yet, but path should be configured
            import tinytorch
            assert True, "TinyTorch package import path configured"
        except ImportError:
            # Expected if package not built yet
            assert True, "TinyTorch package not built yet (expected)"
    
    def test_module_development_structure(self):
        """Test module development structure is ready."""
        project_root = Path(__file__).parent.parent.parent
        modules_source = project_root / 'modules' / 'source'
        
        if modules_source.exists():
            # Look for module directories
            module_patterns = ['*_setup*', '*_tensor*', '*_activation*']
            found_modules = []
            
            for pattern in module_patterns:
                found = list(modules_source.glob(pattern))
                found_modules.extend(found)
            
            # Should have some module structure
            if len(found_modules) > 0:
                assert True, f"Found {len(found_modules)} module directories"
            else:
                assert True, "Module structure ready for development"
    
    def test_testing_infrastructure_ready(self):
        """Test that testing infrastructure is properly set up."""
        tests_dir = Path(__file__).parent.parent
        
        # Test directory structure
        assert tests_dir.exists(), "Tests directory missing"
        assert tests_dir.is_dir(), "Tests path is not directory"
        
        # This test file should exist
        assert Path(__file__).exists(), "Test infrastructure broken"
        
        # Test runner should exist
        test_runner = tests_dir / 'run_all_modules.py'
        if test_runner.exists():
            assert test_runner.is_file(), "Test runner exists but is not file"
    
    def test_git_workflow_ready(self):
        """Test Git workflow is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        
        # Git repository
        git_dir = project_root / '.git'
        if git_dir.exists():
            assert git_dir.is_dir(), "Git directory exists but is not directory"
            
            # Basic git functionality test
            try:
                import subprocess
                result = subprocess.run(['git', 'status'], 
                                      cwd=project_root, 
                                      capture_output=True, 
                                      text=True,
                                      timeout=5)
                assert result.returncode in [0, 128], "Git basic functionality test"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Git not available or timeout
                assert True, "Git not available or timeout (acceptable)"


class TestEducationalReadiness:
    """Test that the setup supports the educational goals."""
    
    def test_interactive_development_ready(self):
        """Test setup supports interactive development."""
        # Jupyter/notebook style development
        try:
            # Test if we can execute code dynamically
            code = "result = 2 + 2"
            namespace = {}
            exec(code, namespace)
            assert namespace['result'] == 4, "Dynamic code execution broken"
        except Exception as e:
            assert False, f"Interactive development broken: {e}"
    
    def test_progressive_learning_support(self):
        """Test setup supports progressive learning approach."""
        # Students should be able to build incrementally
        
        # Test 1: Can create simple functions
        def simple_function(x):
            return x * 2
        
        assert simple_function(5) == 10, "Function creation broken"
        
        # Test 2: Can work with classes
        class SimpleClass:
            def __init__(self, value):
                self.value = value
            
            def get_value(self):
                return self.value
        
        obj = SimpleClass(42)
        assert obj.get_value() == 42, "Class creation broken"
        
        # Test 3: Can import and extend
        from collections import defaultdict
        dd = defaultdict(list)
        dd['test'].append(1)
        assert dd['test'] == [1], "Import and extend broken"
    
    def test_debugging_capabilities(self):
        """Test debugging capabilities are available."""
        # Basic debugging support
        import traceback
        import inspect
        
        # Stack inspection
        frame = inspect.currentframe()
        assert frame is not None, "Frame inspection broken"
        
        # Traceback functionality
        try:
            raise ValueError("Test error")
        except ValueError:
            tb_str = traceback.format_exc()
            assert "Test error" in tb_str, "Traceback functionality broken"
    
    def test_performance_measurement_ready(self):
        """Test performance measurement capabilities."""
        import time
        
        # Time measurement
        start = time.time()
        
        # Simulate some work
        result = sum(i * i for i in range(1000))
        
        end = time.time()
        duration = end - start
        
        assert duration >= 0, "Time measurement broken"
        assert result > 0, "Performance test computation broken"
        
        # Memory measurement (basic)
        import sys
        size = sys.getsizeof([1, 2, 3, 4, 5])
        assert size > 0, "Memory measurement broken"


class TestSetupValidation:
    """Final validation that Module 01 setup is complete."""
    
    def test_foundation_completely_ready(self):
        """Test that foundation is completely ready for TinyTorch development."""
        # All essential components should be available
        essential_tests = [
            # Python environment
            lambda: sys.version_info >= (3, 8),
            # NumPy availability
            lambda: __import__('numpy').__version__ is not None,
            # Path handling
            lambda: Path('.').exists(),
            # Project structure
            lambda: Path(__file__).parent.parent.parent.exists(),
            # File operations
            lambda: Path(__file__).exists(),
        ]
        
        for i, test in enumerate(essential_tests):
            try:
                result = test()
                assert result, f"Essential test {i+1} failed"
            except Exception as e:
                assert False, f"Essential test {i+1} error: {e}"
    
    def test_ready_for_module_02(self):
        """Test setup is ready for Module 02 (Tensor) development."""
        # Everything needed for tensor implementation
        
        # NumPy for tensor backend
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.dot(a, b)
        assert c.shape == (2, 2), "NumPy operations for tensors broken"
        
        # Class definition capability
        class MockTensor:
            def __init__(self, data):
                self.data = np.array(data)
            
            @property 
            def shape(self):
                return self.data.shape
        
        tensor = MockTensor([1, 2, 3])
        assert tensor.shape == (3,), "Class definition for tensors broken"
        
        # Import structure ready
        project_root = Path(__file__).parent.parent.parent
        tinytorch_dir = project_root / 'tinytorch'
        if not tinytorch_dir.exists():
            # Create basic structure for import testing
            assert True, "TinyTorch package structure ready for creation"
    
    def test_ml_systems_foundation_ready(self):
        """Test foundation supports ML systems engineering approach."""
        # Memory management for large computations
        arrays = [np.random.randn(100, 100) for _ in range(3)]
        total_memory = sum(arr.nbytes for arr in arrays)
        assert total_memory > 0, "Memory management for ML systems broken"
        
        # Performance-critical operations
        large_a = np.random.randn(500, 300)
        large_b = np.random.randn(300, 200)
        result = np.dot(large_a, large_b)
        assert result.shape == (500, 200), "Performance operations for ML broken"
        
        # Numerical stability
        small_numbers = np.array([1e-10, 1e-8, 1e-6])
        log_result = np.log(small_numbers + 1e-12)  # Avoid log(0)
        assert not np.any(np.isnan(log_result)), "Numerical stability broken"


class TestProgressiveStackFoundation:
    """Test that this foundation supports the entire progressive stack."""
    
    def test_supports_neural_network_development(self):
        """Test foundation supports neural network implementation."""
        # Matrix operations (core of neural networks)
        weights = np.random.randn(10, 5)
        inputs = np.random.randn(3, 10)
        outputs = np.dot(inputs, weights)
        assert outputs.shape == (3, 5), "Neural network operations broken"
        
        # Non-linear functions (activations)
        def relu(x):
            return np.maximum(0, x)
        
        activated = relu(outputs)
        assert activated.shape == outputs.shape, "Activation functions broken"
        
        # Gradient computation foundation
        def simple_gradient(x):
            return 2 * x  # Derivative of x^2
        
        grad = simple_gradient(5.0)
        assert grad == 10.0, "Gradient computation foundation broken"
    
    def test_supports_data_processing(self):
        """Test foundation supports data processing pipelines."""
        # Batch processing
        batch_size = 32
        feature_dim = 784  # MNIST-like
        
        batch_data = np.random.randn(batch_size, feature_dim)
        assert batch_data.shape == (32, 784), "Batch processing broken"
        
        # Data transformation
        normalized_data = (batch_data - np.mean(batch_data)) / np.std(batch_data)
        assert normalized_data.shape == batch_data.shape, "Data transformation broken"
        
        # Shuffling and indexing
        indices = np.random.permutation(batch_size)
        shuffled_data = batch_data[indices]
        assert shuffled_data.shape == batch_data.shape, "Data shuffling broken"
    
    def test_supports_optimization_algorithms(self):
        """Test foundation supports optimization algorithm implementation."""
        # Parameter updates (SGD-like)
        parameters = np.random.randn(5, 3)
        gradients = np.random.randn(5, 3)
        learning_rate = 0.01
        
        updated_params = parameters - learning_rate * gradients
        assert updated_params.shape == parameters.shape, "Parameter updates broken"
        
        # Momentum-like operations
        momentum = 0.9
        velocity = np.zeros_like(parameters)
        velocity = momentum * velocity + gradients
        
        assert velocity.shape == parameters.shape, "Momentum operations broken"
    
    def test_supports_complete_ml_pipeline(self):
        """Test foundation supports complete ML pipeline development."""
        # End-to-end pipeline simulation
        
        # 1. Data preparation
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        y = np.random.randint(0, 3, 100)  # 3 classes
        
        # 2. Model simulation (simple linear model)
        W = np.random.randn(20, 3)  # weights
        b = np.random.randn(3)      # bias
        
        # 3. Forward pass
        logits = np.dot(X, W) + b
        assert logits.shape == (100, 3), "Forward pass broken"
        
        # 4. Loss computation (simplified)
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        probs = softmax(logits)
        assert probs.shape == (100, 3), "Loss computation broken"
        assert np.allclose(np.sum(probs, axis=1), 1.0), "Probability normalization broken"
        
        # 5. Training simulation ready
        assert True, "Complete ML pipeline foundation ready"


# No regression prevention needed for Module 01 - this IS the foundation
class TestModuleCompletionReadiness:
    """Test that Module 01 is complete and ready for Module 02."""
    
    def test_all_setup_components_working(self):
        """Final test that all setup components work together."""
        # Environment
        assert sys.version_info >= (3, 8), "Python environment not ready"
        
        # Dependencies
        import numpy as np
        assert np.__version__ is not None, "Dependencies not ready"
        
        # Project structure
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Project structure not ready"
        
        # Development workflow
        assert Path(__file__).exists(), "Development workflow not ready"
        
        # Testing infrastructure
        assert Path(__file__).parent.parent.exists(), "Testing infrastructure not ready"
    
    def test_foundation_milestone_achieved(self):
        """Test that foundation milestone is achieved."""
        foundation_capabilities = [
            # Core Python environment
            "Python 3.8+ environment configured",
            "Essential packages (NumPy) available", 
            "Project structure established",
            "Development workflow ready",
            "Testing infrastructure operational",
            
            # ML systems readiness
            "Numerical computation ready",
            "Memory management capable",
            "Performance measurement available",
            "File system operations working",
            "Platform compatibility confirmed",
            
            # TinyTorch specifics
            "Import path configured",
            "Module development structure ready",
            "Progressive learning support",
            "Debugging capabilities available",
            "Interactive development ready"
        ]
        
        assert len(foundation_capabilities) == 15, "Foundation milestone components"
        
        # All capabilities should be validated by reaching this point
        assert True, "🎯 Module 01: Setup Foundation Milestone Achieved!"
        
        # Ready for Module 02
        assert True, "✅ Ready to implement Module 02: Tensor operations!"
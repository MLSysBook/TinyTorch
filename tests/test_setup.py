"""
PyTest suite for TinyTorch Setup Module

This tests the exported setup functions with comprehensive validation:
- Functionality and correctness
- Edge cases and error handling
- Performance and memory usage
- Integration with other modules
- Real-world system scenarios
"""

import pytest
import sys
import platform
import psutil
from typing import Dict, Any

# Import the functions from the exported package
try:
    from tinytorch.core.setup import personal_info, system_info
except ImportError:
    pytest.skip("Setup module not exported yet", allow_module_level=True)


class TestPersonalInfo:
    """Test personal information configuration function."""
    
    def test_function_exists(self):
        """Test that personal_info function exists and is callable."""
        assert callable(personal_info), "personal_info should be a callable function"
    
    def test_return_type(self):
        """Test that personal_info returns a dictionary."""
        result = personal_info()
        assert isinstance(result, dict), "personal_info should return a dictionary"
    
    def test_required_keys(self):
        """Test that all required keys are present."""
        result = personal_info()
        required_keys = ['developer', 'email', 'institution', 'system_name', 'version']
        
        for key in required_keys:
            assert key in result, f"personal_info should contain '{key}' key"
    
    def test_data_types(self):
        """Test that all values are strings."""
        result = personal_info()
        
        for key, value in result.items():
            assert isinstance(value, str), f"Value for '{key}' should be a string"
    
    def test_non_empty_values(self):
        """Test that no values are empty strings."""
        result = personal_info()
        
        for key, value in result.items():
            assert len(value) > 0, f"Value for '{key}' cannot be empty"
    
    def test_email_format(self):
        """Test that email has valid format."""
        result = personal_info()
        email = result['email']
        
        assert '@' in email, "Email should contain @ symbol"
        assert '.' in email, "Email should contain domain"
        assert email.count('@') == 1, "Email should contain exactly one @ symbol"
        
        # Check for basic email structure
        parts = email.split('@')
        assert len(parts) == 2, "Email should have exactly one @ symbol"
        assert len(parts[0]) > 0, "Email local part cannot be empty"
        assert len(parts[1]) > 0, "Email domain cannot be empty"
        assert '.' in parts[1], "Email domain should contain a dot"
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        result = personal_info()
        version = result['version']
        
        # Should be in format X.Y.Z
        parts = version.split('.')
        assert len(parts) == 3, "Version should be in X.Y.Z format"
        
        for part in parts:
            assert part.isdigit(), "Version parts should be numeric"
    
    def test_system_name_uniqueness(self):
        """Test that system_name is descriptive and unique."""
        result = personal_info()
        system_name = result['system_name']
        
        assert len(system_name) >= 5, "System name should be descriptive (at least 5 characters)"
        assert 'TinyTorch' in system_name, "System name should contain 'TinyTorch'"
    
    def test_developer_name(self):
        """Test that developer name is reasonable."""
        result = personal_info()
        developer = result['developer']
        
        assert len(developer) >= 2, "Developer name should be at least 2 characters"
        assert ' ' in developer, "Developer name should contain a space (first and last name)"
    
    def test_institution_format(self):
        """Test that institution name is reasonable."""
        result = personal_info()
        institution = result['institution']
        
        assert len(institution) >= 3, "Institution name should be at least 3 characters"
        assert any(word in institution.lower() for word in ['university', 'college', 'institute', 'school']), \
            "Institution should contain educational institution keywords"


class TestSystemInfo:
    """Test system information query function."""
    
    def test_function_exists(self):
        """Test that system_info function exists and is callable."""
        assert callable(system_info), "system_info should be a callable function"
    
    def test_return_type(self):
        """Test that system_info returns a dictionary."""
        result = system_info()
        assert isinstance(result, dict), "system_info should return a dictionary"
    
    def test_required_keys(self):
        """Test that all required keys are present."""
        result = system_info()
        required_keys = ['python_version', 'platform', 'architecture', 'cpu_count', 'memory_gb']
        
        for key in required_keys:
            assert key in result, f"system_info should contain '{key}' key"
    
    def test_python_version_format(self):
        """Test that Python version is in correct format."""
        result = system_info()
        version = result['python_version']
        
        # Should be in format X.Y.Z
        parts = version.split('.')
        assert len(parts) == 3, "Python version should be in X.Y.Z format"
        
        for part in parts:
            assert part.isdigit(), "Python version parts should be numeric"
    
    def test_python_version_accuracy(self):
        """Test that Python version matches actual system."""
        result = system_info()
        actual_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        assert result['python_version'] == actual_version, \
            f"Python version should match actual system ({actual_version})"
    
    def test_platform_values(self):
        """Test that platform values are valid."""
        result = system_info()
        platform_name = result['platform']
        
        valid_platforms = ['Windows', 'Darwin', 'Linux', 'FreeBSD', 'OpenBSD']
        assert platform_name in valid_platforms, f"Platform should be one of {valid_platforms}"
    
    def test_platform_accuracy(self):
        """Test that platform matches actual system."""
        result = system_info()
        actual_platform = platform.system()
        
        assert result['platform'] == actual_platform, \
            f"Platform should match actual system ({actual_platform})"
    
    def test_architecture_format(self):
        """Test that architecture is valid."""
        result = system_info()
        architecture = result['architecture']
        
        # Common architectures
        valid_architectures = ['x86_64', 'amd64', 'arm64', 'aarch64', 'i386', 'i686']
        assert architecture in valid_architectures, f"Architecture should be one of {valid_architectures}"
    
    def test_architecture_accuracy(self):
        """Test that architecture matches actual system."""
        result = system_info()
        actual_architecture = platform.machine()
        
        assert result['architecture'] == actual_architecture, \
            f"Architecture should match actual system ({actual_architecture})"
    
    def test_cpu_count_validity(self):
        """Test that CPU count is reasonable."""
        result = system_info()
        cpu_count = result['cpu_count']
        
        assert isinstance(cpu_count, int), "CPU count should be an integer"
        assert cpu_count > 0, "CPU count should be positive"
        assert cpu_count <= 128, "CPU count should be reasonable (max 128 cores)"
    
    def test_cpu_count_accuracy(self):
        """Test that CPU count matches actual system."""
        result = system_info()
        actual_cpu_count = psutil.cpu_count()
        
        assert result['cpu_count'] == actual_cpu_count, \
            f"CPU count should match actual system ({actual_cpu_count})"
    
    def test_memory_format(self):
        """Test that memory is in GB and reasonable."""
        result = system_info()
        memory_gb = result['memory_gb']
        
        assert isinstance(memory_gb, (int, float)), "Memory should be a number"
        assert memory_gb > 0, "Memory should be positive"
        assert memory_gb <= 1000, "Memory should be reasonable (max 1000 GB)"
    
    def test_memory_accuracy(self):
        """Test that memory matches actual system."""
        result = system_info()
        actual_memory_bytes = psutil.virtual_memory().total
        actual_memory_gb = round(actual_memory_bytes / (1024**3), 1)
        
        assert abs(result['memory_gb'] - actual_memory_gb) < 0.1, \
            f"Memory should match actual system ({actual_memory_gb} GB)"
    
    def test_data_types(self):
        """Test that all data types are correct."""
        result = system_info()
        
        assert isinstance(result['python_version'], str), "python_version should be string"
        assert isinstance(result['platform'], str), "platform should be string"
        assert isinstance(result['architecture'], str), "architecture should be string"
        assert isinstance(result['cpu_count'], int), "cpu_count should be integer"
        assert isinstance(result['memory_gb'], (int, float)), "memory_gb should be number"


class TestIntegration:
    """Test integration with other modules and real-world scenarios."""
    
    def test_import_from_package(self):
        """Test that functions can be imported from tinytorch package."""
        try:
            from tinytorch.core.setup import personal_info, system_info
            assert callable(personal_info)
            assert callable(system_info)
        except ImportError as e:
            pytest.fail(f"Setup functions should be importable from tinytorch package: {e}")
    
    def test_function_signatures(self):
        """Test that function signatures match expected types."""
        import inspect
        
        # Test personal_info signature
        sig = inspect.signature(personal_info)
        sig_str = str(sig.return_annotation)
        assert 'Dict[str, str]' in sig_str or 'typing.Dict[str, str]' in sig_str, \
            f"personal_info should return Dict[str, str], got {sig_str}"
        
        # Test system_info signature
        sig = inspect.signature(system_info)
        sig_str = str(sig.return_annotation)
        assert 'Dict[str, Any]' in sig_str or 'typing.Dict[str, typing.Any]' in sig_str, \
            f"system_info should return Dict[str, Any], got {sig_str}"
    
    def test_no_side_effects(self):
        """Test that functions don't have side effects."""
        # Call functions multiple times
        result1 = personal_info()
        result2 = personal_info()
        result3 = system_info()
        result4 = system_info()
        
        # Results should be consistent
        assert result1 == result2, "personal_info should be deterministic"
        assert result3 == result4, "system_info should be deterministic"
    
    def test_performance(self):
        """Test that functions complete quickly."""
        import time
        
        # Test personal_info performance
        start_time = time.time()
        personal_info()
        personal_time = time.time() - start_time
        assert personal_time < 0.1, "personal_info should complete in under 0.1 seconds"
        
        # Test system_info performance
        start_time = time.time()
        system_info()
        system_time = time.time() - start_time
        assert system_time < 0.5, "system_info should complete in under 0.5 seconds"
    
    def test_memory_usage(self):
        """Test that functions don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Call functions multiple times
        for _ in range(10):
            personal_info()
            system_info()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 1MB)
        assert memory_increase < 1024 * 1024, \
            f"Functions should not consume excessive memory (increase: {memory_increase} bytes)"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_robust_to_import_errors(self):
        """Test that functions handle missing dependencies gracefully."""
        # This test would require mocking import errors
        # For now, we assume the functions work with standard library modules
        pass
    
    def test_consistent_output_format(self):
        """Test that output format is consistent across calls."""
        result1 = personal_info()
        result2 = personal_info()
        
        # Keys should be the same
        assert set(result1.keys()) == set(result2.keys()), \
            "personal_info should return consistent keys"
        
        result3 = system_info()
        result4 = system_info()
        
        # Keys should be the same
        assert set(result3.keys()) == set(result4.keys()), \
            "system_info should return consistent keys"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"]) 
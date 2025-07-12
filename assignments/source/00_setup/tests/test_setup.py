"""
Test suite for the setup module.
This tests the student implementations to ensure they work correctly.
"""

import pytest
import sys
from pathlib import Path

# Import from the main package
from tinytorch.core.utils import hello_tinytorch, add_numbers, SystemInfo, DeveloperProfile, complex_calculation


class TestSetupFunctions:
    """Test setup module functions."""
    
    def test_hello_tinytorch_executes(self):
        """Test that hello_tinytorch runs without error."""
        # Should not raise any exceptions
        hello_tinytorch()
    
    def test_hello_tinytorch_prints_content(self, capsys):
        """Test that hello_tinytorch prints the expected content."""
        hello_tinytorch()
        captured = capsys.readouterr()
        
        # Should print the branding text
        assert "TinyTorch" in captured.out
        assert "Build ML Systems from Scratch!" in captured.out
    
    def test_complex_calculation_basic(self):
        """Test multi-step calculation with multiple solution blocks."""
        # Test the example from the assignment: complex_calculation(3, 4)
        # Step 1: a_plus_2 = 3+2 = 5, b_plus_2 = 4+2 = 6
        # Step 2: everything_summed = 5+6 = 11
        # Step 3: everything_summed_times_10 = 11*10 = 110
        result = complex_calculation(3, 4)
        expected = 110
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_complex_calculation_different_inputs(self):
        """Test multi-step calculation with different inputs."""
        # Test with different numbers
        result = complex_calculation(1, 2)
        # Step 1: 1+2=3, 2+2=4
        # Step 2: 3+4=7
        # Step 3: 7*10=70
        expected = 70
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_complex_calculation_negative(self):
        """Test multi-step calculation with negative numbers."""
        result = complex_calculation(-1, -2)
        # Step 1: -1+2=1, -2+2=0
        # Step 2: 1+0=1
        # Step 3: 1*10=10
        expected = 10
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_add_numbers_basic(self):
        """Test basic addition functionality."""
        assert add_numbers(2, 3) == 5
        assert add_numbers(10, 15) == 25
        assert add_numbers(0, 0) == 0
    
    def test_add_numbers_negative(self):
        """Test addition with negative numbers."""
        assert add_numbers(-5, 3) == -2
        assert add_numbers(-10, -15) == -25
        assert add_numbers(10, -5) == 5
    
    def test_add_numbers_floats(self):
        """Test addition with floating point numbers."""
        assert abs(add_numbers(2.5, 3.7) - 6.2) < 1e-9
        assert abs(add_numbers(1.1, 2.2) - 3.3) < 1e-9


class TestSystemInfo:
    """Test SystemInfo class."""
    
    def test_system_info_creation(self):
        """Test SystemInfo class instantiation."""
        info = SystemInfo()
        assert hasattr(info, 'python_version')
        assert hasattr(info, 'platform')
        assert hasattr(info, 'machine')
    
    def test_system_info_properties(self):
        """Test SystemInfo properties."""
        info = SystemInfo()
        
        # Check python_version is a string
        assert isinstance(info.python_version, str)
        assert len(info.python_version) > 0
        
        # Check platform is a string
        assert isinstance(info.platform, str)
        assert len(info.platform) > 0
        
        # Check machine is a string
        assert isinstance(info.machine, str)
        assert len(info.machine) > 0
    
    def test_system_info_str(self):
        """Test SystemInfo string representation."""
        info = SystemInfo()
        str_repr = str(info)
        
        assert isinstance(str_repr, str)
        assert "Python" in str_repr
        assert info.python_version in str_repr
        assert info.platform in str_repr
        assert info.machine in str_repr
    
    def test_is_compatible(self):
        """Test SystemInfo compatibility check."""
        info = SystemInfo()
        compatible = info.is_compatible()
        
        # Should return a boolean
        assert isinstance(compatible, bool)
        
        # Since we're running this test, Python should be >= 3.8
        assert compatible is True


class TestDeveloperProfile:
    """Test DeveloperProfile class."""
    
    def test_developer_profile_creation_defaults(self):
        """Test DeveloperProfile with default values."""
        profile = DeveloperProfile()
        
        # Check default values
        assert profile.name == "Student"
        assert profile.email == "student@example.com"
        assert profile.affiliation == "TinyTorch Community"
        assert profile.specialization == "ML Systems"
    
    def test_developer_profile_creation_custom(self):
        """Test DeveloperProfile with custom values."""
        profile = DeveloperProfile(
            name="Test Student",
            email="test@example.com",
            affiliation="Test University",
            specialization="Deep Learning"
        )
        
        assert profile.name == "Test Student"
        assert profile.email == "test@example.com"
        assert profile.affiliation == "Test University"
        assert profile.specialization == "Deep Learning"
    
    def test_developer_profile_str(self):
        """Test DeveloperProfile string representation."""
        profile = DeveloperProfile("Alice", "alice@example.com")
        str_repr = str(profile)
        
        assert isinstance(str_repr, str)
        assert "Alice" in str_repr
        assert "alice@example.com" in str_repr
    
    def test_developer_profile_signature(self):
        """Test DeveloperProfile signature method."""
        profile = DeveloperProfile("Bob", "bob@example.com", "Test University", "Neural Networks")
        signature = profile.get_signature()
        
        assert isinstance(signature, str)
        assert "Bob" in signature
        assert "Test University" in signature
        assert "Neural Networks" in signature
    
    def test_developer_profile_info(self):
        """Test DeveloperProfile get_profile_info method."""
        profile = DeveloperProfile("Charlie", "charlie@example.com", "AI Lab", "Computer Vision")
        info = profile.get_profile_info()
        
        assert isinstance(info, dict)
        assert info['name'] == "Charlie"
        assert info['email'] == "charlie@example.com"
        assert info['affiliation'] == "AI Lab"
        assert info['specialization'] == "Computer Vision"


class TestFileOperations:
    """Test file-related operations."""
    
    def test_hello_tinytorch_handles_missing_file(self, monkeypatch, capsys):
        """Test that hello_tinytorch handles missing ASCII art file gracefully."""
        # Mock Path.exists to return False
        def mock_exists(self):
            return False
        
        monkeypatch.setattr(Path, "exists", mock_exists)
        
        # Should still work without the file
        hello_tinytorch()
        captured = capsys.readouterr()
        
        # Should still print the branding text
        assert "TinyTorch" in captured.out
        assert "Build ML Systems from Scratch!" in captured.out


class TestModuleIntegration:
    """Test integration between different parts of the setup module."""
    
    def test_all_functions_work_together(self):
        """Test that all setup functions work without conflicts."""
        # Test functions
        hello_tinytorch()  # Should not raise
        sum_result = add_numbers(5, 10)
        calc_result = complex_calculation(1, 1)
        
        # Test classes
        info = SystemInfo()
        profile = DeveloperProfile()
        
        # All should work without errors
        assert sum_result == 15
        assert calc_result == 60  # (1+2) + (1+2) = 6, 6*10 = 60
        assert str(info)  # Should not be empty
        assert str(profile)  # Should not be empty
        assert profile.get_signature()  # Should not be empty
    
    def test_no_import_errors(self):
        """Test that imports work correctly."""
        # If we got here, imports worked
        assert callable(hello_tinytorch)
        assert callable(add_numbers)
        assert callable(complex_calculation)
        assert callable(SystemInfo)
        assert callable(DeveloperProfile) 
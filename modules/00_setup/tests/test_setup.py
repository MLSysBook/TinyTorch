"""
Test suite for the setup module.
This tests the student implementations to ensure they work correctly.
"""

import pytest
import numpy as np
import sys
import os

# Import from the main package (rock solid foundation)
from tinytorch.core.utils import hello_tinytorch, add_numbers, SystemInfo, DeveloperProfile


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
        assert "Tiny🔥Torch" in captured.out
        assert "Build ML Systems from Scratch!" in captured.out
    
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
        
        # Check python_version is a version tuple
        assert hasattr(info.python_version, 'major')
        assert hasattr(info.python_version, 'minor')
        assert isinstance(info.python_version.major, int)
        assert isinstance(info.python_version.minor, int)
        
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
        assert str(info.python_version.major) in str_repr
        assert str(info.python_version.minor) in str_repr
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
        assert profile.name == "Vijay Janapa Reddi"
        assert profile.affiliation == "Harvard University"
        assert profile.email == "vj@eecs.harvard.edu"
        assert profile.github_username == "profvjreddi"
        assert profile.ascii_art is not None  # Should have default flame
    
    def test_developer_profile_creation_custom(self):
        """Test DeveloperProfile with custom values."""
        custom_art = """
        Custom ASCII Art
        ****************
        """
        profile = DeveloperProfile(
            name="Test Student",
            affiliation="Test University",
            email="test@example.com",
            github_username="teststudent",
            ascii_art=custom_art
        )
        
        assert profile.name == "Test Student"
        assert profile.affiliation == "Test University"
        assert profile.email == "test@example.com"
        assert profile.github_username == "teststudent"
        assert profile.ascii_art == custom_art
    
    def test_developer_profile_str(self):
        """Test DeveloperProfile string representation."""
        profile = DeveloperProfile()
        str_repr = str(profile)
        
        assert isinstance(str_repr, str)
        assert "👨‍💻" in str_repr
        assert "Vijay Janapa Reddi" in str_repr
        assert "Harvard University" in str_repr
        assert "@profvjreddi" in str_repr
    
    def test_developer_profile_signature(self):
        """Test DeveloperProfile signature method."""
        profile = DeveloperProfile()
        signature = profile.get_signature()
        
        assert isinstance(signature, str)
        assert "Built by" in signature
        assert "Vijay Janapa Reddi" in signature
        assert "@profvjreddi" in signature
    
    def test_developer_profile_ascii_art(self):
        """Test DeveloperProfile ASCII art functionality."""
        # Test default ASCII art
        profile = DeveloperProfile()
        ascii_art = profile.get_ascii_art()
        
        assert isinstance(ascii_art, str)
        assert "Tiny🔥Torch" in ascii_art
        assert "Build ML Systems from Scratch!" in ascii_art
        assert len(ascii_art) > 100  # Should be substantial ASCII art
        
        # Test custom ASCII art
        custom_art = "My Custom Art!"
        custom_profile = DeveloperProfile(ascii_art=custom_art)
        assert custom_profile.get_ascii_art() == custom_art
    
    def test_developer_profile_full_profile(self):
        """Test DeveloperProfile full profile display."""
        profile = DeveloperProfile()
        full_profile = profile.get_full_profile()
        
        assert isinstance(full_profile, str)
        assert "Tiny🔥Torch" in full_profile
        assert "Build ML Systems from Scratch!" in full_profile
        assert "👨‍💻 Developer: Vijay Janapa Reddi" in full_profile
        assert "🏛️  Affiliation: Harvard University" in full_profile
        assert "📧 Email: vj@eecs.harvard.edu" in full_profile
        assert "🐙 GitHub: @profvjreddi" in full_profile
        assert "🔥 Ready to build ML systems from scratch!" in full_profile


class TestFileOperations:
    """Test file-related operations."""
    
    def test_ascii_art_file_exists(self):
        """Test that the ASCII art file exists."""
        art_file = Path(__file__).parent.parent / "tinytorch_flame.txt"
        assert art_file.exists(), "ASCII art file should exist"
        assert art_file.is_file(), "ASCII art should be a file"
    
    def test_ascii_art_file_has_content(self):
        """Test that the ASCII art file has content."""
        art_file = Path(__file__).parent.parent / "tinytorch_flame.txt"
        content = art_file.read_text()
        
        assert len(content) > 0, "ASCII art file should not be empty"
        assert len(content.splitlines()) > 10, "ASCII art should have multiple lines"
    
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
        assert "🔥 TinyTorch 🔥" in captured.out
        assert "Build ML Systems from Scratch!" in captured.out


class TestModuleIntegration:
    """Test integration between different parts of the setup module."""
    
    def test_all_functions_work_together(self):
        """Test that all setup functions work without conflicts."""
        # Test functions
        hello_tinytorch()  # Should not raise
        sum_result = add_numbers(5, 10)
        
        # Test classes
        info = SystemInfo()
        profile = DeveloperProfile()
        
        # All should work without errors
        assert sum_result == 15
        assert str(info)  # Should not be empty
        assert str(profile)  # Should not be empty
        assert profile.get_signature()  # Should not be empty
        assert profile.get_ascii_art()  # Should not be empty
    
    def test_no_import_errors(self):
        """Test that imports work correctly."""
        # If we got here, imports worked
        assert callable(hello_tinytorch)
        assert callable(add_numbers)
        assert callable(SystemInfo)
        assert callable(DeveloperProfile) 
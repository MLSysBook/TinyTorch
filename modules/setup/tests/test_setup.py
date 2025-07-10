"""
Tests for TinyTorch Setup module.

Tests the basic setup functionality including hello function,
arithmetic operations, and system information class.
"""

import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from the module's development file
from setup_dev import hello_tinytorch, add_numbers, SystemInfo, DeveloperProfile


def approx_equal(a, b, tolerance=1e-9):
    """Simple approximation check without pytest."""
    return abs(a - b) < tolerance


class TestSetupFunctions:
    """Test setup module functions."""
    
    def test_hello_tinytorch(self):
        """Test hello_tinytorch function."""
        result = hello_tinytorch()
        assert isinstance(result, str)
        assert "TinyTorch" in result
        assert "🔥" in result  # Should have fire emoji
        assert len(result) > 10  # Should be a meaningful message
    
    def test_add_numbers(self):
        """Test add_numbers function."""
        # Test positive numbers
        assert add_numbers(2, 3) == 5
        assert add_numbers(10, 15) == 25
        
        # Test with zero
        assert add_numbers(0, 5) == 5
        assert add_numbers(7, 0) == 7
        assert add_numbers(0, 0) == 0
        
        # Test negative numbers
        assert add_numbers(-5, 3) == -2
        assert add_numbers(-10, -15) == -25
        assert add_numbers(10, -5) == 5
        
        # Test floats
        assert approx_equal(add_numbers(2.5, 3.7), 6.2)
        assert approx_equal(add_numbers(1.1, 2.2), 3.3)


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
        # (This is a reasonable assumption for TinyTorch)
        assert compatible is True
    
    def test_compatibility_logic(self):
        """Test the compatibility logic more thoroughly."""
        info = SystemInfo()
        
        # Current Python version should be compatible
        current_version = info.python_version
        assert current_version >= (3, 8)
        
        # The is_compatible method should return True for current version
        assert info.is_compatible() is True


class TestDeveloperProfile:
    """Test DeveloperProfile class."""
    
    def test_developer_profile_creation_defaults(self):
        """Test DeveloperProfile with default values."""
        profile = DeveloperProfile()
        
        # Check default values
        assert profile.name == "Vijay Janapa Reddi"
        assert profile.affiliation == "Harvard University"
        assert profile.email == "vijay@seas.harvard.edu"
        assert profile.github_username == "vjreddi"
    
    def test_developer_profile_creation_custom(self):
        """Test DeveloperProfile with custom values."""
        profile = DeveloperProfile(
            name="Test Student",
            affiliation="Test University",
            email="test@example.com",
            github_username="teststudent"
        )
        
        assert profile.name == "Test Student"
        assert profile.affiliation == "Test University"
        assert profile.email == "test@example.com"
        assert profile.github_username == "teststudent"
    
    def test_developer_profile_str(self):
        """Test DeveloperProfile string representation."""
        profile = DeveloperProfile()
        str_repr = str(profile)
        
        assert isinstance(str_repr, str)
        assert "👨‍💻" in str_repr
        assert "Vijay Janapa Reddi" in str_repr
        assert "Harvard University" in str_repr
        assert "@vjreddi" in str_repr
    
    def test_developer_profile_signature(self):
        """Test DeveloperProfile signature method."""
        profile = DeveloperProfile()
        signature = profile.get_signature()
        
        assert isinstance(signature, str)
        assert "Built by" in signature
        assert "Vijay Janapa Reddi" in signature
        assert "@vjreddi" in signature
    
    def test_developer_profile_custom_signature(self):
        """Test DeveloperProfile signature with custom values."""
        profile = DeveloperProfile(
            name="Jane Doe",
            github_username="janedoe"
        )
        signature = profile.get_signature()
        
        assert "Built by Jane Doe (@janedoe)" == signature
    
    def test_developer_profile_partial_customization(self):
        """Test DeveloperProfile with partial customization."""
        profile = DeveloperProfile(
            name="Custom Name",
            github_username="customuser"
        )
        
        # Custom values should be set
        assert profile.name == "Custom Name"
        assert profile.github_username == "customuser"
        
        # Defaults should remain
        assert profile.affiliation == "Harvard University"
        assert profile.email == "vijay@seas.harvard.edu"


class TestModuleIntegration:
    """Test integration between different parts of the setup module."""
    
    def test_all_functions_work_together(self):
        """Test that all setup functions work without conflicts."""
        # Test functions
        hello_msg = hello_tinytorch()
        sum_result = add_numbers(5, 10)
        
        # Test class
        info = SystemInfo()
        info_str = str(info)
        is_compat = info.is_compatible()
        
        # All should work without errors
        assert isinstance(hello_msg, str)
        assert sum_result == 15
        assert isinstance(info_str, str)
        assert isinstance(is_compat, bool)
    
    def test_no_import_errors(self):
        """Test that imports work correctly."""
        # If we got here, imports worked
        assert callable(hello_tinytorch)
        assert callable(add_numbers)
        assert callable(SystemInfo)


def run_setup_tests():
    """
    Run all setup tests without pytest.
    """
    print("🧪 Running Setup Module Tests...")
    print()
    
    # Test functions
    test_functions = TestSetupFunctions()
    tests = [
        ("test_hello_tinytorch", test_functions.test_hello_tinytorch),
        ("test_add_numbers", test_functions.test_add_numbers),
    ]
    
    # Test SystemInfo class
    test_system_info = TestSystemInfo()
    tests.extend([
        ("test_system_info_creation", test_system_info.test_system_info_creation),
        ("test_system_info_properties", test_system_info.test_system_info_properties),
        ("test_system_info_str", test_system_info.test_system_info_str),
        ("test_is_compatible", test_system_info.test_is_compatible),
        ("test_compatibility_logic", test_system_info.test_compatibility_logic),
    ])
    
    # Test DeveloperProfile class
    test_developer_profile = TestDeveloperProfile()
    tests.extend([
        ("test_developer_profile_creation_defaults", test_developer_profile.test_developer_profile_creation_defaults),
        ("test_developer_profile_creation_custom", test_developer_profile.test_developer_profile_creation_custom),
        ("test_developer_profile_str", test_developer_profile.test_developer_profile_str),
        ("test_developer_profile_signature", test_developer_profile.test_developer_profile_signature),
        ("test_developer_profile_custom_signature", test_developer_profile.test_developer_profile_custom_signature),
        ("test_developer_profile_partial_customization", test_developer_profile.test_developer_profile_partial_customization),
    ])

    # Test integration
    test_integration = TestModuleIntegration()
    tests.extend([
        ("test_all_functions_work_together", test_integration.test_all_functions_work_together),
        ("test_no_import_errors", test_integration.test_no_import_errors),
    ])
    
    # Run all tests
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            failed += 1
    
    print()
    print(f"🎉 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Setup Module Tests PASSED!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    # Run tests if script is executed directly
    success = run_setup_tests()
    sys.exit(0 if success else 1) 
"""
Tests for Module 0: Setup & Environment

These tests validate the exported functions from modules/setup/setup_dev.ipynb
"""

import pytest
from tinytorch.core.utils import hello_tinytorch, add_numbers, SystemInfo


def test_hello_tinytorch_exists():
    """Test that the hello_tinytorch function exists and can be imported."""
    assert callable(hello_tinytorch), "hello_tinytorch should be a callable function"


def test_hello_tinytorch_returns_string():
    """Test that hello_tinytorch returns a string."""
    result = hello_tinytorch()
    assert isinstance(result, str), f"hello_tinytorch() should return a string, got {type(result)}"


def test_hello_tinytorch_not_empty():
    """Test that hello_tinytorch returns a non-empty string."""
    result = hello_tinytorch()
    assert len(result.strip()) > 0, "hello_tinytorch() should return a non-empty string"


def test_hello_tinytorch_contains_tinytorch():
    """Test that the greeting contains TinyTorch."""
    result = hello_tinytorch().lower()
    assert 'tinytorch' in result, f"hello_tinytorch() should contain 'TinyTorch'. Got: {hello_tinytorch()}"


def test_hello_tinytorch_has_fire_emoji():
    """Test that the greeting contains the fire emoji (matching brand)."""
    result = hello_tinytorch()
    assert '🔥' in result, f"hello_tinytorch() should contain the 🔥 emoji to match TinyTorch branding. Got: {result}"


def test_add_numbers_basic():
    """Test the add_numbers function with basic inputs."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, 1) == 0
    assert add_numbers(10, -5) == 5


def test_add_numbers_float():
    """Test add_numbers with floating point numbers."""
    assert abs(add_numbers(1.5, 2.5) - 4.0) < 1e-10
    assert abs(add_numbers(0.1, 0.2) - 0.3) < 1e-10


def test_system_info_creation():
    """Test that SystemInfo can be created."""
    info = SystemInfo()
    assert info is not None
    assert hasattr(info, 'python_version')
    assert hasattr(info, 'platform') 
    assert hasattr(info, 'machine')


def test_system_info_string_representation():
    """Test that SystemInfo has a proper string representation."""
    info = SystemInfo()
    info_str = str(info)
    assert isinstance(info_str, str)
    assert len(info_str) > 0
    assert 'Python' in info_str


def test_system_info_compatibility():
    """Test the is_compatible method."""
    info = SystemInfo()
    # Should be compatible since we're running this test
    assert info.is_compatible() == True


def test_function_docstrings():
    """Test that exported functions have proper docstrings."""
    assert hello_tinytorch.__doc__ is not None, "hello_tinytorch should have a docstring"
    assert add_numbers.__doc__ is not None, "add_numbers should have a docstring"
    
    # Check that docstrings are meaningful (not just empty)
    assert len(hello_tinytorch.__doc__.strip()) > 5, "Docstring should be meaningful"
    assert len(add_numbers.__doc__.strip()) > 5, "Docstring should be meaningful" 
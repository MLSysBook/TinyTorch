"""
Tests for Module 01: Setup & Environment

These tests validate the exported functions from notebooks/01_setup.ipynb
"""

import pytest
from tinytorch.core.utils import hello_tinytorch, format_tensor_shape, validate_tensor_shapes


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


def test_hello_tinytorch_contains_welcome():
    """Test that the greeting contains welcoming content."""
    result = hello_tinytorch().lower()
    welcome_words = ['welcome', 'hello', 'tinytorch', 'ready']
    
    assert any(word in result for word in welcome_words), \
        f"hello_tinytorch() should contain welcoming content. Got: {hello_tinytorch()}"


def test_hello_tinytorch_has_fire_emoji():
    """Test that the greeting contains the fire emoji (matching brand)."""
    result = hello_tinytorch()
    assert '🔥' in result, \
        f"hello_tinytorch() should contain the 🔥 emoji to match TinyTorch branding. Got: {result}"


def test_format_tensor_shape():
    """Test the format_tensor_shape utility function."""
    # Test various shapes
    assert format_tensor_shape((3, 4, 5)) == "(3, 4, 5)"
    assert format_tensor_shape((1,)) == "(1)"
    assert format_tensor_shape(()) == "()"
    assert format_tensor_shape((10, 20)) == "(10, 20)"


def test_validate_tensor_shapes():
    """Test the validate_tensor_shapes utility function."""
    # Compatible shapes
    assert validate_tensor_shapes((3, 4), (3, 4)) == True
    assert validate_tensor_shapes((1, 2, 3), (1, 2, 3), (1, 2, 3)) == True
    
    # Incompatible shapes
    assert validate_tensor_shapes((3, 4), (2, 4)) == False
    assert validate_tensor_shapes((1, 2), (1, 3)) == False
    
    # Edge cases
    assert validate_tensor_shapes((3, 4)) == True  # Single shape
    assert validate_tensor_shapes() == True  # No shapes


def test_function_docstrings():
    """Test that exported functions have proper docstrings."""
    assert hello_tinytorch.__doc__ is not None, "hello_tinytorch should have a docstring"
    assert format_tensor_shape.__doc__ is not None, "format_tensor_shape should have a docstring" 
    assert validate_tensor_shapes.__doc__ is not None, "validate_tensor_shapes should have a docstring"
    
    # Check that docstrings are meaningful (not just empty)
    assert len(hello_tinytorch.__doc__.strip()) > 10, "Docstring should be meaningful"
    assert len(format_tensor_shape.__doc__.strip()) > 10, "Docstring should be meaningful"
    assert len(validate_tensor_shapes.__doc__.strip()) > 10, "Docstring should be meaningful" 
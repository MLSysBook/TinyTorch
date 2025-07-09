#!/usr/bin/env python3
"""
Setup Project Tests

This file tests the student's implementation of the hello_tinytorch() function.
Students can run: python -m pytest projects/setup/test_setup.py
"""

import sys
import os
import pytest

# Add the project root to Python path so we can import tinytorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_hello_tinytorch_exists():
    """Test that the hello_tinytorch function exists and can be imported."""
    try:
        from tinytorch.core.utils import hello_tinytorch
    except ImportError:
        pytest.fail("Could not import hello_tinytorch from tinytorch.core.utils. "
                   "Make sure you've implemented the function!")

def test_hello_tinytorch_returns_string():
    """Test that hello_tinytorch returns a string."""
    from tinytorch.core.utils import hello_tinytorch
    
    result = hello_tinytorch()
    assert isinstance(result, str), f"hello_tinytorch() should return a string, got {type(result)}"

def test_hello_tinytorch_not_empty():
    """Test that hello_tinytorch returns a non-empty string."""
    from tinytorch.core.utils import hello_tinytorch
    
    result = hello_tinytorch()
    assert len(result.strip()) > 0, "hello_tinytorch() should return a non-empty string"

def test_hello_tinytorch_contains_welcome():
    """Test that the greeting contains welcoming content."""
    from tinytorch.core.utils import hello_tinytorch
    
    result = hello_tinytorch().lower()
    # Should contain some welcoming words
    welcome_words = ['welcome', 'hello', 'tinytorch', 'ready']
    
    assert any(word in result for word in welcome_words), \
        f"hello_tinytorch() should contain welcoming content. Got: {hello_tinytorch()}"

def test_hello_tinytorch_has_fire_emoji():
    """Test that the greeting contains the fire emoji (matching brand)."""
    from tinytorch.core.utils import hello_tinytorch
    
    result = hello_tinytorch()
    assert '🔥' in result, \
        f"hello_tinytorch() should contain the 🔥 emoji to match TinyTorch branding. Got: {result}"

def test_hello_tinytorch_proper_format():
    """Test that the function has proper docstring and type hints."""
    from tinytorch.core.utils import hello_tinytorch
    import inspect
    
    # Check that function has a docstring
    assert hello_tinytorch.__doc__ is not None, \
        "hello_tinytorch() should have a docstring explaining what it does"
    
    # Check type annotations
    sig = inspect.signature(hello_tinytorch)
    assert sig.return_annotation == str, \
        "hello_tinytorch() should have return type annotation -> str"

if __name__ == "__main__":
    # Allow running this file directly for testing
    pytest.main([__file__, "-v"]) 
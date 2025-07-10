#!/usr/bin/env python3
"""
Tests for the setup module using pytest.
"""

import pytest
import sys
import os
from pathlib import Path
from io import StringIO

# Add the parent directory to the path so we can import setup_dev
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup_dev import hello_tinytorch


class TestSetupModule:
    """Test suite for the setup module."""
    
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
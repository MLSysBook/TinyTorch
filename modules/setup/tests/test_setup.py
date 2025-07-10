#!/usr/bin/env python3
"""
Simple tests for the setup module.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import setup_dev
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup_dev import hello_tinytorch

def test_hello_tinytorch():
    """Test that hello_tinytorch runs without error."""
    try:
        hello_tinytorch()
        print("✅ hello_tinytorch() executed successfully")
        return True
    except Exception as e:
        print(f"❌ hello_tinytorch() failed: {e}")
        return False

def test_ascii_art_file_exists():
    """Test that the ASCII art file exists."""
    art_file = Path(__file__).parent.parent / "tinytorch_flame.txt"
    if art_file.exists():
        print("✅ ASCII art file exists")
        return True
    else:
        print("❌ ASCII art file not found")
        return False

def run_tests():
    """Run all tests."""
    print("Running setup module tests...")
    print("=" * 50)
    
    tests = [
        test_hello_tinytorch,
        test_ascii_art_file_exists,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 
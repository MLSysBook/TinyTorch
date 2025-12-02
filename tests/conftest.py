"""
Pytest configuration for TinyTorch tests.

This file is automatically loaded by pytest and sets up the test environment.
"""

import sys
import os
from pathlib import Path

# Add tests directory to Python path so test_utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Add project root to Python path
project_root = tests_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set quiet mode for tinytorch imports during tests
os.environ['TINYTORCH_QUIET'] = '1'

# Import test utilities to make them available
try:
    from test_utils import setup_integration_test, create_test_tensor, assert_tensors_close
except ImportError:
    pass  # test_utils not yet created or has issues


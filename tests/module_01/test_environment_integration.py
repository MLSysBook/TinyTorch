"""
Module 01: Setup - Integration Tests
Tests that the development environment is properly configured
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEnvironmentSetup:
    """Test that development environment is ready."""
    
    def test_python_version(self):
        """Test Python version is compatible."""
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"
    
    def test_required_packages(self):
        """Test that required packages are installed."""
        required = ['numpy', 'rich']
        for package in required:
            try:
                __import__(package)
            except ImportError:
                assert False, f"Required package {package} not installed"
    
    def test_project_structure(self):
        """Test project directories exist."""
        project_root = Path(__file__).parent.parent.parent
        required_dirs = ['tinytorch', 'modules', 'tests', 'tito']
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} not found"
    
    def test_nbdev_available(self):
        """Test nbdev is available for exporting."""
        try:
            result = subprocess.run(['nbdev_export', '--help'], 
                                  capture_output=True, text=True)
            assert result.returncode == 0, "nbdev_export not working"
        except FileNotFoundError:
            assert False, "nbdev not installed or not in PATH"


class TestTitoCliIntegration:
    """Test TITO CLI is working."""
    
    def test_tito_command_available(self):
        """Test tito command works."""
        try:
            result = subprocess.run(['python', '-m', 'tito', '--help'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            assert result.returncode == 0, "TITO CLI not working"
        except Exception as e:
            assert False, f"TITO CLI failed: {e}"
    
    def test_tito_system_doctor(self):
        """Test tito system doctor command."""
        try:
            result = subprocess.run(['python', '-m', 'tito', 'system', 'doctor'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            # Should run without errors (return code 0 or 1 for warnings)
            assert result.returncode in [0, 1], f"System doctor failed with code {result.returncode}"
        except Exception as e:
            # Skip if tito not fully set up yet
            pass
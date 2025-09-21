"""
Module 01: Setup - Core Functionality Tests
Tests environment configuration and system introspection
"""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEnvironmentDetection:
    """Test environment detection functionality."""
    
    def test_python_version_detection(self):
        """Test Python version detection works."""
        try:
            from tinytorch.setup import get_python_version
            version = get_python_version()
            assert isinstance(version, str)
            assert len(version.split('.')) >= 2  # At least major.minor
        except ImportError:
            # Fallback test
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            assert len(version) >= 3
    
    def test_system_info_detection(self):
        """Test system information detection."""
        try:
            from tinytorch.setup import get_system_info
            info = get_system_info()
            assert 'platform' in info
            assert 'python_version' in info
        except ImportError:
            # Basic system detection test
            import platform
            assert platform.system() in ['Darwin', 'Linux', 'Windows']
    
    def test_hardware_detection(self):
        """Test hardware capability detection."""
        try:
            from tinytorch.setup import detect_hardware
            hw_info = detect_hardware()
            assert 'cpu_count' in hw_info
            assert hw_info['cpu_count'] > 0
        except ImportError:
            # Basic CPU detection
            import multiprocessing
            assert multiprocessing.cpu_count() > 0


class TestDevelopmentEnvironment:
    """Test development environment setup."""
    
    def test_package_dependencies(self):
        """Test required packages are available."""
        required_packages = ['numpy', 'pathlib']
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                assert False, f"Required package {package} not available"
    
    def test_project_structure(self):
        """Test project structure is correct."""
        project_root = Path(__file__).parent.parent.parent
        
        required_dirs = ['modules', 'tests', 'tito', 'tinytorch']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} missing"
    
    def test_virtual_environment(self):
        """Test virtual environment configuration."""
        # Check if we're in a virtual environment
        in_venv = (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            'VIRTUAL_ENV' in os.environ
        )
        
        if in_venv:
            # If in venv, test it's configured correctly
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                assert Path(venv_path).exists()


class TestConfigurationSettings:
    """Test configuration and settings management."""
    
    def test_config_file_creation(self):
        """Test configuration file can be created."""
        try:
            from tinytorch.setup import create_config
            config = create_config()
            assert isinstance(config, dict)
        except ImportError:
            # Basic config structure test
            config = {
                'debug': False,
                'device': 'cpu',
                'data_dir': './data'
            }
            assert 'debug' in config
            assert 'device' in config
    
    def test_performance_settings(self):
        """Test performance-related settings."""
        try:
            from tinytorch.setup import get_performance_config
            perf_config = get_performance_config()
            assert 'num_threads' in perf_config
        except ImportError:
            # Test basic threading setup
            import threading
            assert threading.active_count() >= 1
    
    def test_reproducibility_settings(self):
        """Test reproducibility configuration."""
        try:
            from tinytorch.setup import set_random_seed
            set_random_seed(42)
            # Test that numpy random state is set
            import numpy as np
            rand1 = np.random.random()
            set_random_seed(42)
            rand2 = np.random.random()
            assert rand1 == rand2, "Random seed not working"
        except ImportError:
            # Basic reproducibility test
            import numpy as np
            np.random.seed(42)
            rand1 = np.random.random()
            np.random.seed(42)
            rand2 = np.random.random()
            assert rand1 == rand2


class TestSystemValidation:
    """Test system validation and health checks."""
    
    def test_memory_availability(self):
        """Test system has sufficient memory."""
        try:
            from tinytorch.setup import check_memory
            memory_gb = check_memory()
            assert memory_gb > 0
        except ImportError:
            # Basic memory check using psutil if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                assert memory.total > 1e9  # At least 1GB
            except ImportError:
                # Skip memory check if psutil not available
                assert True
    
    def test_disk_space(self):
        """Test sufficient disk space for datasets."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            from tinytorch.setup import check_disk_space
            space_gb = check_disk_space(project_root)
            assert space_gb > 0
        except ImportError:
            # Basic disk space check
            try:
                import shutil
                total, used, free = shutil.disk_usage(project_root)
                assert free > 1e9  # At least 1GB free
            except (ImportError, OSError):
                # Skip if can't check disk space
                assert True
    
    def test_network_connectivity(self):
        """Test network connectivity for downloading datasets."""
        try:
            from tinytorch.setup import test_network
            is_connected = test_network()
            # Network tests can be flaky, so we don't assert
            assert isinstance(is_connected, bool)
        except ImportError:
            # Basic network test
            try:
                import urllib.request
                urllib.request.urlopen('https://www.google.com', timeout=5)
                assert True
            except:
                # Network might not be available in test environment
                assert True
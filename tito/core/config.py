"""
Configuration management for TinyTorch CLI.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class CLIConfig:
    """Configuration for TinyTorch CLI."""
    
    # Project paths
    project_root: Path
    assignments_dir: Path
    tinytorch_dir: Path
    bin_dir: Path
    modules_dir: Path  # Alias for assignments_dir
    
    # Environment settings
    python_min_version: tuple = (3, 8)
    required_packages: list = None  # type: ignore
    
    # CLI settings
    verbose: bool = False
    no_color: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.required_packages is None:
            self.required_packages = ['numpy', 'pytest', 'rich']
    
    @classmethod
    def from_project_root(cls, project_root: Optional[Path] = None) -> 'CLIConfig':
        """Create config from project root directory."""
        if project_root is None:
            # Auto-detect project root
            current = Path.cwd()
            while current != current.parent:
                if (current / 'pyproject.toml').exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()
        
        modules_path = project_root / 'assignments' / 'source'
        return cls(
            project_root=project_root,
            assignments_dir=modules_path,
            modules_dir=modules_path,  # Same as assignments_dir
            tinytorch_dir=project_root / 'tinytorch',
            bin_dir=project_root / 'bin'
        )
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Check Python version
        if sys.version_info < self.python_min_version:
            issues.append(f"Python {'.'.join(map(str, self.python_min_version))}+ required, "
                         f"found {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check virtual environment (more robust detection)
        in_venv = (
            # Method 1: Check VIRTUAL_ENV environment variable
            os.environ.get('VIRTUAL_ENV') is not None or
            # Method 2: Check sys.prefix vs sys.base_prefix
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            # Method 3: Check for sys.real_prefix (older Python versions)
            hasattr(sys, 'real_prefix') or
            # Method 4: Check if .venv directory exists and packages are available
            (Path('.venv').exists() and self._packages_available())
        )
        if not in_venv:
            issues.append("Virtual environment not activated. Run: source .venv/bin/activate")
        
        # Check required directories
        if not self.assignments_dir.exists():
            issues.append(f"Assignments directory not found: {self.assignments_dir}")
        
        if not self.tinytorch_dir.exists():
            issues.append(f"TinyTorch package not found: {self.tinytorch_dir}")
        
        # Check required packages
        for package in self.required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing dependency: {package}. Run: pip install -r requirements.txt")
        
        return issues
    
    def _packages_available(self) -> bool:
        """Check if required packages are available (helper for venv detection)."""
        try:
            for package in self.required_packages:
                __import__(package)
            return True
        except ImportError:
            return False 
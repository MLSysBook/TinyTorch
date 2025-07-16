"""
Configuration management for TinyTorch CLI.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CLIConfig:
    """Configuration for TinyTorch CLI."""
    
    # Project paths
    project_root: Path
    assignments_dir: Path
    tinytorch_dir: Path
    bin_dir: Path
    
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
        
        return cls(
            project_root=project_root,
            assignments_dir=project_root / 'modules' / 'source',
            tinytorch_dir=project_root / 'tinytorch',
            bin_dir=project_root / 'bin'
        )
    
    def validate(self) -> list[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Check Python version
        if sys.version_info < self.python_min_version:
            issues.append(f"Python {'.'.join(map(str, self.python_min_version))}+ required, "
                         f"found {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check virtual environment (skip in CI environments)
        # CI environments typically have CI=true or GITHUB_ACTIONS=true
        is_ci = os.environ.get('CI', 'false').lower() == 'true' or os.environ.get('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        if not is_ci:
            in_venv = (hasattr(sys, 'real_prefix') or 
                       (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
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
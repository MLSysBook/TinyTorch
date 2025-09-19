"""
User preferences management for TinyTorch CLI.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class UserPreferences:
    """User preferences for TinyTorch CLI."""
    
    # Logo preferences
    logo_theme: str = "standard"  # "standard" or "bright"
    
    # Future preferences can be added here
    # animation_enabled: bool = True
    # color_scheme: str = "auto"
    
    @classmethod
    def load_from_file(cls, config_file: Optional[Path] = None) -> 'UserPreferences':
        """Load preferences from config file."""
        if config_file is None:
            config_file = cls.get_default_config_path()
        
        if not config_file.exists():
            # Return defaults if no config file exists
            return cls()
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Create instance with loaded data, using defaults for missing keys
            return cls(**{
                key: data.get(key, getattr(cls(), key))
                for key in cls.__dataclass_fields__
            })
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Return defaults if config file is corrupted
            return cls()
    
    def save_to_file(self, config_file: Optional[Path] = None) -> None:
        """Save preferences to config file."""
        if config_file is None:
            config_file = self.get_default_config_path()
        
        # Ensure config directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @staticmethod
    def get_default_config_path() -> Path:
        """Get the default config file path."""
        # Look for project root first
        current = Path.cwd()
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                return current / '.tito' / 'config.json'
            current = current.parent
        
        # Fallback to current directory
        return Path.cwd() / '.tito' / 'config.json'
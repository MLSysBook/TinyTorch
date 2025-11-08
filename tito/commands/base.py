"""
Base command class for TinyTorch CLI.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import logging

from ..core.config import CLIConfig
from ..core.virtual_env_manager import get_venv_path
from ..core.console import get_console
from ..core.exceptions import TinyTorchCLIError

logger = logging.getLogger(__name__)

class BaseCommand(ABC):
    """Base class for all CLI commands."""
    
    def __init__(self, config: CLIConfig):
        """Initialize the command with configuration."""
        self.config = config
        self.console = get_console()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the command name."""
        pass

    @property
    def venv_path(self) -> Path:
        """Return the command name."""
        return get_venv_path()
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the command description."""
        pass
    
    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        pass
    
    @abstractmethod
    def run(self, args: Namespace) -> int:
        """Execute the command and return exit code."""
        pass
    
    def validate_args(self, args: Namespace) -> None:
        """Validate command arguments. Override in subclasses if needed."""
        pass
    
    def execute(self, args: Namespace) -> int:
        """Execute the command with error handling."""
        try:
            self.validate_args(args)
            return self.run(args)
        except TinyTorchCLIError as e:
            logger.error(f"Command failed: {e}")
            self.console.print(f"[red]❌ {e}[/red]")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error in command {self.name}")
            self.console.print(f"[red]❌ Unexpected error: {e}[/red]")
            return 1 
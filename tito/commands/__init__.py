"""
CLI Commands package.

Each command is implemented as a separate module with proper separation of concerns.
Commands are organized into logical groups: system, module, and package.
"""

from .base import BaseCommand

# Individual commands (for backward compatibility)
from .notebooks import NotebooksCommand
from .info import InfoCommand
from .test import TestCommand
from .doctor import DoctorCommand
from .sync import SyncCommand
from .reset import ResetCommand
from .jupyter import JupyterCommand
from .nbdev import NbdevCommand
from .status import StatusCommand

# Command groups
from .system import SystemCommand
from .module import ModuleCommand
from .package import PackageCommand

__all__ = [
    'BaseCommand',
    # Individual commands
    'NotebooksCommand',
    'InfoCommand',
    'TestCommand',
    'DoctorCommand',
    'SyncCommand',
    'ResetCommand',
    'JupyterCommand',
    'NbdevCommand',
    'StatusCommand',
    # Command groups
    'SystemCommand',
    'ModuleCommand',
    'PackageCommand',
] 
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
from .export import ExportCommand
from .reset import ResetCommand
from .jupyter import JupyterCommand
from .nbdev import NbdevCommand
from .status import StatusCommand
from .clean import CleanCommand
from .nbgrader import NBGraderCommand
from .book import BookCommand
from .benchmark import BenchmarkCommand
from .community import CommunityCommand

# Command groups
from .system import SystemCommand
from .module_workflow import ModuleWorkflowCommand
from .package import PackageCommand

__all__ = [
    'BaseCommand',
    # Individual commands
    'NotebooksCommand',
    'InfoCommand',
    'TestCommand',
    'DoctorCommand',
    'ExportCommand',
    'ResetCommand',
    'JupyterCommand',
    'NbdevCommand',
    'StatusCommand',
    'CleanCommand',
    'NBGraderCommand',
    'BookCommand',
    'BenchmarkCommand',
    'CommunityCommand',
    # Command groups
    'SystemCommand',
    'ModuleWorkflowCommand',
    'PackageCommand',
] 
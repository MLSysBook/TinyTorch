# TinyTorch CLI Architecture

## 🏗️ Senior Software Engineer Design Principles

This CLI follows industry-standard software engineering practices and patterns used in production systems.

## 📁 Architecture Overview

```
tinytorch/cli/
├── __init__.py              # Package initialization
├── main.py                  # Professional CLI entry point
├── core/                    # Core CLI functionality
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── console.py          # Centralized console output
│   └── exceptions.py       # Exception hierarchy
├── commands/               # Command implementations
│   ├── __init__.py
│   ├── base.py            # Base command class
│   └── notebooks.py       # Notebooks command
└── tools/                 # CLI tools
    ├── __init__.py
    └── py_to_notebook.py  # Conversion tool
```

## 🎯 Design Patterns Applied

### 1. **Command Pattern**
- Each CLI command is a separate class implementing `BaseCommand`
- Consistent interface for all commands
- Easy to add new commands without modifying existing code

### 2. **Dependency Injection**
- Commands receive configuration through constructor
- Testable and loosely coupled
- Easy to mock for testing

### 3. **Template Method Pattern**
- `BaseCommand` provides common functionality
- Subclasses implement specific behavior
- Consistent error handling across all commands

### 4. **Factory Pattern**
- Commands are registered and instantiated dynamically
- Easy to extend with new commands
- Clean separation of command creation and execution

### 5. **Single Responsibility Principle**
- Each module has one clear purpose
- Console output separated from business logic
- Configuration management isolated

## 🔧 Key Features

### Professional Error Handling
```python
class TinyTorchCLIError(Exception):
    """Base exception for all CLI errors."""
    pass

class ValidationError(TinyTorchCLIError):
    """Raised when validation fails."""
    pass
```

### Centralized Configuration
```python
@dataclass
class CLIConfig:
    """Configuration for TinyTorch CLI."""
    project_root: Path
    modules_dir: Path
    tinytorch_dir: Path
    # ... with validation and auto-detection
```

### Logging Integration
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tinytorch-cli.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
```

### Type Safety
- Full type hints throughout
- MyPy configuration for static type checking
- Runtime type validation where needed

## 🚀 Usage Examples

### Professional CLI Interface
```bash
# Modern help system
tito --help
tito notebooks --help

# Global options
tito --verbose notebooks --module tensor
tito --no-color notebooks --dry-run

# Advanced features
tito notebooks --force --module layers
```

### Programmatic Usage
```python
from tinytorch.cli.main import TinyTorchCLI

cli = TinyTorchCLI()
exit_code = cli.run(['notebooks', '--module', 'tensor'])
```

## 🧪 Testing Strategy

### Unit Testing
- Each command class can be tested independently
- Mock configuration and dependencies
- Test error conditions and edge cases

### Integration Testing
- Test full CLI workflows
- Validate configuration loading
- Test subprocess interactions

### Example Test Structure
```python
def test_notebooks_command():
    config = CLIConfig.from_project_root(test_project_root)
    command = NotebooksCommand(config)
    
    # Test with mock arguments
    args = Namespace(module='test', dry_run=True)
    result = command.run(args)
    
    assert result == 0
```

## 📦 Installation & Distribution

### Entry Points
```toml
[project.scripts]
tito = "tinytorch.cli.main:main"
py-to-notebook = "tinytorch.cli.tools.py_to_notebook:main"
```

### Professional Package Structure
- Proper `pyproject.toml` configuration
- Development dependencies separated
- Code quality tools configured (black, isort, mypy)
- Semantic versioning

## 🔄 Extensibility

### Adding New Commands
1. Create command class inheriting from `BaseCommand`
2. Implement required methods
3. Register in `main.py`
4. Add tests

```python
class NewCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "new-command"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--option', help='Command option')
    
    def run(self, args: Namespace) -> int:
        # Implementation here
        return 0
```

## 📊 Benefits Achieved

### For Developers
- **Maintainable**: Clear separation of concerns
- **Testable**: Dependency injection and interfaces
- **Extensible**: Easy to add new features
- **Debuggable**: Proper logging and error handling

### For Users
- **Reliable**: Robust error handling and validation
- **Consistent**: Uniform interface across all commands
- **Helpful**: Clear error messages and help text
- **Fast**: Efficient execution with proper resource management

### For DevOps
- **Installable**: Proper package structure
- **Configurable**: Environment-aware configuration
- **Monitorable**: Comprehensive logging
- **Deployable**: Professional packaging and distribution

## 🎓 Learning Outcomes

This architecture demonstrates:
- How to structure a professional CLI application
- Industry-standard Python packaging
- Design patterns in practice
- Error handling best practices
- Testing strategies for CLI applications
- Configuration management patterns
- Logging and monitoring integration

## 🔮 Future Enhancements

- Plugin system for third-party commands
- Configuration file support (YAML/TOML)
- Shell completion support
- Progress bars for long-running operations
- Parallel command execution
- Remote command execution
- API integration capabilities

This architecture provides a solid foundation for scaling the CLI to enterprise-level requirements while maintaining educational value and ease of use. 
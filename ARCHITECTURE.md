# TinyTorch Project Architecture

## 🏗️ Senior Software Engineer Design Principles

This project follows industry-standard software engineering practices with proper separation of concerns between the **ML framework** and **development tools**.

## 📁 Project Structure Overview

```
TinyTorch/
├── tinytorch/                   # 🧠 Core ML Framework
│   ├── core/                   # Core tensor operations, layers, etc.
│   ├── training/               # Training loops, optimizers
│   ├── models/                 # Pre-built model architectures
│   └── ...                     # Other ML framework components
├── tito/                       # 🔧 Development CLI Tool
│   ├── main.py                 # CLI entry point
│   ├── core/                   # CLI core functionality
│   ├── commands/               # CLI commands
│   └── tools/                  # CLI utilities
├── modules/                    # 📚 Educational modules
├── bin/                        # 🚀 Executable scripts
└── docs/                       # 📖 Documentation
```

## 🎯 Architectural Separation

### **TinyTorch Framework** (`tinytorch/`)
- **Purpose**: Core ML framework for production use
- **Dependencies**: Minimal (numpy, essential ML libraries)
- **Users**: Students, researchers, ML practitioners
- **Scope**: Tensors, layers, training, models, inference

### **Tito CLI Tool** (`tito/`)
- **Purpose**: Development and management tool for building the framework
- **Dependencies**: Rich CLI libraries, development tools
- **Users**: Course instructors, framework developers
- **Scope**: Module generation, testing, notebook conversion, project management

## 🔧 Tito CLI Architecture

```
tito/
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

### 1. **Separation of Concerns**
- **Framework**: Pure ML functionality
- **CLI**: Development and management tools
- **Modules**: Educational content and exercises

### 2. **Command Pattern**
- Each CLI command is a separate class implementing `BaseCommand`
- Consistent interface for all commands
- Easy to add new commands without modifying existing code

### 3. **Dependency Injection**
- Commands receive configuration through constructor
- Testable and loosely coupled
- Easy to mock for testing

### 4. **Single Responsibility Principle**
- Each module has one clear purpose
- CLI separated from framework logic
- Configuration management isolated

## 🚀 Usage Examples

### Framework Usage (Production)
```python
# Core ML framework usage
import tinytorch as tt

# Create tensors
x = tt.Tensor([1, 2, 3])
y = tt.Tensor([4, 5, 6])
z = x + y

# Build models
model = tt.Sequential([
    tt.Linear(784, 128),
    tt.ReLU(),
    tt.Linear(128, 10)
])
```

### CLI Tool Usage (Development)
```bash
# Development and management
tito notebooks --module tensor
tito test --module layers
tito sync --all
tito doctor
```

## 📦 Installation & Distribution

### Core Framework
```bash
pip install tinytorch
```

### Development Tools
```bash
pip install tinytorch[dev]  # Includes tito CLI
```

### Entry Points
```toml
[project.scripts]
tito = "tito.main:main"
```

## 🔄 Benefits of This Architecture

### For Framework Users
- **Clean API**: No CLI dependencies in core framework
- **Lightweight**: Minimal dependencies for production use
- **Focused**: Pure ML functionality without development noise

### For Framework Developers
- **Powerful Tools**: Rich CLI for development tasks
- **Maintainable**: Clear separation of concerns
- **Extensible**: Easy to add new development commands

### For Course Instructors
- **Management Tools**: CLI for course administration
- **Module Generation**: Automated notebook and exercise creation
- **Testing Infrastructure**: Comprehensive testing commands

## 🎓 Educational Benefits

### Clear Mental Model
- **tinytorch**: "The ML framework I'm learning"
- **tito**: "The tool that helps me build and manage the framework"
- **modules**: "The lessons and exercises"

### Professional Practice
- Shows how real software projects separate concerns
- Demonstrates proper package structure
- Teaches dependency management

## 🔮 Future Enhancements

### Framework (`tinytorch/`)
- Advanced tensor operations
- GPU acceleration
- Distributed training
- Model deployment utilities

### CLI Tool (`tito/`)
- Plugin system for custom commands
- Configuration file support
- Shell completion
- Remote development support

## 📊 Comparison: Before vs After

### Before (Mixed Architecture)
```
tinytorch/
├── core/           # ML framework
├── cli/            # CLI mixed in framework
└── ...
```
**Issues**: CLI dependencies pollute framework, unclear separation

### After (Clean Architecture)
```
tinytorch/          # Pure ML framework
tito/              # Development CLI tool
modules/           # Educational content
```
**Benefits**: Clean separation, focused dependencies, clear purpose

## 🎯 Key Takeaways

1. **Separation of Concerns**: Framework and tools are separate
2. **Dependency Management**: Core framework stays lightweight
3. **User Experience**: Clear distinction between using vs building
4. **Professional Practice**: Industry-standard project organization
5. **Educational Value**: Teaches proper software architecture

This architecture demonstrates how senior engineers structure complex projects with multiple concerns, ensuring each component has a clear purpose and minimal dependencies. 
# TinyTorch - Module-first ML Systems Course
# Makefile for convenience commands

.PHONY: help install sync test clean docs jupyter lab setup

# Default target
help:
	@echo "TinyTorch - Build ML Systems from Scratch"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  install     Install dependencies and setup environment"
	@echo "  sync        Export notebook code to Python package"
	@echo "  test        Run all tests"
	@echo "  test-setup  Run setup module tests"
	@echo "  clean       Clean notebook outputs and cache"
	@echo "  docs        Build documentation"
	@echo "  jupyter     Start Jupyter Lab"
	@echo "  info        Show system information"
	@echo "  doctor      Run environment diagnosis"
	@echo ""
	@echo "Development workflow:"
	@echo "  1. make jupyter     # Work in modules/[name]/[name].ipynb"
	@echo "  2. make sync        # Export to tinytorch package"
	@echo "  3. make test        # Run tests"
	@echo "  4. make docs        # Build docs (optional)"

# Install dependencies
install:
	@echo "🔧 Installing TinyTorch dependencies..."
	pip install -r requirements.txt
	pip install nbdev jupyter
	@echo "✅ Installation complete!"

# Export notebooks to Python package
sync:
	@echo "🔄 Exporting notebooks to Python package..."
	python bin/tito.py sync

# Run all tests
test:
	@echo "🧪 Running all tests..."
	python bin/tito.py test --all

# Run setup module tests specifically
test-setup:
	@echo "🧪 Running setup module tests..."
	python bin/tito.py test --module setup

# Clean notebook outputs and Python cache
clean:
	@echo "🧹 Cleaning notebook outputs and cache..."
	python bin/tito.py nbdev --clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Build documentation
docs:
	@echo "📚 Building documentation..."
	python bin/tito.py nbdev --build-docs

# Start Jupyter Lab
jupyter:
	@echo "🚀 Starting Jupyter Lab..."
	python bin/tito.py jupyter --lab

# Alias for jupyter
lab: jupyter

# Show system information
info:
	@echo "ℹ️ System information..."
	python bin/tito.py info

# Run environment diagnosis
doctor:
	@echo "🔬 Running environment diagnosis..."
	python bin/tito.py doctor

# Setup new environment (for first-time users)
setup:
	@echo "🚀 Setting up TinyTorch development environment..."
	@echo "1. Creating virtual environment..."
	python -m venv .venv
	@echo "2. Installing dependencies..."
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install nbdev jupyter
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate the environment: source .venv/bin/activate"
	@echo "  2. Start coding: make jupyter"
	@echo "  3. Test your code: make sync && make test" 
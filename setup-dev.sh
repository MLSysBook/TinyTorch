#!/bin/bash
# TinyTorch Development Environment Setup Script
# This script sets up a complete development environment for TinyTorch

set -e  # Exit on any error

echo "🔧 Setting up TinyTorch development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
print_status "Checking Python installation..."
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8+ required, found Python $python_version"
    exit 1
fi

print_success "Python $python_version found"

# Create virtual environment if it doesn't exist
print_status "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements with error handling
print_status "Installing dependencies..."

# Install core dependencies first (these are most likely to have issues)
print_status "Installing core dependencies..."
pip install setuptools>=70.0.0 wheel>=0.42.0

# Install numerical dependencies
print_status "Installing numerical computing dependencies..."
pip install "numpy>=1.21.0,<2.0.0"

# Install TITO CLI dependencies
print_status "Installing TITO CLI dependencies..."
pip install rich>=13.0.0 jupytext>=1.14.0

# Install testing dependencies
print_status "Installing testing dependencies..."
pip install pytest>=7.0.0

# Install remaining dependencies
print_status "Installing remaining dependencies..."
pip install -r requirements.txt || {
    print_warning "Some dependencies failed to install. Continuing with essential packages..."
}

# Install the package in development mode
print_status "Installing TinyTorch in development mode..."
pip install -e . || {
    print_warning "Failed to install in development mode. You may need to run this manually."
}

# Verify TITO CLI works
print_status "Verifying TITO CLI installation..."
if python -m tito.main --help > /dev/null 2>&1; then
    print_success "TITO CLI is working correctly"
else
    print_error "TITO CLI verification failed"
    exit 1
fi

# Run doctor command to check environment
print_status "Running environment diagnostics..."
python -m tito.main system doctor || {
    print_warning "Some environment issues detected. See output above."
}

# Create helpful aliases
print_status "Setting up helpful aliases..."
echo "# TinyTorch Development Aliases" > .env
echo "alias tito='python -m tito.main'" >> .env
echo "alias nb='python -m tito.main module notebooks'" >> .env
echo "alias export-all='python -m tito.main export --all'" >> .env

# Success message
echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Test TITO CLI: python -m tito.main"
echo "  3. Convert modules to notebooks: python -m tito.main module notebooks"
echo "  4. Open notebooks: jupyter lab"
echo ""
echo "Quick commands:"
echo "  • Convert single module: python -m tito.main module notebooks --module 03_activations"
echo "  • Convert all modules: python -m tito.main module notebooks"
echo "  • Export modules: python -m tito.main export --all"
echo "  • Run tests: python -m tito.main module test"
echo ""
print_success "TinyTorch development environment is ready!"
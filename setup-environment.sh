#!/bin/bash
# TinyTorch Environment Setup
# Single canonical way to set up TinyTorch for development

set -e  # Exit on error

echo "🔥 TinyTorch Environment Setup"
echo "================================"
echo ""

# Detect system
OS=$(uname -s)
ARCH=$(uname -m)

echo "📋 System Info:"
echo "   OS: $OS"
echo "   Architecture: $ARCH"
echo ""

# Check if on Apple Silicon
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "✅ Detected Apple Silicon (arm64)"
    PYTHON_CMD="arch -arm64 /usr/bin/python3"
elif [ "$OS" = "Darwin" ]; then
    echo "⚠️  On macOS but not arm64 - using system Python"
    PYTHON_CMD="/usr/bin/python3"
else
    echo "📦 Using system Python"
    PYTHON_CMD="python3"
fi

# Create venv
echo ""
echo "🐍 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  .venv already exists - removing it"
    rm -rf .venv
fi

$PYTHON_CMD -m venv .venv
echo "✅ Virtual environment created"

# Activate and install
echo ""
echo "📦 Installing dependencies..."

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # On Apple Silicon, use arch prefix for all pip commands
    arch -arm64 .venv/bin/pip install --upgrade pip -q
    arch -arm64 .venv/bin/pip install -r requirements.txt -q
    arch -arm64 .venv/bin/pip install -e . -q
else
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
    .venv/bin/pip install -e . -q
fi

echo "✅ Dependencies installed"

# Verify
echo ""
echo "🔍 Verifying installation..."
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    VENV_ARCH=$(arch -arm64 .venv/bin/python3 -c "import platform; print(platform.machine())")
else
    VENV_ARCH=$(.venv/bin/python3 -c "import platform; print(platform.machine())")
fi
echo "   Python architecture: $VENV_ARCH"

# Create activation helper
cat > activate.sh << 'EOF'
#!/bin/bash
# TinyTorch activation helper
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    # On Apple Silicon, ensure arm64
    export TINYTORCH_ARCH="arm64"
    alias python='arch -arm64 .venv/bin/python3'
    alias pip='arch -arm64 .venv/bin/pip'
    source .venv/bin/activate
    echo "🔥 TinyTorch environment activated (arm64)"
else
    source .venv/bin/activate
    echo "🔥 TinyTorch environment activated"
fi
echo "💡 Try: tito system doctor"
EOF

chmod +x activate.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. source activate.sh          # Activate environment"
echo "   2. tito system doctor           # Verify setup"
echo "   3. tito module view 01_tensor   # Start learning"
echo ""


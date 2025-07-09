#!/bin/bash
# TinyTorch Environment Activation & Setup

# Check if virtual environment exists, create if not
if [ ! -d "tinytorch-env" ]; then
    echo "🆕 First time setup - creating environment..."
    python3 -m venv tinytorch-env || {
        echo "❌ Failed to create virtual environment"
        exit 1
    }
    echo "📦 Installing dependencies..."
    tinytorch-env/bin/pip install -r requirements.txt || {
        echo "❌ Failed to install dependencies"
        exit 1
    }
    echo "✅ Environment created!"
fi

echo "🔥 Activating TinyTorch environment..."
source tinytorch-env/bin/activate

# Create tito alias for convenience
alias tito="python3 bin/tito.py"

echo "✅ Ready to build ML systems!"
echo "💡 Quick commands:"
echo "   tito info      - Check system status"
echo "   tito test      - Run tests" 
echo "   tito doctor    - Diagnose issues"

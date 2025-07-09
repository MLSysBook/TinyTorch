#!/bin/bash
# TinyTorch Environment Activation & Setup

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "🆕 First time setup - creating environment..."
    python3 -m venv .venv || {
        echo "❌ Failed to create virtual environment"
        exit 1
    }
    echo "📦 Installing dependencies..."
    .venv/bin/pip install -r requirements.txt || {
        echo "❌ Failed to install dependencies"
        exit 1
    }
    echo "✅ Environment created!"
fi

echo "🔥 Activating TinyTorch environment..."
source .venv/bin/activate

# Create tito alias for convenience
alias tito="python3 bin/tito.py"

echo "✅ Ready to build ML systems!"
echo "💡 Quick commands:"
echo "   tito info      - Check system status"
echo "   tito test      - Run tests" 
echo "   tito doctor    - Diagnose issues"
echo "   jupyter notebook  - Start Jupyter for interactive development"

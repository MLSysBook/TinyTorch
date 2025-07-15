"""
Test utilities for TinyTorch integration tests.
"""
import subprocess
import sys
from pathlib import Path

def ensure_tinytorch_exported():
    """
    Ensure all modules are properly exported to the tinytorch package.
    This should be called at the beginning of each integration test.
    """
    print("🔄 Ensuring TinyTorch modules are exported...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Run tito export --all to ensure all modules are exported
    try:
        result = subprocess.run([
            sys.executable, "-m", "tito.main", "export", "--all"
        ], cwd=project_root, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Export failed: {result.stderr}")
            raise RuntimeError(f"Failed to export modules: {result.stderr}")
        
        print("✅ All modules exported successfully!")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Export process timed out after 120 seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to export modules: {e}")

def verify_tinytorch_imports():
    """
    Verify that all required tinytorch modules can be imported.
    """
    required_modules = [
        "tinytorch.core.tensor",
        "tinytorch.core.activations", 
        "tinytorch.core.layers",
        "tinytorch.core.networks",
        "tinytorch.core.cnn",
        "tinytorch.core.dataloader",
        "tinytorch.core.autograd",
        "tinytorch.core.optimizers",
        "tinytorch.core.training",
        "tinytorch.core.compression",
        "tinytorch.core.kernels",
        "tinytorch.core.benchmarking",
        "tinytorch.core.mlops"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {e}")
    
    if failed_imports:
        raise ImportError(f"Failed to import modules: {failed_imports}")
    
    print("✅ All tinytorch modules imported successfully!")

def setup_integration_test():
    """
    Complete setup for integration tests.
    Call this at the beginning of each integration test file.
    """
    ensure_tinytorch_exported()
    verify_tinytorch_imports() 
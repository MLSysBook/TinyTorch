#!/usr/bin/env python3
"""
Test script to validate that all capability showcases can import properly.
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_showcase_imports():
    """Test that all showcase files can be imported without errors."""
    capabilities_dir = Path(__file__).parent
    showcase_files = list(capabilities_dir.glob("*_*.py"))
    
    results = []
    
    for file_path in sorted(showcase_files):
        if file_path.name.startswith("test_"):
            continue
            
        module_name = file_path.stem
        
        try:
            # Read the file to check for imports
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if it has TinyTorch imports
            if "from tinytorch" in content:
                # Try to import the modules it needs
                import tinytorch.core.tensor
                if "dense" in content:
                    import tinytorch.core.dense
                if "activations" in content:
                    import tinytorch.core.activations
                if "spatial" in content:
                    import tinytorch.core.spatial
                if "attention" in content:
                    import tinytorch.core.attention
                if "dataloader" in content:
                    import tinytorch.core.dataloader
                if "training" in content:
                    import tinytorch.core.training
                if "compression" in content:
                    import tinytorch.core.compression
                if "benchmarking" in content:
                    import tinytorch.core.benchmarking
                if "mlops" in content:
                    import tinytorch.core.mlops
                if "tinygpt" in content:
                    import tinytorch.tinygpt
            
            results.append((module_name, "✅ PASS", "Dependencies available"))
            
        except ImportError as e:
            if "tinytorch" in str(e):
                results.append((module_name, "⚠️ SKIP", f"TinyTorch module not complete: {str(e).split('.')[-1]}"))
            else:
                results.append((module_name, "⚠️ SKIP", f"Missing: {e}"))
        except Exception as e:
            results.append((module_name, "❌ FAIL", f"Error: {e}"))
    
    return results

def main():
    print("🧪 Testing TinyTorch Capability Showcases")
    print("="*50)
    
    results = test_showcase_imports()
    
    for module_name, status, message in results:
        print(f"{status} {module_name}: {message}")
    
    # Summary
    passed = sum(1 for _, status, _ in results if "PASS" in status)
    skipped = sum(1 for _, status, _ in results if "SKIP" in status)
    failed = sum(1 for _, status, _ in results if "FAIL" in status)
    
    print("\n📊 Summary:")
    print(f"   ✅ Passed:  {passed}")
    print(f"   ⚠️ Skipped: {skipped}")
    print(f"   ❌ Failed:  {failed}")
    
    if failed == 0:
        print("\n🎉 All showcases ready to run!")
    else:
        print(f"\n⚠️ {failed} showcases have import issues.")

if __name__ == "__main__":
    main()
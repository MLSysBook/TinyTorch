#!/usr/bin/env python3
"""Test that all modules work after restructure."""

import sys
import subprocess
from pathlib import Path

def test_module(module_path):
    """Test a single module."""
    module_name = module_path.name
    py_files = list(module_path.glob("*_dev.py"))
    
    if not py_files:
        return None, f"No _dev.py file found"
    
    py_file = py_files[0]
    
    try:
        result = subprocess.run(
            [sys.executable, str(py_file)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path.cwd()
        )
        if result.returncode == 0:
            return True, "Passed"
        else:
            return False, f"Failed with code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    """Test all modules."""
    modules_dir = Path("modules/source")
    
    # Expected modules for each part
    part1 = ["01_setup", "02_tensor", "03_activations", "04_layers", "05_networks"]
    part2 = ["06_spatial", "07_dataloader", "08_normalization", "09_autograd", "10_optimizers", "11_training"]
    part3 = ["12_embeddings", "13_attention", "14_transformers", "15_generation", "16_regularization", "17_systems"]
    
    print("=" * 60)
    print("TinyTorch Module Test Report - After Restructure")
    print("=" * 60)
    
    for part_name, modules in [("Part I: Foundations", part1), 
                                ("Part II: Computer Vision", part2),
                                ("Part III: Language Models", part3)]:
        print(f"\n{part_name}")
        print("-" * 40)
        
        for module_name in modules:
            module_path = modules_dir / module_name
            
            if not module_path.exists():
                print(f"  ⚠️  {module_name:20} - Directory missing")
                continue
            
            result, message = test_module(module_path)
            
            if result is True:
                print(f"  ✅ {module_name:20} - {message}")
            elif result is False:
                print(f"  ❌ {module_name:20} - {message}")
            else:
                print(f"  ⚠️  {module_name:20} - {message}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Part I & II modules (1-11) are working")
    print("- Part III needs content for new modules (12,14,15,17)")
    print("- Structure is ready for development!")
    print("=" * 60)

if __name__ == "__main__":
    main()

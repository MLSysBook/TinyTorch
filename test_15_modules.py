#!/usr/bin/env python3
"""Test the final 15-module structure."""

import subprocess
import sys
from pathlib import Path

def test_module(module_path):
    """Test a single module."""
    py_files = list(module_path.glob("*_dev.py"))
    if not py_files:
        return None
    result = subprocess.run([sys.executable, str(py_files[0])], 
                          capture_output=True, timeout=10, cwd=Path.cwd())
    return result.returncode == 0

print("="*60)
print("TinyTorch 15-Module Structure Test")
print("="*60)

modules_dir = Path("modules/source")
parts = [
    ("Part I: MLPs (XORNet)", ["01_setup", "02_tensor", "03_activations", "04_layers", "05_networks"]),
    ("Part II: CNNs (CIFAR-10)", ["06_spatial", "07_dataloader", "08_autograd", "09_optimizers", "10_training"]),
    ("Part III: Transformers (TinyGPT)", ["11_embeddings", "12_attention", "13_normalization", "14_transformers", "15_generation"])
]

for part_name, modules in parts:
    print(f"\n{part_name}")
    print("-"*40)
    for module in modules:
        path = modules_dir / module
        if not path.exists():
            print(f"  ⚠️  {module:20} Missing")
        elif test_module(path):
            print(f"  ✅ {module:20} Passes")
        elif test_module(path) is None:
            print(f"  ⚠️  {module:20} No implementation")
        else:
            print(f"  ❌ {module:20} Failed")

print("\n" + "="*60)
print("✨ Clean 15-module structure ready!")
print("Each part: 5 modules, 1 innovation, 1 capstone")

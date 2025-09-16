#!/usr/bin/env python3
"""
Test script to validate module structure without numpy dependency
"""

import ast
import sys
from pathlib import Path

def validate_module_structure(filepath):
    """Validate that a module has the correct structure"""
    print(f"\n🔍 Validating: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        
        # Check for required classes
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Check module sections (markdown cells)
        has_sections = "Module Introduction" in content
        has_math = "Mathematical Background" in content  
        has_implementation = "Implementation" in content or "Core Implementation" in content
        has_testing = "Testing" in content
        has_ml_systems = "ML Systems Thinking" in content
        has_summary = "Module Summary" in content
        
        results = {
            "Classes found": len(classes),
            "Functions found": len(functions),
            "Has Introduction": has_sections,
            "Has Math Background": has_math,
            "Has Implementation": has_implementation,
            "Has Testing": has_testing,
            "Has ML Systems Questions": has_ml_systems,
            "Has Summary": has_summary
        }
        
        # Print results
        all_good = True
        for key, value in results.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
                if not value:
                    all_good = False
            else:
                status = "✅" if value > 0 else "⚠️"
            print(f"  {status} {key}: {value}")
        
        # Module-specific validation
        if "compression" in filepath.name.lower():
            has_profiler = "CompressionSystemsProfiler" in classes
            print(f"  {'✅' if has_profiler else '❌'} Has CompressionSystemsProfiler: {has_profiler}")
            if not has_profiler:
                all_good = False
                
        elif "kernels" in filepath.name.lower():
            has_profiler = "KernelOptimizationProfiler" in classes
            print(f"  {'✅' if has_profiler else '❌'} Has KernelOptimizationProfiler: {has_profiler}")
            if not has_profiler:
                all_good = False
                
        elif "benchmarking" in filepath.name.lower():
            has_profiler = "ProductionBenchmarkingProfiler" in classes
            print(f"  {'✅' if has_profiler else '❌'} Has ProductionBenchmarkingProfiler: {has_profiler}")
            if not has_profiler:
                all_good = False
                
        elif "mlops" in filepath.name.lower():
            has_profiler = "ProductionMLOpsProfiler" in classes
            print(f"  {'✅' if has_profiler else '❌'} Has ProductionMLOpsProfiler: {has_profiler}")
            if not has_profiler:
                all_good = False
                
        elif "capstone" in filepath.name.lower():
            has_profiler = "ProductionMLSystemProfiler" in classes
            print(f"  {'✅' if has_profiler else '❌'} Has ProductionMLSystemProfiler: {has_profiler}")
            if not has_profiler:
                all_good = False
        
        return all_good
        
    except SyntaxError as e:
        print(f"  ❌ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Test all modified modules"""
    print("=" * 60)
    print("🧪 Testing TinyTorch Module Structures")
    print("=" * 60)
    
    modules_to_test = [
        "modules/source/12_compression/compression_dev.py",
        "modules/source/13_kernels/kernels_dev.py", 
        "modules/source/14_benchmarking/benchmarking_dev.py",
        "modules/source/15_mlops/mlops_dev.py",
        "modules/source/16_capstone/capstone_dev.py"
    ]
    
    all_passed = True
    
    for module_path in modules_to_test:
        filepath = Path(module_path)
        if filepath.exists():
            passed = validate_module_structure(filepath)
            if not passed:
                all_passed = False
        else:
            print(f"\n❌ Module not found: {module_path}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All module structure tests passed!")
    else:
        print("❌ Some tests failed. Please review the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
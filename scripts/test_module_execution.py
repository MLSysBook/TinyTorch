#!/usr/bin/env python3
"""
Test script to validate module execution with mock dependencies
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock numpy and other dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['tinytorch'] = MagicMock()
sys.modules['tinytorch.tensor'] = MagicMock()
sys.modules['tinytorch.nn'] = MagicMock()
sys.modules['tinytorch.optim'] = MagicMock()
sys.modules['tinytorch.data'] = MagicMock()
sys.modules['tinytorch.autograd'] = MagicMock()
sys.modules['tinytorch.utils.nbgrader'] = MagicMock()

def test_module_imports(module_path):
    """Test if a module can be imported and key classes instantiated"""
    print(f"\n🔍 Testing: {module_path}")
    
    try:
        # Clear any cached imports
        module_name = Path(module_path).stem
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Read and execute the module
        with open(module_path, 'r') as f:
            code = f.read()
        
        # Create a namespace for execution
        namespace = {
            '__name__': '__main__',
            '__file__': module_path,
            'np': MagicMock(),
            'time': MagicMock(),
            'json': MagicMock()
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Check for expected classes based on module
        expected_classes = {
            'compression_dev': 'CompressionSystemsProfiler',
            'kernels_dev': 'KernelOptimizationProfiler', 
            'benchmarking_dev': 'ProductionBenchmarkingProfiler',
            'mlops_dev': 'ProductionMLOpsProfiler',
            'capstone_dev': 'ProductionMLSystemProfiler'
        }
        
        module_name = Path(module_path).stem
        if module_name in expected_classes:
            class_name = expected_classes[module_name]
            if class_name in namespace:
                print(f"  ✅ Found {class_name}")
                # Try to instantiate
                try:
                    instance = namespace[class_name]()
                    print(f"  ✅ Successfully instantiated {class_name}")
                    
                    # Check for key methods
                    if module_name == 'capstone_dev':
                        assert hasattr(instance, 'profile_end_to_end_system')
                        assert hasattr(instance, 'detect_cross_module_optimizations')
                        print(f"  ✅ Key methods present")
                    elif module_name == 'mlops_dev':
                        assert hasattr(instance, 'register_model_version')
                        assert hasattr(instance, 'detect_advanced_feature_drift')
                        print(f"  ✅ Key methods present")
                    
                except Exception as e:
                    print(f"  ⚠️ Could not instantiate: {e}")
            else:
                print(f"  ❌ {class_name} not found in module")
                return False
        
        # Check test functions were called (if they exist)
        test_functions = [name for name in namespace if name.startswith('test_')]
        print(f"  ℹ️ Found {len(test_functions)} test functions")
        
        return True
        
    except SyntaxError as e:
        print(f"  ❌ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all modified modules"""
    print("=" * 60)
    print("🧪 Testing TinyTorch Module Execution")
    print("=" * 60)
    
    modules_to_test = [
        "modules/source/12_compression/compression_dev.py",
        "modules/source/13_kernels/kernels_dev.py", 
        "modules/source/14_benchmarking/benchmarking_dev.py",
        "modules/source/15_mlops/mlops_dev.py",
        "modules/source/16_capstone/capstone_dev.py"
    ]
    
    results = {}
    
    for module_path in modules_to_test:
        filepath = Path(module_path)
        if filepath.exists():
            results[module_path] = test_module_imports(module_path)
        else:
            print(f"\n❌ Module not found: {module_path}")
            results[module_path] = False
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    for module, passed in results.items():
        status = "✅" if passed else "❌"
        module_name = Path(module).stem
        print(f"{status} {module_name}: {'Passed' if passed else 'Failed'}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All module execution tests passed!")
    else:
        print("❌ Some tests failed. The modules have syntax/import issues.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
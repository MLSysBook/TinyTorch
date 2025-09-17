"""
Integration test for Module 01: Setup

Validates that the setup module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
from pathlib import Path


def test_setup_module_integration():
    """Test that setup module integrates correctly with package."""
    
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")
    
    results = {
        "module_name": "01_setup",
        "integration_type": "setup_validation",
        "tests": [],
        "success": True,
        "errors": []
    }
    
    try:
        # Test 1: Package structure exists
        try:
            import tinytorch
            results["tests"].append({
                "name": "package_import",
                "status": "✅ PASS",
                "description": "TinyTorch package imports successfully"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "package_import", 
                "status": "❌ FAIL",
                "description": f"TinyTorch package import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Package import error: {e}")
        
        # Test 2: Core package structure
        try:
            import tinytorch.core
            results["tests"].append({
                "name": "core_structure",
                "status": "✅ PASS", 
                "description": "Core package structure exists"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "core_structure",
                "status": "❌ FAIL",
                "description": f"Core structure missing: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Core structure error: {e}")
        
        # Test 3: Essential directories exist
        package_path = Path("tinytorch")
        essential_dirs = ["core", "utils", "datasets"]
        
        missing_dirs = []
        for dir_name in essential_dirs:
            if not (package_path / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            results["tests"].append({
                "name": "directory_structure",
                "status": "✅ PASS",
                "description": "All essential directories present"
            })
        else:
            results["tests"].append({
                "name": "directory_structure", 
                "status": "❌ FAIL",
                "description": f"Missing directories: {missing_dirs}"
            })
            results["success"] = False
            results["errors"].append(f"Missing directories: {missing_dirs}")
        
        # Test 4: No import conflicts
        try:
            # Try importing multiple parts to check for conflicts
            import tinytorch
            import tinytorch.core
            import tinytorch.utils
            
            results["tests"].append({
                "name": "no_conflicts",
                "status": "✅ PASS",
                "description": "No import conflicts detected"
            })
        except Exception as e:
            results["tests"].append({
                "name": "no_conflicts",
                "status": "❌ FAIL", 
                "description": f"Import conflict detected: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Import conflict: {e}")
            
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error in setup integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })
    
    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_setup_module_integration()


if __name__ == "__main__":
    # Run test when script is executed directly
    result = run_integration_test()
    
    print(f"=== Integration Test: {result['module_name']} ===")
    print(f"Type: {result['integration_type']}")
    print(f"Overall Success: {result['success']}")
    print("\nTest Results:")
    
    for test in result["tests"]:
        print(f"  {test['status']} {test['name']}: {test['description']}")
    
    if result["errors"]:
        print(f"\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)
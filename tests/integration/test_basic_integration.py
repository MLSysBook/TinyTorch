"""
Basic integration test that doesn't require external dependencies.

Tests the Package Manager integration system itself.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_integration_system():
    """Test that the integration system itself works."""
    
    results = {
        "integration_system_test": True,
        "tests": [],
        "success": True,
        "errors": []
    }
    
    try:
        # Test 1: Import the Package Manager integration system
        try:
            # Import using file path since module path doesn't work
            import importlib.util
            
            integration_file = Path(__file__).parent / "package_manager_integration.py"
            spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
            integration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(integration_module)
            
            PackageManagerIntegration = integration_module.PackageManagerIntegration
            results["tests"].append({
                "name": "system_import",
                "status": "✅ PASS",
                "description": "Package Manager integration system imports successfully"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "system_import",
                "status": "❌ FAIL",
                "description": f"System import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"System import error: {e}")
            return results
        
        # Test 2: Create manager instance
        try:
            manager = PackageManagerIntegration()
            results["tests"].append({
                "name": "manager_creation",
                "status": "✅ PASS",
                "description": "Package Manager can be instantiated"
            })
        except Exception as e:
            results["tests"].append({
                "name": "manager_creation",
                "status": "❌ FAIL",
                "description": f"Manager creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Manager creation error: {e}")
            return results
        
        # Test 3: Check module mappings exist
        try:
            assert hasattr(manager, 'module_mappings'), "Manager should have module_mappings"
            assert len(manager.module_mappings) > 0, "Should have module mappings configured"
            
            results["tests"].append({
                "name": "module_mappings",
                "status": "✅ PASS",
                "description": f"Module mappings configured ({len(manager.module_mappings)} modules)"
            })
        except Exception as e:
            results["tests"].append({
                "name": "module_mappings",
                "status": "❌ FAIL",
                "description": f"Module mappings test failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Module mappings error: {e}")
        
        # Test 4: Test normalization function
        try:
            normalized = manager._normalize_module_name("tensor")
            if normalized == "02_tensor":
                results["tests"].append({
                    "name": "name_normalization",
                    "status": "✅ PASS",
                    "description": "Module name normalization works"
                })
            else:
                results["tests"].append({
                    "name": "name_normalization",
                    "status": "❌ FAIL",
                    "description": f"Expected '02_tensor', got '{normalized}'"
                })
                results["success"] = False
                results["errors"].append(f"Name normalization error: expected '02_tensor', got '{normalized}'")
        except Exception as e:
            results["tests"].append({
                "name": "name_normalization",
                "status": "❌ FAIL",
                "description": f"Name normalization failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Name normalization error: {e}")
        
        # Test 5: Test package validation (basic)
        try:
            validation = manager.validate_package_state()
            assert isinstance(validation, dict), "Validation should return dict"
            assert 'overall_health' in validation, "Should include overall health"
            
            results["tests"].append({
                "name": "package_validation",
                "status": "✅ PASS",
                "description": f"Package validation works (health: {validation['overall_health']})"
            })
        except Exception as e:
            results["tests"].append({
                "name": "package_validation",
                "status": "❌ FAIL",
                "description": f"Package validation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Package validation error: {e}")
            
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error in integration system test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })
    
    return results


if __name__ == "__main__":
    result = test_integration_system()
    
    print("=== Package Manager Integration System Test ===")
    print(f"Overall Success: {result['success']}")
    print("\nTest Results:")
    
    for test in result["tests"]:
        print(f"  {test['status']} {test['name']}: {test['description']}")
    
    if result["errors"]:
        print(f"\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    sys.exit(0 if result["success"] else 1)
#!/usr/bin/env python3
"""
Manual verification script for the Tensor module.

This script provides human-readable feedback on the student's
tensor implementation progress and helps identify what needs to be done.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import from tinytorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def print_header():
    """Print the verification header."""
    print("🔥 TinyTorch Tensor Verification 🔥")
    print("=" * 50)
    print()

def check_tensor_import():
    """Check if the Tensor class can be imported."""
    print("📦 Checking Tensor Implementation...")
    
    try:
        from tinytorch.core.tensor import Tensor
        print("✅ Tensor class found and imported successfully!")
        return True, Tensor
    except ImportError as e:
        print("❌ Tensor class not found!")
        print(f"   Error: {e}")
        print("💡 Make sure to implement the Tensor class in tinytorch/core/tensor.py")
        return False, None

def check_basic_functionality(Tensor):
    """Check basic tensor functionality."""
    print("\n🔧 Testing Basic Functionality...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Scalar creation
    total_tests += 1
    try:
        t = Tensor(5)
        if hasattr(t, 'shape') and hasattr(t, 'size') and hasattr(t, 'data'):
            print("✅ Scalar tensor creation: PASSED")
            tests_passed += 1
        else:
            print("❌ Scalar tensor creation: FAILED (missing properties)")
    except Exception as e:
        print(f"❌ Scalar tensor creation: FAILED ({e})")
    
    # Test 2: List creation
    total_tests += 1
    try:
        t = Tensor([1, 2, 3, 4])
        if t.shape == (4,) and t.size == 4:
            print("✅ List tensor creation: PASSED")
            tests_passed += 1
        else:
            print("❌ List tensor creation: FAILED (incorrect shape/size)")
    except Exception as e:
        print(f"❌ List tensor creation: FAILED ({e})")
    
    # Test 3: Matrix creation
    total_tests += 1
    try:
        t = Tensor([[1, 2], [3, 4]])
        if t.shape == (2, 2) and t.size == 4:
            print("✅ Matrix tensor creation: PASSED")
            tests_passed += 1
        else:
            print("❌ Matrix tensor creation: FAILED (incorrect shape/size)")
    except Exception as e:
        print(f"❌ Matrix tensor creation: FAILED ({e})")
    
    # Test 4: Data type handling
    total_tests += 1
    try:
        t = Tensor([1, 2, 3], dtype=np.float32)
        if t.dtype == np.float32:
            print("✅ Data type handling: PASSED")
            tests_passed += 1
        else:
            print("❌ Data type handling: FAILED (incorrect dtype)")
    except Exception as e:
        print(f"❌ Data type handling: FAILED ({e})")
    
    return tests_passed, total_tests

def check_arithmetic_operations(Tensor):
    """Check arithmetic operations."""
    print("\n🔢 Testing Arithmetic Operations...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Addition
    total_tests += 1
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        expected = np.array([5, 7, 9])
        if np.array_equal(c.data, expected):
            print("✅ Tensor addition: PASSED")
            tests_passed += 1
        else:
            print("❌ Tensor addition: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Tensor addition: FAILED ({e})")
    
    # Test 2: Scalar addition
    total_tests += 1
    try:
        a = Tensor([1, 2, 3])
        c = a + 5
        expected = np.array([6, 7, 8])
        if np.array_equal(c.data, expected):
            print("✅ Scalar addition: PASSED")
            tests_passed += 1
        else:
            print("❌ Scalar addition: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Scalar addition: FAILED ({e})")
    
    # Test 3: Multiplication
    total_tests += 1
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        expected = np.array([4, 10, 18])
        if np.array_equal(c.data, expected):
            print("✅ Tensor multiplication: PASSED")
            tests_passed += 1
        else:
            print("❌ Tensor multiplication: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Tensor multiplication: FAILED ({e})")
    
    # Test 4: Scalar multiplication
    total_tests += 1
    try:
        a = Tensor([1, 2, 3])
        c = a * 3
        expected = np.array([3, 6, 9])
        if np.array_equal(c.data, expected):
            print("✅ Scalar multiplication: PASSED")
            tests_passed += 1
        else:
            print("❌ Scalar multiplication: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Scalar multiplication: FAILED ({e})")
    
    return tests_passed, total_tests

def check_utility_methods(Tensor):
    """Check utility methods."""
    print("\n🛠️ Testing Utility Methods...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Reshape
    total_tests += 1
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        reshaped = t.reshape(3, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        if np.array_equal(reshaped.data, expected) and reshaped.shape == (3, 2):
            print("✅ Reshape method: PASSED")
            tests_passed += 1
        else:
            print("❌ Reshape method: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Reshape method: FAILED ({e})")
    
    # Test 2: Transpose
    total_tests += 1
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        transposed = t.transpose()
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        if np.array_equal(transposed.data, expected) and transposed.shape == (3, 2):
            print("✅ Transpose method: PASSED")
            tests_passed += 1
        else:
            print("❌ Transpose method: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Transpose method: FAILED ({e})")
    
    # Test 3: Sum
    total_tests += 1
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum()
        expected = np.array(21)
        if np.array_equal(result.data, expected):
            print("✅ Sum method: PASSED")
            tests_passed += 1
        else:
            print("❌ Sum method: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Sum method: FAILED ({e})")
    
    # Test 4: Mean
    total_tests += 1
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean()
        expected = np.array(3.5)
        if np.allclose(result.data, expected):
            print("✅ Mean method: PASSED")
            tests_passed += 1
        else:
            print("❌ Mean method: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Mean method: FAILED ({e})")
    
    # Test 5: Flatten
    total_tests += 1
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        flattened = t.flatten()
        expected = np.array([1, 2, 3, 4, 5, 6])
        if np.array_equal(flattened.data, expected) and flattened.shape == (6,):
            print("✅ Flatten method: PASSED")
            tests_passed += 1
        else:
            print("❌ Flatten method: FAILED (incorrect result)")
    except Exception as e:
        print(f"❌ Flatten method: FAILED ({e})")
    
    return tests_passed, total_tests

def check_error_handling(Tensor):
    """Check error handling."""
    print("\n🚨 Testing Error Handling...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Invalid data type
    total_tests += 1
    try:
        Tensor("invalid")
        print("❌ Invalid data type: FAILED (should raise ValueError)")
    except ValueError:
        print("✅ Invalid data type: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Invalid data type: FAILED (wrong exception: {e})")
    
    # Test 2: Invalid operation
    total_tests += 1
    try:
        a = Tensor([1, 2, 3])
        a + "invalid"
        print("❌ Invalid operation: FAILED (should raise TypeError)")
    except TypeError:
        print("✅ Invalid operation: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Invalid operation: FAILED (wrong exception: {e})")
    
    return tests_passed, total_tests

def print_summary(basic_passed, basic_total, arithmetic_passed, arithmetic_total, 
                 utility_passed, utility_total, error_passed, error_total):
    """Print a summary of all tests."""
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_passed = basic_passed + arithmetic_passed + utility_passed + error_passed
    total_tests = basic_total + arithmetic_total + utility_total + error_total
    
    print(f"Basic Functionality: {basic_passed}/{basic_total} tests passed")
    print(f"Arithmetic Operations: {arithmetic_passed}/{arithmetic_total} tests passed")
    print(f"Utility Methods: {utility_passed}/{utility_total} tests passed")
    print(f"Error Handling: {error_passed}/{error_total} tests passed")
    print()
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 CONGRATULATIONS! All tests passed!")
        print("✅ Your Tensor implementation is complete and working correctly!")
        print("\n📋 Next steps:")
        print("   1. Run: tito test --module tensor")
        print("   2. Submit: tito submit --module tensor")
        print("   3. Move to next module: cd ../mlp/")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed.")
        print("💡 Review the failed tests above and fix your implementation.")
        print("\n📚 Need help? Check the tutorial notebooks in tutorials/")

def main():
    """Main verification function."""
    print_header()
    
    # Check if Tensor class exists
    tensor_available, Tensor = check_tensor_import()
    if not tensor_available:
        print("\n💡 To get started:")
        print("   1. Open: modules/tensor/notebook/tensor_dev.ipynb")
        print("   2. Implement the Tensor class step by step")
        print("   3. Run this verification again")
        return False
    
    # Run all verification checks
    basic_passed, basic_total = check_basic_functionality(Tensor)
    arithmetic_passed, arithmetic_total = check_arithmetic_operations(Tensor)
    utility_passed, utility_total = check_utility_methods(Tensor)
    error_passed, error_total = check_error_handling(Tensor)
    
    # Print summary
    print_summary(basic_passed, basic_total, arithmetic_passed, arithmetic_total,
                 utility_passed, utility_total, error_passed, error_total)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
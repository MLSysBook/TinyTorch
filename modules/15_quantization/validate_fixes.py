#!/usr/bin/env python
"""
Validation script to verify quantization module fixes.

This script checks that:
1. Test functions are defined but not called at module level
2. NBGrader metadata is present
3. __main__ guards are in place
"""

import re
import sys

def validate_quantization_module():
    """Validate that all fixes were applied correctly."""

    print("=" * 70)
    print("QUANTIZATION MODULE VALIDATION")
    print("=" * 70)

    with open('quantization_dev.py', 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # Check 1: Test functions should NOT be called at module level
    print("\n1. Checking test execution protection...")
    test_functions = [
        'test_unit_quantize_int8',
        'test_unit_dequantize_int8',
        'test_unit_quantized_linear',
        'test_unit_quantize_model',
        'test_unit_compare_model_sizes',
        'test_module'
    ]

    issues = []
    protected = []

    for i, line in enumerate(lines, 1):
        for test_func in test_functions:
            # Check for unprotected calls (not in if __main__)
            if re.match(rf'^{test_func}\(\)', line.strip()):
                # Look back to see if there's an if __main__ before this
                has_guard = False
                for j in range(max(0, i-5), i):
                    if 'if __name__ ==' in lines[j]:
                        has_guard = True
                        break

                if not has_guard:
                    issues.append(f"Line {i}: {test_func}() called without __main__ guard")
                else:
                    protected.append(f"Line {i}: {test_func}() properly protected")

    if issues:
        print("❌ FAILED: Found unprotected test calls:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ PASSED: All test functions are protected")
        for p in protected:
            print(f"   ✓ {p}")

    # Check 2: NBGrader metadata presence
    print("\n2. Checking NBGrader metadata...")

    nbgrader_tests = {
        'test-quantize-int8': False,
        'test-dequantize-int8': False,
        'test-quantized-linear': False,
        'test-quantize-model': False,
        'test-compare-sizes': False,
        'test_module': False
    }

    for line in lines:
        for grade_id in nbgrader_tests.keys():
            if f'grade_id": "{grade_id}"' in line or f"'grade_id': '{grade_id}'" in line:
                nbgrader_tests[grade_id] = True

    missing = [k for k, v in nbgrader_tests.items() if not v and k != 'test_module']

    if missing:
        print(f"⚠️  WARNING: Missing NBGrader metadata for: {', '.join(missing)}")
    else:
        print("✅ PASSED: All unit tests have NBGrader metadata")
        for grade_id in nbgrader_tests:
            if nbgrader_tests[grade_id]:
                print(f"   ✓ {grade_id}")

    # Check 3: Demo functions protected
    print("\n3. Checking demo function protection...")

    demo_functions = [
        'demo_motivation_profiling',
        'analyze_quantization_memory',
        'analyze_quantization_accuracy',
        'demo_quantization_with_profiler'
    ]

    demo_protected = []
    demo_issues = []

    for i, line in enumerate(lines, 1):
        for demo_func in demo_functions:
            if re.match(rf'^{demo_func}\(\)', line.strip()):
                # Look back for if __main__ guard
                has_guard = False
                for j in range(max(0, i-5), i):
                    if 'if __name__ ==' in lines[j]:
                        has_guard = True
                        break

                if not has_guard:
                    demo_issues.append(f"Line {i}: {demo_func}() not protected")
                else:
                    demo_protected.append(f"Line {i}: {demo_func}() protected")

    if demo_issues:
        print("❌ FAILED: Found unprotected demo calls:")
        for issue in demo_issues:
            print(f"   {issue}")
    else:
        print("✅ PASSED: All demo functions are protected")
        for p in demo_protected:
            print(f"   ✓ {p}")

    # Check 4: No print statements at module level
    print("\n4. Checking for module-level print statements...")

    unprotected_prints = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('print(') and 'def ' not in lines[max(0,i-10):i][-1]:
            # Check if it's in a function or protected
            in_function = False
            has_main_guard = False

            for j in range(max(0, i-20), i):
                if lines[j].strip().startswith('def '):
                    in_function = True
                if 'if __name__ ==' in lines[j]:
                    has_main_guard = True

            if not in_function and not has_main_guard:
                unprotected_prints.append((i, line.strip()))

    if unprotected_prints:
        print("⚠️  WARNING: Found unprotected print statements:")
        for line_num, stmt in unprotected_prints:
            print(f"   Line {line_num}: {stmt[:60]}...")
    else:
        print("✅ PASSED: No unprotected print statements")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = not issues and not demo_issues and not missing

    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nThe module is now:")
        print("  • Safe to import (no test execution)")
        print("  • NBGrader compliant")
        print("  • Ready for export with TITO")
        print("  • Can be used as dependency by future modules")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the issues above and apply fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(validate_quantization_module())

#!/usr/bin/env python3
"""
Phase 1: Test all TinyTorch examples to ensure they learn.
Tests each example and logs results.
"""

import subprocess
import sys
import time
from datetime import datetime

def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def test_example(name, command, success_criteria):
    """Test an example and return success status."""
    log(f"Testing {name}...")
    log(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        output = result.stdout + result.stderr
        
        # Check success criteria
        success = all(criterion in output for criterion in success_criteria)
        
        if success:
            log(f"✅ {name} PASSED - All criteria met")
        else:
            log(f"❌ {name} FAILED - Missing criteria")
            log(f"Output preview: {output[-500:]}")
            
        return success, output
        
    except subprocess.TimeoutExpired:
        log(f"⏱️ {name} TIMEOUT - Took too long")
        return False, "TIMEOUT"
    except Exception as e:
        log(f"❌ {name} ERROR - {str(e)}")
        return False, str(e)

def main():
    """Test all examples in order of complexity."""
    
    log("="*60)
    log("PHASE 1: TESTING ALL EXAMPLES FOR LEARNING")
    log("="*60)
    
    results = []
    
    # 1. Perceptron (simplest)
    log("\n1. PERCEPTRON (1957)")
    success, output = test_example(
        "Perceptron",
        "python examples/perceptron_1957/rosenblatt_perceptron.py --epochs 100",
        ["SUCCESS", "100.0%", "Loss"]
    )
    results.append(("Perceptron", success))
    
    # 2. XOR (multi-layer)
    log("\n2. XOR (1969)")
    success, output = test_example(
        "XOR",
        "python examples/xor_1969/minsky_xor_problem.py --epochs 200",
        ["SUCCESS", "Training Complete", "Val"]
    )
    results.append(("XOR", success))
    
    # 3. MNIST MLP (deep network)
    log("\n3. MNIST MLP (1986)")
    success, output = test_example(
        "MNIST",
        "python examples/mnist_mlp_1986/train_mlp.py --epochs 2 --batch-size 32",
        ["SUCCESS", "Training", "Test"]
    )
    results.append(("MNIST", success))
    
    # 4. CIFAR CNN (convolutional)
    log("\n4. CIFAR CNN (Modern)")
    success, output = test_example(
        "CIFAR",
        "python examples/cifar_cnn_modern/train_cnn.py --quick-test --epochs 2",
        ["SUCCESS", "Forward pass", "CNN"]
    )
    results.append(("CIFAR", success))
    
    # 5. TinyGPT (transformer)
    log("\n5. TINYGPT (2018)")
    success, output = test_example(
        "TinyGPT",
        "python examples/gpt_2018/train_gpt.py",
        ["Success", "transformer", "Loss"]
    )
    results.append(("TinyGPT", success))
    
    # Summary
    log("\n" + "="*60)
    log("PHASE 1 SUMMARY")
    log("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        log(f"{name:15} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        log("\n🎉 ALL EXAMPLES LEARNING SUCCESSFULLY!")
        log("Ready for Phase 2: Optimization Testing")
    else:
        log("\n⚠️ Some examples need fixing before optimization")
        log("Fix failing examples first")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

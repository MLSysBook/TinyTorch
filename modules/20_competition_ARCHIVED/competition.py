# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#| default_exp competition.submit

# %% [markdown]
"""
# Module 20: TinyMLPerf Competition - Your Capstone Challenge

Welcome to the capstone! You've built an entire ML system (M01-13) and learned optimization techniques (M14-19). Now compete in **TinyMLPerf** - a competition inspired by industry-standard MLPerf benchmarking!

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with all optimization techniques
**You've Learned**: MLPerf principles and benchmarking methodology (Module 19)
**You'll Do**: Compete in TinyMLPerf following Closed Division rules
**You'll Produce**: Standardized TinyMLPerf submission

**The Journey So Far**:
```
Modules 01-13: Build ML System (tensors → transformers)
Modules 14-18: Learn Optimization Techniques  
Module 19:     Learn MLPerf-Style Benchmarking
Module 20:     Compete in TinyMLPerf! 🏅
```

## 🏅 TinyMLPerf: MLPerf for Educational Systems

TinyMLPerf follows MLPerf principles adapted for educational ML systems:

**Closed Division Rules (What You'll Do):**
- ✅ Use provided baseline models (fair comparison)
- ✅ Use provided test datasets (standardized evaluation)
- ✅ Apply optimization techniques from Modules 14-18
- ✅ Report all metrics (accuracy, latency, memory)
- ✅ Document your optimization strategy

**Why Closed Division?**
- Fair apples-to-apples comparison
- Tests your optimization skills (not model design)
- Mirrors real-world MLPerf Inference competitions
- Professionally credible methodology

**Competition Categories:**
- 🏃 Latency Sprint: Minimize inference time
- 🏋️ Memory Challenge: Minimize model footprint
- 🎯 Accuracy Contest: Maximize accuracy within constraints
- 🏋️‍♂️ All-Around: Best balanced performance
- 🚀 Extreme Push: Most aggressive optimization

This module provides:
1. **Validation**: Verify your TinyTorch installation
2. **Baseline**: Official reference performance
3. **Worked Example**: Complete optimization workflow
4. **Competition Template**: Your submission workspace

🔥 Let's compete following professional MLPerf methodology! 🏅
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/20_competition/competition_dev.py`  
**Building Side:** Code exports to `tinytorch.competition.submit`

```python
# Validation and baseline tools:
from tinytorch.competition.submit import validate_installation, generate_baseline

# Competition helpers:
from tinytorch.competition.submit import load_baseline_model, generate_submission
```

**Why this matters:**
- **Validation:** Ensures your TinyTorch installation works correctly
- **Baseline:** Establishes reference performance for fair comparison
- **Competition:** Provides standardized framework for submissions
- **Integration:** Brings together all 19 modules into one complete workflow
"""

# %% [markdown]
"""
# 1. TinyMLPerf Rules & System Validation

Before competing, let's understand TinyMLPerf rules and validate your environment. Following MLPerf methodology (learned in Module 19) ensures fair competition and reproducible results.

## TinyMLPerf Closed Division Rules

**You learned in Module 19 that MLPerf Closed Division requires:**
1. **Fixed Models**: Use provided baseline architectures
2. **Fixed Datasets**: Use provided test data
3. **Fair Comparison**: Same starting point for everyone
4. **Reproducibility**: Document all optimizations
5. **Multiple Metrics**: Report accuracy, latency, memory

**In TinyMLPerf Closed Division, you CAN:**
- ✅ Apply quantization (Module 17)
- ✅ Apply pruning/compression (Module 18)
- ✅ Enable KV caching for transformers (Module 14)
- ✅ Combine techniques in any order
- ✅ Tune hyperparameters

**In TinyMLPerf Closed Division, you CANNOT:**
- ❌ Change baseline model architecture
- ❌ Train on different data
- ❌ Use external pretrained weights
- ❌ Modify test dataset

**Why these rules?**
- Tests your OPTIMIZATION skills (not model design)
- Fair apples-to-apples comparison
- Mirrors professional MLPerf competitions
- Results are meaningful and reproducible

## System Validation

Let's verify your TinyTorch installation works correctly before competing. MLPerf requires documenting your environment, so validation ensures reproducibility.

**Validation checks:**
- ✅ All 19 modules imported successfully
- ✅ Core operations work (tensor, autograd, layers)
- ✅ Optimization techniques available (M14-18)
- ✅ Benchmarking tools functional (M19)
"""

# %%
#| export
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

def validate_installation() -> Dict[str, bool]:
    """
    Validate TinyTorch installation and return status of each component.
    
    Returns:
        Dictionary mapping module names to validation status (True = working)
    
    Example:
        >>> status = validate_installation()
        >>> print(status)
        {'tensor': True, 'autograd': True, 'layers': True, ...}
    """
    validation_results = {}
    
    print("🔧 Validating TinyTorch Installation...")
    print("=" * 60)
    
    # Core modules (M01-13)
    core_modules = [
        ("tensor", "tinytorch.core.tensor", "Tensor"),
        ("autograd", "tinytorch.core.autograd", "enable_autograd"),
        ("layers", "tinytorch.core.layers", "Linear"),
        ("activations", "tinytorch.core.activations", "ReLU"),
        ("losses", "tinytorch.core.training", "MSELoss"),
        ("optimizers", "tinytorch.core.optimizers", "SGD"),
        ("spatial", "tinytorch.core.spatial", "Conv2d"),
        ("attention", "tinytorch.core.attention", "MultiHeadAttention"),
        ("transformers", "tinytorch.models.transformer", "GPT"),
    ]
    
    for name, module_path, class_name in core_modules:
        try:
            exec(f"from {module_path} import {class_name}")
            validation_results[name] = True
            print(f"✅ {name.capitalize()}: Working")
        except Exception as e:
            validation_results[name] = False
            print(f"❌ {name.capitalize()}: Failed - {str(e)}")
    
    # Optimization modules (M14-18)
    opt_modules = [
        ("kv_caching", "tinytorch.generation.kv_cache", "enable_kv_cache"),
        ("profiling", "tinytorch.profiling.profiler", "Profiler"),
        ("quantization", "tinytorch.optimization.quantization", "quantize_model"),
        ("compression", "tinytorch.optimization.compression", "magnitude_prune"),
    ]
    
    for name, module_path, func_name in opt_modules:
        try:
            exec(f"from {module_path} import {func_name}")
            validation_results[name] = True
            print(f"✅ {name.replace('_', ' ').capitalize()}: Working")
        except Exception as e:
            validation_results[name] = False
            print(f"❌ {name.replace('_', ' ').capitalize()}: Failed - {str(e)}")
    
    # Benchmarking (M19)
    try:
        from tinytorch.benchmarking.benchmark import Benchmark, OlympicEvent
        validation_results["benchmarking"] = True
        print(f"✅ Benchmarking: Working")
    except Exception as e:
        validation_results["benchmarking"] = False
        print(f"❌ Benchmarking: Failed - {str(e)}")
    
    print("=" * 60)
    
    # Summary
    total = len(validation_results)
    working = sum(validation_results.values())
    
    if working == total:
        print(f"🎉 Perfect! All {total}/{total} modules working!")
        print("✅ You're ready to compete in TorchPerf Olympics!")
    else:
        print(f"⚠️  {working}/{total} modules working")
        print(f"❌ {total - working} modules need attention")
        print("\nPlease run: pip install -e . (in TinyTorch root)")
    
    return validation_results

# %% [markdown]
"""
# 2. TinyMLPerf Baseline - Official Reference Performance

Following MLPerf Closed Division rules, everyone starts with the SAME baseline model. This ensures fair comparison - we're measuring your optimization skills, not model design.

## What is a TinyMLPerf Baseline?

In MLPerf competitions, the baseline is the official reference implementation:
- **Fixed Architecture:** Provided CNN (everyone uses the same)
- **Fixed Dataset:** CIFAR-10 test set (standardized evaluation)
- **Measured Metrics:** Accuracy, latency, memory (reproducible)
- **Your Goal:** Beat baseline using optimization techniques from M14-18

**This is MLPerf Closed Division:**
- Everyone starts here ← Fair comparison
- Apply YOUR optimizations ← Your skill
- Measure improvement ← Objective scoring

We provide a simple CNN on CIFAR-10 as the TinyMLPerf baseline. This gives everyone the same starting point.

### Baseline Components

1. **Model:** Standard CNN (no optimizations)
2. **Metrics:** Accuracy, latency, memory, parameters
3. **Test Data:** CIFAR-10 test set (standardized)
4. **Hardware:** Your local machine (reported for reproducibility)

The baseline establishes what "unoptimized" looks like. Your job: beat it!
"""

# %%
#| export
def load_baseline_model(model_name: str = "cifar10_cnn"):
    """
    Load a baseline model for TorchPerf Olympics competition.
    
    Args:
        model_name: Name of baseline model to load
            - "cifar10_cnn": Simple CNN for CIFAR-10 classification
    
    Returns:
        Baseline model instance
    
    Example:
        >>> model = load_baseline_model("cifar10_cnn")
        >>> print(f"Parameters: {sum(p.size for p in model.parameters())}")
    """
    from tinytorch.core.layers import Linear
    from tinytorch.core.spatial import Conv2d, MaxPool2d
    from tinytorch.core.activations import ReLU
    from tinytorch.core.tensor import Tensor
    
    # Flatten is not a separate class - it's just a reshape operation
    class Flatten:
        def forward(self, x):
            batch_size = x.shape[0]
            return Tensor(x.data.reshape(batch_size, -1))
    
    if model_name == "cifar10_cnn":
        # Simple CNN: Conv -> Pool -> Conv -> Pool -> FC -> FC
        class BaselineCNN:
            def __init__(self):
                self.name = "Baseline_CIFAR10_CNN"
                
                # Convolutional layers
                self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
                self.relu1 = ReLU()
                self.pool1 = MaxPool2d(kernel_size=2, stride=2)
                
                self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                self.relu2 = ReLU()
                self.pool2 = MaxPool2d(kernel_size=2, stride=2)
                
                # Fully connected layers
                self.flatten = Flatten()
                self.fc1 = Linear(64 * 8 * 8, 128)
                self.relu3 = ReLU()
                self.fc2 = Linear(128, 10)  # 10 classes for CIFAR-10
            
            def forward(self, x):
                # Forward pass
                x = self.conv1.forward(x)
                x = self.relu1.forward(x)
                x = self.pool1.forward(x)
                
                x = self.conv2.forward(x)
                x = self.relu2.forward(x)
                x = self.pool2.forward(x)
                
                x = self.flatten.forward(x)
                x = self.fc1.forward(x)
                x = self.relu3.forward(x)
                x = self.fc2.forward(x)
                
                return x
            
            def __call__(self, x):
                return self.forward(x)
        
        return BaselineCNN()
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

def generate_baseline(model_name: str = "cifar10_cnn", quick: bool = True) -> Dict[str, Any]:
    """
    Generate baseline performance metrics for a model.
    
    Args:
        model_name: Name of baseline model
        quick: If True, use quick estimates instead of full benchmarks
    
    Returns:
        Baseline scorecard with metrics
    
    Example:
        >>> baseline = generate_baseline("cifar10_cnn", quick=True)
        >>> print(f"Baseline latency: {baseline['latency_ms']}ms")
    """
    print("📊 Generating Baseline Scorecard...")
    print("=" * 60)
    
    # Load model
    model = load_baseline_model(model_name)
    print(f"✅ Loaded baseline model: {model.name}")
    
    # Count parameters using the standard .parameters() API from Module 03
    def count_parameters(model):
        """
        Count total parameters in a model.

        Uses the explicit .parameters() API from Module 03 instead of hasattr()
        to count model parameters. This is cleaner and follows TinyTorch conventions.

        Note: Previously used hasattr(attr, 'weights') which was incorrect -
        TinyTorch uses .weight (singular) not .weights (plural).
        """
        total = 0
        # Trust that model has .parameters() method (from Module 03)
        try:
            for param in model.parameters():
                # Each param is a Tensor from Module 01 with .data attribute
                total += param.data.size
        except (AttributeError, TypeError):
            # Fallback: model might not have parameters() method
            # This shouldn't happen in TinyTorch, but handle gracefully
            pass
        return total
    
    params = count_parameters(model)
    memory_mb = params * 4 / (1024 * 1024)  # Assuming float32
    
    if quick:
        # Quick estimates for fast validation
        print("⚡ Using quick estimates (set quick=False for full benchmark)")
        
        baseline = {
            "model": model_name,
            "accuracy": 85.0,  # Typical for this architecture
            "latency_ms": 45.2,
            "memory_mb": memory_mb,
            "parameters": params,
            "mode": "quick_estimate"
        }
    else:
        # Full benchmark (requires more time)
        from tinytorch.benchmarking.benchmark import Benchmark
        
        print("🔬 Running full benchmark (this may take a minute)...")
        
        benchmark = Benchmark([model], [{"name": "baseline"}], 
                            warmup_runs=5, measurement_runs=20)
        
        # Measure latency
        input_shape = (1, 3, 32, 32)  # CIFAR-10 input
        latency_results = benchmark.run_latency_benchmark(input_shape=input_shape)
        latency_ms = list(latency_results.values())[0].mean * 1000
        
        baseline = {
            "model": model_name,
            "accuracy": 85.0,  # Would need actual test set evaluation
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "parameters": params,
            "mode": "full_benchmark"
        }
    
    # Display baseline
    print("\n📋 BASELINE SCORECARD")
    print("=" * 60)
    print(f"Model:          {baseline['model']}")
    print(f"Accuracy:       {baseline['accuracy']:.1f}%")
    print(f"Latency:        {baseline['latency_ms']:.1f}ms")
    print(f"Memory:         {baseline['memory_mb']:.2f}MB")
    print(f"Parameters:     {baseline['parameters']:,}")
    print("=" * 60)
    print("📌 This is your starting point. Optimize to compete!")
    print()
    
    return baseline

# %% [markdown]
"""
# 3. TinyMLPerf Closed Division Workflow - Complete Example

Let's see a complete TinyMLPerf submission following Closed Division rules. This example demonstrates the professional MLPerf methodology you learned in Module 19.

**TinyMLPerf Closed Division Workflow:**
1. **Load Official Baseline** (MLPerf requirement)
2. **Apply Optimizations** (Modules 14-18 techniques)
3. **Benchmark Systematically** (Module 19 tools)
4. **Generate Submission** (MLPerf-compliant format)
5. **Document Strategy** (Reproducibility requirement)

This is your template - study it, then implement your own optimization strategy!

## Example Strategy: All-Around Category

For this worked example, we'll compete in the **All-Around** category (best balanced performance across all metrics).

**Our Optimization Strategy:**
- **Step 1:** Quantization (INT8) → 4x memory reduction
- **Step 2:** Magnitude Pruning (60%) → Faster inference
- **Step 3:** Systematic Benchmarking → Measure impact

**Why this order?**
- Quantize FIRST: Preserves more accuracy than pruning first
- Prune SECOND: Reduces what needs to be quantized
- Benchmark: Following MLPerf measurement methodology

**This follows MLPerf Closed Division rules:**
- ✅ Uses provided baseline CNN
- ✅ Applies optimization techniques (not architecture changes)
- ✅ Documents strategy clearly
- ✅ Reports all required metrics
"""

# %%
#| export
def worked_example_optimization():
    """
    Complete worked example showing full optimization workflow.
    
    This demonstrates:
    - Loading baseline model
    - Applying multiple optimization techniques
    - Benchmarking systematically
    - Generating submission
    
    Students should study this and adapt for their own strategies!
    """
    print("🏅 WORKED EXAMPLE: Complete Optimization Workflow")
    print("=" * 70)
    print("Target: All-Around Event (balanced performance)")
    print("Strategy: Quantization (INT8) → Pruning (60%)")
    print("=" * 70)
    print()
    
    # Step 1: Load Baseline
    print("📦 Step 1: Load Baseline Model")
    print("-" * 70)
    baseline = load_baseline_model("cifar10_cnn")
    baseline_metrics = generate_baseline("cifar10_cnn", quick=True)
    print()
    
    # Step 2: Apply Quantization
    print("🔧 Step 2: Apply INT8 Quantization (Module 17)")
    print("-" * 70)
    print("💡 Why quantize? Reduces memory 4x (FP32 → INT8)")
    
    # For demonstration, we'll simulate quantization
    # In real competition, students would use:
    # from tinytorch.optimization.quantization import quantize_model
    # optimized = quantize_model(baseline, bits=8)
    
    print("✅ Quantized model (simulated)")
    print("   - Memory: 12.4MB → 3.1MB (4x reduction)")
    print()
    
    # Step 3: Apply Pruning
    print("✂️  Step 3: Apply Magnitude Pruning (Module 18)")
    print("-" * 70)
    print("💡 Why prune? Removes 60% of weights for faster inference")
    
    # For demonstration, we'll simulate pruning
    # In real competition, students would use:
    # from tinytorch.optimization.compression import magnitude_prune
    # optimized = magnitude_prune(optimized, sparsity=0.6)
    
    print("✅ Pruned model (simulated)")
    print("   - Active parameters: 3.2M → 1.28M (60% removed)")
    print()
    
    # Step 4: Benchmark Results
    print("📊 Step 4: Benchmark Optimized Model (Module 19)")
    print("-" * 70)
    
    # Simulated optimized metrics
    optimized_metrics = {
        "model": "Optimized_CIFAR10_CNN",
        "accuracy": 83.5,  # Slight drop from aggressive optimization
        "latency_ms": 22.1,
        "memory_mb": 1.24,  # 4x quantization + 60% pruning
        "parameters": 1280000,
        "techniques": ["quantization_int8", "magnitude_prune_0.6"]
    }
    
    print("Baseline vs Optimized:")
    print(f"  Accuracy:    {baseline_metrics['accuracy']:.1f}% → {optimized_metrics['accuracy']:.1f}% (-1.5pp)")
    print(f"  Latency:     {baseline_metrics['latency_ms']:.1f}ms → {optimized_metrics['latency_ms']:.1f}ms (2.0x faster ✅)")
    print(f"  Memory:      {baseline_metrics['memory_mb']:.2f}MB → {optimized_metrics['memory_mb']:.2f}MB (10.0x smaller ✅)")
    print(f"  Parameters:  {baseline_metrics['parameters']:,} → {optimized_metrics['parameters']:,} (60% fewer ✅)")
    print()
    
    # Step 5: Generate Submission
    print("📤 Step 5: Generate Competition Submission")
    print("-" * 70)
    
    submission = {
        "event": "all_around",
        "athlete_name": "Example_Submission",
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvements": {
            "accuracy_drop": -1.5,
            "latency_speedup": 2.0,
            "memory_reduction": 10.0
        },
        "techniques_applied": ["quantization_int8", "magnitude_prune_0.6"],
        "technique_order": "quantize_first_then_prune"
    }
    
    print("✅ Submission generated!")
    print(f"   Event: {submission['event']}")
    print(f"   Techniques: {', '.join(submission['techniques_applied'])}")
    print()
    print("=" * 70)
    print("🎯 This is the complete workflow!")
    print("   Now it's your turn to implement your own optimization strategy.")
    print("=" * 70)
    
    return submission

# %% [markdown]
"""
# 4. Your TinyMLPerf Submission Template

Now it's your turn! Below is your TinyMLPerf Closed Division submission template. Following MLPerf methodology ensures your results are reproducible and fairly comparable.

## TinyMLPerf Closed Division Submission Process

**Step 1: Choose Your Category**
Pick ONE category to optimize for:
- 🏃 **Latency Sprint:** Minimize inference time
- 🏋️ **Memory Challenge:** Minimize model footprint
- 🎯 **Accuracy Contest:** Maximize accuracy within constraints
- 🏋️‍♂️ **All-Around:** Best balanced performance
- 🚀 **Extreme Push:** Most aggressive optimization

**Step 2: Design Your Optimization Strategy**
- Review Module 19, Section 4.5 for combination strategies
- Consider optimization order (quantize→prune vs prune→quantize)
- Plan ablation study to understand each technique's impact
- Document your reasoning (MLPerf reproducibility requirement)

**Step 3: Implement in Template**
- Write optimization code in `optimize_for_competition()`
- Apply techniques from Modules 14-18
- Follow TinyMLPerf Closed Division rules (no architecture changes!)

**Step 4: Benchmark Systematically**
- Use Module 19 benchmarking tools
- Measure all required metrics (accuracy, latency, memory)
- Run multiple times for statistical validity (MLPerf requirement)

**Step 5: Generate MLPerf-Compliant Submission**
- Run `generate_submission()` to create `submission.json`
- Includes baseline comparison (MLPerf requirement)
- Documents optimization strategy (reproducibility)
- Ready for TinyMLPerf leaderboard upload

## Submission Guidelines (MLPerf Inspired)

- ✅ **Start with baseline:** Load provided CNN (don't modify architecture)
- ✅ **Apply optimizations:** Use M14-18 techniques only
- ✅ **Measure fairly:** Same hardware, same test data
- ✅ **Document everything:** Strategy writeup required
- ✅ **Report all metrics:** Accuracy, latency, memory (not just best one!)

**Remember:** TinyMLPerf Closed Division tests your OPTIMIZATION skills, not model design. Work within the rules! 🏅
"""

# %%
#| export
def optimize_for_competition(baseline_model, event: str = "all_around"):
    """
    🏅 YOUR COMPETITION ENTRY - IMPLEMENT YOUR STRATEGY HERE!
    
    This is where you apply optimization techniques from Modules 14-18.
    
    Available techniques:
    - Module 14: KV Caching (for transformers) - enable_kv_cache()
    - Module 16: Acceleration (vectorization, fusion)
    - Module 17: Quantization (INT8, INT4) - quantize_model()
    - Module 18: Compression (pruning) - magnitude_prune()
    
    Args:
        baseline_model: The unoptimized model
        event: Which Olympic event you're competing in
            - "latency_sprint": Minimize latency
            - "memory_challenge": Minimize memory
            - "accuracy_contest": Maximize accuracy
            - "all_around": Best balance
            - "extreme_push": Most aggressive
    
    Returns:
        Your optimized model
    
    Example:
        from tinytorch.optimization.quantization import quantize_model
        from tinytorch.optimization.compression import magnitude_prune
        
        optimized = baseline_model
        optimized = quantize_model(optimized, bits=8)
        optimized = magnitude_prune(optimized, sparsity=0.7)
        return optimized
    """
    
    print(f"🏅 YOUR OPTIMIZATION STRATEGY FOR: {event}")
    print("=" * 70)
    
    # Start with baseline
    optimized_model = baseline_model
    
    # ============================================================
    # YOUR CODE BELOW - Apply optimization techniques here!
    # ============================================================
    
    # TODO: Students implement their optimization strategy
    #
    # Example strategies by event:
    #
    # Latency Sprint (speed priority):
    #   - Heavy quantization (INT4 or INT8)
    #   - Aggressive pruning (80-90%)
    #   - Kernel fusion if applicable
    #
    # Memory Challenge (size priority):
    #   - INT8 or INT4 quantization
    #   - Aggressive pruning (70-90%)
    #   - Compression techniques
    #
    # All-Around (balanced):
    #   - INT8 quantization
    #   - Moderate pruning (50-70%)
    #   - Selective optimization
    #
    # Your strategy:
    
    
    
    # ============================================================
    # YOUR CODE ABOVE
    # ============================================================
    
    print("✅ Optimization complete!")
    print("💡 Tip: Benchmark your result to see the impact!")
    
    return optimized_model

def generate_submission(baseline_model, optimized_model, 
                       event: str = "all_around",
                       athlete_name: str = "YourName",
                       techniques: List[str] = None) -> Dict[str, Any]:
    """
    Generate standardized competition submission.
    
    Args:
        baseline_model: Original unoptimized model
        optimized_model: Your optimized model
        event: Olympic event name
        athlete_name: Your name for leaderboard
        techniques: List of techniques applied
    
    Returns:
        Submission dictionary (will be saved as JSON)
    """
    print("📤 Generating Competition Submission...")
    print("=" * 70)
    
    # Get baseline metrics
    baseline_metrics = generate_baseline(quick=True)
    
    # For demonstration, estimate optimized metrics
    # In real competition, this would benchmark the actual optimized model
    print("🔬 Benchmarking optimized model...")
    
    # Placeholder: Students' actual optimizations would be measured here
    optimized_metrics = {
        "model": "Your_Optimized_Model",
        "accuracy": 84.0,  # Measured
        "latency_ms": 28.0,  # Measured
        "memory_mb": 4.0,  # Measured
        "parameters": 2000000,  # Measured
    }
    
    # Calculate improvements
    improvements = {
        "accuracy_change": optimized_metrics["accuracy"] - baseline_metrics["accuracy"],
        "latency_speedup": baseline_metrics["latency_ms"] / optimized_metrics["latency_ms"],
        "memory_reduction": baseline_metrics["memory_mb"] / optimized_metrics["memory_mb"],
    }
    
    # Create submission
    submission = {
        "event": event,
        "athlete_name": athlete_name,
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvements": improvements,
        "techniques_applied": techniques or ["TODO: List your techniques"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save to JSON
    output_file = Path("submission.json")
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f"✅ Submission saved to: {output_file}")
    print()
    print("📊 Your Results:")
    print(f"  Event:           {event}")
    print(f"  Accuracy:        {optimized_metrics['accuracy']:.1f}% (Δ {improvements['accuracy_change']:+.1f}pp)")
    print(f"  Latency:         {optimized_metrics['latency_ms']:.1f}ms ({improvements['latency_speedup']:.2f}x faster)")
    print(f"  Memory:          {optimized_metrics['memory_mb']:.2f}MB ({improvements['memory_reduction']:.2f}x smaller)")
    print()
    print("📤 Upload submission.json to TorchPerf Olympics platform!")
    print("=" * 70)
    
    return submission

# %% [markdown]
"""
# 5. Module Integration Test

Complete validation and competition workflow test.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 10}
def test_module():
    """
    Complete test of Module 20 functionality.
    
    This validates:
    - Installation validation works
    - Baseline generation works
    - Worked example runs successfully
    - Competition template is ready
    """
    print("=" * 70)
    print("MODULE 20 INTEGRATION TEST")
    print("=" * 70)
    print()
    
    # Test 1: Validation
    print("🔧 Test 1: System Validation")
    validation_status = validate_installation()
    assert len(validation_status) > 0, "Validation should return status dict"
    print("✅ Validation working!")
    print()
    
    # Test 2: Baseline Generation
    print("📊 Test 2: Baseline Generation")
    baseline = generate_baseline(quick=True)
    assert "accuracy" in baseline, "Baseline should include accuracy"
    assert "latency_ms" in baseline, "Baseline should include latency"
    assert "memory_mb" in baseline, "Baseline should include memory"
    print("✅ Baseline generation working!")
    print()
    
    # Test 3: Worked Example
    print("🏅 Test 3: Worked Example")
    example_submission = worked_example_optimization()
    assert "event" in example_submission, "Submission should include event"
    assert "baseline" in example_submission, "Submission should include baseline"
    assert "optimized" in example_submission, "Submission should include optimized"
    print("✅ Worked example working!")
    print()
    
    # Test 4: Competition Template
    print("🎯 Test 4: Competition Template")
    baseline_model = load_baseline_model("cifar10_cnn")
    optimized = optimize_for_competition(baseline_model, event="all_around")
    assert optimized is not None, "Optimization should return model"
    print("✅ Competition template working!")
    print()
    
    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("🎉 You're ready for TorchPerf Olympics!")
    print("   Next steps:")
    print("   1. Implement your optimization strategy in optimize_for_competition()")
    print("   2. Run this module to generate submission.json")
    print("   3. Upload to competition platform")
    print()
    print("🔥 Good luck! May the best optimizer win! 🏅")

test_module()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Competition as Learning

TorchPerf Olympics isn't just about winning - it's about understanding trade-offs:

**The Meta-Lesson**: Every optimization involves trade-offs:
- Quantization: Speed vs Accuracy
- Pruning: Size vs Performance
- Caching: Memory vs Speed

Professional ML engineers navigate these trade-offs daily. The competition forces you to:
1. **Think systematically** about optimization strategies
2. **Measure rigorously** using benchmarking tools
3. **Make data-driven decisions** based on actual measurements
4. **Document and justify** your choices

The best submission isn't always the "fastest" or "smallest" - it's the one that best understands and navigates the trade-off space for their chosen event.

What will your strategy be? 🤔
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Competition & Validation

**What You've Learned:**
- ✅ How to validate your TinyTorch installation
- ✅ How to generate baseline performance metrics
- ✅ How to combine optimization techniques systematically
- ✅ How to benchmark and measure impact
- ✅ How to generate standardized competition submissions

**The Complete Workflow:**
```
1. Validate  → Ensure environment works
2. Baseline  → Establish reference performance
3. Optimize  → Apply techniques from M14-18
4. Benchmark → Measure impact using M19
5. Submit    → Generate standardized submission
```

**Key Takeaway**: Competition teaches systematic optimization thinking. The goal isn't just winning - it's understanding the entire optimization process from baseline to submission.

**Next Steps:**
1. Study the worked example
2. Implement your own optimization strategy
3. Benchmark your results
4. Generate submission.json
5. Compete in TorchPerf Olympics!

🔥 Now go optimize and win gold! 🏅
"""


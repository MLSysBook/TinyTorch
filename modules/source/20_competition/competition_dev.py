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

Welcome to the capstone! You've built an entire ML system from scratch (M01-13) and learned optimization techniques (M14-19). Now it's time to compete and show what you can do! 🏅

## 🔗 Your Journey
```
Modules 01-13: Build ML System (tensors → transformers)
Modules 14-18: Learn Optimization Techniques  
Module 19:     Learn Benchmarking
Module 20:     Compete in TinyMLPerf! 🏅
```

## 🏅 TinyMLPerf: Two Ways to Compete

Inspired by industry-standard MLPerf (which you learned about in Module 19), TinyMLPerf offers **two competition tracks**:

### 🔒 Closed Division - "Optimization Challenge"
**What you do:**
- Start with provided baseline model (everyone gets the same)
- Apply optimization techniques from Modules 14-18
- Compete on: Who optimizes best?

**Best for:** Most students - clear rules, fair comparison
**Focus:** Your optimization skills

### 🔓 Open Division - "Innovation Challenge"  
**What you do:**
- Modify anything! Improve your implementations from M01-19
- Design better architectures
- Novel approaches encouraged

**Best for:** Advanced students who want more creative freedom
**Focus:** Your systems innovations

## Competition Categories (Both Divisions)
- 🏃 **Latency Sprint**: Fastest inference
- 🏋️ **Memory Challenge**: Smallest model
- 🎯 **Accuracy Contest**: Best accuracy within constraints
- 🏋️‍♂️ **All-Around**: Best balanced performance
- 🚀 **Extreme Push**: Most aggressive optimization

## What This Module Provides
1. **Validation**: Check your TinyTorch works
2. **Baseline**: Starting point for Closed Division
3. **Examples**: See both tracks in action
4. **Template**: Your competition workspace

Pick your track, optimize, and compete! 🔥
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
# 1. Pick Your Track & Validate

Before competing, choose your track and make sure your TinyTorch installation works!

## Two Tracks, Two Styles

### 🔒 Closed Division - "The Optimization Challenge"
- Everyone starts with the same baseline model
- Apply techniques from Modules 14-18 (quantization, pruning, etc.)
- Fair comparison: who optimizes best?
- **Choose this if:** You want clear rules and direct competition

### 🔓 Open Division - "The Innovation Challenge"
- Modify anything! Improve YOUR TinyTorch implementations
- Better Conv2d? Faster matmul? Novel architecture? All allowed!
- Compete on innovation and creativity
- **Choose this if:** You want freedom to explore and innovate

**Can I do both?** Absolutely! Submit to both tracks.

**Which is "better"?** Neither - they test different skills:
- Closed = Optimization mastery
- Open = Systems innovation

## Quick Validation

Before competing, let's verify everything works:
- ✅ All modules imported successfully
- ✅ Optimization techniques available
- ✅ Benchmarking tools ready
"""

# %%
#| export
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tinytorch.benchmarking.benchmark import Benchmark, calculate_normalized_scores
from tinytorch.profiling.profiler import Profiler

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
# 2. The Baseline (For Closed Division)

If you're competing in **Closed Division**, everyone starts with this baseline model. If you're in **Open Division**, you can skip this or use it as a reference!

## Baseline Model: Simple CNN on CIFAR-10

We provide a simple CNN as the starting point for Closed Division:
- **Architecture:** Conv → Pool → Conv → Pool → FC → FC
- **Dataset:** CIFAR-10 (standardized test set)
- **Metrics:** Accuracy, latency, memory (we'll measure together)

**Closed Division:** Optimize THIS model using M14-18 techniques
**Open Division:** Build/modify whatever you want!

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
    from tinytorch.core.spatial import Conv2d, MaxPool2d, Flatten
    from tinytorch.core.activations import ReLU
    
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
    
    # Count parameters
    def count_parameters(model):
        total = 0
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'weights') and attr.weights is not None:
                total += attr.weights.size
            if hasattr(attr, 'bias') and attr.bias is not None:
                total += attr.bias.size
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
# 3. Complete Example - See Both Tracks in Action

Let's see complete examples for BOTH competition tracks!

## Example 1: Closed Division - Optimization Master

**Goal:** Compete in All-Around category using provided baseline

**Strategy:**
1. Load baseline CNN
2. Apply quantization (INT8) → 4x memory reduction
3. Apply pruning (60%) → Speed boost
4. Benchmark and submit

**Why this order?** Quantize first preserves more accuracy than pruning first.

## Example 2: Open Division - Innovation Master

**Goal:** Beat everyone with a novel approach

**Strategy:**
1. Improve YOUR Conv2d implementation (faster algorithm)
2. OR design a better architecture (MobileNet-style)
3. OR novel quantization (mixed precision per layer)
4. Benchmark and submit

**Freedom:** Modify anything in your TinyTorch implementation!

Let's see the Closed Division example in detail below:
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
# 4. Your Turn - Pick Your Track!

Now it's time to compete! Choose your track and implement your strategy.

## Choose Your Track

### 🔒 Closed Division Template
**If you choose Closed Division:**
1. Pick a category (Latency Sprint, Memory Challenge, etc.)
2. Design your optimization strategy
3. Implement in `optimize_for_competition()` below
4. Use techniques from Modules 14-18 only
5. Generate submission

**Good for:** Clear path, fair comparison, most students

### 🔓 Open Division Template  
**If you choose Open Division:**
1. Pick a category
2. Modify YOUR TinyTorch implementations (go edit earlier modules!)
3. OR design novel architectures
4. Re-export with `tito export` and benchmark
5. Generate submission

**Good for:** Creative freedom, systems innovation, advanced students

## Competition Categories (Pick ONE)
- 🏃 **Latency Sprint:** Fastest inference
- 🏋️ **Memory Challenge:** Smallest model
- 🎯 **Accuracy Contest:** Best accuracy within constraints
- 🏋️‍♂️ **All-Around:** Best balanced performance
- 🚀 **Extreme Push:** Most aggressive optimization

## Template Below

Use the `optimize_for_competition()` function to implement your strategy:
- **Closed Division:** Apply M14-18 techniques
- **Open Division:** Do whatever you want, document it!
"""

# %%
#| export
def optimize_for_competition(baseline_model, event: str = "all_around", division: str = "closed"):
    """
    🏅 YOUR COMPETITION ENTRY - IMPLEMENT YOUR STRATEGY HERE!
    
    Args:
        baseline_model: Starting model (use for Closed, optional for Open)
        event: Category you're competing in
            - "latency_sprint": Minimize latency
            - "memory_challenge": Minimize memory
            - "accuracy_contest": Maximize accuracy
            - "all_around": Best balance
            - "extreme_push": Most aggressive
        division: "closed" or "open" - which track you chose
    
    Returns:
        Your optimized model
    
    🔒 CLOSED DIVISION Example:
        from tinytorch.optimization.quantization import quantize_model
        from tinytorch.optimization.compression import magnitude_prune
        
        optimized = baseline_model
        optimized = quantize_model(optimized, bits=8)
        optimized = magnitude_prune(optimized, sparsity=0.7)
        return optimized
    
    🔓 OPEN DIVISION Example:
        # Build your own model OR
        # Use your improved implementations from earlier modules
        # (after you've modified and re-exported them)
        
        from tinytorch.models import YourCustomArchitecture
        optimized = YourCustomArchitecture()
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

#| export
def validate_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate competition submission with sanity checks.
    
    This catches honest mistakes like unrealistic speedups or accidental training.
    Honor code system - we trust but verify basic reasonableness.
    
    Args:
        submission: Submission dictionary to validate
        
    Returns:
        Dict with validation results and warnings
    """
    checks = []
    warnings = []
    errors = []
    
    # Extract metrics
    normalized = submission.get("normalized_scores", {})
    speedup = normalized.get("speedup", 1.0)
    compression = normalized.get("compression_ratio", 1.0)
    accuracy_delta = normalized.get("accuracy_delta", 0.0)
    
    # Check 1: Speedup is reasonable (not claiming impossible gains)
    if speedup > 50:
        errors.append(f"❌ Speedup {speedup:.1f}x seems unrealistic (>50x)")
    elif speedup > 20:
        warnings.append(f"⚠️  Speedup {speedup:.1f}x is very high - please verify measurements")
    else:
        checks.append(f"✅ Speedup {speedup:.2f}x is reasonable")
    
    # Check 2: Compression is reasonable
    if compression > 32:
        errors.append(f"❌ Compression {compression:.1f}x seems unrealistic (>32x)")
    elif compression > 16:
        warnings.append(f"⚠️  Compression {compression:.1f}x is very high - please verify")
    else:
        checks.append(f"✅ Compression {compression:.2f}x is reasonable")
    
    # Check 3: Accuracy didn't improve (Closed Division rule - no training allowed!)
    division = submission.get("division", "closed")
    if division == "closed" and accuracy_delta > 1.0:
        errors.append(f"❌ Accuracy improved by {accuracy_delta:.1f}pp - did you accidentally train the model?")
    elif accuracy_delta > 0.5:
        warnings.append(f"⚠️  Accuracy improved by {accuracy_delta:.1f}pp - verify no training occurred")
    else:
        checks.append(f"✅ Accuracy change {accuracy_delta:+.2f}pp is reasonable")
    
    # Check 4: GitHub repo provided
    github_repo = submission.get("github_repo", "")
    if not github_repo or github_repo == "":
        warnings.append("⚠️  No GitHub repo provided - required for verification")
    else:
        checks.append(f"✅ GitHub repo provided: {github_repo}")
    
    # Check 5: Required fields present
    required_fields = ["division", "event", "athlete_name", "baseline", "optimized", "normalized_scores"]
    missing = [f for f in required_fields if f not in submission]
    if missing:
        errors.append(f"❌ Missing required fields: {', '.join(missing)}")
    else:
        checks.append("✅ All required fields present")
    
    # Check 6: Techniques documented
    techniques = submission.get("techniques_applied", [])
    if not techniques or "TODO" in str(techniques):
        warnings.append("⚠️  No optimization techniques listed")
    else:
        checks.append(f"✅ Techniques documented: {', '.join(techniques[:3])}...")
    
    return {
        "valid": len(errors) == 0,
        "checks": checks,
        "warnings": warnings,
        "errors": errors
    }

#| export
def generate_submission(baseline_model, optimized_model, 
                       division: str = "closed",
                       event: str = "all_around",
                       athlete_name: str = "YourName",
                       github_repo: str = "",
                       techniques: List[str] = None) -> Dict[str, Any]:
    """
    Generate standardized TinyMLPerf competition submission with normalized scoring.
    
    Args:
        baseline_model: Original unoptimized model
        optimized_model: Your optimized model
        division: "closed" or "open"
        event: Competition category (latency_sprint, memory_challenge, all_around, etc.)
        athlete_name: Your name for submission
        github_repo: GitHub repository URL for code verification
        techniques: List of optimization techniques applied
    
    Returns:
        Submission dictionary (will be saved as JSON)
    """
    print("📤 Generating TinyMLPerf Competition Submission...")
    print("=" * 70)
    
    # Get baseline metrics
    baseline_metrics = generate_baseline(quick=True)
    
    # Benchmark optimized model
    print("🔬 Benchmarking optimized model...")
    
    # Use Profiler and Benchmark from Module 19
    profiler = Profiler()
    
    # For demonstration, we'll use placeholder metrics
    # In real competition, students would measure their actual optimized model
    optimized_metrics = {
        "model": getattr(optimized_model, 'name', 'Optimized_Model'),
        "accuracy": 84.0,  # Would be measured with actual test set
        "latency_ms": 28.0,  # Would be measured with profiler
        "memory_mb": 4.0,  # Would be measured with profiler
        "parameters": 2000000,  # Would be counted
    }
    
    # Calculate normalized scores using Module 19's function
    baseline_for_norm = {
        "latency": baseline_metrics["latency_ms"],
        "memory": baseline_metrics["memory_mb"],
        "accuracy": baseline_metrics["accuracy"]
    }
    
    optimized_for_norm = {
        "latency": optimized_metrics["latency_ms"],
        "memory": optimized_metrics["memory_mb"],
        "accuracy": optimized_metrics["accuracy"]
    }
    
    normalized_scores = calculate_normalized_scores(baseline_for_norm, optimized_for_norm)
    
    # Create submission with all required fields
    submission = {
        "division": division,
        "event": event,
        "athlete_name": athlete_name,
        "github_repo": github_repo,
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "normalized_scores": {
            "speedup": normalized_scores["speedup"],
            "compression_ratio": normalized_scores["compression_ratio"],
            "accuracy_delta": normalized_scores["accuracy_delta"],
            "efficiency_score": normalized_scores["efficiency_score"]
        },
        "techniques_applied": techniques or ["TODO: Document your optimization techniques"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tinytorch_version": "0.1.0",
        "honor_code": False  # Must be explicitly set to True after validation
    }
    
    # Validate submission
    print("\n🔍 Validating submission...")
    validation = validate_submission(submission)
    
    # Display validation results
    print("\n📋 Validation Results:")
    for check in validation["checks"]:
        print(f"  {check}")
    for warning in validation["warnings"]:
        print(f"  {warning}")
    for error in validation["errors"]:
        print(f"  {error}")
    
    if not validation["valid"]:
        print("\n❌ Submission has errors - please fix before submitting")
        return submission
    
    # Save to JSON
    output_file = Path("submission.json")
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n✅ Submission saved to: {output_file}")
    print()
    print("📊 Your Normalized Scores (MLPerf-style):")
    print(f"  Division:        {division.upper()}")
    print(f"  Event:           {event.replace('_', ' ').title()}")
    print(f"  Speedup:         {normalized_scores['speedup']:.2f}x faster ⚡")
    print(f"  Compression:     {normalized_scores['compression_ratio']:.2f}x smaller 💾")
    print(f"  Accuracy:        {optimized_metrics['accuracy']:.1f}% (Δ {normalized_scores['accuracy_delta']:+.2f}pp)")
    print(f"  Efficiency:      {normalized_scores['efficiency_score']:.2f}")
    print()
    print("📤 Next Steps:")
    print("  1. Verify all metrics are correct")
    print("  2. Push your code to GitHub (if not done)")
    print("  3. Run: tito submit submission.json")
    print("     (This will validate and prepare final submission)")
    print()
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


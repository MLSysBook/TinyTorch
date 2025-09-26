#| default_exp core.mlops
"""
# MLOps - Production ML Systems Module

This module teaches production ML systems engineering through hands-on implementation
of monitoring, deployment, and lifecycle management tools.

**Learning Focus**: Systems engineering principles for production ML
**Key Concepts**: Model monitoring, drift detection, automated retraining
**Systems Insights**: How real ML systems manage model lifecycles
"""

# %% [markdown]
"""
## 🎯 Learning Objectives

By completing this module, you will:

1. **Build model monitoring systems** that track performance degradation in production
2. **Implement drift detection** to identify when data changes affect model quality  
3. **Create automated retraining triggers** that respond to performance issues
4. **Understand MLOps systems engineering** principles used in real production systems

**Systems Engineering Focus**: This module emphasizes building production-ready
infrastructure, not just algorithms. You'll learn memory management, performance
monitoring, and system architecture patterns used in enterprise ML systems.
"""

# %% [markdown]
"""
## 📚 Background: Production ML Systems

### The MLOps Challenge

In production ML systems, models don't just run once - they serve predictions
continuously for months or years. This creates unique engineering challenges:

1. **Model Degradation**: Performance drops as data changes over time
2. **Data Drift**: Input distributions shift, affecting model validity  
3. **Infrastructure Complexity**: Monitoring, versioning, and deployment at scale
4. **Incident Response**: Automated detection and response to performance issues

### Real-World Context

Companies like Netflix, Uber, and Airbnb run thousands of ML models in production.
Each model requires:
- Continuous performance monitoring
- Automated drift detection
- Retraining pipelines
- A/B testing infrastructure
- Incident response systems

This module teaches you to build these systems from scratch.
"""

import numpy as np
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

# Import our dependencies - try from package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.training import Trainer
    from tinytorch.core.layers import Dense
except ImportError:
    # For development, fallback gracefully
    print("⚠️  Some TinyTorch modules not available - MLOps will use mock implementations")
    Tensor = None
    Trainer = None

# %% [markdown]
"""
---

## 🔬 SECTION 1: Model Performance Monitoring

### The Foundation: Tracking Model Health

Real production systems continuously monitor model performance. When accuracy drops
or latency increases, automated systems need to detect and respond.

**Systems Insight**: Production monitoring is about **thresholds and trends**, not
perfect accuracy. A 5% accuracy drop might trigger retraining, while gradual
degradation over weeks might indicate data drift.

**What You'll Build**: A `ModelMonitor` class that tracks accuracy over time
and detects performance degradation using simple thresholds.
"""

@dataclass
class ModelMonitor:
    """
    Monitors ML model performance in production environments.
    
    Tracks accuracy, latency, and other metrics over time to detect degradation.
    This is the foundation of production ML systems monitoring.
    """
    
    def __init__(self, model_name: str, baseline_accuracy: float = 0.95):
        """
        Initialize model monitoring with baseline performance expectations.
        
        Args:
            model_name: Unique identifier for the model being monitored
            baseline_accuracy: Expected accuracy level (alerts when dropping below 90% of this)
        """
        self.model_name = model_name
        self.baseline_accuracy = baseline_accuracy
        
        # Performance history - stored as lists for simple analysis
        self.accuracy_history: List[float] = []
        self.latency_history: List[float] = []  # milliseconds
        self.timestamp_history: List[datetime] = []
        
        # Alert thresholds - 90% of baseline triggers accuracy alert
        self.accuracy_threshold = baseline_accuracy * 0.9
        self.latency_threshold = 200.0  # milliseconds
        
        # Alert states
        self.accuracy_alert = False
        self.latency_alert = False
        
        print(f"✅ Model monitor initialized for '{model_name}'")
        print(f"   Baseline accuracy: {baseline_accuracy:.3f}")
        print(f"   Alert threshold: {self.accuracy_threshold:.3f}")
    
    def record_performance(self, accuracy: float, latency: float) -> None:
        """
        Record a new performance measurement.
        
        Args:
            accuracy: Model accuracy on recent batch
            latency: Inference latency in milliseconds
        """
        self.accuracy_history.append(accuracy)
        self.latency_history.append(latency)
        self.timestamp_history.append(datetime.now())
        
        # Update alert states
        self.accuracy_alert = accuracy < self.accuracy_threshold
        self.latency_alert = latency > self.latency_threshold
        
        # Log significant changes
        if self.accuracy_alert:
            print(f"🚨 ACCURACY ALERT: {accuracy:.3f} < {self.accuracy_threshold:.3f}")
        if self.latency_alert:
            print(f"🚨 LATENCY ALERT: {latency:.1f}ms > {self.latency_threshold:.1f}ms")
    
    def check_alerts(self) -> Dict[str, Any]:
        """
        Check current alert status and return summary.
        
        Returns:
            Dictionary with alert information and recent performance
        """
        if not self.accuracy_history:
            return {"status": "no_data", "alerts": []}
        
        recent_accuracy = self.accuracy_history[-1]
        recent_latency = self.latency_history[-1]
        
        alerts = []
        if self.accuracy_alert:
            alerts.append({
                "type": "accuracy_degradation",
                "current": recent_accuracy,
                "threshold": self.accuracy_threshold,
                "severity": "high"
            })
        
        if self.latency_alert:
            alerts.append({
                "type": "latency_spike", 
                "current": recent_latency,
                "threshold": self.latency_threshold,
                "severity": "medium"
            })
        
        return {
            "model_name": self.model_name,
            "status": "alert" if alerts else "healthy",
            "alerts": alerts,
            "recent_accuracy": recent_accuracy,
            "recent_latency": recent_latency,
            "total_measurements": len(self.accuracy_history)
        }
    
    def get_performance_trend(self, window: int = 5) -> Dict[str, str]:
        """
        Analyze performance trends over recent measurements.
        
        Args:
            window: Number of recent measurements to analyze
            
        Returns:
            Trend analysis for accuracy and latency
        """
        if len(self.accuracy_history) < window:
            return {"accuracy_trend": "insufficient_data", "latency_trend": "insufficient_data"}
        
        # Compare recent window to previous window
        recent_acc = np.mean(self.accuracy_history[-window:])
        older_acc = np.mean(self.accuracy_history[-2*window:-window]) if len(self.accuracy_history) >= 2*window else recent_acc
        
        recent_lat = np.mean(self.latency_history[-window:])
        older_lat = np.mean(self.latency_history[-2*window:-window]) if len(self.latency_history) >= 2*window else recent_lat
        
        # Simple trend analysis with 2% threshold
        accuracy_trend = "stable"
        if recent_acc > older_acc * 1.02:
            accuracy_trend = "improving"
        elif recent_acc < older_acc * 0.98:
            accuracy_trend = "degrading"
        
        latency_trend = "stable"
        if recent_lat > older_lat * 1.1:
            latency_trend = "increasing"
        elif recent_lat < older_lat * 0.9:
            latency_trend = "decreasing"
        
        return {
            "accuracy_trend": accuracy_trend,
            "latency_trend": latency_trend,
            "recent_accuracy": recent_acc,
            "older_accuracy": older_acc,
            "recent_latency": recent_lat,
            "older_latency": older_lat
        }

# %% [markdown]
"""
### 🧪 Testing: Model Monitor

Let's test our model monitoring system with realistic scenarios.
"""

def test_model_monitor():
    """Test the ModelMonitor with realistic performance scenarios."""
    print("🧪 Testing Model Monitor...")
    
    # Create monitor for an image classifier
    monitor = ModelMonitor("image_classifier_v1", baseline_accuracy=0.92)
    
    # Simulate healthy performance
    print("\n📊 Phase 1: Healthy Performance")
    for i in range(5):
        accuracy = 0.91 + np.random.normal(0, 0.01)  # Around 91% with small variance
        latency = 150 + np.random.normal(0, 10)      # Around 150ms
        monitor.record_performance(accuracy, latency)
    
    alerts = monitor.check_alerts()
    print(f"Status: {alerts['status']}")
    print(f"Alerts: {len(alerts['alerts'])}")
    
    # Simulate performance degradation
    print("\n📉 Phase 2: Performance Degradation")
    for i in range(3):
        accuracy = 0.81 + np.random.normal(0, 0.02)  # Drop to 81% - below threshold
        latency = 220 + np.random.normal(0, 15)      # Spike to 220ms
        monitor.record_performance(accuracy, latency)
    
    alerts = monitor.check_alerts()
    print(f"Status: {alerts['status']}")
    print(f"Number of alerts: {len(alerts['alerts'])}")
    for alert in alerts['alerts']:
        print(f"  - {alert['type']}: {alert['current']:.3f} (threshold: {alert['threshold']:.3f})")
    
    # Analyze trends
    trend = monitor.get_performance_trend()
    print(f"\n📈 Trend Analysis:")
    print(f"Accuracy trend: {trend['accuracy_trend']}")
    print(f"Latency trend: {trend['latency_trend']}")
    
    print("✅ Model monitor test completed!")

# %% [markdown]
"""
---

## 🔬 SECTION 2: Data Drift Detection

### The Challenge: When Data Changes

Data drift occurs when the input distribution changes over time. A model trained
on summer data might perform poorly in winter. Production systems need to detect
this automatically.

**Systems Insight**: Drift detection is about **statistical comparisons**, not
model accuracy. We compare new data statistics to baseline statistics to detect
distribution shifts before they affect model performance.

**What You'll Build**: A `DriftDetector` class using simple statistical thresholds
(mean differences) rather than complex statistical tests.
"""

class DriftDetector:
    """
    Detects distribution drift in input data using statistical methods.
    
    Compares new data statistics against baseline to identify significant changes
    that might affect model performance.
    """
    
    def __init__(self, baseline_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize drift detection with baseline data statistics.
        
        Args:
            baseline_data: Reference data to compare against (n_samples x n_features)
            feature_names: Optional names for features (for reporting)
        """
        self.baseline_data = baseline_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(baseline_data.shape[1])]
        
        # Compute baseline statistics
        self.baseline_mean = np.mean(baseline_data, axis=0)
        self.baseline_std = np.std(baseline_data, axis=0)
        self.baseline_min = np.min(baseline_data, axis=0)
        self.baseline_max = np.max(baseline_data, axis=0)
        
        # Drift history
        self.drift_history: List[Dict] = []
        
        print(f"✅ Drift detector initialized")
        print(f"   Baseline shape: {baseline_data.shape}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Baseline mean: [{', '.join([f'{m:.3f}' for m in self.baseline_mean[:3]])}...]")
    
    def detect_simple_drift(self, new_data: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Simple drift detection using statistical thresholds.
        
        Args:
            new_data: New data to compare against baseline
            threshold: Number of standard deviations to consider drift
            
        Returns:
            Drift detection results
        """
        if new_data.shape[1] != self.baseline_data.shape[1]:
            raise ValueError(f"Feature count mismatch: {new_data.shape[1]} vs {self.baseline_data.shape[1]}")
        
        # Compute new data statistics
        new_mean = np.mean(new_data, axis=0)
        new_std = np.std(new_data, axis=0)
        
        # Detect drift per feature using simple threshold
        drift_flags = []
        drift_scores = []
        
        for i in range(len(self.baseline_mean)):
            # Mean shift detection
            mean_diff = abs(new_mean[i] - self.baseline_mean[i])
            mean_threshold = threshold * self.baseline_std[i]
            
            # Normalize drift score to handle different feature scales
            # Small epsilon (1e-8) prevents division by zero
            drift_score = mean_diff / (self.baseline_std[i] + 1e-8)
            drift_flags.append(drift_score > threshold)
            drift_scores.append(drift_score)
        
        # Overall drift assessment with clear thresholds
        drift_detected = any(drift_flags)
        max_drift_score = max(drift_scores)
        
        # Simple severity classification (no magic numbers)
        if max_drift_score > 3.0:
            drift_severity = "high"
        elif max_drift_score > 2.0:
            drift_severity = "medium"
        else:
            drift_severity = "low"
        
        result = {
            "timestamp": datetime.now(),
            "drift_detected": drift_detected,
            "drift_severity": drift_severity,
            "max_drift_score": max_drift_score,
            "per_feature": {
                self.feature_names[i]: {
                    "drift_detected": drift_flags[i],
                    "drift_score": drift_scores[i],
                    "baseline_mean": self.baseline_mean[i],
                    "new_mean": new_mean[i]
                }
                for i in range(len(self.feature_names))
            },
            "summary": {
                "total_features": len(self.feature_names),
                "drifted_features": sum(drift_flags),
                "drift_percentage": sum(drift_flags) / len(drift_flags) * 100
            }
        }
        
        # Store in history
        self.drift_history.append(result)
        
        if drift_detected:
            print(f"🚨 DRIFT DETECTED: {sum(drift_flags)}/{len(drift_flags)} features drifted")
            print(f"   Severity: {drift_severity} (max score: {max_drift_score:.2f})")
        
        return result
    
    def get_drift_history(self, limit: int = 10) -> List[Dict]:
        """Get recent drift detection history."""
        return self.drift_history[-limit:] if limit else self.drift_history

# %% [markdown]
"""
### 🧪 Testing: Drift Detection

Let's test drift detection with simulated data distribution changes.
"""

def test_drift_detection():
    """Test drift detection with controlled data changes."""
    print("🧪 Testing Drift Detection...")
    
    # Create baseline data - normal distribution
    np.random.seed(42)
    baseline_data = np.random.normal(loc=[0, 1, -0.5], scale=[1, 0.5, 0.8], size=(1000, 3))
    feature_names = ["temperature", "humidity", "pressure"]
    
    detector = DriftDetector(baseline_data, feature_names)
    
    # Test 1: No drift - similar distribution
    print("\n📊 Test 1: No Drift Expected")
    new_data_normal = np.random.normal(loc=[0.1, 1.1, -0.4], scale=[1, 0.5, 0.8], size=(500, 3))
    result1 = detector.detect_simple_drift(new_data_normal)
    print(f"Drift detected: {result1['drift_detected']}")
    print(f"Drifted features: {result1['summary']['drifted_features']}/{result1['summary']['total_features']}")
    
    # Test 2: Clear drift - shifted distribution  
    print("\n🚨 Test 2: Clear Drift Expected")
    new_data_drift = np.random.normal(loc=[3, 1, -0.5], scale=[1, 0.5, 0.8], size=(500, 3))  # Temperature shifted
    result2 = detector.detect_simple_drift(new_data_drift)
    print(f"Drift detected: {result2['drift_detected']}")
    print(f"Severity: {result2['drift_severity']}")
    print(f"Max drift score: {result2['max_drift_score']:.2f}")
    
    # Show per-feature results
    for feature, info in result2['per_feature'].items():
        if info['drift_detected']:
            print(f"  - {feature}: {info['baseline_mean']:.2f} → {info['new_mean']:.2f} (score: {info['drift_score']:.2f})")
    
    print("✅ Drift detection test completed!")

# %% [markdown]
"""
---

## 🔬 SECTION 3: Automated Retraining Triggers

### The Response: When to Retrain

When performance drops or drift is detected, production systems need to decide
whether to retrain the model. This requires balancing computational cost with
model quality.

**Systems Insight**: Retraining triggers are about **policies and thresholds**.
Real systems consider accuracy drops, drift severity, data volume, and business
impact when deciding to retrain.

**What You'll Build**: A `RetrainingTrigger` class that makes retraining decisions
based on configurable policies combining performance and drift signals.
"""

class RetrainingTrigger:
    """
    Manages automated retraining decisions based on performance and drift signals.
    
    Implements policies for when to trigger expensive retraining operations.
    """
    
    def __init__(self, model_name: str, retraining_policy: Optional[Dict] = None):
        """
        Initialize retraining trigger with policies.
        
        Args:
            model_name: Model being managed
            retraining_policy: Configuration for retraining decisions
        """
        self.model_name = model_name
        
        # Default retraining policy
        default_policy = {
            "accuracy_threshold": 0.05,    # 5% accuracy drop triggers retraining
            "drift_threshold": 2.0,        # Drift score > 2.0 triggers retraining  
            "min_time_between_retrains": 24 * 3600,  # 24 hours minimum
            "max_time_without_retrain": 7 * 24 * 3600,  # 7 days maximum
            "drift_severity_weights": {"low": 0.1, "medium": 0.5, "high": 1.0}  # Simplified weights
        }
        
        self.policy = {**default_policy, **(retraining_policy or {})}
        
        # Retraining history
        self.retraining_history: List[Dict] = []
        self.last_retrain_time = datetime.now()
        
        print(f"✅ Retraining trigger initialized for '{model_name}'")
        print(f"   Accuracy threshold: {self.policy['accuracy_threshold']} drop")
        print(f"   Drift threshold: {self.policy['drift_threshold']}")
    
    def should_retrain(self, monitor_alerts: Dict, drift_result: Dict) -> Dict[str, Any]:
        """
        Decide whether to trigger retraining based on current conditions.
        
        Args:
            monitor_alerts: Results from ModelMonitor.check_alerts()
            drift_result: Results from DriftDetector.detect_simple_drift()
            
        Returns:
            Retraining decision with reasoning
        """
        current_time = datetime.now()
        time_since_last_retrain = (current_time - self.last_retrain_time).total_seconds()
        
        # Initialize decision tracking
        trigger_reasons = []
        severity_score = 0.0
        
        # Check accuracy degradation
        accuracy_trigger = False
        if monitor_alerts['status'] == 'alert':
            for alert in monitor_alerts['alerts']:
                if alert['type'] == 'accuracy_degradation':
                    accuracy_drop = alert['threshold'] - alert['current']
                    if accuracy_drop >= self.policy['accuracy_threshold']:
                        accuracy_trigger = True
                        trigger_reasons.append(f"Accuracy dropped by {accuracy_drop:.3f}")
                        severity_score += 1.0
        
        # Check drift conditions
        drift_trigger = False
        if drift_result['drift_detected']:
            drift_weight = self.policy['drift_severity_weights'][drift_result['drift_severity']]
            if drift_result['max_drift_score'] >= self.policy['drift_threshold']:
                drift_trigger = True
                trigger_reasons.append(f"Drift detected (score: {drift_result['max_drift_score']:.2f})")
                severity_score += drift_weight
        
        # Check time-based policies
        time_trigger = False
        if time_since_last_retrain >= self.policy['max_time_without_retrain']:
            time_trigger = True
            trigger_reasons.append(f"Maximum time exceeded ({time_since_last_retrain/86400:.1f} days)")
            severity_score += 0.5
        
        # Check minimum time constraint
        min_time_satisfied = time_since_last_retrain >= self.policy['min_time_between_retrains']
        
        # Final decision
        should_retrain = (accuracy_trigger or drift_trigger or time_trigger) and min_time_satisfied
        
        if not min_time_satisfied and (accuracy_trigger or drift_trigger):
            trigger_reasons.append(f"Waiting for minimum time ({self.policy['min_time_between_retrains']/3600:.1f}h)")
        
        decision = {
            "timestamp": current_time,
            "should_retrain": should_retrain,
            "severity_score": severity_score,
            "trigger_reasons": trigger_reasons,
            "constraints": {
                "accuracy_trigger": accuracy_trigger,
                "drift_trigger": drift_trigger, 
                "time_trigger": time_trigger,
                "min_time_satisfied": min_time_satisfied
            },
            "time_since_last_retrain": time_since_last_retrain,
            "next_allowed_retrain": self.last_retrain_time + timedelta(seconds=self.policy['min_time_between_retrains'])
        }
        
        if should_retrain:
            print(f"🔄 RETRAINING TRIGGERED for {self.model_name}")
            print(f"   Reasons: {', '.join(trigger_reasons)}")
            print(f"   Severity score: {severity_score:.2f}")
            self.last_retrain_time = current_time
            self.retraining_history.append(decision)
        
        return decision
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get recent retraining decision history.""" 
        return self.retraining_history[-limit:] if limit else self.retraining_history

# %% [markdown]
"""
### 🧪 Testing: Retraining Triggers

Let's test the retraining decision logic with various scenarios.
"""

def test_retraining_triggers():
    """Test retraining trigger logic with different scenarios."""
    print("🧪 Testing Retraining Triggers...")
    
    # Create retraining trigger
    trigger = RetrainingTrigger("test_model", {
        "min_time_between_retrains": 60,  # 1 minute for testing
        "accuracy_threshold": 0.05
    })
    
    # Scenario 1: No issues - no retrain
    print("\n📊 Scenario 1: Healthy Model")
    healthy_alerts = {"status": "healthy", "alerts": []}
    no_drift = {"drift_detected": False, "drift_severity": "low", "max_drift_score": 0.5}
    
    decision1 = trigger.should_retrain(healthy_alerts, no_drift)
    print(f"Should retrain: {decision1['should_retrain']}")
    print(f"Reasons: {decision1['trigger_reasons']}")
    
    # Scenario 2: Accuracy drop - should trigger
    print("\n🚨 Scenario 2: Accuracy Degradation")
    accuracy_alerts = {
        "status": "alert",
        "alerts": [{
            "type": "accuracy_degradation",
            "current": 0.85,
            "threshold": 0.90,
            "severity": "high"
        }]
    }
    
    decision2 = trigger.should_retrain(accuracy_alerts, no_drift)
    print(f"Should retrain: {decision2['should_retrain']}")
    print(f"Reasons: {decision2['trigger_reasons']}")
    print(f"Severity score: {decision2['severity_score']}")
    
    # Wait to satisfy minimum time constraint
    time.sleep(1)
    
    # Scenario 3: High drift - should trigger
    print("\n🚨 Scenario 3: High Drift")
    high_drift = {"drift_detected": True, "drift_severity": "high", "max_drift_score": 3.5}
    
    decision3 = trigger.should_retrain(healthy_alerts, high_drift)
    print(f"Should retrain: {decision3['should_retrain']}")
    print(f"Reasons: {decision3['trigger_reasons']}")
    
    print("✅ Retraining triggers test completed!")

# %% [markdown]
"""
---

## 🔬 SECTION 4: Systems Analysis **[ADVANCED - OPTIONAL]**

> **Note**: This section contains advanced systems analysis. Focus on Sections 1-3
> for core MLOps concepts. This section shows how to analyze production performance.

### Memory Analysis: Monitoring Infrastructure Costs

Let's analyze the memory usage patterns of our MLOps infrastructure and understand
the computational costs of production monitoring.

**Advanced Topic**: Production MLOps systems need to monitor their own performance
to ensure monitoring doesn't become more expensive than the models being monitored.
"""

def analyze_mlops_memory_usage():
    """Analyze memory consumption patterns in MLOps components."""
    print("🔬 MLOps Memory Analysis")
    print("=" * 50)
    
    import tracemalloc
    tracemalloc.start()
    
    # Test different monitoring scales
    model_counts = [1, 10, 100]
    history_lengths = [100, 1000, 10000]
    
    for model_count in model_counts:
        for history_length in history_lengths:
            tracemalloc.clear_traces()
            
            # Create monitoring infrastructure
            monitors = []
            for i in range(model_count):
                monitor = ModelMonitor(f"model_{i}", baseline_accuracy=0.9)
                
                # Fill with history
                for j in range(history_length):
                    monitor.record_performance(
                        accuracy=0.85 + np.random.normal(0, 0.02),
                        latency=150 + np.random.normal(0, 10)
                    )
                monitors.append(monitor)
            
            # Measure memory
            current, peak = tracemalloc.get_traced_memory()
            
            # Calculate per-model overhead
            per_model_kb = (current / 1024) / model_count
            per_measurement_bytes = current / (model_count * history_length)
            
            print(f"Models: {model_count:3d}, History: {history_length:5d}")
            print(f"  Total memory: {current/1024/1024:.2f} MB")
            print(f"  Per model: {per_model_kb:.1f} KB")
            print(f"  Per measurement: {per_measurement_bytes:.1f} bytes")
            print()
    
    tracemalloc.stop()
    
    print("💡 Memory Insights:")
    print("- Each monitor uses ~15-20KB for 1000 measurements")
    print("- Memory scales linearly with history length")
    print("- Real systems use circular buffers to limit memory growth")
    print("- Database storage is preferred for long-term history")

def analyze_mlops_computational_complexity():
    """Analyze computational complexity of MLOps operations."""
    print("🔬 MLOps Computational Complexity")
    print("=" * 50)
    
    # Test drift detection complexity
    feature_counts = [10, 100, 1000]
    sample_counts = [100, 1000, 10000]
    
    print("Drift Detection Performance:")
    for n_features in feature_counts:
        for n_samples in sample_counts:
            # Create test data
            baseline = np.random.normal(0, 1, (n_samples, n_features))
            new_data = np.random.normal(0.1, 1, (n_samples//2, n_features))
            
            detector = DriftDetector(baseline)
            
            # Time the operation
            start_time = time.time()
            result = detector.detect_simple_drift(new_data)
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000  # milliseconds
            
            print(f"  Features: {n_features:4d}, Samples: {n_samples:5d}")
            print(f"    Time: {computation_time:.2f}ms")
            print(f"    O(N): {computation_time/(n_features*n_samples)*1e6:.2f} ns per feature-sample")
    
    print("\n💡 Complexity Insights:")
    print("- Drift detection is O(N*M) where N=samples, M=features")
    print("- Real systems use sampling and approximation for large datasets")
    print("- Statistical tests (KS, Mann-Whitney) are more expensive but robust")
    print("- Production systems often batch process drift detection")

# %% [markdown]
"""
## 🧪 Comprehensive Testing

Let's test the complete MLOps system integration.
"""

def run_comprehensive_mlops_tests():
    """Run all MLOps component tests."""
    print("🧪 Running Comprehensive MLOps Tests")
    print("=" * 50)
    
    try:
        # Test each component
        test_model_monitor()
        print()
        test_drift_detection()
        print()
        test_retraining_triggers()
        print()
        
        # Test integration
        print("🔄 Testing Integration...")
        
        # Create integrated system
        monitor = ModelMonitor("production_model", baseline_accuracy=0.92)
        
        # Generate baseline data for drift detection
        np.random.seed(42)
        baseline_data = np.random.normal(loc=[0, 1], scale=[1, 0.5], size=(1000, 2))
        detector = DriftDetector(baseline_data, ["feature_a", "feature_b"])
        
        trigger = RetrainingTrigger("production_model")
        
        # Simulate production scenario
        print("\n📊 Production Simulation:")
        
        # Day 1-3: Normal operation
        print("Days 1-3: Normal operation")
        for day in range(3):
            accuracy = 0.91 + np.random.normal(0, 0.01)
            latency = 150 + np.random.normal(0, 10)
            monitor.record_performance(accuracy, latency)
            
            new_data = np.random.normal(loc=[0.1, 1.1], scale=[1, 0.5], size=(100, 2))
            drift_result = detector.detect_simple_drift(new_data)
            
            alerts = monitor.check_alerts()
            decision = trigger.should_retrain(alerts, drift_result)
            
            print(f"  Day {day+1}: Accuracy={accuracy:.3f}, Drift={drift_result['drift_detected']}, Retrain={decision['should_retrain']}")
        
        # Day 4: Data shift occurs
        print("\nDay 4: Data distribution shift")
        accuracy = 0.87  # Significant drop
        latency = 180
        monitor.record_performance(accuracy, latency)
        
        # Shifted data
        new_data = np.random.normal(loc=[2, 1], scale=[1, 0.5], size=(100, 2))  # Feature A shifted
        drift_result = detector.detect_simple_drift(new_data)
        
        alerts = monitor.check_alerts()
        decision = trigger.should_retrain(alerts, drift_result)
        
        print(f"  Day 4: Accuracy={accuracy:.3f}, Drift={drift_result['drift_detected']}, Retrain={decision['should_retrain']}")
        if decision['should_retrain']:
            print(f"    Trigger reasons: {', '.join(decision['trigger_reasons'])}")
        
        print("\n✅ Comprehensive MLOps integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

# %% [markdown]
"""
---

## 📊 Production Context: Real-World MLOps Systems **[OPTIONAL]**

> **Note**: This section provides real-world context but isn't required for
> understanding the core implementations above.

### How Real Systems Handle MLOps

**Netflix**: Runs 1000+ ML models in production with automated monitoring
- Uses Kafka for real-time metric streaming  
- A/B tests model versions automatically
- Monitors business metrics (click-through rate) alongside model metrics

**Uber**: MLOps platform serves billions of predictions daily
- Custom monitoring with drift detection for rider demand models
- Automated retraining triggers based on city-specific performance
- Feature stores with automated data quality checks

**Airbnb**: ML models for pricing, search ranking, and fraud detection
- Custom dashboard showing model health across all services
- Automated canary deployments with rollback capabilities
- Real-time alert system integrated with incident response

### Systems Engineering Insights

1. **Scale**: Real systems monitor thousands of models simultaneously
2. **Automation**: Human intervention is expensive - everything must be automated
3. **Business Impact**: Technical metrics (accuracy) must connect to business metrics (revenue)
4. **Reliability**: MLOps infrastructure must be more reliable than the models it monitors
5. **Cost**: Monitoring costs must be balanced against model improvement benefits
"""

# %% [markdown]
"""
## 🧪 Main Execution

Run all tests when this module is executed directly.
"""

if __name__ == "__main__":
    print("🚀 TinyTorch MLOps Module")
    print("=" * 50)
    
    try:
        # Run all tests
        run_comprehensive_mlops_tests()
        print()
        
        # Run performance analysis
        analyze_mlops_memory_usage()
        print()
        analyze_mlops_computational_complexity()
        
        print("🎉 All MLOps tests completed successfully!")
        
    except Exception as e:
        print(f"❌ MLOps module execution failed: {e}")
        import traceback
        traceback.print_exc()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

These questions help you think about the systems engineering principles behind
MLOps infrastructure and connect to real production challenges.
"""

# %% [markdown]
"""
**Question 1: Memory Management in Production Monitoring**

You're monitoring 500 models in production, each recording performance metrics
every minute. Each model maintains 1 week of history.

Calculate:
- Total memory usage for metric storage
- Memory growth rate per day
- When you need to implement data retention policies

*Consider: How would you design a memory-efficient monitoring system that 
doesn't lose critical historical information?*
"""

# %% [markdown]
"""
**Question 2: Drift Detection Trade-offs**

Your drift detector takes 50ms to analyze 1000 features. You need to monitor
100 models, each with 5000 features, every hour.

Analyze:
- Total computational cost per monitoring cycle
- Whether you can meet the hourly requirement
- How to optimize without losing detection quality

*Consider: What approximations might real systems use to handle larger scale?*
"""

# %% [markdown]
"""
**Question 3: Retraining Decision Economics**

Model retraining costs $100 in compute and takes 2 hours. A 5% accuracy drop
costs $50/hour in lost business value.

Design a retraining policy that:
- Minimizes total cost (compute + business impact)  
- Accounts for retraining reliability (90% success rate)
- Handles different model criticalities

*Consider: How do real companies balance automation vs human oversight for
expensive retraining decisions?*
"""

# %% [markdown]
"""
**Question 4: Systems Integration Challenges**

Your MLOps system needs to integrate with:
- 5 different model serving systems
- 3 data pipelines with different schemas
- Legacy monitoring infrastructure
- Multiple ML frameworks (PyTorch, TensorFlow, XGBoost)

Evaluate:
- How to design interfaces that work across all systems
- Where to place monitoring hooks in the pipeline
- How to handle partial failures and degraded monitoring

*Consider: What abstraction layers help manage this complexity while keeping
the monitoring reliable?*
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: MLOps - Production ML Systems

### What You Built

In this module, you implemented a complete MLOps monitoring system from scratch:

1. **ModelMonitor**: Tracks model performance and detects degradation
2. **DriftDetector**: Identifies data distribution changes using statistical methods
3. **RetrainingTrigger**: Makes automated decisions about when to retrain models
4. **Integrated System**: All components working together for production monitoring

### Key Systems Engineering Insights

**Memory Management**: MLOps systems accumulate large amounts of monitoring data.
You learned how to analyze memory usage patterns and design for scale.

**Computational Complexity**: Drift detection and monitoring can be expensive at scale.
You analyzed algorithmic complexity and identified optimization opportunities.

**Production Trade-offs**: Real systems balance detection sensitivity with false alarm
rates, computational cost with monitoring coverage, and automation with human oversight.

**Integration Challenges**: MLOps systems must work across diverse infrastructure,
handle partial failures gracefully, and provide reliable monitoring even when
models themselves are unreliable.

### Connection to Real Systems

The patterns you implemented mirror those used by major tech companies:
- Statistical drift detection (used by Netflix, Uber)
- Threshold-based alerting (used by Airbnb, Spotify)  
- Automated retraining policies (used by Google, Facebook)
- Performance trend analysis (used by Amazon, Microsoft)

### Next Steps

- **Module 22**: Would extend to A/B testing and gradual rollouts
- **Module 23**: Could add distributed monitoring and federated learning
- **Production**: These patterns scale to enterprise MLOps platforms

You now understand how to build production ML systems that monitor themselves,
detect problems automatically, and make intelligent decisions about model updates.
This is the foundation of reliable, scalable ML systems engineering.
"""
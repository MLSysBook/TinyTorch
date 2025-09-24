#!/usr/bin/env python3
"""
Backend Integration Example: Drop-in Performance Optimization

This demonstrates how the backend system integrates with existing TinyTorch
code to provide dramatic performance improvements without changing APIs.
"""

import numpy as np
import sys
import os

# Add the kernels module to path
sys.path.append('/Users/VJ/GitHub/TinyTorch/modules/13_kernels')
from kernels_dev import set_backend, benchmark, run_performance_comparison

# Import existing TinyTorch components  
sys.path.append('/Users/VJ/GitHub/TinyTorch/modules/02_tensor')
sys.path.append('/Users/VJ/GitHub/TinyTorch/modules/04_layers')

try:
    from tensor_dev import Tensor
    from layers_dev import Dense, Module
except ImportError:
    print("Creating minimal tensor/layer classes for demo...")
    
    class Tensor:
        def __init__(self, data):
            self.data = np.array(data, dtype=np.float32)
            self.shape = self.data.shape
        
        def __str__(self):
            return f"Tensor(shape={self.shape})"
    
    class Dense:
        def __init__(self, in_features, out_features):
            self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1)
            self.bias = Tensor(np.zeros(out_features))
        
        def forward(self, x):
            # This would normally call tinytorch.matmul, but we'll simulate
            result = x.data @ self.weight.data + self.bias.data
            return Tensor(result)

# Now import our optimized functions
from kernels_dev import fast_matmul

def demo_same_code_different_performance():
    """Demonstrate same code achieving different performance"""
    
    print("🎯 DEMONSTRATION: Same Code, Different Performance")
    print("=" * 70)
    
    # Create a simple neural network model
    class SimpleNet:
        def __init__(self):
            self.layer1 = Dense(784, 512)
            self.layer2 = Dense(512, 256) 
            self.layer3 = Dense(256, 10)
        
        def forward(self, x):
            x = self.layer1.forward(x)
            x = self.layer2.forward(x) 
            x = self.layer3.forward(x)
            return x
    
    # Create model and data
    model = SimpleNet()
    batch_data = Tensor(np.random.randn(128, 784))  # Batch of 128 images
    
    def run_model():
        """Run the same model forward pass"""
        output = model.forward(batch_data)
        return output
    
    # This is the magic - SAME CODE, different performance!
    results = run_performance_comparison("Neural Network Forward Pass", run_model)
    
    return results

def demo_competition_scenario():
    """Demonstrate a competition scenario"""
    
    print("\n🏆 COMPETITION SCENARIO: Matrix Multiplication Optimization")
    print("=" * 70)
    
    # Different student "submissions" 
    def student_alice_submission():
        """Alice's optimized implementation"""
        set_backend('optimized')
        a = Tensor(np.random.randn(400, 300))
        b = Tensor(np.random.randn(300, 200))
        return fast_matmul(a, b)
    
    def student_bob_submission():
        """Bob still using naive implementation"""
        set_backend('naive')
        a = Tensor(np.random.randn(400, 300))
        b = Tensor(np.random.randn(300, 200))
        return fast_matmul(a, b)
    
    # Simulate competition submissions
    from kernels_dev import submit_to_competition, competition
    
    print("Student submissions:")
    submit_to_competition("Alice", "Matrix Multiplication", student_alice_submission)
    submit_to_competition("Bob", "Matrix Multiplication", student_bob_submission)
    
    # Show leaderboard
    competition.show_leaderboard("Matrix Multiplication")

def demo_real_world_scenario():
    """Demonstrate real-world ML training scenario"""
    
    print("\n🌍 REAL-WORLD SCENARIO: Training Speed Comparison")
    print("=" * 70)
    
    # Simulate training step computation  
    def training_step():
        """Simulate one training step with multiple operations"""
        
        # Forward pass operations
        batch_size, seq_len, hidden_dim = 32, 128, 512
        
        # Attention computation (the expensive part)
        queries = Tensor(np.random.randn(batch_size, seq_len, hidden_dim))
        keys = Tensor(np.random.randn(batch_size, seq_len, hidden_dim))
        values = Tensor(np.random.randn(batch_size, seq_len, hidden_dim))
        
        # Attention weights: Q @ K^T  
        attention_weights = fast_matmul(queries, keys)  # This gets optimized!
        
        # Attention output: weights @ V
        attention_output = fast_matmul(attention_weights, values)  # This too!
        
        # Feed-forward layers
        ff1 = Dense(hidden_dim, hidden_dim * 4)
        ff2 = Dense(hidden_dim * 4, hidden_dim)
        
        ff_output = ff1.forward(attention_output)
        final_output = ff2.forward(ff_output)
        
        return final_output
    
    # Compare training speeds
    results = run_performance_comparison("Transformer Training Step", training_step)
    
    # Calculate training time implications
    naive_time = results['naive'].time_ms
    opt_time = results['optimized'].time_ms
    
    print(f"\n📊 Training Time Analysis:")
    print(f"Time per step: Naive={naive_time:.1f}ms, Optimized={opt_time:.1f}ms")
    
    steps_per_epoch = 1000
    naive_epoch_time = (naive_time * steps_per_epoch) / 1000 / 60  # minutes
    opt_epoch_time = (opt_time * steps_per_epoch) / 1000 / 60    # minutes
    
    print(f"Time per epoch: Naive={naive_epoch_time:.1f}min, Optimized={opt_epoch_time:.1f}min")
    print(f"Training 100 epochs: Naive={naive_epoch_time*100/60:.1f}hrs, Optimized={opt_epoch_time*100/60:.1f}hrs")
    
    time_saved = (naive_epoch_time - opt_epoch_time) * 100 / 60  # hours saved over 100 epochs
    print(f"⚡ Time saved: {time_saved:.1f} hours over 100 epochs!")

if __name__ == "__main__":
    print("🚀 TinyTorch Backend Integration Demo")
    print("Demonstrating competition-ready optimization without API changes")
    print("=" * 80)
    
    # Run all demonstrations
    demo_same_code_different_performance()
    demo_competition_scenario()  
    demo_real_world_scenario()
    
    print("\n" + "=" * 80)
    print("🎯 KEY INSIGHTS:")
    print("• Same APIs, dramatically different performance")
    print("• Backend switching enables both learning AND competition")
    print("• Real ML training can be 10-100x faster with proper optimization")
    print("• Students see immediate impact of systems engineering")
    print("=" * 80)
#!/usr/bin/env python3
"""
Automatic Differentiation with TinyTorch

Demonstrates how automatic differentiation works in YOUR TinyTorch,
showing gradient computation through complex computational graphs.

This is the magic that makes deep learning possible!
"""

import numpy as np
import tinytorch as tt
from tinytorch.core import Tensor
from tinytorch.core.autograd import Variable, backward


def simple_gradient_example():
    """Show basic gradient computation."""
    print("📐 Example 1: Simple Gradient")
    print("-" * 40)
    
    # Create variables that track gradients
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    
    # Forward computation: z = x*y + x^2
    z = x * y + x ** 2
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x*y + x^2 = {z.data}")
    print()
    
    # Backward pass
    z.backward()
    
    print("Gradients after backward():")
    print(f"∂z/∂x = {x.grad}  (should be y + 2x = 3 + 4 = 7)")
    print(f"∂z/∂y = {y.grad}  (should be x = 2)")
    print()


def computational_graph_example():
    """Show gradients through a computational graph."""
    print("🌳 Example 2: Computational Graph")
    print("-" * 40)
    
    # Build a more complex graph
    a = Variable(1.0, requires_grad=True)
    b = Variable(2.0, requires_grad=True)
    c = Variable(3.0, requires_grad=True)
    
    # Forward: d = (a + b) * c
    d = a + b
    e = d * c
    
    print("Computational graph:")
    print("  a = 1.0")
    print("  b = 2.0")  
    print("  c = 3.0")
    print("  d = a + b = 3.0")
    print("  e = d * c = 9.0")
    print()
    
    # Backward
    e.backward()
    
    print("Gradients:")
    print(f"∂e/∂a = {a.grad}  (flows through d)")
    print(f"∂e/∂b = {b.grad}  (flows through d)")
    print(f"∂e/∂c = {c.grad}  (direct multiplication)")
    print()


def neural_network_gradients():
    """Show gradients in a neural network layer."""
    print("🧠 Example 3: Neural Network Layer")
    print("-" * 40)
    
    # Input and weights
    X = Variable(Tensor([[1.0, 2.0]]), requires_grad=True)
    W = Variable(Tensor([[0.5, -0.3], [0.2, 0.8]]), requires_grad=True)
    b = Variable(Tensor([0.1, 0.2]), requires_grad=True)
    
    print("Neural network layer: y = X @ W + b")
    print(f"Input X: {X.data}")
    print(f"Weights W:\n{W.data}")
    print(f"Bias b: {b.data}")
    print()
    
    # Forward pass
    y = X @ W + b
    print(f"Output y: {y.data}")
    
    # Define a simple loss (sum of outputs)
    loss = y.sum()
    print(f"Loss (sum): {loss.data}")
    print()
    
    # Backward pass
    loss.backward()
    
    print("Gradients for weight update:")
    print(f"∂L/∂W:\n{W.grad}")
    print(f"∂L/∂b: {b.grad}")
    print(f"∂L/∂X: {X.grad}")
    print()


def gradient_descent_visualization():
    """Visualize gradient descent optimization."""
    print("📉 Example 4: Gradient Descent")
    print("-" * 40)
    
    # Optimize: minimize f(x) = (x - 3)^2
    x = Variable(0.0, requires_grad=True)
    learning_rate = 0.1
    
    print("Minimizing f(x) = (x - 3)^2")
    print("Starting at x = 0.0")
    print()
    
    history = []
    for step in range(10):
        # Forward: compute loss
        loss = (x - 3.0) ** 2
        
        # Backward: compute gradient
        x.zero_grad()  # Clear previous gradients
        loss.backward()
        
        # Update: gradient descent
        x.data = x.data - learning_rate * x.grad
        history.append((x.data, loss.data))
        
        if step % 2 == 0:
            print(f"Step {step}: x = {x.data:.3f}, f(x) = {loss.data:.3f}, grad = {x.grad:.3f}")
    
    print()
    print(f"Converged to x = {x.data:.3f} (optimal is 3.0)")
    print()


def chain_rule_demonstration():
    """Demonstrate the chain rule in action."""
    print("⛓️ Example 5: Chain Rule")
    print("-" * 40)
    
    x = Variable(2.0, requires_grad=True)
    
    # Composite function: f(g(h(x)))
    # h(x) = x^2
    # g(h) = h + 1  
    # f(g) = sqrt(g)
    
    h = x ** 2          # h = 4
    g = h + 1           # g = 5
    f = g ** 0.5        # f = sqrt(5) ≈ 2.236
    
    print("Composite function: f(g(h(x))) = sqrt(x^2 + 1)")
    print(f"x = {x.data}")
    print(f"h = x^2 = {h.data}")
    print(f"g = h + 1 = {g.data}")
    print(f"f = sqrt(g) = {f.data:.3f}")
    print()
    
    # Backward uses chain rule
    f.backward()
    
    print("Chain rule gradient:")
    print(f"∂f/∂x = ∂f/∂g · ∂g/∂h · ∂h/∂x")
    print(f"      = (1/2√g) · 1 · 2x")
    print(f"      = x/√(x^2 + 1)")
    print(f"      = {x.grad:.3f}")
    
    # Manual computation for verification
    manual_grad = x.data / np.sqrt(x.data**2 + 1)
    print(f"Manual: {manual_grad:.3f} ✓")
    print()


def main():
    print("=" * 50)
    print("🔬 Automatic Differentiation with TinyTorch")
    print("=" * 50)
    print()
    print("This demonstrates the autograd engine you built!")
    print("Every modern deep learning framework needs this.")
    print()
    
    # Run all examples
    simple_gradient_example()
    computational_graph_example()
    neural_network_gradients()
    gradient_descent_visualization()
    chain_rule_demonstration()
    
    print("=" * 50)
    print("🎯 Key Takeaways:")
    print("-" * 50)
    print("✅ Automatic differentiation computes exact gradients")
    print("✅ Chain rule enables gradients through deep networks")
    print("✅ Computational graphs track operations for backprop")
    print("✅ This is the foundation of all deep learning!")
    print()
    print("🎉 Your autograd engine makes training possible!")
    
    return True


if __name__ == "__main__":
    success = main()
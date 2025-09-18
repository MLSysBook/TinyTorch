#!/usr/bin/env python3
"""
TinyTorch Demo 02: Matrix Math Magic
Demonstrates tensor operations solving real linear algebra problems
"""

import sys
import numpy as np

def demo_tensor_math():
    """Demo tensor operations with practical linear algebra"""
    
    try:
        # Import TinyTorch tensor module
        import tinytorch.core.tensor as tt
        print("🧮 TinyTorch Tensor Math Demo")
        print("=" * 40)
        print("Solving real linear algebra with tensors!")
        print()
        
        # Demo 1: Solve system of linear equations
        print("📐 Demo 1: Solving Linear System")
        print("System: 2x + 3y = 13")
        print("        1x + 1y = 5")
        print()
        
        # Coefficient matrix A and result vector b
        A = tt.Tensor([[2, 3], [1, 1]])
        b = tt.Tensor([[13], [5]])
        
        print(f"Matrix A:\n{A.data}")
        print(f"Vector b:\n{b.data}")
        print()
        
        # Solve using matrix operations (simplified inverse)
        print("🔍 Solving A @ x = b...")
        
        # Manual 2x2 inverse for demo
        det = A.data[0,0] * A.data[1,1] - A.data[0,1] * A.data[1,0]
        A_inv_data = np.array([[A.data[1,1], -A.data[0,1]], 
                               [-A.data[1,0], A.data[0,0]]]) / det
        A_inv = tt.Tensor(A_inv_data)
        
        # Solve: x = A_inv @ b
        x = tt.Tensor(A_inv.data @ b.data)
        
        print(f"Solution: x = {x.data[0,0]:.1f}, y = {x.data[1,0]:.1f}")
        
        # Verify solution
        verification = tt.Tensor(A.data @ x.data)
        print(f"Verification: A @ x = {verification.data.flatten()}")
        print(f"Original b = {b.data.flatten()}")
        print("✅ Solution verified!" if np.allclose(verification.data, b.data) else "❌ Solution incorrect")
        print()
        
        # Demo 2: Matrix transformation (rotation)
        print("🌀 Demo 2: 2D Rotation Matrix")
        angle = np.pi / 4  # 45 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = tt.Tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        original_point = tt.Tensor([[1], [0]])  # Point (1, 0)
        
        print(f"Rotating point (1, 0) by 45°...")
        print(f"Rotation matrix:\n{rotation_matrix.data}")
        
        rotated_point = tt.Tensor(rotation_matrix.data @ original_point.data)
        print(f"Rotated point: ({rotated_point.data[0,0]:.3f}, {rotated_point.data[1,0]:.3f})")
        print(f"Expected: (0.707, 0.707)")
        print()
        
        # Demo 3: Batch matrix operations
        print("⚡ Demo 3: Batch Processing")
        print("Processing multiple vectors simultaneously...")
        
        # Multiple 2D points
        points = tt.Tensor([[1, 0, -1], [0, 1, 0]])  # 3 points: (1,0), (0,1), (-1,0)
        print(f"Original points:\n{points.data}")
        
        rotated_points = tt.Tensor(rotation_matrix.data @ points.data)
        print(f"All points rotated by 45°:\n{rotated_points.data}")
        print()
        
        # Demo 4: Neural network weights preview
        print("🧠 Demo 4: Neural Network Preview")
        print("This is how tensors will power neural networks...")
        
        # Simulate a simple linear layer: y = W @ x + b
        weights = tt.Tensor([[0.5, -0.3, 0.8], [0.2, 0.9, -0.1]])  # 2 neurons, 3 inputs
        bias = tt.Tensor([[0.1], [0.05]])
        input_data = tt.Tensor([[1.0], [0.5], [-0.2]])  # 3D input
        
        print(f"Weights (2×3):\n{weights.data}")
        print(f"Input (3×1):\n{input_data.data.flatten()}")
        
        output = tt.Tensor(weights.data @ input_data.data + bias.data)
        print(f"Output (2×1): {output.data.flatten()}")
        print("🔮 Soon we'll add activations to make this a real neuron!")
        print()
        
        print("🏆 TinyTorch Tensor Math Demo Complete!")
        print("🎯 Achievements:")
        print("  • Solved linear systems with matrix operations")
        print("  • Performed geometric transformations")
        print("  • Processed multiple data points in parallel")
        print("  • Previewed neural network computations")
        print()
        print("🔥 Next: Add activations for real neural networks!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import TinyTorch tensor module: {e}")
        print("💡 Make sure to run: tito export 02_tensor")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_tensor_math()
    sys.exit(0 if success else 1)
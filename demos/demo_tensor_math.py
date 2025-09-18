#!/usr/bin/env python3
"""
TinyTorch Demo 02: Matrix Math Magic
Demonstrates tensor operations solving real linear algebra problems
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def demo_tensor_math():
    """Demo tensor operations with practical linear algebra"""
    
    console = Console()
    
    try:
        # Import TinyTorch tensor module
        import tinytorch.core.tensor as tt
        
        # Main header
        console.print(Panel.fit(
            "🧮 TinyTorch Tensor Math Demo\nSolving real linear algebra with tensors!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # What this demo shows
        console.print(Panel(
            "[bold yellow]What This Demo Shows:[/bold yellow]\n\n"
            "Tensors are the foundation of all neural networks - they're just multi-dimensional arrays\n"
            "that can represent scalars, vectors, matrices, and higher dimensions. You'll see:\n\n"
            "• Solving systems of linear equations (finding x in Ax = b)\n"
            "• Geometric transformations with rotation matrices\n"
            "• Batch processing - operating on multiple data points simultaneously\n"
            "• How neural network weights are just matrices doing transformations\n\n"
            "[bold cyan]Key Insight:[/bold cyan] Every neural network operation is matrix multiplication at its core.\n"
            "Understanding tensors means understanding how neural networks compute!",
            title="📚 Understanding This Demo",
            style="blue"
        ))
        console.print()
        
        # Demo 1: Solve system of linear equations
        console.print(Panel(
            "System: 2x + 3y = 13\n        1x + 1y = 5",
            title="📐 Demo 1: Solving Linear System",
            style="green"
        ))
        
        # Coefficient matrix A and result vector b
        A = tt.Tensor([[2, 3], [1, 1]])
        b = tt.Tensor([[13], [5]])
        
        # Create table for matrices
        matrix_table = Table(show_header=True, header_style="bold magenta")
        matrix_table.add_column("Matrix A", style="cyan")
        matrix_table.add_column("Vector b", style="yellow")
        matrix_table.add_row(str(A.data), str(b.data))
        console.print(matrix_table)
        console.print()
        
        # Solve using matrix operations (simplified inverse)
        console.print("🔍 [bold yellow]Solving A @ x = b...[/bold yellow]")
        
        # Manual 2x2 inverse for demo
        det = A.data[0,0] * A.data[1,1] - A.data[0,1] * A.data[1,0]
        A_inv_data = np.array([[A.data[1,1], -A.data[0,1]], 
                               [-A.data[1,0], A.data[0,0]]]) / det
        A_inv = tt.Tensor(A_inv_data)
        
        # Solve: x = A_inv @ b
        x = tt.Tensor(A_inv.data @ b.data)
        
        # Solution panel
        solution_text = f"x = {x.data[0,0]:.1f}, y = {x.data[1,0]:.1f}"
        console.print(Panel(solution_text, title="✨ Solution", style="bold green"))
        
        # Verify solution
        verification = tt.Tensor(A.data @ x.data)
        verify_table = Table(show_header=True, header_style="bold magenta")
        verify_table.add_column("Verification: A @ x", style="cyan")
        verify_table.add_column("Original b", style="yellow") 
        verify_table.add_column("Status", style="green")
        status = "✅ Verified!" if np.allclose(verification.data, b.data) else "❌ Incorrect"
        verify_table.add_row(str(verification.data.flatten()), str(b.data.flatten()), status)
        console.print(verify_table)
        console.print()
        
        console.print("[dim]💡 [bold]What Just Happened:[/bold] We solved for x=2, y=3 using matrix operations![/dim]")
        console.print("[dim]   This is exactly how neural networks solve for optimal weights during training.[/dim]")
        console.print()
        
        # Demo 2: Matrix transformation (rotation)
        console.print(Panel(
            "Rotating point (1, 0) by 45°...",
            title="🌀 Demo 2: 2D Rotation Matrix",
            style="blue"
        ))
        
        angle = np.pi / 4  # 45 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = tt.Tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        original_point = tt.Tensor([[1], [0]])  # Point (1, 0)
        
        # Rotation table
        rotation_table = Table(show_header=True, header_style="bold magenta")
        rotation_table.add_column("Rotation Matrix", style="cyan")
        rotation_table.add_column("Original Point", style="yellow")
        rotation_table.add_row(str(rotation_matrix.data), str(original_point.data))
        console.print(rotation_table)
        
        rotated_point = tt.Tensor(rotation_matrix.data @ original_point.data)
        
        # Results table
        result_table = Table(show_header=True, header_style="bold magenta")
        result_table.add_column("Rotated Point", style="green")
        result_table.add_column("Expected", style="yellow")
        result_table.add_row(
            f"({rotated_point.data[0,0]:.3f}, {rotated_point.data[1,0]:.3f})",
            "(0.707, 0.707)"
        )
        console.print(result_table)
        console.print()
        
        # Demo 3: Batch matrix operations
        console.print(Panel(
            "Processing multiple vectors simultaneously...",
            title="⚡ Demo 3: Batch Processing",
            style="yellow"
        ))
        
        # Multiple 2D points
        points = tt.Tensor([[1, 0, -1], [0, 1, 0]])  # 3 points: (1,0), (0,1), (-1,0)
        
        batch_table = Table(show_header=True, header_style="bold magenta")
        batch_table.add_column("Original Points", style="cyan")
        batch_table.add_column("Rotated Points", style="green")
        
        rotated_points = tt.Tensor(rotation_matrix.data @ points.data)
        batch_table.add_row(str(points.data), str(rotated_points.data))
        console.print(batch_table)
        console.print()
        
        # Demo 4: Neural network weights preview
        console.print(Panel(
            "This is how tensors will power neural networks...",
            title="🧠 Demo 4: Neural Network Preview",
            style="magenta"
        ))
        
        # Simulate a simple linear layer: y = W @ x + b
        weights = tt.Tensor([[0.5, -0.3, 0.8], [0.2, 0.9, -0.1]])  # 2 neurons, 3 inputs
        bias = tt.Tensor([[0.1], [0.05]])
        input_data = tt.Tensor([[1.0], [0.5], [-0.2]])  # 3D input
        
        nn_table = Table(show_header=True, header_style="bold magenta")
        nn_table.add_column("Weights (2×3)", style="cyan")
        nn_table.add_column("Input (3×1)", style="yellow")
        nn_table.add_column("Output (2×1)", style="green")
        
        output = tt.Tensor(weights.data @ input_data.data + bias.data)
        nn_table.add_row(
            str(weights.data), 
            str(input_data.data.flatten()),
            str(output.data.flatten())
        )
        console.print(nn_table)
        
        console.print("\n🔮 [italic]Soon we'll add activations to make this a real neuron![/italic]")
        console.print()
        
        # Success panel
        console.print(Panel.fit(
            "🎯 Achievements:\n• Solved linear systems with matrix operations\n• Performed geometric transformations\n• Processed multiple data points in parallel\n• Previewed neural network computations\n\n🔥 Next: Add activations for real neural networks!",
            title="🏆 TinyTorch Tensor Math Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch tensor module: {e}\n\n💡 Make sure to run: tito export 02_tensor",
            title="❌ Import Error",
            style="bold red"
        ))
        return False
    except Exception as e:
        console.print(Panel(
            f"Demo failed: {e}",
            title="❌ Error",
            style="bold red"
        ))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_tensor_math()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
🚀 CAPABILITY SHOWCASE: Tensor Operations
After Module 02 (Tensor)

"Look what you built!" - Your tensors can do linear algebra!
"""

import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.align import Align

# Import from YOUR TinyTorch implementation
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    print("❌ TinyTorch not found. Make sure you've completed Module 02 (Tensor)!")
    sys.exit(1)

console = Console()

def ascii_matrix(matrix_data, title="Matrix"):
    """Create ASCII visualization of a matrix."""
    table = Table(title=title, show_header=False, show_edge=False)
    
    # Add columns based on matrix width
    for _ in range(len(matrix_data[0])):
        table.add_column(justify="center", style="cyan")
    
    # Add rows
    for row in matrix_data:
        table.add_row(*[f"{val:6.2f}" for val in row])
    
    return table

def demonstrate_tensor_creation():
    """Show tensor creation and basic operations."""
    console.print(Panel.fit("📊 TENSOR CREATION", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating tensors with YOUR code...", total=None)
        time.sleep(1)
        
        # Create tensors using student's implementation
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([[7, 8], [9, 10], [11, 12]])
        
        progress.update(task, description="✅ Tensors created!")
        time.sleep(0.5)
    
    console.print("\n🎯 Matrix A:")
    console.print(ascii_matrix(a.data, "Your Tensor A"))
    
    console.print("\n🎯 Matrix B:")
    console.print(ascii_matrix(b.data, "Your Tensor B"))
    
    return a, b

def demonstrate_matrix_multiplication(a, b):
    """Show matrix multiplication with visual explanation."""
    console.print(Panel.fit("⚡ MATRIX MULTIPLICATION", style="bold green"))
    
    console.print("🧮 Computing A @ B using YOUR implementation...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Multiplying matrices...", total=None)
        time.sleep(1)
        
        # Use student's matrix multiplication
        result = a.matmul(b)
        
        progress.update(task, description="✅ Matrix multiplication complete!")
        time.sleep(0.5)
    
    console.print(f"\n🎯 Result Shape: {result.shape}")
    console.print("\n📊 A @ B =")
    console.print(ascii_matrix(result.data, "Matrix Multiplication Result"))
    
    # Show the math visually
    console.print("\n🔍 What happened:")
    console.print("   [1×7 + 2×9 + 3×11] [1×8 + 2×10 + 3×12]")
    console.print("   [4×7 + 5×9 + 6×11] [4×8 + 5×10 + 6×12]")
    console.print("         ↓                    ↓")
    console.print("        [58]                [64]")
    console.print("       [139]               [154]")
    
    return result

def demonstrate_tensor_operations():
    """Show various tensor operations."""
    console.print(Panel.fit("🔧 TENSOR OPERATIONS", style="bold yellow"))
    
    # Create a simple tensor
    x = Tensor([[2, 4, 6], [8, 10, 12]])
    
    console.print("🎯 Original Tensor:")
    console.print(ascii_matrix(x.data, "Tensor X"))
    
    # Transpose - check if available
    try:
        console.print("\n🔄 Transpose:")
        x_t = x.T if hasattr(x, 'T') else Tensor(np.array(x.data).T.tolist())
        console.print(ascii_matrix(x_t.data, "X.T"))
    except:
        console.print("\n🔄 Transpose not yet implemented (coming soon!)")
    
    # Element-wise operations (if implemented)
    try:
        console.print("\n➕ Addition (X + 5):")
        x_plus = x.add(5)
        console.print(ascii_matrix(x_plus.data, "X + 5"))
    except:
        console.print("\n➕ Addition not yet implemented (coming in later modules!)")
    
    try:
        console.print("\n✖️ Multiplication (X * 2):")
        x_mul = x.multiply(2)
        console.print(ascii_matrix(x_mul.data, "X * 2"))
    except:
        console.print("\n✖️ Multiplication not yet implemented (coming in later modules!)")

def show_neural_network_preview():
    """Preview how tensors will be used in neural networks."""
    console.print(Panel.fit("🧠 NEURAL NETWORK PREVIEW", style="bold magenta"))
    
    console.print("🔮 Coming soon in your TinyTorch journey:")
    console.print("   🎯 These tensors will become neural network weights")
    console.print("   🎯 Matrix multiplication will compute layer outputs")
    console.print("   🎯 You'll train networks to recognize images and text")
    console.print("   🎯 Eventually you'll build GPT from scratch!")
    
    # Simple preview calculation
    weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    inputs = Tensor([[1], [2], [3]])
    
    console.print(f"\n🔍 Preview - Neural layer calculation:")
    console.print("   Weights @ Inputs = Layer Output")
    
    output = weights.matmul(inputs)
    console.print(f"   Result shape: {output.shape}")
    console.print("   (This will make sense after Module 05!)")

def main():
    """Main showcase function."""
    console.clear()
    
    # Header
    layout = Layout()
    header = Panel.fit(
        "[bold cyan]🚀 CAPABILITY SHOWCASE: TENSOR OPERATIONS[/bold cyan]\n"
        "[yellow]After Module 02 (Tensor)[/yellow]\n\n"
        "[green]\"Look what you built!\" - Your tensors can do linear algebra![/green]",
        border_style="bright_blue"
    )
    console.print(Align.center(header))
    console.print()
    
    try:
        # Demonstrate tensor capabilities
        a, b = demonstrate_tensor_creation()
        console.print("\n" + "="*60)
        
        result = demonstrate_matrix_multiplication(a, b)
        console.print("\n" + "="*60)
        
        demonstrate_tensor_operations()
        console.print("\n" + "="*60)
        
        show_neural_network_preview()
        
        # Celebration
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 CONGRATULATIONS! 🎉[/bold green]\n\n"
            "[cyan]Your Tensor class is the foundation of all machine learning![/cyan]\n"
            "[white]Every neural network, from simple classifiers to GPT,[/white]\n"
            "[white]starts with the tensor operations YOU just implemented.[/white]\n\n"
            "[yellow]Next up: Activations (Module 03) - Adding intelligence to your tensors![/yellow]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error running showcase: {e}")
        console.print("💡 Make sure you've completed Module 02 and your Tensor class works!")

if __name__ == "__main__":
    main()
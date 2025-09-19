#!/usr/bin/env python3
"""
🚀 CAPABILITY SHOWCASE: Image Processing 
After Module 06 (Spatial)

"Look what you built!" - Your convolutions can see patterns!
"""

import sys
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.align import Align

# Import from YOUR TinyTorch implementation
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.spatial import Conv2D, MaxPool2D
except ImportError:
    print("❌ TinyTorch spatial layers not found. Make sure you've completed Module 06 (Spatial)!")
    sys.exit(1)

console = Console()

def create_sample_image():
    """Create a sample image with clear features for edge detection."""
    # 8x8 image with a square in the middle
    image = np.zeros((8, 8))
    image[2:6, 2:6] = 1.0  # White square in center
    return image

def create_noisy_image():
    """Create an image with noise to show filtering effects."""
    # Create a diagonal line with noise
    image = np.random.random((8, 8)) * 0.3  # Background noise
    for i in range(8):
        if i < 8:
            image[i, i] = 1.0  # Diagonal line
    return image

def ascii_image(image, chars=" ░▒▓█"):
    """Convert image to ASCII art."""
    lines = []
    for row in image:
        line = ""
        for pixel in row:
            # Normalize pixel value to char index
            char_idx = int(pixel * (len(chars) - 1))
            char_idx = max(0, min(char_idx, len(chars) - 1))
            line += chars[char_idx]
        lines.append(line)
    return "\n".join(lines)

def display_image_comparison(original, filtered, title, filter_name):
    """Display original and filtered images side by side."""
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    
    # Create side-by-side display
    table = Table(show_header=True, show_edge=False)
    table.add_column("Original Image", style="white")
    table.add_column("After " + filter_name, style="yellow")
    
    orig_lines = ascii_image(original).split('\n')
    filt_lines = ascii_image(filtered).split('\n')
    
    for orig_line, filt_line in zip(orig_lines, filt_lines):
        table.add_row(orig_line, filt_line)
    
    console.print(table)

def demonstrate_edge_detection():
    """Show edge detection with convolution."""
    console.print(Panel.fit("🔍 EDGE DETECTION WITH YOUR CONVOLUTIONS", style="bold green"))
    
    # Create edge detection kernel (vertical edges)
    edge_kernel = np.array([
        [[-1, 0, 1],
         [-2, 0, 2], 
         [-1, 0, 1]]
    ])
    
    console.print("🧮 Edge Detection Kernel (Sobel):")
    console.print("   [-1  0  1]")
    console.print("   [-2  0  2]")
    console.print("   [-1  0  1]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating convolution layer...", total=None)
        time.sleep(1)
        
        # Create convolution layer with YOUR implementation
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        
        # Set the edge detection kernel
        conv.weights = Tensor([[edge_kernel]])  # Shape: (1, 1, 3, 3)
        conv.bias = Tensor([0])
        
        progress.update(task, description="✅ Edge detector ready!")
        time.sleep(0.5)
    
    # Test on sample image
    sample_image = create_sample_image()
    console.print("\n📸 Testing on sample image...")
    
    # Reshape for convolution (add batch and channel dimensions)
    input_tensor = Tensor([[sample_image.tolist()]])  # Shape: (1, 1, 8, 8)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Applying convolution...", total=None)
        time.sleep(1)
        
        # Apply YOUR convolution
        output = conv.forward(input_tensor)
        filtered_image = np.array(output.data[0][0])  # Extract image from tensor
        
        progress.update(task, description="✅ Edge detection complete!")
        time.sleep(0.5)
    
    # Normalize for display
    filtered_image = np.abs(filtered_image)  # Take absolute value
    if filtered_image.max() > 0:
        filtered_image = filtered_image / filtered_image.max()
    
    display_image_comparison(sample_image, filtered_image, 
                           "Edge Detection Results", "Sobel Filter")
    
    console.print("\n💡 [bold]What happened:[/bold]")
    console.print("   🎯 Vertical edges were detected and highlighted")
    console.print("   🎯 The convolution found brightness changes")
    console.print("   🎯 This is how CNNs 'see' features in images!")

def demonstrate_blur_filter():
    """Show blur/smoothing with convolution."""
    console.print(Panel.fit("🌫️ NOISE REDUCTION WITH BLUR FILTER", style="bold blue"))
    
    # Create blur kernel (Gaussian-like)
    blur_kernel = np.array([
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]
    ]) / 16.0  # Normalize
    
    console.print("🧮 Blur Kernel (Gaussian-like):")
    console.print("   [1  2  1]    / 16")
    console.print("   [2  4  2]   ")
    console.print("   [1  2  1]   ")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating blur filter...", total=None)
        time.sleep(1)
        
        # Create convolution layer for blurring
        blur_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        blur_conv.weights = Tensor([[blur_kernel]])
        blur_conv.bias = Tensor([0])
        
        progress.update(task, description="✅ Blur filter ready!")
        time.sleep(0.5)
    
    # Test on noisy image
    noisy_image = create_noisy_image()
    console.print("\n📸 Testing on noisy image...")
    
    input_tensor = Tensor([[noisy_image.tolist()]])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Applying blur filter...", total=None)
        time.sleep(1)
        
        output = blur_conv.forward(input_tensor)
        blurred_image = np.array(output.data[0][0])
        
        progress.update(task, description="✅ Image smoothed!")
        time.sleep(0.5)
    
    display_image_comparison(noisy_image, blurred_image,
                           "Noise Reduction Results", "Blur Filter")
    
    console.print("\n💡 [bold]What happened:[/bold]")
    console.print("   🎯 Random noise was smoothed out")
    console.print("   🎯 The diagonal line is preserved")
    console.print("   🎯 This is preprocessing for better feature detection!")

def demonstrate_pooling():
    """Show max pooling for downsampling."""
    console.print(Panel.fit("📉 DOWNSAMPLING WITH MAX POOLING", style="bold yellow"))
    
    console.print("🔧 Max Pooling Operation:")
    console.print("   Takes maximum value in each 2×2 region")
    console.print("   Reduces spatial dimensions by half")
    console.print("   Keeps strongest features")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating max pooling layer...", total=None)
        time.sleep(1)
        
        # Create max pooling layer
        maxpool = MaxPool2D(kernel_size=2, stride=2)
        
        progress.update(task, description="✅ Max pooling ready!")
        time.sleep(0.5)
    
    # Create test image with clear patterns
    test_image = np.array([
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1]
    ])
    
    input_tensor = Tensor([[test_image.tolist()]])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Applying max pooling...", total=None)
        time.sleep(1)
        
        pooled_output = maxpool.forward(input_tensor)
        pooled_image = np.array(pooled_output.data[0][0])
        
        progress.update(task, description="✅ Downsampling complete!")
        time.sleep(0.5)
    
    display_image_comparison(test_image, pooled_image,
                           f"Max Pooling Results (8×8 → {pooled_image.shape[0]}×{pooled_image.shape[1]})",
                           "Max Pool 2×2")
    
    console.print("\n💡 [bold]What happened:[/bold]")
    console.print("   🎯 Image size reduced from 8×8 to 4×4")
    console.print("   🎯 Important features were preserved")
    console.print("   🎯 This makes CNNs more efficient and translation-invariant!")

def show_cnn_architecture_preview():
    """Preview how these operations combine in CNNs."""
    console.print(Panel.fit("🏗️ CNN ARCHITECTURE PREVIEW", style="bold magenta"))
    
    console.print("🧠 Your spatial operations are the building blocks of CNNs:")
    console.print()
    console.print("   📥 Input Image")
    console.print("   ↓")
    console.print("   🔍 Conv2D + ReLU  ← [bold cyan]Feature Detection[/bold cyan]")
    console.print("   ↓")  
    console.print("   📉 MaxPool2D      ← [bold yellow]Spatial Reduction[/bold yellow]")
    console.print("   ↓")
    console.print("   🔍 Conv2D + ReLU  ← [bold cyan]Higher-level Features[/bold cyan]")
    console.print("   ↓")
    console.print("   📉 MaxPool2D      ← [bold yellow]Further Reduction[/bold yellow]")
    console.print("   ↓")
    console.print("   🧮 Dense Layers   ← [bold green]Classification[/bold green]")
    console.print("   ↓")
    console.print("   📤 Predictions")
    console.print()
    console.print("🎯 [bold]Real CNN Examples:[/bold]")
    console.print("   • LeNet-5: Handwritten digit recognition")
    console.print("   • AlexNet: ImageNet classification breakthrough") 
    console.print("   • ResNet: Deep networks with skip connections")
    console.print("   • U-Net: Medical image segmentation")

def show_production_applications():
    """Show real-world applications of convolutions."""
    console.print(Panel.fit("🌐 PRODUCTION APPLICATIONS", style="bold red"))
    
    console.print("🚀 Your convolution operations power:")
    console.print()
    console.print("   📱 [bold]Computer Vision:[/bold]")
    console.print("      • Photo apps (Instagram filters)")
    console.print("      • Medical imaging (X-ray analysis)")
    console.print("      • Autonomous vehicles (object detection)")
    console.print("      • Security systems (face recognition)")
    console.print()
    console.print("   🏭 [bold]Industrial Applications:[/bold]")
    console.print("      • Quality control in manufacturing")
    console.print("      • Satellite image analysis")
    console.print("      • Document processing (OCR)")
    console.print("      • Agricultural monitoring")
    console.print()
    console.print("   ⚡ [bold]Performance Optimizations:[/bold]")
    console.print("      • GPU acceleration (thousands of parallel ops)")
    console.print("      • Winograd convolution algorithms")
    console.print("      • Quantization for mobile deployment")
    console.print("      • TensorRT optimization for inference")

def main():
    """Main showcase function."""
    console.clear()
    
    # Header
    header = Panel.fit(
        "[bold cyan]🚀 CAPABILITY SHOWCASE: IMAGE PROCESSING[/bold cyan]\n"
        "[yellow]After Module 06 (Spatial)[/yellow]\n\n"
        "[green]\"Look what you built!\" - Your convolutions can see patterns![/green]",
        border_style="bright_blue"
    )
    console.print(Align.center(header))
    console.print()
    
    try:
        demonstrate_edge_detection()
        console.print("\n" + "="*60)
        
        demonstrate_blur_filter()
        console.print("\n" + "="*60)
        
        demonstrate_pooling()
        console.print("\n" + "="*60)
        
        show_cnn_architecture_preview()
        console.print("\n" + "="*60)
        
        show_production_applications()
        
        # Celebration
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 COMPUTER VISION MASTERY! 🎉[/bold green]\n\n"
            "[cyan]Your Conv2D and MaxPool2D layers are the foundation[/cyan]\n"
            "[cyan]of EVERY modern computer vision system![/cyan]\n\n"
            "[white]These same operations power:[/white]\n"
            "[white]• Self-driving cars[/white]\n"
            "[white]• Medical diagnosis AI[/white]\n"
            "[white]• Photo recognition apps[/white]\n"
            "[white]• Industrial quality control[/white]\n\n"
            "[yellow]Next up: Attention (Module 07) - The transformer revolution![/yellow]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error running showcase: {e}")
        console.print("💡 Make sure you've completed Module 06 and your spatial layers work!")
        import traceback
        console.print(f"Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
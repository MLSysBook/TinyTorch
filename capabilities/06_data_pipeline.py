#!/usr/bin/env python3
"""
🚀 CAPABILITY SHOWCASE: Data Pipeline
After Module 09 (DataLoader)

"Look what you built!" - Your data pipeline can feed neural networks!
"""

import sys
import time
import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.align import Align

# Import from YOUR TinyTorch implementation
try:
    from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset
except ImportError:
    print("❌ TinyTorch DataLoader not found. Make sure you've completed Module 09 (DataLoader)!")
    sys.exit(1)

console = Console()

def ascii_image_small(image_data, width=16, height=8):
    """Convert image to small ASCII representation."""
    if len(image_data.shape) == 3:  # RGB image
        # Convert to grayscale
        gray = np.mean(image_data, axis=2)
    else:
        gray = image_data
    
    # Resize to display size
    h, w = gray.shape
    step_h, step_w = h // height, w // width
    
    if step_h == 0: step_h = 1
    if step_w == 0: step_w = 1
    
    small = gray[::step_h, ::step_w][:height, :width]
    
    # Convert to ASCII
    chars = " ░▒▓█"
    lines = []
    for row in small:
        line = ""
        for pixel in row:
            char_idx = int(pixel * (len(chars) - 1))
            char_idx = max(0, min(char_idx, len(chars) - 1))
            line += chars[char_idx]
        lines.append(line)
    return "\n".join(lines)

def demonstrate_cifar10_loading():
    """Show CIFAR-10 dataset loading capabilities."""
    console.print(Panel.fit("📊 CIFAR-10 DATASET LOADING", style="bold green"))
    
    console.print("🎯 Loading real CIFAR-10 dataset with YOUR DataLoader...")
    console.print("   📁 32×32 color images")
    console.print("   🏷️ 10 classes: planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing CIFAR-10 dataset...", total=None)
        time.sleep(1)
        
        try:
            # Use YOUR CIFAR-10 dataset implementation
            dataset = CIFAR10Dataset(train=True, download=True)
            
            progress.update(task, description="✅ Dataset loaded!")
            time.sleep(0.5)
            
        except Exception as e:
            progress.update(task, description="⚠️ Using sample data (CIFAR-10 not available)")
            time.sleep(0.5)
            # Create sample data for demo
            dataset = create_sample_dataset()
    
    console.print(f"📈 Dataset size: {len(dataset)} training images")
    
    return dataset

def create_sample_dataset():
    """Create sample dataset if CIFAR-10 not available."""
    class SampleDataset:
        def __init__(self):
            self.data = []
            self.labels = []
            
            # Create sample 32x32 images
            np.random.seed(42)  # For reproducible demo
            for i in range(100):  # Small sample
                # Create simple colored patterns
                image = np.random.random((32, 32, 3)) * 0.3
                
                # Add some patterns based on class
                class_id = i % 10
                if class_id == 0:  # Airplane - horizontal lines
                    image[10:15, :, :] = 0.8
                elif class_id == 1:  # Car - rectangle
                    image[12:20, 8:24, :] = 0.7
                elif class_id == 2:  # Bird - circular pattern
                    center = (16, 16)
                    for y in range(32):
                        for x in range(32):
                            if (x-center[0])**2 + (y-center[1])**2 < 64:
                                image[y, x, :] = 0.6
                
                self.data.append(image)
                self.labels.append(class_id)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    return SampleDataset()

def demonstrate_batching():
    """Show batching capabilities."""
    console.print(Panel.fit("📦 BATCH PROCESSING", style="bold blue"))
    
    dataset = create_sample_dataset()
    
    console.print("🔄 Creating DataLoader with YOUR implementation...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating DataLoader...", total=None)
        time.sleep(1)
        
        # Create DataLoader with YOUR implementation
        batch_size = 8
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        progress.update(task, description="✅ DataLoader ready!")
        time.sleep(0.5)
    
    console.print(f"⚙️ Configuration:")
    console.print(f"   📦 Batch size: {batch_size}")
    console.print(f"   🔀 Shuffling: Enabled")
    console.print(f"   📊 Total batches: {len(dataloader)}")
    console.print()
    
    # Show first batch
    console.print("🎯 Loading first batch...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching batch...", total=None)
        time.sleep(1)
        
        batch_images, batch_labels = next(iter(dataloader))
        
        progress.update(task, description="✅ Batch loaded!")
        time.sleep(0.5)
    
    # Display batch info
    console.print(f"📊 [bold]Batch Information:[/bold]")
    console.print(f"   📷 Images shape: {np.array(batch_images).shape}")
    console.print(f"   🏷️ Labels: {batch_labels}")
    
    return batch_images, batch_labels

def visualize_batch_samples(batch_images, batch_labels):
    """Visualize some samples from the batch."""
    console.print(Panel.fit("👀 BATCH VISUALIZATION", style="bold yellow"))
    
    # CIFAR-10 class names
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    console.print("🖼️ Sample images from current batch:")
    console.print()
    
    # Show first 4 images from batch
    for i in range(min(4, len(batch_images))):
        image = np.array(batch_images[i])
        label = batch_labels[i]
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        
        console.print(f"📷 [bold]Image {i+1}: {class_name} (label: {label})[/bold]")
        ascii_art = ascii_image_small(image)
        console.print(ascii_art)
        console.print()

def demonstrate_data_augmentation():
    """Show data augmentation concepts."""
    console.print(Panel.fit("🔄 DATA AUGMENTATION PREVIEW", style="bold magenta"))
    
    console.print("🎯 Data augmentation improves model generalization:")
    console.print()
    
    console.print("   🖼️ [bold]Image Transformations:[/bold]")
    console.print("      • Rotation: ±15 degrees")
    console.print("      • Horizontal flip: 50% chance")
    console.print("      • Random crop: 32×32 from 40×40")
    console.print("      • Color jitter: brightness, contrast")
    console.print("      • Normalization: mean=[0.485, 0.456, 0.406]")
    console.print()
    
    console.print("   📊 [bold]Why Augmentation Works:[/bold]")
    console.print("      • Increases effective dataset size")
    console.print("      • Teaches invariance to transformations")
    console.print("      • Reduces overfitting")
    console.print("      • Improves real-world performance")
    console.print()
    
    # Simulate augmentation pipeline
    console.print("🔄 [bold]Typical Training Pipeline:[/bold]")
    console.print("   1️⃣ Load image from disk")
    console.print("   2️⃣ Apply random transformations")
    console.print("   3️⃣ Convert to tensor")
    console.print("   4️⃣ Normalize pixel values")
    console.print("   5️⃣ Batch together")
    console.print("   6️⃣ Send to GPU")
    console.print("   7️⃣ Feed to neural network")

def show_production_data_pipeline():
    """Show production data pipeline considerations."""
    console.print(Panel.fit("🏭 PRODUCTION DATA PIPELINES", style="bold red"))
    
    console.print("🚀 Your DataLoader scales to production systems:")
    console.print()
    
    console.print("   ⚡ [bold]Performance Optimizations:[/bold]")
    console.print("      • Multi-process data loading (num_workers=8)")
    console.print("      • Prefetching next batch while training")
    console.print("      • Memory mapping large datasets")
    console.print("      • GPU-CPU pipeline overlap")
    console.print()
    
    console.print("   💾 [bold]Storage Systems:[/bold]")
    console.print("      • HDF5 for large scientific datasets")
    console.print("      • TFRecord for TensorFlow ecosystems")
    console.print("      • Parquet for structured data")
    console.print("      • Cloud storage (S3, GCS) integration")
    console.print()
    
    console.print("   📊 [bold]Data Processing at Scale:[/bold]")
    console.print("      • Apache Spark for distributed preprocessing")
    console.print("      • Ray for parallel data loading")
    console.print("      • Kubernetes for container orchestration")
    console.print("      • Data versioning with DVC")
    console.print()
    
    # Performance metrics table
    table = Table(title="Data Loading Performance Targets")
    table.add_column("Dataset Size", style="cyan")
    table.add_column("Batch Size", style="yellow")
    table.add_column("Target Speed", style="green")
    table.add_column("Optimization", style="magenta")
    
    table.add_row("ImageNet", "256", ">1000 img/sec", "Multi-GPU + prefetch")
    table.add_row("COCO", "32", ">500 img/sec", "SSD + memory mapping")
    table.add_row("Custom", "64", ">2000 img/sec", "Preprocessing pipeline")
    
    console.print(table)

def main():
    """Main showcase function."""
    console.clear()
    
    # Header
    header = Panel.fit(
        "[bold cyan]🚀 CAPABILITY SHOWCASE: DATA PIPELINE[/bold cyan]\n"
        "[yellow]After Module 09 (DataLoader)[/yellow]\n\n"
        "[green]\"Look what you built!\" - Your data pipeline can feed neural networks![/green]",
        border_style="bright_blue"
    )
    console.print(Align.center(header))
    console.print()
    
    try:
        dataset = demonstrate_cifar10_loading()
        console.print("\n" + "="*60)
        
        batch_images, batch_labels = demonstrate_batching()
        console.print("\n" + "="*60)
        
        visualize_batch_samples(batch_images, batch_labels)
        console.print("\n" + "="*60)
        
        demonstrate_data_augmentation()
        console.print("\n" + "="*60)
        
        show_production_data_pipeline()
        
        # Celebration
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 DATA PIPELINE MASTERY! 🎉[/bold green]\n\n"
            "[cyan]Your DataLoader is the foundation of ALL machine learning![/cyan]\n\n"
            "[white]No neural network can train without efficient data loading.[/white]\n"
            "[white]Your pipeline powers:[/white]\n"
            "[white]• Computer vision training (ImageNet, COCO)[/white]\n"
            "[white]• NLP model training (massive text corpora)[/white]\n"
            "[white]• Recommendation systems (user behavior data)[/white]\n"
            "[white]• Scientific ML (sensor data, simulations)[/white]\n\n"
            "[yellow]Next up: Training loops (Module 11) - Putting it all together![/yellow]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error running showcase: {e}")
        console.print("💡 Make sure you've completed Module 09 and your DataLoader works!")
        import traceback
        console.print(f"Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
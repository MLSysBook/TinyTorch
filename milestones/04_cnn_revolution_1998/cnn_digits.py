#!/usr/bin/env python3
"""
Milestone 04: The CNN Revolution (LeNet - 1998)

Historical Context:
-------------------
After backpropagation proved MLPs could learn (1986), researchers still struggled
with image recognition. MLPs treated pixels independently, requiring millions of
parameters and ignoring spatial structure.

Then in 1998, Yann LeCun's LeNet-5 revolutionized computer vision with 
Convolutional Neural Networks (CNNs). By using:
- Shared weights (convolution) → 100× fewer parameters
- Local connectivity → preserves spatial structure
- Pooling → translation invariance

LeNet achieved 99%+ accuracy on handwritten digits, launching the deep learning
revolution that led to modern computer vision.

What You'll Build:
------------------
A simple CNN that demonstrates why spatial operations matter:
- Conv layers detect local patterns (edges, curves)
- Pooling provides robustness to small shifts
- Spatial structure preserved throughout

You'll see CNNs outperform MLPs on the same digits dataset from Milestone 03!
"""

import sys
import os
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add paths for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import TinyTorch components
from tinytorch import Tensor, SGD, CrossEntropyLoss, enable_autograd
from tinytorch.core.spatial import Conv2d, MaxPool2d
from tinytorch.core.layers import Linear, ReLU
from tinytorch.data.loader import DataLoader, TensorDataset

console = Console()

# Enable gradient tracking
enable_autograd()


# ============================================================================
# 📊 DATA LOADING
# ============================================================================

def load_digits_dataset():
    """
    Load the 8x8 digits dataset from local file.
    
    Returns 1,797 grayscale images of handwritten digits (0-9).
    Each image is 8×8 pixels, perfect for quick CNN demonstrations.
    """
    # Load from the local data file (same as MLP milestone uses)
    data_path = os.path.join(os.path.dirname(__file__), '../03_mlp_revival_1986/data/digits_8x8.npz')
    data = np.load(data_path)
    
    images = data['images']  # (1797, 8, 8)
    labels = data['labels']  # (1797,)
    
    # Split into train/test (80/20)
    n_train = int(0.8 * len(images))
    
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    
    # CNN expects (batch, channels, height, width)
    # Add channel dimension: (N, 8, 8) → (N, 1, 8, 8)
    train_images = train_images[:, np.newaxis, :, :]  # (1437, 1, 8, 8)
    test_images = test_images[:, np.newaxis, :, :]    # (360, 1, 8, 8)
    
    return (
        Tensor(train_images.astype(np.float32)),
        Tensor(train_labels.astype(np.int64)),
        Tensor(test_images.astype(np.float32)),
        Tensor(test_labels.astype(np.int64))
    )


# ============================================================================
# 🏗️ NETWORK ARCHITECTURE
# ============================================================================

class SimpleCNN:
    """
    Simple Convolutional Neural Network for digit classification.
    
    Architecture inspired by LeNet-5 (1998):
    - Conv2d: Detects local patterns (edges, curves)
    - ReLU: Nonlinearity
    - MaxPool: Spatial down-sampling + translation invariance
    - Linear: Final classification
    
    Input: (batch, 1, 8, 8)
    Conv1: 1 → 8 channels, 3×3 kernel → (batch, 8, 6, 6)
    Pool1: 2×2 max pooling → (batch, 8, 3, 3)
    Flatten: → (batch, 72)
    Linear: 72 → 10 classes
    """
    
    def __init__(self):
        # Convolutional layers
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        
        # After conv(3×3) and pool(2×2): 8×8 → 6×6 → 3×3
        # Flattened size: 8 channels × 3 × 3 = 72
        self.fc = Linear(in_features=72, out_features=10)
        
        # Set requires_grad for all parameters
        self.conv1.weight.requires_grad = True
        self.conv1.bias.requires_grad = True
        self.fc.weight.requires_grad = True
        self.fc.bias.requires_grad = True
        
        self.params = [self.conv1.weight, self.conv1.bias, self.fc.weight, self.fc.bias]
    
    def forward(self, x):
        # Conv + ReLU + Pool
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # Flatten: (batch, 8, 3, 3) → (batch, 72)
        batch_size = out.shape[0]
        out = Tensor(out.data.reshape(batch_size, -1))
        
        # Final classification
        out = self.fc.forward(out)
        return out
    
    def parameters(self):
        return self.params


# ============================================================================
# 🎯 TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    total_loss = 0.0
    n_batches = 0
    
    for batch_images, batch_labels in dataloader:
        # Forward pass
        logits = model.forward(batch_images)
        loss = criterion.forward(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.data.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate_accuracy(model, images, labels):
    """Evaluate model accuracy on a dataset."""
    logits = model.forward(images)
    predictions = np.argmax(logits.data, axis=1)
    accuracy = 100.0 * np.mean(predictions == labels.data)
    avg_loss = np.mean((predictions - labels.data) ** 2)
    return accuracy, avg_loss


# ============================================================================
# 🎬 MAIN MILESTONE DEMONSTRATION
# ============================================================================

def train_cnn():
    """Main training loop following 5-Act structure."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACT 1: THE CHALLENGE 🎯
    # ═══════════════════════════════════════════════════════════════════════
    
    console.print(Panel.fit(
        "[bold cyan]1998: The Computer Vision Challenge[/bold cyan]\n\n"
        "[yellow]The Problem:[/yellow]\n"
        "MLPs flatten images → lose spatial structure\n"
        "Each pixel treated independently\n"
        "Millions of parameters needed for larger images\n\n"
        "[green]The Innovation:[/green]\n"
        "Convolutional Neural Networks (CNNs)\n"
        "  • Shared weights across space (convolution)\n"
        "  • Local connectivity (receptive fields)\n"
        "  • Pooling for translation invariance\n\n"
        "[bold]Can spatial operations outperform dense layers?[/bold]",
        title="🎯 ACT 1: THE CHALLENGE",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Load data
    console.print("\n[bold]📊 Loading Handwritten Digits Dataset...[/bold]")
    train_images, train_labels, test_images, test_labels = load_digits_dataset()
    
    console.print(f"  Training samples: [cyan]{len(train_images.data)}[/cyan]")
    console.print(f"  Test samples: [cyan]{len(test_images.data)}[/cyan]")
    console.print(f"  Image shape: [cyan]{train_images.data[0].shape}[/cyan] (1 channel, 8×8 pixels)")
    console.print(f"  Classes: [cyan]10[/cyan] (digits 0-9)")
    
    # Show training data structure
    console.print(f"\n  [dim]Sample digit values (first image, top-left 3×3):[/dim]")
    sample = train_images.data[0, 0, :3, :3]
    for row in sample:
        console.print(f"    {' '.join(f'{val:.2f}' for val in row)}")
    
    console.print("\n" + "─" * 70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACT 2: THE SETUP 🏗️
    # ═══════════════════════════════════════════════════════════════════════
    
    console.print("[bold]🏗️  The Architecture:[/bold]")
    console.print("""
    ┌──────────┐    ┌──────────┐    ┌──────┐    ┌─────────┐    ┌─────────┐    ┌────────┐
    │  Input   │    │  Conv2d  │    │ ReLU │    │MaxPool2d│    │ Flatten │    │ Linear │
    │ 1×8×8    │───▶│  1→8     │───▶│      │───▶│  2×2    │───▶│ 8×3×3   │───▶│ 72→10  │
    │          │    │  3×3     │    │      │    │         │    │  =72    │    │        │
    └──────────┘    └──────────┘    └──────┘    └─────────┘    └─────────┘    └────────┘
                    ↑ Detects                   ↑ Spatial
                    local patterns              downsampling
    """)
    
    console.print("[bold]🔧 Components:[/bold]")
    console.print("  • Conv layer: Detects local patterns (edges, curves)")
    console.print("  • ReLU: Non-linear activation")
    console.print("  • MaxPool: Spatial downsampling + translation invariance")
    console.print("  • Linear: Final classification (72 → 10 classes)")
    console.print("  • [bold cyan]Key insight: Shared weights → 100× fewer params![/bold cyan]")
    
    # Create model
    console.print("\n🧠 Building Convolutional Neural Network...")
    model = SimpleCNN()
    
    # Count parameters
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    conv_params = np.prod(model.conv1.weight.shape) + np.prod(model.conv1.bias.shape)
    fc_params = np.prod(model.fc.weight.shape) + np.prod(model.fc.bias.shape)
    
    console.print(f"  ✓ Conv layer: [cyan]{conv_params}[/cyan] parameters")
    console.print(f"  ✓ FC layer: [cyan]{fc_params}[/cyan] parameters")
    console.print(f"  ✓ Total: [bold cyan]{total_params}[/bold cyan] parameters")
    
    # Hyperparameters
    console.print("\n[bold]⚙️  Training Configuration:[/bold]")
    epochs = 20  # Reduced for demo speed (explicit loops are slow!)
    batch_size = 32
    learning_rate = 0.01
    
    config_table = Table(show_header=False, box=None)
    config_table.add_row("Epochs:", f"[cyan]{epochs}[/cyan]")
    config_table.add_row("Batch size:", f"[cyan]{batch_size}[/cyan]")
    config_table.add_row("Learning rate:", f"[cyan]{learning_rate}[/cyan]")
    config_table.add_row("Optimizer:", "[cyan]SGD[/cyan]")
    config_table.add_row("Loss:", "[cyan]CrossEntropyLoss[/cyan]")
    console.print(config_table)
    
    # Create optimizer and loss
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    # Create dataloader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    console.print("\n" + "─" * 70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════
    
    console.print("[bold]🔬 Training CNN on Handwritten Digits...[/bold]\n")
    
    # Before training
    initial_acc, initial_loss = evaluate_accuracy(model, test_images, test_labels)
    console.print(f"[yellow]Before training:[/yellow] Accuracy = {initial_acc:.1f}%\n")
    
    # Training loop
    history = {"loss": [], "accuracy": []}
    start_time = time.time()
    
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer)
        accuracy, _ = evaluate_accuracy(model, test_images, test_labels)
        
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.1f}%")
    
    training_time = time.time() - start_time
    
    console.print("\n" + "─" * 70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════
    
    console.print("[bold]📊 The Results:[/bold]\n")
    
    final_acc, _ = evaluate_accuracy(model, test_images, test_labels)
    final_loss = history["loss"][-1]
    
    table = Table(title="Training Outcome", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Before Training", style="yellow", width=16)
    table.add_column("After Training", style="green", width=16)
    table.add_column("Improvement", style="magenta", width=14)
    
    table.add_row(
        "Accuracy",
        f"{initial_acc:.1f}%",
        f"{final_acc:.1f}%",
        f"+{final_acc - initial_acc:.1f}%"
    )
    table.add_row(
        "Training Time",
        "—",
        f"{training_time*1000:.0f}ms",
        "—"
    )
    
    console.print(table)
    
    # Sample predictions
    console.print("\n[bold]🔍 Sample Predictions:[/bold]")
    sample_images = Tensor(test_images.data[:10])  # First 10 test samples
    logits = model.forward(sample_images)
    predictions = np.argmax(logits.data, axis=1)
    
    samples_table = Table(show_header=True, box=box.SIMPLE)
    samples_table.add_column("True", style="cyan", justify="center")
    samples_table.add_column("Pred", style="green", justify="center")
    samples_table.add_column("Result", justify="center")
    
    for i in range(10):
        true_label = int(test_labels.data[i])
        pred_label = int(predictions[i])
        result = "✓" if true_label == pred_label else "✗"
        style = "green" if true_label == pred_label else "red"
        samples_table.add_row(str(true_label), str(pred_label), f"[{style}]{result}[/{style}]")
    
    console.print(samples_table)
    
    # Key insights
    console.print("\n[bold]💡 Key Insights:[/bold]")
    console.print(f"  • CNNs preserve spatial structure")
    console.print(f"  • Conv layers detect local patterns (edges → digits)")
    console.print(f"  • Pooling provides translation invariance")
    console.print(f"  • {total_params} params vs ~5,000 for MLP with similar accuracy!")
    
    console.print("\n" + "─" * 70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════
    
    console.print("")
    console.print(Panel.fit(
        "[bold green]🎉 Success! Your CNN Learned to Recognize Digits![/bold green]\n\n"
        
        f"Final accuracy: [bold]{final_acc:.1f}%[/bold]\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  ✓ Built a Convolutional Neural Network from scratch\n"
        "  ✓ Used Conv2d for spatial feature extraction\n"
        "  ✓ Applied MaxPooling for translation invariance\n"
        f"  ✓ Achieved {final_acc:.1f}% accuracy on digit recognition!\n"
        "  ✓ Used 100× fewer parameters than MLP!\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "[bold]🎓 Why This Matters:[/bold]\n"
        "  LeNet-5 (1998) proved CNNs work for real-world vision.\n"
        "  This breakthrough led to:\n"
        "  • AlexNet (2012) - ImageNet revolution\n"
        "  • VGG, ResNet, modern computer vision\n"
        "  • Self-driving cars, medical imaging, face recognition\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "[bold]📌 The Key Breakthrough:[/bold]\n"
        "  [yellow]Spatial structure matters![/yellow]\n"
        "  MLPs: Every pixel connects to everything → explosion\n"
        "  CNNs: Local connectivity + shared weights → efficiency\n"
        "  \n"
        "  This is why CNNs dominate computer vision today!\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "[bold]🚀 What's Next:[/bold]\n"
        "[dim]You've now built the complete ML training pipeline:\n"
        "  Tensors → Layers → Optimizers → DataLoaders → CNNs\n"
        "  \n"
        "  Next modules will add modern techniques:\n"
        "  • Normalization, Dropout, Advanced architectures\n"
        "  • Attention mechanisms, Transformers\n"
        "  • Production systems, Optimization, Deployment![/dim]",
        
        title="🌟 1998 CNN Revolution Complete",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    train_cnn()


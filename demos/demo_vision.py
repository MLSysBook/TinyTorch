#!/usr/bin/env python3
"""
TinyTorch Demo 06: Computer Vision - Image Processing Revolution
Shows convolutional networks processing images like edge detection and pattern recognition!
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def demo_vision():
    """Demo computer vision with convolutional operations and pattern recognition"""
    
    console = Console()
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        import tinytorch.core.layers as layers
        import tinytorch.core.dense as dense
        import tinytorch.core.spatial as spatial
        
        # Main header
        console.print(Panel.fit(
            "👁️ TinyTorch Computer Vision Demo\nFrom raw pixels to intelligent pattern recognition!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # What this demo shows
        console.print(Panel(
            "[bold yellow]What This Demo Shows:[/bold yellow]\n\n"
            "Convolutional neural networks (CNNs) revolutionized computer vision by learning to detect\n"
            "visual patterns hierarchically. You'll understand:\n\n"
            "• How digital images are just 2D arrays of numbers (tensors)\n"
            "• How convolution operations scan images to detect local patterns\n"
            "• Why edge detection is fundamental - edges define object boundaries\n"
            "• How multiple filters create different 'views' of the same image\n"
            "• Why CNNs build hierarchical features: edges → textures → shapes → objects\n\n"
            "[bold cyan]Key Insight:[/bold cyan] CNNs automatically learn which patterns matter for your task.\n"
            "Early layers detect simple edges, deeper layers combine them into complex features!",
            title="📚 Understanding This Demo",
            style="blue"
        ))
        console.print()
        
        # Demo 1: The Image Processing Foundation
        print("🖼️ Demo 1: Digital Images as Tensors")
        print("Understanding how computers see...")
        print()
        
        # Create a simple 5x5 image
        image = tt.Tensor([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ])
        
        print("Simple 5×5 image (diamond pattern):")
        for row in image.data:
            print("  " + " ".join("█" if pixel else "·" for pixel in row))
        print()
        
        print(f"Image tensor shape: {image.data.shape}")
        print(f"Pixel values: {np.unique(image.data)} (0=black, 1=white)")
        print()
        
        console.print("[dim]💡 [bold]How to Read This:[/bold] Each symbol represents a pixel value:[/dim]")
        console.print("[dim]   • █ = 1 (white/bright pixel), · = 0 (black/dark pixel)[/dim]")
        console.print("[dim]   • This diamond pattern is what the computer 'sees' as numbers[/dim]")
        console.print("[dim]   • Real images have values 0-255, but the principle is the same[/dim]")
        console.print()
        
        # Demo 2: Edge Detection - Computer Vision's Foundation
        print("🔍 Demo 2: Edge Detection - How Computers Find Shapes")
        print("Using convolution to detect edges...")
        print()
        
        # Sobel edge detection kernels
        sobel_x = tt.Tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])  # Detects vertical edges
        
        sobel_y = tt.Tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])  # Detects horizontal edges
        
        print("Sobel X kernel (vertical edge detector):")
        for row in sobel_x.data:
            print(f"  {row}")
        print()
        
        # Apply edge detection
        edge_x = spatial.conv2d_naive(image.data, sobel_x.data)
        edge_y = spatial.conv2d_naive(image.data, sobel_y.data)
        
        print("Vertical edges detected:")
        for row in edge_x:
            print("  " + " ".join(f"{val:2.0f}" for val in row))
        print()
        
        print("Horizontal edges detected:")
        for row in edge_y:
            print("  " + " ".join(f"{val:2.0f}" for val in row))
        print()
        
        console.print("[dim]💡 [bold]Interpreting Edge Detection:[/bold] The numbers show edge strength:[/dim]")
        console.print("[dim]   • Positive values = bright-to-dark transitions[/dim]")
        console.print("[dim]   • Negative values = dark-to-bright transitions[/dim]")
        console.print("[dim]   • Zero = no edge (uniform area)[/dim]")
        console.print("[dim]   • Larger absolute values = stronger edges[/dim]")
        console.print()
        
        # Combine edges
        edge_magnitude = tt.Tensor(np.sqrt(edge_x**2 + edge_y**2))
        print("Combined edge magnitude:")
        for row in edge_magnitude.data:
            print("  " + " ".join(f"{val:2.0f}" for val in row))
        print()
        
        # Demo 3: Pattern Recognition with Conv2D
        print("🎯 Demo 3: Learning Pattern Detectors")
        print("Training convolutional filters to recognize patterns...")
        print()
        
        # Create a Conv2D layer
        conv_layer = spatial.Conv2D(kernel_size=(3, 3))
        
        # Set weights to detect different patterns
        # Pattern 1: Corner detector
        corner_kernel = tt.Tensor([
            [1, 1, 0],
            [1, 0, -1],
            [0, -1, -1]
        ])
        conv_layer.kernel = corner_kernel.data
        
        print("Corner detection kernel:")
        for row in corner_kernel.data:
            print(f"  {row}")
        print()
        
        # Apply corner detection
        corner_response = conv_layer.forward(image)
        print("Corner detection response:")
        for row in corner_response.data:
            print("  " + " ".join(f"{val:2.0f}" for val in row))
        print()
        
        console.print("[dim]💡 [bold]Understanding Feature Detection:[/bold] Each filter learns to detect specific patterns:[/dim]")
        console.print("[dim]   • High positive values = strong match to the pattern[/dim]")
        console.print("[dim]   • Near zero = pattern not present[/dim]")
        console.print("[dim]   • In real CNNs, hundreds of filters learn different features automatically[/dim]")
        console.print()
        
        # Demo 4: Multi-layer Feature Extraction
        print("🏗️ Demo 4: Deep Feature Extraction")
        print("Building feature hierarchy like real CNNs...")
        print()
        
        # Create simple CNN architecture
        cnn = dense.Sequential([
            spatial.Conv2D(kernel_size=(3, 3)), # Feature extraction
            act.ReLU(),                         # Nonlinearity
            spatial.flatten,                    # Flatten for dense layer
            layers.Dense(9, 5),                 # Feature combination
            act.ReLU(),
            layers.Dense(5, 1),                 # Classification
            act.Sigmoid()
        ])
        
        print("CNN Architecture:")
        print("  Input(5×5) → Conv2D(3×3) → ReLU → Flatten → Dense(9→5) → ReLU → Dense(5→1) → Sigmoid")
        print()
        
        console.print("[dim]💡 [bold]Architecture Flow:[/bold] Data transforms through the network:[/dim]")
        console.print("[dim]   • Conv2D: Extracts spatial features (edges, corners)[/dim]")
        console.print("[dim]   • ReLU: Adds nonlinearity for complex patterns[/dim]")
        console.print("[dim]   • Flatten: Converts 2D features to 1D for classification[/dim]")
        console.print("[dim]   • Dense layers: Combine features for final decision[/dim]")
        console.print()
        
        # Set known good weights for demonstration
        cnn.layers[0].kernel = corner_kernel.data  # Use corner detector
        
        # Forward pass
        input_image = image.data.reshape(1, 5, 5)  # Add batch dimension
        result = cnn.forward(tt.Tensor(input_image))
        
        print(f"CNN processes image: {input_image.shape} → {result.data.shape}")
        print(f"Classification score: {result.data[0, 0]:.3f}")
        print(f"Prediction: {'Pattern Detected!' if result.data[0, 0] > 0.5 else 'No Pattern'}")
        print()
        
        # Demo 5: Real-world Image Classification Setup
        print("📱 Demo 5: Production Image Classification")
        print("How this scales to real images...")
        print()
        
        # Simulate processing a real image (32x32, RGB)
        print("Real image classification scenario:")
        print("  Input: 32×32×3 RGB image (3,072 pixels)")
        print("  Conv1: 32 filters, 5×5 kernel → 28×28×32 (25,088 features)")
        print("  MaxPool: 2×2 → 14×14×32 (6,272 features)")
        print("  Conv2: 64 filters, 3×3 → 12×12×64 (9,216 features)")
        print("  MaxPool: 2×2 → 6×6×64 (2,304 features)")
        print("  Flatten → 2,304 → Dense(512) → Dense(10 classes)")
        print()
        
        # Demonstrate memory calculations
        print("Memory analysis:")
        print("  Input: 32×32×3 = 3,072 values × 4 bytes = 12.3 KB")
        print("  Conv1 weights: 5×5×3×32 = 2,400 params × 4 bytes = 9.6 KB")
        print("  Conv2 weights: 3×3×32×64 = 18,432 params × 4 bytes = 73.7 KB")
        print("  Dense weights: 2,304×512 = 1.18M params × 4 bytes = 4.7 MB")
        print("  Total: ~5 MB parameters + activations")
        print()
        
        console.print("[dim]💡 [bold]Scaling Insights:[/bold] Notice how parameters grow:[/dim]")
        console.print("[dim]   • Conv layers: Few parameters but powerful feature extraction[/dim]")
        console.print("[dim]   • Dense layers: Most parameters are here (fully connected)[/dim]")
        console.print("[dim]   • This is why modern CNNs minimize dense layers![/dim]")
        console.print()
        
        # Demo 6: Feature Visualization
        print("👁️ Demo 6: What CNNs Actually Learn")
        print("Visualizing learned features...")
        print()
        
        # Show different specialized kernels
        kernels = {
            "Horizontal Edge": tt.Tensor([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
            "Vertical Edge": tt.Tensor([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
            "Diagonal Edge": tt.Tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),
            "Corner Detector": corner_kernel
        }
        
        print("Specialized feature detectors:")
        for name, kernel in kernels.items():
            print(f"\n{name}:")
            for row in kernel.data:
                print(f"  {row}")
            
            # Show response to our test image
            response = spatial.conv2d_naive(image.data, kernel.data)
            max_response = np.max(np.abs(response))
            print(f"  Max response: {max_response:.1f}")
        
        print()
        
        # Demo 7: Training Process Simulation
        print("🎓 Demo 7: How CNNs Learn Features")
        print("From random filters to intelligent pattern detectors...")
        print()
        
        # Show evolution of learning
        learning_stages = [
            ("Random Init", "Filters detect noise and random patterns"),
            ("Early Training", "Filters start detecting simple edges"),
            ("Mid Training", "Filters specialize in different edge orientations"),
            ("Late Training", "Filters detect complex patterns like corners, curves"),
            ("Converged", "Filters detect object-specific features (wheels, faces, etc.)")
        ]
        
        print("Training evolution:")
        for stage, description in learning_stages:
            print(f"  {stage}: {description}")
        
        print()
        
        console.print("[dim]💡 [bold]Learning Process:[/bold] CNNs discover features automatically:[/dim]")
        console.print("[dim]   • No need to hand-design edge detectors[/dim]")
        console.print("[dim]   • The network learns what patterns matter for your task[/dim]")
        console.print("[dim]   • Different tasks learn different features from same architecture![/dim]")
        console.print()
        
        print("🏆 TinyTorch Computer Vision Demo Complete!")
        print("🎯 Achievements:")
        print("  • Processed images as numerical tensors")
        print("  • Applied edge detection with Sobel operators")
        print("  • Built pattern recognition with Conv2D layers")
        print("  • Created multi-layer feature extraction pipeline")
        print("  • Analyzed real-world image classification architectures")
        print("  • Visualized what CNNs actually learn to detect")
        print("  • Simulated the training process for feature learning")
        print()
        print("🔥 Next: Attention mechanisms for sequence understanding!")
        
        # Success summary
        console.print(Panel.fit(
            "🎯 Achievements:\n"
            "• Understood images as tensors and pixel arrays\n"
            "• Implemented edge detection with convolution filters\n"
            "• Built complete CNN architecture from scratch\n"
            "• Processed real image data with spatial operations\n"
            "• Connected local features to global understanding\n"
            "• Demonstrated the computer vision revolution\n\n"
            "🔥 Next: Attention mechanisms and transformers!",
            title="🏆 TinyTorch Computer Vision Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch modules: {e}\n\n💡 Make sure to run: tito export 06_spatial",
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
    success = demo_vision()
    sys.exit(0 if success else 1)
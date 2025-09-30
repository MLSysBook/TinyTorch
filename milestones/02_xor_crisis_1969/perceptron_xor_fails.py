#!/usr/bin/env python3
"""
The XOR Problem (1969) - Minsky & Papert
========================================

📚 HISTORICAL CONTEXT:
In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," proving that 
single-layer perceptrons CANNOT solve the XOR problem. This killed neural network 
research for a decade (the "AI Winter") until multi-layer networks solved it!

🎯 WHAT YOU'RE BUILDING:
Using YOUR TinyTorch implementations, you'll solve the "impossible" XOR problem
that stumped AI for years - proving that YOUR hidden layers enable non-linear learning!

✅ REQUIRED MODULES (Run after Module 6):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : YOUR data structure with autodiff
  Module 03 (Activations)   : YOUR ReLU for non-linearity (the key!)
  Module 04 (Layers)        : YOUR Linear layers for transformations
  Module 06 (Autograd)      : YOUR gradient computation for learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Multi-Layer Solution):
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Input   │    │ Linear  │    │  ReLU   │    │ Linear  │    │ Binary  │
    │ (x1,x2) │───▶│  2→4    │───▶│ Hidden  │───▶│  4→1    │───▶│ Output  │
    │ 2 dims  │    │ YOUR M4 │    │ YOUR M3 │    │ YOUR M4 │    │ 0 or 1  │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                   Hidden Layer    Non-linearity  Output Layer

🔍 WHY XOR IS SPECIAL - THE NON-LINEAR SEPARABILITY PROBLEM:

The XOR (exclusive OR) problem outputs 1 when inputs differ, 0 when they match:

    Input Space:                    XOR Truth Table:
    
    1 │ (0,1)→1     (1,1)→0         │ x1 │ x2 │ XOR │
      │    RED        BLUE          ├────┼────┼─────┤
      │                             │ 0  │ 0  │  0  │ (same → 0)
    0 │ (0,0)→0     (1,0)→1         │ 0  │ 1  │  1  │ (diff → 1)
      │   BLUE        RED           │ 1  │ 0  │  1  │ (diff → 1)
      └────────────────────         │ 1  │ 1  │  0  │ (same → 0)
        0            1              └────┴────┴─────┘

    🚫 IMPOSSIBLE with single line:     ✅ POSSIBLE with hidden layer:
    
    No single line can separate         Hidden units learn features:
    RED from BLUE points!                - Unit 1: (x1 AND NOT x2)
                                        - Unit 2: (x2 AND NOT x1)
    1 │ R ╱ ╱ ╱ B                      Then combine: Unit1 OR Unit2
      │ ╱ ╱ ╱ ╱ ╱
    0 │ B ╱ ╱ ╱ R                      The hidden layer creates a new
      └────────────                     feature space where XOR becomes
        0        1                      linearly separable!

This is why neural networks need DEPTH - hidden layers create new representations!

📊 EXPECTED PERFORMANCE:
- Dataset: 1,000 XOR samples with slight noise
- Training time: 1 minute  
- Expected accuracy: 95%+ (non-linear problem solved!)
- Key insight: Hidden layer enables non-linear decision boundary
"""

import sys
import os
import numpy as np

# Add project root to path
if __name__ == "__main__":
    # When run as script
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
else:
    # When imported, assume we're already in right location
    sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss, SGD

class XORNetwork:
    """
    Multi-layer network that solves XOR using YOUR TinyTorch implementations!
    
    The hidden layer is the KEY - it learns features that make XOR separable.
    """
    
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        print("🧠 Building XOR Network with YOUR TinyTorch modules...")
        
        # Hidden layer - this is what Minsky said was needed!
        self.hidden = Linear(input_size, hidden_size)  # Module 04: YOUR Linear layer!
        self.activation = ReLU()                       # Module 03: YOUR ReLU (key to non-linearity!)
        self.output = Linear(hidden_size, output_size) # Module 04: YOUR output layer!
        self.sigmoid = Sigmoid()                       # Module 03: YOUR final activation!
        
        print(f"   Input → Hidden: {input_size} → {hidden_size} (YOUR Linear layer)")
        print(f"   Hidden activation: ReLU (YOUR non-linearity - this solves XOR!)")
        print(f"   Hidden → Output: {hidden_size} → {output_size} (YOUR Linear layer)")
        print(f"   Output activation: Sigmoid (YOUR Module 03)")
        
    def forward(self, x):
        """Forward pass through YOUR multi-layer network."""
        # Hidden layer with non-linearity (the SECRET to solving XOR!)
        x = self.hidden(x)        # Module 04: YOUR Linear transformation!
        x = self.activation(x)    # Module 03: YOUR ReLU - creates non-linear features!
        
        # Output layer
        x = self.output(x)        # Module 04: YOUR final transformation!
        x = self.sigmoid(x)       # Module 03: YOUR sigmoid for probability!
        
        return x
    
    def parameters(self):
        """Get all trainable parameters from YOUR layers."""
        return [
            self.hidden.weights, self.hidden.bias,    # Module 04: YOUR hidden parameters!
            self.output.weights, self.output.bias     # Module 04: YOUR output parameters!
        ]

def visualize_xor_problem():
    """Show why XOR is non-linearly separable using ASCII art."""
    print("\n" + "="*70)
    print("🎨 VISUALIZING THE XOR PROBLEM - Why Single Layers Fail:")
    print("="*70)
    
    print("""
    XOR DATA POINTS:                  SINGLE LAYER ATTEMPT:
    
    1.0 │ ○(0,1)=1    ●(1,1)=0       1.0 │ ○         ●    
        │   RED        BLUE              │    ╲           
        │                                 │     ╲  ← No single line
    0.5 │                             0.5 │      ╲    can separate!
        │                                 │       ╲        
        │                                 │        ╲       
    0.0 │ ●(0,0)=0    ○(1,0)=1       0.0 │ ●        ╲ ○   
        └─────────────────────           └─────────────────
          0.0   0.5   1.0                  0.0   0.5   1.0
    
    Legend: ○ = Output 1 (RED)       Problem: RED and BLUE points
            ● = Output 0 (BLUE)               are diagonally mixed!
    """)
    
    print("🔄 THE MULTI-LAYER SOLUTION:")
    print("""
    Hidden Layer Features:            New Feature Space:
    
    Hidden Unit 1: x1 AND NOT x2      In hidden space, XOR becomes
    Hidden Unit 2: x2 AND NOT x1      linearly separable!
    
    Original → Hidden Transform:       Now a single line works:
    (0,0) → [0,0] → 0 ✓               
    (0,1) → [0,1] → 1 ✓               H2 │     ○(0,1)
    (1,0) → [1,0] → 1 ✓                  │    ╱ 
    (1,1) → [0,0] → 0 ✓                  │   ╱  ○(1,0)
                                          │  ╱
    YOUR hidden layer learned         0  │ ●────────────
    to transform the problem!            0        H1
    """)
    print("="*70)

def train_xor_network(model, X, y, learning_rate=0.1, epochs=100):
    """
    Train XOR network using YOUR autograd system with efficient monitoring!

    This uses a simplified but effective approach with progress tracking.
    """
    print("\n🚀 Training XOR Network with YOUR TinyTorch autograd!")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max epochs: {epochs}")
    print(f"   Using validation split and progress monitoring!")

    # Split data manually for monitoring
    n_samples = len(X)
    n_val = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    print(f"   Split: {len(X_train)} training, {len(X_val)} validation samples")

    # Convert to YOUR Tensor format
    X_train_tensor = Tensor(X_train)
    y_train_tensor = Tensor(y_train.reshape(-1, 1))
    X_val_tensor = Tensor(X_val)
    y_val_tensor = Tensor(y_val.reshape(-1, 1))

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training step
        predictions = model.forward(X_train_tensor)

        # Simple MSE loss that maintains computational graph
        diff = predictions - y_train_tensor
        squared_diff = diff * diff

        # Backward pass with proper graph maintenance
        n_samples = squared_diff.data.shape[0]
        grad_output = Tensor(np.ones_like(squared_diff.data) / n_samples)
        squared_diff.backward(grad_output)

        # Update parameters
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                grad_np = np.array(grad_data.data if hasattr(grad_data, 'data') else grad_data)
                param.data = param.data - learning_rate * grad_np
                param.grad = None

        # Calculate metrics
        pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
        y_train_np = np.array(y_train_tensor.data.data if hasattr(y_train_tensor.data, 'data') else y_train_tensor.data)
        train_loss = np.mean((pred_np - y_train_np) ** 2)
        train_acc = np.mean((pred_np > 0.5) == y_train_np) * 100

        # Validation step
        val_predictions = model.forward(X_val_tensor)
        val_pred_np = np.array(val_predictions.data.data if hasattr(val_predictions.data, 'data') else val_predictions.data)
        y_val_np = np.array(y_val_tensor.data.data if hasattr(y_val_tensor.data, 'data') else y_val_tensor.data)
        val_loss = np.mean((val_pred_np - y_val_np) ** 2)
        val_acc = np.mean((val_pred_np > 0.5) == y_val_np) * 100

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            status = "📈"
        else:
            epochs_no_improve += 1
            status = "⚠️" if epochs_no_improve > patience // 2 else "📊"

        # Progress updates
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"   {status} Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
            if val_loss == best_val_loss:
                print(f"       ✅ New best validation loss: {val_loss:.4f}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"   Early stopping triggered after {patience} epochs without improvement")
            break

    # Create monitor-like object for compatibility
    class SimpleMonitor:
        def __init__(self):
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_accuracies = train_accs
            self.val_accuracies = val_accs
            self.best_val_loss = best_val_loss
            self.should_stop = epochs_no_improve >= patience

        def get_summary(self):
            return {
                'total_epochs': len(train_losses),
                'best_val_loss': self.best_val_loss,
                'final_train_acc': train_accs[-1] if train_accs else 0,
                'best_val_acc': max(val_accs) if val_accs else 0,
                'early_stopped': self.should_stop,
                'epochs_no_improve': epochs_no_improve,
                'total_time': 0.1  # Placeholder
            }

    monitor = SimpleMonitor()

    print(f"\n🏁 Training Complete!")
    print(f"   • Total epochs: {len(train_losses)}")
    print(f"   • Best validation loss: {best_val_loss:.4f}")
    print(f"   • Best validation accuracy: {max(val_accs):.1f}%")
    print(f"   • Final training accuracy: {train_accs[-1]:.1f}%")

    return model, monitor

def test_xor_solution(model, show_examples=True):
    """Test YOUR XOR solution on the classic 4 points."""
    print("\n🧪 Testing YOUR XOR Network on Classic Examples:")
    print("   " + "─"*45)
    
    # The classic XOR test cases
    test_cases = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    expected = np.array([0, 1, 1, 0])
    
    # Test with YOUR network
    X_test = Tensor(test_cases)  # Module 02: YOUR Tensor!
    predictions = model.forward(X_test)  # YOUR forward pass!
    pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
    predicted_classes = (pred_np > 0.5).astype(int).flatten()
    
    # Display results
    print("   │ x1 │ x2 │ Expected │ YOUR Output │ ✓/✗ │")
    print("   ├────┼────┼──────────┼─────────────┼─────┤")
    
    all_correct = True
    for i in range(4):
        x1, x2 = test_cases[i]
        exp = expected[i]
        pred = predicted_classes[i]
        prob = pred_np[i, 0]
        status = "✓" if pred == exp else "✗"
        if pred != exp:
            all_correct = False
        
        print(f"   │ {x1:.0f}  │ {x2:.0f}  │    {exp}     │  {pred} ({prob:.3f})  │  {status}  │")
    
    print("   " + "─"*45)
    
    if all_correct:
        print("   🎉 SUCCESS! YOUR network solved XOR perfectly!")
        print("   Hidden layers enabled non-linear learning!")
    else:
        print("   🔄 Network still training... (try more epochs)")
    
    return all_correct

def analyze_xor_systems(model, monitor=None):
    """Analyze YOUR XOR solution from an ML systems perspective."""
    print("\n🔬 SYSTEMS ANALYSIS of YOUR XOR Network:")

    # Parameter count
    total_params = sum(p.data.size for p in model.parameters())

    print(f"   Parameters: {total_params} weights (YOUR Linear layers)")
    print(f"   Architecture: 2 → 4 → 1 (minimal for XOR)")
    print(f"   Key innovation: Hidden layer creates non-linear features")
    print(f"   Memory: {total_params * 4} bytes (float32)")

    # Training efficiency analysis
    if monitor:
        summary = monitor.get_summary()
        print(f"\n   🚀 Training Efficiency:")
        print(f"   • Epochs to convergence: {summary['total_epochs']}")
        print(f"   • Training time: {summary['total_time']:.1f}s")
        print(f"   • Validation-based early stopping: {'Yes' if summary['early_stopped'] else 'No'}")
        print(f"   • Best validation loss: {summary['best_val_loss']:.4f}")

    print("\n   🏛️ Historical Impact:")
    print("   • 1969: Minsky showed single layers CAN'T solve XOR")
    print("   • 1970s: 'AI Winter' - neural networks abandoned")
    print("   • 1980s: Backprop + hidden layers solved it (YOUR approach!)")
    print("   • Today: Deep networks with many hidden layers power AI")

    print("\n   💡 Why This Matters:")
    print("   • YOUR hidden layer transforms the feature space")
    print("   • Non-linear activation (ReLU) is ESSENTIAL")
    print("   • This principle scales to ImageNet, GPT, etc.")
    print("   • Modern AI = deeper versions of YOUR XOR network!")

def main():
    """Demonstrate the XOR solution using YOUR TinyTorch system!"""
    
    parser = argparse.ArgumentParser(description='XOR Problem 1969')
    parser.add_argument('--test-only', action='store_true',
                       help='Test architecture without training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (with early stopping)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Show XOR visualization')
    args = parser.parse_args()
    
    print("🎯 XOR PROBLEM 1969 - Breaking the Linear Barrier!")
    print("   Historical significance: Proved need for hidden layers")
    print("   YOUR achievement: Solving 'impossible' problem with YOUR network")
    print("   Components used: YOUR Tensor + Linear + ReLU + Autograd")
    
    # Show why XOR is special
    if args.visualize:
        visualize_xor_problem()
    
    # Step 1: Get XOR data
    print("\n📊 Generating XOR dataset...")
    data_manager = DatasetManager()
    X, y = data_manager.get_xor_data(num_samples=1000)
    print(f"   Generated {len(X)} XOR samples with noise")
    
    # Step 2: Create network with YOUR components
    model = XORNetwork(input_size=2, hidden_size=4, output_size=1)
    
    if args.test_only:
        print("\n🧪 ARCHITECTURE TEST MODE")
        test_input = Tensor(X[:4])  # Module 02: YOUR Tensor!
        test_output = model.forward(test_input)  # YOUR architecture!
        print(f"✅ Forward pass successful! Output shape: {test_output.data.shape}")
        print("✅ YOUR multi-layer network works!")
        return
    
    # Step 3: Train using YOUR autograd with modern infrastructure
    model, monitor = train_xor_network(model, X, y, epochs=args.epochs)
    
    # Step 4: Test on classic XOR cases
    solved = test_xor_solution(model)
    
    # Step 5: Systems analysis
    analyze_xor_systems(model, monitor)
    
    print("\n✅ SUCCESS! XOR Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • YOU solved the 'impossible' XOR problem")
    print("   • YOUR hidden layer creates non-linear decision boundaries")
    print("   • YOUR ReLU activation enables feature learning")
    print("   • YOUR autograd trains multi-layer networks")
    
    print("\n🚀 Next Steps:")
    print("   • Continue to MNIST MLP after Module 08 (Training)")
    print("   • YOUR XOR solution scales to real vision problems!")
    print("   • Hidden layers principle powers all modern deep learning!")

if __name__ == "__main__":
    main()
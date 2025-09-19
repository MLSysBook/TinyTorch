#!/usr/bin/env python3
"""
Detailed validation script for TinyTorch demos.
Checks for specific expected outputs and functionality.
"""

import sys
import subprocess
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class DemoValidator:
    """Validates demos against expected outputs and patterns."""
    
    def __init__(self):
        self.console = Console()
        self.validations = []
    
    def run_demo(self, demo_file: str) -> str:
        """Run a demo and return its output."""
        demo_path = Path("demos") / demo_file
        
        try:
            result = subprocess.run(
                [sys.executable, str(demo_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_pattern(self, output: str, pattern: str, description: str) -> bool:
        """Check if a pattern exists in the output."""
        found = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        return found is not None
    
    def validate_demo_tensor_math(self):
        """Validate tensor math demo."""
        output = self.run_demo("demo_tensor_math.py")
        
        checks = [
            ("Linear system solution", r"x = 2\.0, y = 3\.0", "Correct solution to linear system"),
            ("Matrix rotation", r"0\.707.*0\.707", "Rotation matrix applied correctly"),
            ("Batch processing", r"Batch Processing", "Batch operations demonstrated"),
            ("Neural network preview", r"Neural Network Preview", "NN preview shown"),
            ("Success completion", r"Demo Complete", "Demo completed successfully"),
            ("Understanding panel", r"Understanding This Demo", "Educational content present"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_tensor_math.py", results
    
    def validate_demo_activations(self):
        """Validate activations demo."""
        output = self.run_demo("demo_activations.py")
        
        checks = [
            ("ReLU function", r"ReLU\(x\)", "ReLU activation demonstrated"),
            ("Sigmoid function", r"Sigmoid\(x\)", "Sigmoid activation demonstrated"),
            ("XOR problem", r"XOR", "XOR problem explained"),
            ("Softmax", r"Softmax", "Softmax for classification shown"),
            ("Success completion", r"Demo Complete", "Demo completed successfully"),
            ("Interpretation guides", r"💡.*How to Interpret", "Interpretation guides present"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_activations.py", results
    
    def validate_demo_single_neuron(self):
        """Validate single neuron demo."""
        output = self.run_demo("demo_single_neuron.py")
        
        checks = [
            ("AND gate table", r"AND Output", "AND gate truth table shown"),
            ("Weight updates", r"Weight 1.*Weight 2.*Bias", "Weight updates displayed"),
            ("Training progress", r"Training.*Neuron", "Training process shown"),
            ("Decision boundary", r"Decision.*boundary", "Decision boundary explained"),
            ("Dense layer", r"Dense", "TinyTorch Dense layer used"),
            ("Learning insights", r"💡.*What's Happening", "Learning process explained"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_single_neuron.py", results
    
    def validate_demo_xor_network(self):
        """Validate XOR network demo."""
        output = self.run_demo("demo_xor_network.py")
        
        checks = [
            ("XOR truth table", r"XOR Output", "XOR truth table displayed"),
            ("Hidden layer", r"Hidden.*layer", "Hidden layer explanation"),
            ("Multi-layer solution", r"Multi-[Ll]ayer", "Multi-layer network shown"),
            ("Sequential model", r"Sequential", "Sequential model demonstrated"),
            ("Success completion", r"Demo Complete", "Demo completed successfully"),
            ("Key insights", r"Key Insight", "Educational insights provided"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_xor_network.py", results
    
    def validate_demo_vision(self):
        """Validate computer vision demo."""
        output = self.run_demo("demo_vision.py")
        
        checks = [
            ("Image as tensor", r"5×5.*diamond", "Image representation shown"),
            ("Edge detection", r"[Ee]dge [Dd]etection", "Edge detection demonstrated"),
            ("Convolution", r"Conv", "Convolution operations shown"),
            ("CNN architecture", r"CNN", "CNN architecture explained"),
            ("Feature maps", r"[Ff]eature", "Feature extraction discussed"),
            ("Scaling insights", r"💡.*Scaling", "Scaling analysis provided"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_vision.py", results
    
    def validate_demo_attention(self):
        """Validate attention mechanisms demo."""
        output = self.run_demo("demo_attention.py")
        
        checks = [
            ("Attention scores", r"[Aa]ttention.*scores", "Attention scores computed"),
            ("Multi-head", r"Multi-[Hh]ead", "Multi-head attention shown"),
            ("Self-attention", r"Self-[Aa]ttention", "Self-attention explained"),
            ("Transformer", r"Transformer", "Transformer architecture shown"),
            ("Q, K, V", r"Q.*K.*V", "Query, Key, Value explained"),
            ("Scaling analysis", r"O\(n²\)", "Computational complexity discussed"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_attention.py", results
    
    def validate_demo_training(self):
        """Validate training demo."""
        output = self.run_demo("demo_training.py")
        
        checks = [
            ("Dataset creation", r"Dataset.*samples", "Dataset created"),
            ("Model architecture", r"Model architecture", "Architecture described"),
            ("Training loop", r"Training", "Training loop demonstrated"),
            ("Loss tracking", r"Loss", "Loss values shown"),
            ("Accuracy metrics", r"[Aa]ccuracy", "Accuracy tracked"),
            ("Production context", r"[Pp]roduction", "Production considerations discussed"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_training.py", results
    
    def validate_demo_language(self):
        """Validate language generation demo."""
        output = self.run_demo("demo_language.py")
        
        checks = [
            ("Tokenization", r"Token", "Tokenization explained"),
            ("Embeddings", r"Embedding", "Word embeddings shown"),
            ("Autoregressive", r"[Aa]utoregressive", "Autoregressive generation explained"),
            ("TinyGPT", r"TinyGPT", "TinyGPT architecture discussed"),
            ("Scaling laws", r"GPT-[1234]", "Model scaling shown"),
            ("Journey complete", r"Journey|journey", "Learning journey summarized"),
        ]
        
        results = []
        for name, pattern, desc in checks:
            passed = self.check_pattern(output, pattern, desc)
            results.append((name, passed, desc))
        
        return "demo_language.py", results
    
    def validate_all(self):
        """Run all validations."""
        
        self.console.print(Panel.fit(
            "🔬 TinyTorch Demo Deep Validation\nChecking specific outputs and functionality",
            style="bold cyan",
            border_style="bright_blue"
        ))
        self.console.print()
        
        # Run each validation
        validators = [
            self.validate_demo_tensor_math,
            self.validate_demo_activations,
            self.validate_demo_single_neuron,
            self.validate_demo_xor_network,
            self.validate_demo_vision,
            self.validate_demo_attention,
            self.validate_demo_training,
            self.validate_demo_language,
        ]
        
        all_results = []
        
        self.console.print("🧪 Running detailed validations...")
        self.console.print()
        
        for validator in validators:
            demo_name, results = validator()
            all_results.append((demo_name, results))
            
            # Show progress
            passed = sum(1 for _, p, _ in results if p)
            total = len(results)
            status = "✅" if passed == total else "⚠️"
            self.console.print(f"{status} {demo_name}: {passed}/{total} checks passed")
        
        self.console.print()
        
        # Detailed results table
        self.console.print("📋 Detailed Validation Results:")
        self.console.print()
        
        for demo_name, results in all_results:
            table = Table(show_header=True, header_style="bold magenta", title=demo_name)
            table.add_column("Check", style="cyan", width=25)
            table.add_column("Status", style="green", width=8)
            table.add_column("Description", style="yellow", width=45)
            
            for check_name, passed, description in results:
                status = "✅ PASS" if passed else "❌ FAIL"
                status_style = "green" if passed else "red"
                table.add_row(
                    check_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    description
                )
            
            self.console.print(table)
            self.console.print()
        
        # Summary
        total_checks = sum(len(results) for _, results in all_results)
        passed_checks = sum(sum(1 for _, p, _ in results if p) for _, results in all_results)
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        if success_rate == 100:
            self.console.print(Panel.fit(
                f"🎉 Perfect! All {total_checks} validation checks passed!",
                style="bold green",
                border_style="bright_green"
            ))
        elif success_rate >= 90:
            self.console.print(Panel.fit(
                f"✅ Excellent! {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)",
                style="bold green",
                border_style="green"
            ))
        elif success_rate >= 70:
            self.console.print(Panel.fit(
                f"⚠️  Good but needs work: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)",
                style="bold yellow",
                border_style="yellow"
            ))
        else:
            self.console.print(Panel.fit(
                f"❌ Needs attention: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)",
                style="bold red",
                border_style="red"
            ))
        
        return 0 if success_rate == 100 else 1

if __name__ == "__main__":
    validator = DemoValidator()
    sys.exit(validator.validate_all())
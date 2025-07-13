#!/usr/bin/env python3
"""
TinyTorch Module Analysis Wrapper

Simple wrapper to run the module analyzer from the root directory.
"""

import sys
import os
from pathlib import Path

# Add instructor tools to path
sys.path.insert(0, str(Path(__file__).parent / "instructor" / "tools"))

# Import and run the analyzer
from tinytorch_module_analyzer import TinyTorchModuleAnalyzer
import argparse

def main():
    parser = argparse.ArgumentParser(description="TinyTorch Module Analyzer & Report Card Generator")
    parser.add_argument("--module", help="Analyze specific module (e.g., 02_activations)")
    parser.add_argument("--all", action="store_true", help="Analyze all modules")
    parser.add_argument("--compare", nargs="+", help="Compare multiple modules")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="Output format")
    parser.add_argument("--save", action="store_true", help="Save report cards to files")
    
    args = parser.parse_args()
    
    # Use correct path from root directory
    analyzer = TinyTorchModuleAnalyzer("modules/source")
    
    if args.module:
        # Analyze single module
        print(f"🔍 Analyzing module: {args.module}")
        try:
            report_card = analyzer.analyze_module(args.module)
            print(f"\n📊 Report Card for {args.module}:")
            print(f"Overall Grade: {report_card.overall_grade}")
            print(f"Scaffolding Quality: {report_card.scaffolding_quality}/5")
            print(f"Critical Issues: {len(report_card.critical_issues)}")
            
            if args.save:
                saved_files = analyzer.save_report_card(report_card, args.format)
                print(f"💾 Saved to: {', '.join(saved_files)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif args.all:
        # Analyze all modules
        print("🔍 Analyzing all modules...")
        results = analyzer.analyze_all_modules()
        
        print("\n📊 Summary Report:")
        for name, rc in results.items():
            print(f"{name}: Grade {rc.overall_grade} | Scaffolding {rc.scaffolding_quality}/5")
            
        if args.save:
            for name, rc in results.items():
                saved_files = analyzer.save_report_card(rc, args.format)
                print(f"💾 {name} saved to: {', '.join(saved_files)}")
    
    elif args.compare:
        # Compare modules
        print(f"🔍 Comparing modules: {', '.join(args.compare)}")
        comparison = analyzer.compare_modules(args.compare)
        print(f"\n{comparison}")
        
        if args.save:
            from datetime import datetime
            with open(f"instructor/reports/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", 'w') as f:
                f.write(comparison)
            print("💾 Comparison saved to instructor/reports/")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
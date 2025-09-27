#!/usr/bin/env python3
"""
Educational Enhancement Integration Tests for TinyTorch
Tests visual teaching elements, systems insights, and pedagogical coherence
"""

import sys
import os
import re
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class EducationalIntegrationTester:
    def __init__(self):
        self.results = {}
        self.module_paths = [
            "/Users/VJ/GitHub/TinyTorch/modules/02_tensor/tensor_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/03_activations/activations_dev.py", 
            "/Users/VJ/GitHub/TinyTorch/modules/04_layers/layers_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/05_losses/losses_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/06_autograd/autograd_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/07_optimizers/optimizers_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/08_training/training_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/09_spatial/spatial_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/10_dataloader/dataloader_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/12_embeddings/embeddings_dev.py",
            "/Users/VJ/GitHub/TinyTorch/modules/13_attention/attention_dev.py"
        ]
        
    def print_section(self, title):
        """Print section header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        
    def test_all_educational_integration(self):
        """Run all educational integration tests"""
        self.print_section("EDUCATIONAL ENHANCEMENT INTEGRATION TESTS")
        
        # Test 1: Visual Teaching Elements Consistency
        self.test_visual_teaching_consistency()
        
        # Test 2: Systems Insights Integration
        self.test_systems_insights_integration()
        
        # Test 3: Graduated Comment Strategy
        self.test_graduated_comment_strategy()
        
        # Test 4: ML Systems Thinking Questions
        self.test_ml_systems_thinking_questions()
        
        # Test 5: Pedagogical Flow Consistency
        self.test_pedagogical_flow_consistency()
        
        # Test 6: NBGrader Metadata Integration
        self.test_nbgrader_metadata_integration()
        
        # Generate report
        self.generate_educational_report()
        
    def test_visual_teaching_consistency(self):
        """Test consistency of visual teaching elements across modules"""
        print("\n1. Testing Visual Teaching Elements Consistency...")
        
        visual_elements = {
            "emojis_in_headers": 0,
            "progress_indicators": 0,
            "visual_separators": 0,
            "step_by_step_numbering": 0,
            "memory_diagrams": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for visual elements
                if re.search(r'#{1,3}.*[🔥🧪🎯🚀💡⚡🔬🎉📈🔍]', content):
                    visual_elements["emojis_in_headers"] += 1
                    
                if "Progress:" in content or "progress:" in content:
                    visual_elements["progress_indicators"] += 1
                    
                if "=" * 20 in content or "-" * 20 in content:
                    visual_elements["visual_separators"] += 1
                    
                if re.search(r'\d+\.\s+\*\*', content):  # Numbered bold steps
                    visual_elements["step_by_step_numbering"] += 1
                    
                if "memory" in content.lower() and ("diagram" in content.lower() or "layout" in content.lower()):
                    visual_elements["memory_diagrams"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for element, count in visual_elements.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {element}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["visual_teaching"] = {
            "total_modules": total_modules,
            "elements": visual_elements,
            "average_coverage": sum(visual_elements.values()) / (len(visual_elements) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def test_systems_insights_integration(self):
        """Test systems insights integration across modules"""
        print("\n2. Testing Systems Insights Integration...")
        
        systems_concepts = {
            "memory_analysis": 0,
            "performance_complexity": 0,
            "scaling_behavior": 0,
            "production_context": 0,
            "hardware_implications": 0,
            "memory_profiling": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check for systems concepts
                if any(term in content for term in ["memory", "ram", "allocation", "footprint"]):
                    systems_concepts["memory_analysis"] += 1
                    
                if any(term in content for term in ["complexity", "o(n", "performance", "timing"]):
                    systems_concepts["performance_complexity"] += 1
                    
                if any(term in content for term in ["scaling", "bottleneck", "larger", "scaling"]):
                    systems_concepts["scaling_behavior"] += 1
                    
                if any(term in content for term in ["pytorch", "tensorflow", "production", "deployment"]):
                    systems_concepts["production_context"] += 1
                    
                if any(term in content for term in ["gpu", "cpu", "vectorization", "parallel"]):
                    systems_concepts["hardware_implications"] += 1
                    
                if any(term in content for term in ["tracemalloc", "profiling", "memory_info", "psutil"]):
                    systems_concepts["memory_profiling"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for concept, count in systems_concepts.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {concept}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["systems_insights"] = {
            "total_modules": total_modules,
            "concepts": systems_concepts,
            "average_coverage": sum(systems_concepts.values()) / (len(systems_concepts) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def test_graduated_comment_strategy(self):
        """Test graduated comment strategy across modules"""
        print("\n3. Testing Graduated Comment Strategy...")
        
        comment_patterns = {
            "detailed_explanations": 0,
            "inline_comments": 0,
            "docstring_examples": 0,
            "parameter_explanations": 0,
            "step_by_step_comments": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for comment patterns
                if len(re.findall(r'""".*?"""', content, re.DOTALL)) > 5:
                    comment_patterns["detailed_explanations"] += 1
                    
                if len(re.findall(r'#.*\n', content)) > 20:
                    comment_patterns["inline_comments"] += 1
                    
                if ">>>" in content and "Example" in content:
                    comment_patterns["docstring_examples"] += 1
                    
                if "Args:" in content or "Parameters:" in content:
                    comment_patterns["parameter_explanations"] += 1
                    
                if re.search(r'# Step \d+', content):
                    comment_patterns["step_by_step_comments"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for pattern, count in comment_patterns.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {pattern}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["graduated_comments"] = {
            "total_modules": total_modules,
            "patterns": comment_patterns,
            "average_coverage": sum(comment_patterns.values()) / (len(comment_patterns) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def test_ml_systems_thinking_questions(self):
        """Test ML Systems Thinking questions across modules"""
        print("\n4. Testing ML Systems Thinking Questions...")
        
        question_types = {
            "memory_performance": 0,
            "systems_architecture": 0,
            "production_engineering": 0,
            "scaling_analysis": 0,
            "interactive_questions": 0,
            "reflection_questions": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check for question types
                if "memory" in content and "performance" in content and "?" in content:
                    question_types["memory_performance"] += 1
                    
                if "architecture" in content and "?" in content:
                    question_types["systems_architecture"] += 1
                    
                if "production" in content and "?" in content:
                    question_types["production_engineering"] += 1
                    
                if "scaling" in content and "?" in content:
                    question_types["scaling_analysis"] += 1
                    
                if "ml systems thinking" in content:
                    question_types["interactive_questions"] += 1
                    
                if "reflection" in content and "?" in content:
                    question_types["reflection_questions"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for qtype, count in question_types.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {qtype}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["ml_systems_questions"] = {
            "total_modules": total_modules,
            "question_types": question_types,
            "average_coverage": sum(question_types.values()) / (len(question_types) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def test_pedagogical_flow_consistency(self):
        """Test pedagogical flow consistency across modules"""
        print("\n5. Testing Pedagogical Flow Consistency...")
        
        flow_elements = {
            "learning_objectives": 0,
            "build_use_reflect": 0,
            "implementation_first": 0,
            "immediate_testing": 0,
            "systems_analysis": 0,
            "module_summary": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for flow elements
                if "Learning Goals" in content or "learning objectives" in content:
                    flow_elements["learning_objectives"] += 1
                    
                if "Build → Use → Reflect" in content or "build.*use.*reflect" in content.lower():
                    flow_elements["build_use_reflect"] += 1
                    
                if re.search(r'def\s+\w+.*:', content) and re.search(r'test.*:', content):
                    flow_elements["implementation_first"] += 1
                    
                if "Unit Test:" in content:
                    flow_elements["immediate_testing"] += 1
                    
                if "Systems Analysis" in content or "systems analysis" in content.lower():
                    flow_elements["systems_analysis"] += 1
                    
                if "MODULE SUMMARY" in content or "module summary" in content.lower():
                    flow_elements["module_summary"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for element, count in flow_elements.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {element}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["pedagogical_flow"] = {
            "total_modules": total_modules,
            "elements": flow_elements,
            "average_coverage": sum(flow_elements.values()) / (len(flow_elements) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def test_nbgrader_metadata_integration(self):
        """Test NBGrader metadata integration across modules"""
        print("\n6. Testing NBGrader Metadata Integration...")
        
        nbgrader_elements = {
            "solution_blocks": 0,
            "grade_cells": 0,
            "locked_cells": 0,
            "schema_version_3": 0,
            "grade_ids": 0,
            "assessment_questions": 0
        }
        
        total_modules = 0
        
        for module_path in self.module_paths:
            if os.path.exists(module_path):
                total_modules += 1
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for NBGrader elements
                if "BEGIN SOLUTION" in content and "END SOLUTION" in content:
                    nbgrader_elements["solution_blocks"] += 1
                    
                if '"grade": true' in content:
                    nbgrader_elements["grade_cells"] += 1
                    
                if '"locked": true' in content:
                    nbgrader_elements["locked_cells"] += 1
                    
                if '"schema_version": 3' in content:
                    nbgrader_elements["schema_version_3"] += 1
                    
                if '"grade_id"' in content:
                    nbgrader_elements["grade_ids"] += 1
                    
                if any(term in content for term in ["assessment", "question", "quiz", "grade"]):
                    nbgrader_elements["assessment_questions"] += 1
        
        print(f"  Total modules analyzed: {total_modules}")
        for element, count in nbgrader_elements.items():
            percentage = (count / total_modules) * 100 if total_modules > 0 else 0
            print(f"  {element}: {count}/{total_modules} modules ({percentage:.1f}%)")
            
        self.results["nbgrader_metadata"] = {
            "total_modules": total_modules,
            "elements": nbgrader_elements,
            "average_coverage": sum(nbgrader_elements.values()) / (len(nbgrader_elements) * total_modules) * 100 if total_modules > 0 else 0
        }
        
    def generate_educational_report(self):
        """Generate comprehensive educational integration report"""
        self.print_section("EDUCATIONAL INTEGRATION SUMMARY REPORT")
        
        print(f"\nOVERALL EDUCATIONAL INTEGRATION ANALYSIS:")
        print(f"{'='*50}")
        
        for category, data in self.results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Coverage: {data['average_coverage']:.1f}%")
            print(f"  Modules analyzed: {data['total_modules']}")
            
            # Show top elements/concepts
            if 'elements' in data:
                items = data['elements']
            elif 'concepts' in data:
                items = data['concepts']
            elif 'patterns' in data:
                items = data['patterns']
            elif 'question_types' in data:
                items = data['question_types']
            else:
                items = {}
                
            if items:
                top_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top elements:")
                for item, count in top_items:
                    percentage = (count / data['total_modules']) * 100 if data['total_modules'] > 0 else 0
                    print(f"    - {item}: {count} modules ({percentage:.1f}%)")
        
        # Overall assessment
        avg_coverage = sum(data['average_coverage'] for data in self.results.values()) / len(self.results)
        
        print(f"\n{'='*50}")
        print(f"OVERALL EDUCATIONAL INTEGRATION SCORE: {avg_coverage:.1f}%")
        
        if avg_coverage >= 80:
            print("🎉 EXCELLENT: Strong educational integration across modules")
        elif avg_coverage >= 60:
            print("✅ GOOD: Solid educational integration with room for improvement")
        elif avg_coverage >= 40:
            print("⚠️  MODERATE: Inconsistent educational integration")
        else:
            print("❌ NEEDS WORK: Educational integration requires significant improvement")
            
        # Specific recommendations
        print(f"\nRECOMMENDATIONS:")
        for category, data in self.results.items():
            if data['average_coverage'] < 60:
                print(f"  - Improve {category.replace('_', ' ')} consistency across modules")
                
        return avg_coverage >= 60

if __name__ == "__main__":
    print("Starting Educational Enhancement Integration Tests...")
    
    tester = EducationalIntegrationTester()
    success = tester.test_all_educational_integration()
    
    sys.exit(0 if success else 1)
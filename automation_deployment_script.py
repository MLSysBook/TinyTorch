#!/usr/bin/env python3
"""
NBGrader Text Response Automation Script
========================================

This script automates the deployment of interactive NBGrader text response cells
across all TinyTorch modules for ML Systems Thinking sections.

Usage:
    python automation_deployment_script.py --module 02_tensor
    python automation_deployment_script.py --all
    python automation_deployment_script.py --validate
"""

import os
import re
import json
import argparse
from pathlib import Path

# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

MODULE_CONFIGS = {
    "02_tensor": [
        {
            "title": "Memory Management in Production ML",
            "context": "Your tensor implementation creates a new result for every operation, copying data each time.",
            "question": "When training large language models like GPT-4 with billions of parameters, memory management becomes critical. Analyze how your simple tensor design would impact production systems. What are the trade-offs between memory efficiency and implementation simplicity? How do you think frameworks like PyTorch handle this challenge?",
            "focus_areas": "discussing memory implications, production considerations, and framework design choices",
            "points": 10
        },
        {
            "title": "Hardware Abstraction Design", 
            "context": "Your tensor works on CPU through NumPy, providing a simple interface for mathematical operations.",
            "question": "Modern ML frameworks must run the same code on CPUs, GPUs, and specialized chips like TPUs. How does abstracting tensor operations (like your `+` operator) enable this hardware flexibility? What challenges do framework designers face when ensuring consistent behavior across different hardware platforms?",
            "focus_areas": "analyzing hardware abstraction, framework design challenges, and cross-platform consistency",
            "points": 10
        },
        {
            "title": "API Design Philosophy",
            "context": "You implemented property-style access (tensor.shape) rather than function calls (tensor.get_shape()).",
            "question": "This seemingly small design choice affects how millions of developers interact with tensors daily. How do API design decisions like this impact developer productivity when building complex neural architectures? Compare your approach to how different frameworks handle tensor interfaces.",
            "focus_areas": "analyzing API design impact, developer experience, and framework comparison",
            "points": 10
        }
    ],
    
    "03_activations": [
        {
            "title": "Computational Efficiency in Production",
            "context": "Your ReLU implementation uses simple NumPy operations: `np.maximum(0, x)`, while your Softmax requires exponentials and normalization with overflow protection.",
            "question": "In production neural networks with billions of activations computed per forward pass, every operation matters. How might the computational complexity differences between ReLU and Softmax impact training speed and memory usage in large-scale deployments? What specific optimizations do you think GPU kernels implement for these activation functions, and why has ReLU become the dominant choice in deep learning?",
            "focus_areas": "analyzing computational efficiency, GPU optimizations, and production performance implications",
            "points": 10
        },
        {
            "title": "Numerical Stability in Large Systems",
            "context": "Your Softmax implementation includes overflow protection by clipping large values, preventing `exp(x)` from causing numerical overflow.",
            "question": "In production systems training massive language models with hundreds of layers, numerical instability can cascade and destroy training. How do frameworks like PyTorch handle numerical stability for activation functions at scale? What happens when a single unstable activation propagates through a deep network, and how do production systems prevent this? Consider both forward pass stability and gradient computation implications.",
            "focus_areas": "discussing numerical stability challenges, cascading effects, and production solutions",
            "points": 10
        },
        {
            "title": "Hardware Abstraction and API Design",
            "context": "Your activation functions use callable classes (`relu(x)`) that provide a consistent interface regardless of the underlying mathematical complexity.",
            "question": "Modern ML frameworks must run the same activation code on CPUs, GPUs, TPUs, and other specialized hardware. How does your simple, consistent API design enable this hardware flexibility? What challenges do framework designers face when ensuring that `relu(x)` produces identical results whether running on a laptop CPU or a datacenter GPU cluster? Consider precision, parallelization, and hardware-specific optimizations.",
            "focus_areas": "analyzing hardware abstraction, cross-platform consistency, and framework design challenges",
            "points": 10
        }
    ],
    
    "04_layers": [
        {
            "title": "Layer Abstraction in Framework Design",
            "context": "Your Layer base class provides a common interface for all neural network components through forward() methods.",
            "question": "This abstraction pattern is fundamental to every major ML framework. How does providing a unified layer interface enable complex neural architectures to be built compositionally? What challenges arise when trying to optimize execution across heterogeneous layer types in production systems?",
            "focus_areas": "analyzing abstraction benefits, compositional design, and execution optimization",
            "points": 10
        },
        {
            "title": "Parameter Management at Scale",
            "context": "Your layers manage their own parameters and state, providing methods for accessing weights and biases.",
            "question": "In production models with billions of parameters spread across thousands of layers, parameter management becomes critical. How do frameworks handle parameter synchronization, gradient accumulation, and memory-efficient parameter updates at scale? What happens when parameter updates fail to synchronize properly in distributed training?",
            "focus_areas": "discussing distributed parameter management, synchronization challenges, and production reliability",
            "points": 10
        }
    ],
    
    "06_spatial": [
        {
            "title": "Convolution Performance Optimization",
            "context": "Your convolution implementation uses nested loops and manual sliding window operations.",
            "question": "Production computer vision models process millions of images daily with extremely tight latency requirements. How do frameworks like PyTorch optimize convolution operations for GPU acceleration? What specific techniques (like im2col, FFT convolutions, or Winograd algorithms) make convolutions fast enough for real-time applications?",
            "focus_areas": "analyzing convolution optimization techniques, GPU acceleration, and real-time performance",
            "points": 10
        },
        {
            "title": "Memory Access Patterns",
            "context": "Your convolution iterates through spatial dimensions, accessing different memory locations for each filter application.",
            "question": "Memory access patterns dramatically affect performance in GPU computing. How do the memory access patterns in convolution operations impact cache efficiency and memory bandwidth utilization? Why might frameworks reorganize data layout specifically for convolution operations, and what are the trade-offs?",
            "focus_areas": "discussing memory access optimization, cache efficiency, and data layout strategies",
            "points": 10
        }
    ],
    
    "07_attention": [
        {
            "title": "Attention Scaling Challenges",
            "context": "Your attention implementation computes all pairwise similarities, resulting in O(N²) memory and computation for sequence length N.",
            "question": "This quadratic scaling becomes prohibitive for long sequences in production language models. How do modern attention mechanisms (like Flash Attention, sparse attention, or linear attention) address these scaling challenges? What are the trade-offs between attention quality and computational efficiency?",
            "focus_areas": "analyzing attention scaling solutions, memory optimization, and quality trade-offs",
            "points": 10
        },
        {
            "title": "Multi-Head Attention Parallelization",
            "context": "Your multi-head attention processes each head independently, suggesting natural parallelization opportunities.",
            "question": "In production transformer models with dozens of attention heads, efficient parallelization is crucial. How do frameworks parallelize multi-head attention across GPU cores and multiple devices? What challenges arise when synchronizing gradients across parallel attention computations?",
            "focus_areas": "discussing parallelization strategies, GPU utilization, and gradient synchronization",
            "points": 10
        }
    ],
    
    "10_optimizers": [
        {
            "title": "Optimizer Memory Overhead",
            "context": "Your Adam optimizer maintains momentum and velocity buffers, effectively tripling memory usage compared to model parameters.",
            "question": "For models with billions of parameters, this 3× memory overhead can exceed available GPU memory. How do production systems handle optimizer state management at scale? What techniques (like gradient checkpointing, optimizer state partitioning, or mixed precision) help manage memory constraints?",
            "focus_areas": "analyzing memory optimization techniques, distributed optimizer state, and production constraints",
            "points": 10
        },
        {
            "title": "Learning Rate Scheduling",
            "context": "Your optimizers use fixed learning rates, but production training often requires complex scheduling strategies.",
            "question": "Learning rate scheduling can make the difference between successful training and divergence in large models. How do production ML systems implement adaptive learning rate strategies? What happens when learning rate schedules are poorly tuned in distributed training scenarios?",
            "focus_areas": "discussing learning rate strategies, adaptive scheduling, and distributed training challenges",
            "points": 10
        }
    ]
}

# =============================================================================
# GRADING RUBRIC TEMPLATES
# =============================================================================

STANDARD_RUBRIC_TEMPLATE = """GRADING CRITERIA ({points} points total):

EXCELLENT ({excellent_min}-{points} points):
- Demonstrates deep understanding of {topic_area}
- Makes specific connections between implementation and production ML systems
- Shows awareness of real-world optimization strategies and trade-offs
- Discusses challenges thoughtfully with concrete examples
- Writing is clear, technical, and insightful

GOOD ({good_min}-{good_max} points):
- Shows good understanding of {topic_area} 
- Makes some connections to production systems
- Discusses trade-offs but may lack depth or specificity
- Generally accurate technical understanding

SATISFACTORY ({satisfactory_min}-{satisfactory_max} points):
- Basic understanding of core concepts
- Limited connection to production systems
- General discussion without specific insights
- May contain minor technical inaccuracies

NEEDS IMPROVEMENT (1-{needs_improvement_max} points):
- Minimal understanding of {topic_area}
- Few or no connections to real systems
- Unclear or inaccurate technical content

NO CREDIT (0 points):
- No response or completely off-topic
- Factually incorrect fundamental concepts"""

def generate_rubric(points, topic_area):
    """Generate standardized grading rubric"""
    excellent_min = max(1, int(points * 0.9))
    good_min = max(1, int(points * 0.7))
    good_max = excellent_min - 1
    satisfactory_min = max(1, int(points * 0.5))
    satisfactory_max = good_min - 1
    needs_improvement_max = satisfactory_min - 1
    
    return STANDARD_RUBRIC_TEMPLATE.format(
        points=points,
        excellent_min=excellent_min,
        good_min=good_min,
        good_max=good_max,
        satisfactory_min=satisfactory_min,
        satisfactory_max=satisfactory_max,
        needs_improvement_max=needs_improvement_max,
        topic_area=topic_area
    )

# =============================================================================
# TEMPLATE GENERATION
# =============================================================================

def generate_ml_systems_section(module_name, questions_config):
    """Generate complete ML Systems Thinking section"""
    
    section_intro = f'''# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Reflection

Now that you've implemented core functionality in the {module_name} module, let's explore how this connects 
to real-world ML systems through focused questions requiring thoughtful analysis.

**Instructions:** 
- Provide thoughtful 150-300 word responses to each question
- Draw connections between your implementation and production ML systems
- Use specific examples from your code and real-world scenarios
- These responses will be manually graded for insight and understanding
"""'''

    questions_section = ""
    
    for i, question_config in enumerate(questions_config, 1):
        # Generate rubric if not provided
        if 'grading_rubric' not in question_config:
            question_config['grading_rubric'] = generate_rubric(
                question_config['points'], 
                question_config['title'].lower()
            )
        
        task_cell = f'''
# %% [markdown] nbgrader={{"grade": false, "grade_id": "systems-thinking-task-{i}", "locked": true, "schema_version": 3, "solution": false, "task": true}}
"""
### Question {i}: {question_config['title']}

**Context:** {question_config['context']}

**Question:** {question_config['question']}

**Expected Response:** 150-300 words {question_config['focus_areas']}.
"""'''

        response_cell = f'''
# %% [markdown] nbgrader={{"grade": true, "grade_id": "systems-thinking-response-{i}", "locked": false, "schema_version": 3, "solution": true, "task": false, "points": {question_config['points']}}}
"""
=== BEGIN MARK SCHEME ===
{question_config['grading_rubric']}
=== END MARK SCHEME ===

**Your Response:**
[Student writes their analysis here - this cell will be editable by students]
"""'''
        
        questions_section += task_cell + response_cell
    
    conclusion = '''
# %% [markdown]
"""
**💡 Systems Insight**: The functionality you've implemented represents fundamental building blocks that must work reliably across every major computing platform powering modern AI. Your implementations demonstrate the elegant abstractions that hide incredible complexity in optimization, hardware acceleration, and distributed computing underneath simple, mathematical interfaces.
"""'''
    
    return section_intro + questions_section + conclusion

# =============================================================================
# FILE PROCESSING
# =============================================================================

def find_ml_systems_section(content):
    """Find the ML Systems Thinking section in module content"""
    patterns = [
        r"## 🤔 ML Systems Thinking.*?(?=# MODULE SUMMARY|$)",
        r"## ML Systems Thinking.*?(?=# MODULE SUMMARY|$)",
        r"ML Systems Thinking.*?(?=# MODULE SUMMARY|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.start(), match.end()
    
    return None, None

def extract_module_name(module_path):
    """Extract clean module name from path"""
    folder_name = Path(module_path).parent.name
    if '_' in folder_name:
        return folder_name.split('_', 1)[1].replace('_', ' ').title()
    return folder_name.title()

def deploy_to_module(module_path, module_config):
    """Deploy interactive questions to a specific module"""
    
    if not os.path.exists(module_path):
        print(f"❌ Module not found: {module_path}")
        return False
    
    # Read existing content
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {module_path}: {e}")
        return False
    
    # Find ML Systems section
    start_pos, end_pos = find_ml_systems_section(content)
    
    if start_pos is None:
        print(f"❌ No ML Systems Thinking section found in {module_path}")
        return False
    
    # Extract module name
    module_name = extract_module_name(module_path)
    
    # Generate new section
    new_section = generate_ml_systems_section(module_name, module_config)
    
    # Replace the section
    new_content = content[:start_pos] + new_section + content[end_pos:]
    
    # Write back
    try:
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Successfully updated {module_path} with interactive questions")
        return True
    except Exception as e:
        print(f"❌ Error writing {module_path}: {e}")
        return False

# =============================================================================
# VALIDATION
# =============================================================================

def validate_nbgrader_metadata(module_path):
    """Validate NBGrader metadata in module file"""
    
    with open(module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all NBGrader cells
    nbgrader_pattern = r'nbgrader=\{[^}]+\}'
    matches = re.findall(nbgrader_pattern, content)
    
    issues = []
    grade_ids = []
    
    for match in matches:
        try:
            # Parse the metadata (simplified)
            if '"grade_id"' in match:
                grade_id_match = re.search(r'"grade_id":\s*"([^"]+)"', match)
                if grade_id_match:
                    grade_id = grade_id_match.group(1)
                    if grade_id in grade_ids:
                        issues.append(f"Duplicate grade_id: {grade_id}")
                    grade_ids.append(grade_id)
                    
            # Check required fields for graded cells
            if '"grade": true' in match:
                required_fields = ['grade_id', 'points', 'solution']
                for field in required_fields:
                    if f'"{field}"' not in match:
                        issues.append(f"Missing {field} in graded cell")
                        
        except Exception as e:
            issues.append(f"Error parsing metadata: {e}")
    
    return issues

def validate_all_modules():
    """Validate NBGrader metadata across all modules"""
    
    modules_dir = Path("modules/source")
    issues_by_module = {}
    
    for module_dir in modules_dir.iterdir():
        if module_dir.is_dir() and not module_dir.name.startswith('.'):
            module_file = module_dir / f"{module_dir.name}_dev.py"
            if module_file.exists():
                issues = validate_nbgrader_metadata(module_file)
                if issues:
                    issues_by_module[module_dir.name] = issues
    
    return issues_by_module

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deploy NBGrader text response cells to TinyTorch modules')
    parser.add_argument('--module', help='Specific module to update (e.g., 02_tensor)')
    parser.add_argument('--all', action='store_true', help='Update all configured modules')
    parser.add_argument('--validate', action='store_true', help='Validate NBGrader metadata across modules')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if args.validate:
        print("🔍 Validating NBGrader metadata across all modules...")
        issues = validate_all_modules()
        if issues:
            print("❌ Validation issues found:")
            for module, module_issues in issues.items():
                print(f"\n{module}:")
                for issue in module_issues:
                    print(f"  - {issue}")
        else:
            print("✅ All modules pass validation")
        return
    
    modules_to_process = []
    
    if args.module:
        if args.module in MODULE_CONFIGS:
            modules_to_process = [args.module]
        else:
            print(f"❌ No configuration found for module: {args.module}")
            print(f"Available modules: {', '.join(MODULE_CONFIGS.keys())}")
            return
    elif args.all:
        modules_to_process = list(MODULE_CONFIGS.keys())
    else:
        print("❌ Must specify --module, --all, or --validate")
        parser.print_help()
        return
    
    success_count = 0
    
    for module in modules_to_process:
        module_path = f"modules/source/{module}/{module.split('_', 1)[1]}_dev.py"
        
        if args.dry_run:
            print(f"🔍 Would update: {module_path}")
            continue
        
        print(f"🔄 Processing {module}...")
        
        if deploy_to_module(module_path, MODULE_CONFIGS[module]):
            success_count += 1
        else:
            print(f"❌ Failed to update {module}")
    
    if not args.dry_run:
        print(f"\n🎉 Successfully updated {success_count}/{len(modules_to_process)} modules")
    
    print("\n📋 Next steps:")
    print("1. Review generated questions for clarity and appropriateness")
    print("2. Test NBGrader generation: ./bin/tito nbgrader generate MODULE_NAME")
    print("3. Validate student experience with generated notebooks")
    print("4. Train graders on rubrics for consistent assessment")

if __name__ == "__main__":
    main()
# NBGrader Text Response Implementation Pattern for TinyTorch
# Technical implementation for interactive ML Systems Thinking questions
# Module Developer: Complete implementation pattern for deployment

"""
NBGRADER TEXT RESPONSE CELLS - TECHNICAL IMPLEMENTATION PATTERN
================================================================

This file provides the complete technical pattern for implementing NBGrader 
text response cells in TinyTorch modules for ML Systems Thinking sections.

EDUCATION ARCHITECT RECOMMENDATION IMPLEMENTED:
- 3-4 interactive questions per module (reduced from 10-15 passive questions)
- NBGrader text response cells with manual grading
- Focus on System Design, Production Integration, Performance Analysis
- 150-300 word responses per question
"""

# =============================================================================
# 1. CELL STRUCTURE PATTERN - Copy this for each text response question
# =============================================================================

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Reflection

Now that you've implemented [SPECIFIC MODULE FEATURE], let's explore how this connects 
to real-world ML systems through focused questions requiring thoughtful analysis.

**Instructions:** 
- Provide thoughtful 150-300 word responses to each question
- Draw connections between your implementation and production ML systems
- Use specific examples from your code and real-world scenarios
- These responses will be manually graded for insight and understanding
"""

# %% [markdown] nbgrader={"grade": false, "grade_id": "systems-thinking-task-1", "locked": true, "schema_version": 3, "solution": false, "task": true}
"""
### Question 1: System Design Analysis

**Context:** Your tensor implementation creates a new result for every operation, copying data each time.

**Question:** When training large language models like GPT-4 with billions of parameters, memory management becomes critical. Analyze how your simple tensor design would impact production systems. What are the trade-offs between memory efficiency and implementation simplicity? How do you think frameworks like PyTorch handle this challenge?

**Expected Response:** 150-300 words discussing memory implications, production considerations, and framework design choices.
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-response-1", "locked": false, "schema_version": 3, "solution": true, "task": false, "points": 10}
"""
=== BEGIN MARK SCHEME ===
GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Demonstrates deep understanding of memory implications in production ML
- Makes specific connections between simple tensor design and real-world challenges
- Shows awareness of framework optimization strategies (in-place operations, memory pooling, etc.)
- Discusses trade-offs thoughtfully with concrete examples
- Writing is clear, technical, and insightful

GOOD (7-8 points):
- Shows good understanding of memory challenges in large-scale ML
- Makes some connections to production systems
- Discusses trade-offs but may lack depth or specificity
- Generally accurate technical understanding

SATISFACTORY (5-6 points):
- Basic understanding of memory issues
- Limited connection to production systems
- General discussion without specific insights
- May contain minor technical inaccuracies

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of memory implications
- Few or no connections to real systems
- Unclear or inaccurate technical content

NO CREDIT (0 points):
- No response or completely off-topic
- Factually incorrect fundamental concepts
=== END MARK SCHEME ===

**Your Response:**
[Student writes their analysis here - this cell will be editable by students]
"""

# =============================================================================
# 2. COMPLETE TEMPLATE FOR ML SYSTEMS THINKING SECTION
# =============================================================================

def generate_ml_systems_thinking_section(module_name, module_feature, questions_config):
    """
    Generate complete ML Systems Thinking section with NBGrader text responses
    
    Args:
        module_name: str - Name of the TinyTorch module (e.g., "Tensor", "Activations")
        module_feature: str - Key feature implemented (e.g., "tensor operations", "ReLU function")
        questions_config: list - Configuration for each question
    
    Returns:
        str - Complete section with NBGrader metadata
    """
    
    section_intro = f'''# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Reflection

Now that you've implemented {module_feature} in the {module_name} module, let's explore how this connects 
to real-world ML systems through focused questions requiring thoughtful analysis.

**Instructions:** 
- Provide thoughtful 150-300 word responses to each question
- Draw connections between your implementation and production ML systems
- Use specific examples from your code and real-world scenarios
- These responses will be manually graded for insight and understanding
"""'''

    questions_section = ""
    
    for i, question_config in enumerate(questions_config, 1):
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
    
    return section_intro + questions_section

# =============================================================================
# 3. EXAMPLE CONFIGURATIONS FOR DIFFERENT MODULES
# =============================================================================

# Example 1: Tensor Module Configuration
TENSOR_MODULE_CONFIG = [
    {
        "title": "Memory Management in Production ML",
        "context": "Your tensor implementation creates a new result for every operation, copying data each time.",
        "question": "When training large language models like GPT-4 with billions of parameters, memory management becomes critical. Analyze how your simple tensor design would impact production systems. What are the trade-offs between memory efficiency and implementation simplicity? How do you think frameworks like PyTorch handle this challenge?",
        "focus_areas": "discussing memory implications, production considerations, and framework design choices",
        "points": 10,
        "grading_rubric": """GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Demonstrates deep understanding of memory implications in production ML
- Makes specific connections between simple tensor design and real-world challenges
- Shows awareness of framework optimization strategies (in-place operations, memory pooling, etc.)
- Discusses trade-offs thoughtfully with concrete examples
- Writing is clear, technical, and insightful

GOOD (7-8 points):
- Shows good understanding of memory challenges in large-scale ML
- Makes some connections to production systems
- Discusses trade-offs but may lack depth or specificity
- Generally accurate technical understanding

SATISFACTORY (5-6 points):
- Basic understanding of memory issues
- Limited connection to production systems
- General discussion without specific insights
- May contain minor technical inaccuracies

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of memory implications
- Few or no connections to real systems
- Unclear or inaccurate technical content

NO CREDIT (0 points):
- No response or completely off-topic
- Factually incorrect fundamental concepts"""
    },
    {
        "title": "Hardware Abstraction Design",
        "context": "Your tensor works on CPU through NumPy, providing a simple interface for mathematical operations.",
        "question": "Modern ML frameworks must run the same code on CPUs, GPUs, and specialized chips like TPUs. How does abstracting tensor operations (like your `+` operator) enable this hardware flexibility? What challenges do framework designers face when ensuring consistent behavior across different hardware platforms?",
        "focus_areas": "analyzing hardware abstraction, framework design challenges, and cross-platform consistency",
        "points": 10,
        "grading_rubric": """GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Demonstrates understanding of hardware abstraction principles
- Discusses specific challenges in cross-platform ML frameworks
- Shows awareness of performance implications across different hardware
- Makes connections to real framework design decisions
- Clear technical communication

GOOD (7-8 points):
- Good understanding of hardware abstraction concepts
- Some awareness of cross-platform challenges
- Generally accurate technical content
- Makes some connections to real systems

SATISFACTORY (5-6 points):
- Basic understanding of hardware differences
- Limited insight into framework design challenges
- General discussion without specific examples

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of hardware abstraction
- Unclear or inaccurate technical content

NO CREDIT (0 points):
- No response or completely off-topic"""
    },
    {
        "title": "API Design Philosophy",
        "context": "You implemented property-style access (tensor.shape) rather than function calls (tensor.get_shape()).",
        "question": "This seemingly small design choice affects how millions of developers interact with tensors daily. How do API design decisions like this impact developer productivity when building complex neural architectures? Compare your approach to how different frameworks handle tensor interfaces.",
        "focus_areas": "analyzing API design impact, developer experience, and framework comparison",
        "points": 10,
        "grading_rubric": """GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Insightful analysis of API design impact on developer experience
- Thoughtful comparison of different framework approaches
- Understanding of how small design choices scale to large codebases
- Clear examples of developer productivity implications

GOOD (7-8 points):
- Good understanding of API design principles
- Some comparison of framework approaches
- Generally accurate insights about developer experience

SATISFACTORY (5-6 points):
- Basic understanding of API design concepts
- Limited framework comparison
- General discussion without deep insights

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of API design impact
- Unclear analysis

NO CREDIT (0 points):
- No response or off-topic"""
    }
]

# Example 2: Activations Module Configuration
ACTIVATIONS_MODULE_CONFIG = [
    {
        "title": "Computational Efficiency in Production",
        "context": "Your ReLU implementation uses simple NumPy operations: np.maximum(0, x).",
        "question": "In production neural networks with billions of activations computed per forward pass, every operation matters. How might the simplicity of ReLU (compared to sigmoid or tanh) impact training speed and memory usage in large-scale deployments? What optimizations do you think GPU kernels implement for activation functions?",
        "focus_areas": "analyzing computational efficiency, GPU optimizations, and production performance",
        "points": 10,
        "grading_rubric": """GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Deep understanding of computational complexity in neural networks
- Insightful analysis of ReLU's efficiency advantages
- Shows awareness of GPU kernel optimizations
- Makes connections to real-world performance implications

GOOD (7-8 points):
- Good understanding of activation function efficiency
- Some awareness of GPU optimizations
- Generally accurate technical content

SATISFACTORY (5-6 points):
- Basic understanding of computational differences
- Limited insight into production optimizations

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of efficiency implications

NO CREDIT (0 points):
- No response or inaccurate content"""
    }
]

# =============================================================================
# 4. AUTOMATION SCRIPT FOR DEPLOYMENT
# =============================================================================

def deploy_interactive_questions_to_module(module_path, module_config):
    """
    Automatically add interactive NBGrader questions to a TinyTorch module
    
    Args:
        module_path: str - Path to the module_dev.py file
        module_config: list - Question configuration for the module
    """
    
    # Read existing module content
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Find the ML Systems Thinking section
    ml_systems_start = content.find("## 🤔 ML Systems Thinking")
    if ml_systems_start == -1:
        print(f"No ML Systems Thinking section found in {module_path}")
        return False
    
    # Find the section end (next major section or end of file)
    section_end = content.find("# MODULE SUMMARY", ml_systems_start)
    if section_end == -1:
        section_end = len(content)
    
    # Extract module name and feature from existing content
    module_name = extract_module_name(module_path)
    module_feature = extract_module_feature(content)
    
    # Generate new interactive section
    new_section = generate_ml_systems_thinking_section(
        module_name, module_feature, module_config
    )
    
    # Replace the section
    new_content = (
        content[:ml_systems_start] + 
        new_section + 
        content[section_end:]
    )
    
    # Write back to file
    with open(module_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Successfully updated {module_path} with interactive questions")
    return True

def extract_module_name(module_path):
    """Extract module name from file path"""
    import os
    folder_name = os.path.basename(os.path.dirname(module_path))
    return folder_name.split('_', 1)[1].title() if '_' in folder_name else folder_name

def extract_module_feature(content):
    """Extract key feature from module content"""
    # Simple heuristic - look for main class or function names
    lines = content.split('\n')
    for line in lines:
        if line.strip().startswith('class ') and 'Test' not in line:
            class_name = line.split('class ')[1].split('(')[0].split(':')[0]
            return f"{class_name} implementation"
    return "core functionality"

# =============================================================================
# 5. TECHNICAL LIMITATIONS AND CONSIDERATIONS
# =============================================================================

"""
TECHNICAL LIMITATIONS AND CONSIDERATIONS:

1. MANUAL GRADING REQUIREMENT:
   - Text responses cannot be auto-graded
   - Requires instructor time for meaningful assessment
   - Rubrics help standardize grading but still need human judgment

2. NBGRADER METADATA FRAGILITY:
   - Metadata must be precisely formatted
   - Cell IDs must be unique across entire assignment
   - Schema version compatibility required

3. CELL EXECUTION ORDER:
   - Text response cells don't execute code
   - No validation of student understanding beyond written response
   - Cannot test implementation knowledge directly

4. JUPYTEXT COMPATIBILITY:
   - NBGrader metadata must be preserved in .py files
   - Conversion between formats may affect metadata
   - Cell structure must be maintained

5. SCALING CONSIDERATIONS:
   - Manual grading doesn't scale to large class sizes
   - Requires multiple graders for consistency
   - Feedback generation is time-intensive

6. STUDENT EXPERIENCE:
   - Students need clear instructions about response expectations
   - Word count guidelines help scope responses
   - Grading rubrics should be transparent

RECOMMENDED IMPLEMENTATION STRATEGY:
1. Start with 2-3 modules as pilot test
2. Develop standardized rubrics
3. Train multiple graders for consistency
4. Collect student feedback on question clarity
5. Iterate based on grading time and student outcomes
"""

# =============================================================================
# 6. DEPLOYMENT CHECKLIST
# =============================================================================

"""
PRE-DEPLOYMENT CHECKLIST:

□ Verify NBGrader metadata format compliance
□ Ensure unique grade_id for each cell across all modules
□ Test cell execution in NBGrader environment
□ Validate mark scheme syntax (=== BEGIN/END ===)
□ Confirm point values align with course grading scheme
□ Review question clarity and scope
□ Test student cell editability
□ Verify instructor rubric visibility
□ Check integration with existing module structure
□ Validate jupytext conversion compatibility

POST-DEPLOYMENT TESTING:

□ Generate assignment with tito nbgrader generate
□ Verify student version removes mark schemes
□ Test manual grading workflow
□ Confirm feedback generation works
□ Validate gradebook integration
□ Check cell locking behavior
□ Test response submission workflow
"""

if __name__ == "__main__":
    print("NBGrader Text Response Implementation Pattern")
    print("=" * 50)
    print("This file provides the complete technical implementation")
    print("for adding interactive ML Systems Thinking questions")
    print("to TinyTorch modules using NBGrader text response cells.")
    print("\nKey Components:")
    print("1. Cell structure pattern with proper NBGrader metadata")
    print("2. Complete template generator function")
    print("3. Example configurations for different modules")
    print("4. Automation script for deployment")
    print("5. Technical limitations and considerations")
    print("6. Comprehensive deployment checklist")
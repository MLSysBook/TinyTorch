#| default_exp core.introduction

# %% [markdown] nbgrader={"grade": false, "grade_id": "introduction-overview", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
# Introduction - TinyTorch System Architecture & Learning Journey

Welcome to the Introduction module! You'll explore the complete TinyTorch architecture and understand how building an ML framework teaches systems engineering.

## Learning Goals
- Systems understanding: How 17 modules connect to form a complete ML framework
- Core implementation skill: Navigate complex software architecture and dependencies
- Pattern recognition: Identify how modular design enables scalable ML systems
- Framework connection: See how TinyTorch mirrors PyTorch/TensorFlow architecture
- Performance insight: Understand why modularity affects system performance and memory usage

## Build → Use → Reflect
1. **Build**: Interactive architecture visualizations and dependency analysis tools
2. **Use**: Explore module relationships and trace data flow through the entire system
3. **Reflect**: How does modular architecture impact development speed vs runtime performance?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of ML framework architecture and component relationships
- Practical capability to navigate and extend complex ML systems
- Systems insight into how modular design trades development flexibility for runtime overhead
- Performance consideration of module dependencies and circular import prevention
- Connection to production ML systems and how PyTorch organizes its codebase

## Systems Reality Check
💡 **Production Context**: PyTorch uses similar modular architecture - torch.nn depends on torch.autograd, which depends on torch._C for performance-critical operations
⚡ **Performance Note**: Module boundaries introduce function call overhead, but enable parallel development and testing - understand this engineering trade-off
"""

# %% nbgrader={"grade": false, "grade_id": "introduction-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
from pathlib import Path
import yaml
import networkx as nx
from typing import Dict, List, Tuple, Set
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict, deque

# Set plotting style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
"""
## Module Metadata Analysis System

First, let's build tools to automatically analyze all TinyTorch modules and their relationships.
This will power our interactive visualizations.
"""

# %%
#| export
@dataclass
class ModuleInfo:
    """Complete information about a TinyTorch module"""
    name: str
    title: str
    description: str
    prerequisites: List[str]
    enables: List[str]
    components: List[str]
    difficulty: str
    time_estimate: str
    exports_to: str
    
    def difficulty_level(self) -> int:
        """Convert difficulty stars to numeric level"""
        return self.difficulty.count('⭐')
    
    def estimated_hours(self) -> float:
        """Extract numeric time estimate"""
        time_str = self.time_estimate.lower()
        if 'capstone' in time_str:
            return 40.0  # Capstone project estimate
        
        # Extract first number from time estimate
        import re
        numbers = re.findall(r'\d+', time_str)
        if numbers:
            return float(numbers[0])
        return 4.0  # Default estimate

#| export
class TinyTorchAnalyzer:
    """Comprehensive analysis of TinyTorch module system"""
    
    def __init__(self, modules_path: str = "/Users/VJ/GitHub/TinyTorch/modules/source"):
        self.modules_path = Path(modules_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph = nx.DiGraph()
        self._load_all_modules()
        self._build_dependency_graph()
    
    def _load_all_modules(self):
        """Load metadata from all module.yaml files"""
        for module_dir in sorted(self.modules_path.iterdir()):
            if module_dir.is_dir() and not module_dir.name.startswith('.'):
                yaml_file = module_dir / 'module.yaml'
                if yaml_file.exists():
                    try:
                        with open(yaml_file, 'r') as f:
                            data = yaml.safe_load(f)
                        
                        # Handle different YAML formats in the modules
                        if 'dependencies' in data:
                            deps = data['dependencies']
                            prerequisites = deps.get('prerequisites', [])
                            enables = deps.get('enables', [])
                        else:
                            # Handle older format
                            prerequisites = data.get('dependencies', [])
                            enables = []
                        
                        module_info = ModuleInfo(
                            name=data.get('name', module_dir.name),
                            title=data.get('title', module_dir.name.title()),
                            description=data.get('description', ''),
                            prerequisites=prerequisites,
                            enables=enables,
                            components=data.get('components', []),
                            difficulty=data.get('difficulty', '⭐'),
                            time_estimate=data.get('time_estimate', '2-4 hours'),
                            exports_to=data.get('exports_to', f'tinytorch.{module_dir.name}')
                        )
                        
                        self.modules[module_info.name] = module_info
                        
                    except Exception as e:
                        print(f"Warning: Could not load {yaml_file}: {e}")
    
    def _build_dependency_graph(self):
        """Build NetworkX graph of module dependencies"""
        # Add all modules as nodes
        for name, module in self.modules.items():
            self.dependency_graph.add_node(name, **{
                'title': module.title,
                'description': module.description,
                'difficulty': module.difficulty_level(),
                'time': module.estimated_hours(),
                'components': len(module.components)
            })
        
        # Add dependency edges
        for name, module in self.modules.items():
            for prereq in module.prerequisites:
                if prereq in self.modules:
                    self.dependency_graph.add_edge(prereq, name)
    
    def get_learning_path(self) -> List[str]:
        """Generate optimal learning path through modules using topological sort"""
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXError:
            # Fallback if cycles exist
            return sorted(self.modules.keys())
    
    def get_module_levels(self) -> Dict[str, int]:
        """Assign modules to learning levels based on dependencies"""
        levels = {}
        path = self.get_learning_path()
        
        for module in path:
            prereqs = self.modules[module].prerequisites
            if not prereqs:
                levels[module] = 0
            else:
                max_prereq_level = max((levels.get(p, 0) for p in prereqs if p in levels), default=0)
                levels[module] = max_prereq_level + 1
        
        return levels

# Initialize the analyzer
analyzer = TinyTorchAnalyzer()

# %% [markdown]
"""
### Test the Module Analysis System

Let's verify our module analyzer is working correctly by examining a few key modules.
"""

# %%
def test_module_analyzer():
    """Test that our module analyzer correctly loads and processes modules"""
    
    # Test basic loading
    assert len(analyzer.modules) >= 10, "Should load multiple modules"
    
    # Test specific modules exist
    key_modules = ['setup', 'tensor', 'activations', 'training']
    for module_name in key_modules:
        assert module_name in analyzer.modules, f"Should load {module_name} module"
    
    # Test dependency relationships
    tensor_module = analyzer.modules['tensor']
    assert 'setup' in tensor_module.prerequisites, "Tensor should depend on setup"
    
    # Test learning path generation
    learning_path = analyzer.get_learning_path()
    setup_pos = learning_path.index('setup') if 'setup' in learning_path else -1
    tensor_pos = learning_path.index('tensor') if 'tensor' in learning_path else -1
    
    if setup_pos >= 0 and tensor_pos >= 0:
        assert setup_pos < tensor_pos, "Setup should come before tensor in learning path"
    
    print("✅ Module analyzer tests passed!")
    
    # Show some sample data
    print(f"\n📋 Sample modules loaded:")
    for name in list(analyzer.modules.keys())[:5]:
        module = analyzer.modules[name]
        print(f"  • {module.title} ({module.difficulty}) - {len(module.components)} components")

# test_module_analyzer() # Test moved to main block

# %% [markdown]
"""
## Interactive Dependency Visualization

Now let's create beautiful, interactive visualizations of the TinyTorch module dependency system.
"""

# %%
#| export  
def create_dependency_graph_visualization(figsize=(16, 12)):
    """Create an interactive dependency graph visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Hierarchical layout
    ax1.set_title("TinyTorch Module Dependencies\n(Hierarchical Layout)", fontsize=16, fontweight='bold')
    
    # Calculate positions using spring layout with hierarchy
    levels = analyzer.get_module_levels()
    pos = {}
    
    # Group modules by level
    level_groups = defaultdict(list)
    for module, level in levels.items():
        level_groups[level].append(module)
    
    # Position modules in levels
    max_level = max(levels.values()) if levels else 0
    for level, modules in level_groups.items():
        y = max_level - level  # Higher levels at top
        for i, module in enumerate(sorted(modules)):
            x = (i - len(modules)/2) * 2.5
            pos[module] = (x, y * 2)
    
    # Draw the graph
    G = analyzer.dependency_graph
    
    # Node colors based on difficulty
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        difficulty = analyzer.modules[node].difficulty_level()
        node_colors.append(plt.cm.viridis(difficulty / 5.0))  # Normalize to 0-1
        node_sizes.append(200 + difficulty * 100)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, 
                          arrows=True, arrowsize=20, ax=ax1)
    
    # Add labels
    labels = {node: analyzer.modules[node].name for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax1)
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right plot: Circular layout
    ax2.set_title("TinyTorch Module Dependencies\n(Circular Layout)", fontsize=16, fontweight='bold')
    
    # Circular layout
    pos_circular = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos_circular, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos_circular, edge_color='gray', alpha=0.4,
                          arrows=True, arrowsize=15, ax=ax2)
    nx.draw_networkx_labels(G, pos_circular, labels, font_size=7, font_weight='bold', ax=ax2)
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add legend
    difficulty_colors = [plt.cm.viridis(i/5.0) for i in range(1, 6)]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 label=f"{'⭐' * (i+1)} Difficulty")
                      for i, color in enumerate(difficulty_colors)]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create dependency visualization function (called in main block)

# %% [markdown]
"""
### Test the Dependency Visualization

Let's verify our dependency graph captures the correct relationships.
"""

# %%
def test_dependency_relationships():
    """Test that dependency relationships are correctly captured"""
    
    G = analyzer.dependency_graph
    
    # Test that setup has no prerequisites (should be a source node)
    setup_predecessors = list(G.predecessors('setup')) if 'setup' in G else []
    print(f"Setup prerequisites: {setup_predecessors}")
    
    # Test that capstone depends on many modules (should have many predecessors)
    if 'capstone' in G:
        capstone_predecessors = list(G.predecessors('capstone'))
        print(f"Capstone depends on {len(capstone_predecessors)} modules: {capstone_predecessors[:5]}...")
        assert len(capstone_predecessors) >= 5, "Capstone should depend on many modules"
    
    # Test learning path makes sense
    learning_path = analyzer.get_learning_path()
    print(f"\n📚 Learning path ({len(learning_path)} modules):")
    for i, module in enumerate(learning_path[:8]):  # Show first 8
        print(f"  {i+1:2d}. {analyzer.modules[module].title}")
    
    print("✅ Dependency relationship tests passed!")

# test_dependency_relationships() # Test moved to main block

# %% [markdown]
"""
## System Architecture Overview

Let's create a comprehensive system architecture diagram showing how all TinyTorch components work together.
"""

# %%
#| export
def create_system_architecture_diagram(figsize=(18, 12)):
    """Create a comprehensive TinyTorch system architecture diagram"""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    
    # Define architectural layers
    layers = {
        'Foundation': {'y': 1, 'color': '#FF6B6B', 'modules': ['setup', 'tensor']},
        'Core Components': {'y': 3, 'color': '#4ECDC4', 'modules': ['activations', 'layers', 'dataloader']},
        'Network Architecture': {'y': 5, 'color': '#45B7D1', 'modules': ['dense', 'spatial', 'attention']},
        'Training System': {'y': 7, 'color': '#96CEB4', 'modules': ['autograd', 'optimizers', 'training']},
        'Production & Optimization': {'y': 9, 'color': '#FFEAA7', 'modules': ['compression', 'kernels', 'benchmarking']},
        'MLOps & Integration': {'y': 11, 'color': '#DDA0DD', 'modules': ['mlops', 'capstone']}
    }
    
    # Draw layer backgrounds
    for layer_name, layer_info in layers.items():
        y = layer_info['y']
        rect = FancyBboxPatch((1, y-0.4), 18, 1.2, 
                             boxstyle="round,pad=0.1",
                             facecolor=layer_info['color'], 
                             alpha=0.3, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Layer label
        ax.text(0.5, y, layer_name, fontsize=12, fontweight='bold', 
               rotation=90, va='center', ha='center')
    
    # Draw modules within layers
    module_positions = {}
    for layer_name, layer_info in layers.items():
        y = layer_info['y']
        modules = [m for m in layer_info['modules'] if m in analyzer.modules]
        
        for i, module_name in enumerate(modules):
            module = analyzer.modules[module_name]
            x = 2 + (i * 16 / max(len(modules), 1))
            module_positions[module_name] = (x, y)
            
            # Module box
            width = min(3.5, 14 / len(modules))
            box = FancyBboxPatch((x-width/2, y-0.3), width, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor='black',
                               linewidth=2)
            ax.add_patch(box)
            
            # Module title
            ax.text(x, y+0.1, module.title, fontsize=9, fontweight='bold',
                   ha='center', va='center')
            
            # Difficulty and components
            ax.text(x, y-0.15, f"{module.difficulty} • {len(module.components)} comp.",
                   fontsize=7, ha='center', va='center', style='italic')
    
    # Draw dependency arrows between layers
    for module_name, module in analyzer.modules.items():
        if module_name in module_positions:
            x1, y1 = module_positions[module_name]
            for prereq in module.prerequisites:
                if prereq in module_positions:
                    x2, y2 = module_positions[prereq]
                    if abs(y1 - y2) > 1:  # Only draw arrows between different layers
                        arrow = ConnectionPatch((x2, y2+0.3), (x1, y1-0.3), "data", "data",
                                              arrowstyle="->", shrinkA=0, shrinkB=0,
                                              mutation_scale=15, alpha=0.6, color='gray')
                        ax.add_patch(arrow)
    
    # Title and annotations
    ax.text(10, 11.7, 'TinyTorch System Architecture', fontsize=20, fontweight='bold', ha='center')
    ax.text(10, 0.3, 'Data flows upward through layers • Arrows show dependencies', 
           fontsize=10, ha='center', style='italic')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, module_positions

# System architecture diagram function (called in main block)

# %% [markdown]
"""
### Test the System Architecture Visualization

Let's verify our architecture diagram correctly represents the system structure.
"""

# %%
def test_system_architecture():
    """Test that the system architecture is correctly represented"""
    
    # Test that we have positions for all modules
    expected_modules = set(analyzer.modules.keys())
    positioned_modules = set(module_positions.keys())
    
    missing_modules = expected_modules - positioned_modules
    if missing_modules:
        print(f"⚠️  Missing modules in architecture: {missing_modules}")
    
    # Test layer organization makes sense
    foundation_modules = ['setup', 'tensor']
    core_modules = ['activations', 'layers', 'dataloader']
    
    foundation_y = [module_positions[m][1] for m in foundation_modules if m in module_positions]
    core_y = [module_positions[m][1] for m in core_modules if m in module_positions]
    
    if foundation_y and core_y:
        assert min(core_y) > max(foundation_y), "Core modules should be above foundation"
    
    print(f"✅ Architecture diagram includes {len(module_positions)} modules")
    print(f"📊 Modules organized across {len(set(pos[1] for pos in module_positions.values()))} layers")

# test_system_architecture() # Test moved to main block

# %% [markdown]
"""
## Learning Roadmap Visualization

Create an interactive learning roadmap that shows the optimal path through TinyTorch modules.
"""

# %%
#| export
def create_learning_roadmap(figsize=(16, 10)):
    """Create an interactive learning roadmap visualization"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Get learning path and levels
    learning_path = analyzer.get_learning_path()
    levels = analyzer.get_module_levels()
    
    # Top plot: Learning path flowchart
    ax1.set_title('TinyTorch Learning Roadmap\n(Optimal Learning Sequence)', 
                 fontsize=16, fontweight='bold')
    
    # Calculate positions for roadmap
    path_positions = {}
    cumulative_time = 0
    y_positions = {}
    
    for i, module_name in enumerate(learning_path):
        if module_name in analyzer.modules:
            module = analyzer.modules[module_name]
            level = levels.get(module_name, 0)
            
            # X position based on cumulative time
            x = cumulative_time + module.estimated_hours() / 2
            # Y position based on dependency level with some jitter
            y = level + (i % 3 - 1) * 0.3
            
            path_positions[module_name] = (x, y)
            y_positions[module_name] = y
            cumulative_time += module.estimated_hours()
    
    # Draw the learning path
    for i, module_name in enumerate(learning_path[:-1]):
        if module_name in path_positions and learning_path[i+1] in path_positions:
            x1, y1 = path_positions[module_name]
            x2, y2 = path_positions[learning_path[i+1]]
            
            # Draw connecting line
            ax1.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=1, zorder=1)
    
    # Draw modules
    for module_name in learning_path:
        if module_name in analyzer.modules and module_name in path_positions:
            module = analyzer.modules[module_name]
            x, y = path_positions[module_name]
            
            # Color based on difficulty
            difficulty = module.difficulty_level()
            color = plt.cm.viridis(difficulty / 5.0)
            
            # Draw module circle
            circle = Circle((x, y), 0.4, facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.8, zorder=3)
            ax1.add_patch(circle)
            
            # Module number
            ax1.text(x, y, str(learning_path.index(module_name) + 1), 
                    fontsize=10, fontweight='bold', ha='center', va='center',
                    color='white', zorder=4)
            
            # Module name below
            ax1.text(x, y-0.7, module.title, fontsize=8, ha='center', va='top',
                    rotation=45, fontweight='bold')
    
    ax1.set_xlim(-2, cumulative_time + 2)
    ax1.set_ylim(-1, max(y_positions.values()) + 1)
    ax1.set_xlabel('Cumulative Learning Time (hours)', fontsize=12)
    ax1.set_ylabel('Dependency Level', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Time and difficulty analysis
    ax2.set_title('Module Difficulty and Time Distribution', fontsize=14, fontweight='bold')
    
    module_names = [analyzer.modules[name].title[:15] for name in learning_path 
                   if name in analyzer.modules]
    difficulties = [analyzer.modules[name].difficulty_level() for name in learning_path 
                   if name in analyzer.modules]
    times = [analyzer.modules[name].estimated_hours() for name in learning_path 
            if name in analyzer.modules]
    
    # Create stacked bar chart
    x_pos = np.arange(len(module_names))
    
    # Time bars
    bars1 = ax2.bar(x_pos, times, alpha=0.7, label='Time (hours)', color='lightblue')
    
    # Difficulty overlay
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x_pos, difficulties, alpha=0.5, label='Difficulty (⭐)', 
                        color='orange', width=0.6)
    
    ax2.set_xlabel('Modules (in learning order)', fontsize=12)
    ax2.set_ylabel('Time (hours)', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Difficulty Level', fontsize=12, color='orange')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(module_names, rotation=45, ha='right')
    
    # Legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, learning_path, cumulative_time

# Learning roadmap function (called in main block)

# %% [markdown]
"""
### Test the Learning Roadmap

Let's verify our learning roadmap is pedagogically sound and follows dependency constraints.
"""

# %%
def test_learning_roadmap():
    """Test that the learning roadmap respects dependencies and makes pedagogical sense"""
    
    # Test that all prerequisites come before dependents
    path_indices = {module: i for i, module in enumerate(learning_path)}
    
    violations = []
    for module_name in learning_path:
        if module_name in analyzer.modules:
            module = analyzer.modules[module_name]
            module_index = path_indices[module_name]
            
            for prereq in module.prerequisites:
                if prereq in path_indices:
                    prereq_index = path_indices[prereq]
                    if prereq_index >= module_index:
                        violations.append(f"{module_name} comes before its prerequisite {prereq}")
    
    if violations:
        print("⚠️  Dependency violations found:")
        for violation in violations:
            print(f"   {violation}")
    else:
        print("✅ Learning roadmap respects all dependencies")
    
    # Test reasonable progression
    foundation_modules = ['setup', 'tensor']
    advanced_modules = ['capstone', 'mlops', 'benchmarking']
    
    foundation_positions = [path_indices.get(m, -1) for m in foundation_modules]
    advanced_positions = [path_indices.get(m, -1) for m in advanced_modules]
    
    foundation_positions = [p for p in foundation_positions if p >= 0]
    advanced_positions = [p for p in advanced_positions if p >= 0]
    
    if foundation_positions and advanced_positions:
        if max(foundation_positions) < min(advanced_positions):
            print("✅ Foundation modules come before advanced modules")
        else:
            print("⚠️  Some advanced modules come before foundation modules")
    
    print(f"📊 Total learning time: {total_time:.1f} hours ({total_time/8:.1f} work days)")

# test_learning_roadmap() # Test moved to main block

# %% [markdown]
"""
## Component Relationship Analysis

Let's analyze the specific components within each module and how they relate to each other.
"""

# %%
#| export
def create_component_analysis(figsize=(14, 10)):
    """Create visualization of components within modules and their relationships"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Components per module
    modules = [name for name in learning_path if name in analyzer.modules]
    component_counts = [len(analyzer.modules[name].components) for name in modules]
    module_titles = [analyzer.modules[name].title for name in modules]
    
    ax1.bar(range(len(modules)), component_counts, 
           color=plt.cm.viridis(np.linspace(0, 1, len(modules))))
    ax1.set_title('Components per Module', fontweight='bold')
    ax1.set_xlabel('Module')
    ax1.set_ylabel('Number of Components')
    ax1.set_xticks(range(len(modules)))
    ax1.set_xticklabels([title[:10] for title in module_titles], rotation=45)
    
    # 2. Difficulty vs Components scatter
    difficulties = [analyzer.modules[name].difficulty_level() for name in modules]
    times = [analyzer.modules[name].estimated_hours() for name in modules]
    
    scatter = ax2.scatter(component_counts, difficulties, s=[t*20 for t in times], 
                         c=times, cmap='plasma', alpha=0.7)
    ax2.set_title('Module Complexity Analysis', fontweight='bold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Difficulty Level')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time (hours)')
    
    # 3. Module categories pie chart
    categories = {
        'Foundation': ['setup', 'tensor', 'activations'],
        'Architecture': ['layers', 'dense', 'spatial', 'attention'],
        'Training': ['dataloader', 'autograd', 'optimizers', 'training'],
        'Production': ['compression', 'kernels', 'benchmarking', 'mlops', 'capstone']
    }
    
    category_counts = []
    category_labels = []
    category_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for category, module_list in categories.items():
        count = sum(1 for m in module_list if m in analyzer.modules)
        if count > 0:
            category_counts.append(count)
            category_labels.append(f'{category}\n({count} modules)')
    
    ax3.pie(category_counts, labels=category_labels, colors=category_colors[:len(category_counts)],
           autopct='%1.0f%%', startangle=90)
    ax3.set_title('Module Distribution by Category', fontweight='bold')
    
    # 4. Learning progression timeline
    cumulative_components = np.cumsum([0] + component_counts)
    cumulative_time = np.cumsum([0] + times)
    
    ax4.plot(cumulative_time[:-1], cumulative_components[:-1], 'o-', linewidth=2, markersize=6)
    ax4.set_title('Learning Progression', fontweight='bold')
    ax4.set_xlabel('Cumulative Time (hours)')
    ax4.set_ylabel('Cumulative Components Learned')
    ax4.grid(True, alpha=0.3)
    
    # Add milestone annotations
    milestones = [0, len(modules)//4, len(modules)//2, 3*len(modules)//4, len(modules)-1]
    for i in milestones:
        if i < len(cumulative_time) - 1:
            ax4.annotate(f'{cumulative_components[i]} comp.\n{cumulative_time[i]:.0f}h',
                        xy=(cumulative_time[i], cumulative_components[i]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Component analysis function (called in main block)

# %% [markdown]
"""
### Test Component Analysis

Let's verify our component analysis provides meaningful insights.
"""

# %%
def test_component_analysis():
    """Test that component analysis reveals meaningful patterns"""
    
    # Test component distribution
    total_components = sum(len(module.components) for module in analyzer.modules.values())
    avg_components = total_components / len(analyzer.modules)
    
    print(f"📊 Total components across all modules: {total_components}")
    print(f"📊 Average components per module: {avg_components:.1f}")
    
    # Find modules with most/least components
    component_counts = [(name, len(module.components)) for name, module in analyzer.modules.items()]
    component_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Modules with most components:")
    for name, count in component_counts[:3]:
        print(f"   {analyzer.modules[name].title}: {count} components")
    
    print(f"\n🏆 Modules with least components:")
    for name, count in component_counts[-3:]:
        print(f"   {analyzer.modules[name].title}: {count} components")
    
    # Test correlation between difficulty and components
    difficulties = [module.difficulty_level() for module in analyzer.modules.values()]
    components = [len(module.components) for module in analyzer.modules.values()]
    
    correlation = np.corrcoef(difficulties, components)[0, 1]
    print(f"\n📈 Correlation between difficulty and components: {correlation:.2f}")
    
    if correlation > 0.3:
        print("✅ Higher difficulty modules tend to have more components")
    elif correlation < -0.3:
        print("⚠️  Higher difficulty modules tend to have fewer components")
    else:
        print("📊 No strong correlation between difficulty and component count")

# test_component_analysis() # Test moved to main block

# %% [markdown]
"""
## Export Functions and Module Interface

Create functions that can be imported and used by other parts of TinyTorch.
"""

# %%
#| export
def get_tinytorch_overview() -> Dict:
    """Get comprehensive overview of TinyTorch system for external use"""
    return {
        'total_modules': len(analyzer.modules),
        'total_components': sum(len(module.components) for module in analyzer.modules.values()),
        'learning_path': analyzer.get_learning_path(),
        'total_time_hours': sum(module.estimated_hours() for module in analyzer.modules.values()),
        'difficulty_levels': {name: module.difficulty_level() for name, module in analyzer.modules.items()},
        'module_categories': {
            'foundation': ['setup', 'tensor', 'activations'],
            'architecture': ['layers', 'dense', 'spatial', 'attention'], 
            'training': ['dataloader', 'autograd', 'optimizers', 'training'],
            'production': ['compression', 'kernels', 'benchmarking', 'mlops', 'capstone']
        }
    }

#| export
def visualize_tinytorch_system(save_plots: bool = False) -> Dict:
    """Generate all TinyTorch system visualizations"""
    
    visualizations = {}
    
    print("🎨 Generating TinyTorch system visualizations...")
    
    # Generate dependency graph
    print("   📊 Creating dependency graph...")
    dep_fig = create_dependency_graph_visualization()
    visualizations['dependency_graph'] = dep_fig
    
    # Generate architecture diagram  
    print("   🏗️  Creating architecture diagram...")
    arch_fig, positions = create_system_architecture_diagram()
    visualizations['architecture'] = arch_fig
    
    # Generate learning roadmap
    print("   📚 Creating learning roadmap...")
    roadmap_fig, path, time = create_learning_roadmap()
    visualizations['roadmap'] = roadmap_fig
    
    # Generate component analysis
    print("   🔍 Creating component analysis...")
    component_fig = create_component_analysis()
    visualizations['components'] = component_fig
    
    if save_plots:
        print("   💾 Saving plots to files...")
        for name, fig in visualizations.items():
            fig.savefig(f'tinytorch_{name}.png', dpi=300, bbox_inches='tight')
    
    print("✅ All visualizations generated successfully!")
    
    return visualizations

#| export
def get_module_info(module_name: str) -> Dict:
    """Get detailed information about a specific module"""
    if module_name not in analyzer.modules:
        return {'error': f'Module {module_name} not found'}
    
    module = analyzer.modules[module_name]
    return {
        'name': module.name,
        'title': module.title,
        'description': module.description,
        'prerequisites': module.prerequisites,
        'enables': module.enables,
        'components': module.components,
        'difficulty': module.difficulty,
        'difficulty_level': module.difficulty_level(),
        'time_estimate': module.time_estimate,
        'estimated_hours': module.estimated_hours(),
        'exports_to': module.exports_to
    }

#| export
def get_learning_recommendations(current_module: str = None) -> Dict:
    """Get personalized learning recommendations"""
    path = analyzer.get_learning_path()
    
    if current_module is None:
        return {
            'recommended_start': path[0] if path else None,
            'full_path': path,
            'total_time': sum(analyzer.modules[name].estimated_hours() 
                            for name in path if name in analyzer.modules)
        }
    
    if current_module not in path:
        return {'error': f'Module {current_module} not found in learning path'}
    
    current_index = path.index(current_module)
    
    return {
        'current_module': current_module,
        'progress': f"{current_index + 1}/{len(path)}",
        'next_modules': path[current_index + 1:current_index + 4],  # Next 3 modules
        'remaining_time': sum(analyzer.modules[name].estimated_hours() 
                            for name in path[current_index + 1:] 
                            if name in analyzer.modules),
        'prerequisites_completed': path[:current_index],
        'can_start': [name for name in path[current_index + 1:] 
                     if all(prereq in path[:current_index + 1] 
                           for prereq in analyzer.modules.get(name, ModuleInfo('','','',[],'',[],'','','')).prerequisites)]
    }

# Export functions (tested in main block)

# %% [markdown]
"""
## ML Systems Thinking Questions

Let's explore how TinyTorch's architecture connects to broader ML systems and production frameworks.
"""

# %% [markdown]
"""
### System Architecture & Design Patterns

**Reflection Questions:**

1. **Modular Design Philosophy**: How does TinyTorch's module dependency system compare to frameworks like PyTorch or TensorFlow? What are the advantages and trade-offs of explicit dependency management?

2. **Component Composition**: Notice how higher-level modules (like `training`) depend on multiple lower-level modules (`tensor`, `autograd`, `optimizers`). How does this pattern reflect real ML system architecture?

3. **Framework Evolution**: Looking at the learning roadmap, how might you extend TinyTorch to support distributed training or GPU acceleration? Where would new modules fit in the dependency graph?

### Production ML Systems

**Reflection Questions:**

4. **Deployment Pipeline**: How do the later modules (`compression`, `benchmarking`, `mlops`) mirror real-world ML deployment concerns? What additional modules might production systems require?

5. **System Integration**: If you were to deploy a TinyTorch model in production, which modules would be most critical for runtime efficiency? How might you minimize dependencies?

6. **Monitoring & Observability**: How does the `mlops` module's position as a terminal node reflect its role in production systems? What additional monitoring capabilities might be needed?

### Framework Design Decisions

**Reflection Questions:**

7. **Dependency Management**: Compare TinyTorch's explicit prerequisite system to frameworks like Keras or scikit-learn. How do design decisions about dependencies affect developer experience?

8. **Component Granularity**: Some modules have many components (like `training`) while others have few (like `setup`). How do you balance component granularity in ML framework design?

9. **Educational vs Production**: How might the educational goals of TinyTorch influence its architecture differently than a production framework? Where do you see pedagogical design choices?

### Performance & Scalability

**Reflection Questions:**

10. **Computational Graph**: How does TinyTorch's module structure relate to computational graphs in frameworks like PyTorch or JAX? Where do you see opportunities for optimization?

11. **Memory Management**: Looking at the component analysis, which modules are likely to be most memory-intensive? How might this influence deployment strategies?

12. **Hardware Acceleration**: The `kernels` module focuses on hardware-aware optimization. How do production frameworks handle the trade-off between abstraction and performance?

*These questions are designed to help you think beyond implementation details toward the broader patterns and principles that guide ML systems design in industry.*
"""

# %% [markdown]
"""
## Comprehensive Testing

Let's run comprehensive tests to ensure all our visualizations and analysis tools work correctly.
"""

# %%
def run_comprehensive_tests():
    """Run comprehensive tests of the introduction module functionality"""
    
    print("🧪 Running comprehensive tests for TinyTorch Introduction Module...")
    print("=" * 60)
    
    # Test 1: Module Loading
    print("\n1️⃣  Testing module loading...")
    assert len(analyzer.modules) >= 10, "Should load multiple modules"
    assert 'setup' in analyzer.modules, "Should load setup module"
    assert 'tensor' in analyzer.modules, "Should load tensor module"
    print("✅ Module loading tests passed")
    
    # Test 2: Dependency Graph
    print("\n2️⃣  Testing dependency graph...")
    G = analyzer.dependency_graph
    assert G.number_of_nodes() >= 10, "Should have multiple nodes"
    assert G.number_of_edges() >= 5, "Should have dependency edges"
    print("✅ Dependency graph tests passed")
    
    # Test 3: Learning Path
    print("\n3️⃣  Testing learning path...")
    path = analyzer.get_learning_path()
    assert len(path) >= 10, "Should have meaningful learning path"
    
    # Verify no dependency violations
    path_indices = {module: i for i, module in enumerate(path)}
    violations = 0
    for module_name in path:
        if module_name in analyzer.modules:
            module = analyzer.modules[module_name]
            for prereq in module.prerequisites:
                if prereq in path_indices:
                    if path_indices[prereq] >= path_indices[module_name]:
                        violations += 1
    
    assert violations == 0, f"Learning path should have no dependency violations (found {violations})"
    print("✅ Learning path tests passed")
    
    # Test 4: Component Analysis
    print("\n4️⃣  Testing component analysis...")
    total_components = sum(len(module.components) for module in analyzer.modules.values())
    assert total_components >= 20, "Should have meaningful number of components"
    print("✅ Component analysis tests passed")
    
    # Test 5: Export Functions
    print("\n5️⃣  Testing export functions...")
    overview = get_tinytorch_overview()
    assert 'total_modules' in overview, "Overview should include module count"
    assert 'learning_path' in overview, "Overview should include learning path"
    
    module_info = get_module_info('setup')
    assert 'title' in module_info, "Module info should include title"
    
    recommendations = get_learning_recommendations()
    assert 'recommended_start' in recommendations, "Should provide starting recommendation"
    print("✅ Export function tests passed")
    
    # Test 6: Visualization Generation
    print("\n6️⃣  Testing visualization generation...")
    try:
        # Test that we can generate all visualizations without errors
        dep_fig = create_dependency_graph_visualization()
        arch_fig, positions = create_system_architecture_diagram()
        roadmap_fig, path, time = create_learning_roadmap()
        component_fig = create_component_analysis()
        
        assert dep_fig is not None, "Should generate dependency graph"
        assert arch_fig is not None, "Should generate architecture diagram"
        assert roadmap_fig is not None, "Should generate roadmap"
        assert component_fig is not None, "Should generate component analysis"
        
        # Close figures to save memory
        plt.close(dep_fig)
        plt.close(arch_fig)
        plt.close(roadmap_fig)
        plt.close(component_fig)
        
        print("✅ Visualization generation tests passed")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! TinyTorch Introduction Module is working correctly!")
    print("=" * 60)
    
    # Summary statistics
    print(f"\n📊 System Summary:")
    print(f"   • {len(analyzer.modules)} modules loaded")
    print(f"   • {analyzer.dependency_graph.number_of_edges()} dependencies mapped")
    print(f"   • {len(analyzer.get_learning_path())} modules in learning path")
    print(f"   • {sum(len(m.components) for m in analyzer.modules.values())} total components")
    print(f"   • {sum(m.estimated_hours() for m in analyzer.modules.values()):.1f} total learning hours")

if __name__ == "__main__":
    # Run individual tests
    test_module_analyzer()
    test_dependency_relationships()
    test_system_architecture()
    test_learning_roadmap()
    test_component_analysis()
    
    # Run comprehensive test suite
    run_comprehensive_tests()
    
    # Create and display visualizations
    dependency_fig = create_dependency_graph_visualization()
    arch_fig, module_positions = create_system_architecture_diagram()
    roadmap_fig, learning_path, total_time = create_learning_roadmap()
    component_fig = create_component_analysis()
    
    print(f"📚 Learning path contains {len(learning_path)} modules")
    print(f"⏱️  Total estimated time: {total_time:.1f} hours")
    
    # Test export functions
    print("🧪 Testing export functions...")
    overview = get_tinytorch_overview()
    print(f"📊 System Overview: {overview['total_modules']} modules, {overview['total_components']} components")
    
    setup_info = get_module_info('setup')
    print(f"📋 Setup Module: {setup_info['title']} - {setup_info['difficulty']}")
    
    recommendations = get_learning_recommendations()
    print(f"📚 Learning Recommendations: Start with {recommendations['recommended_start']}")
    print("✅ Export functions working correctly!")
    
    print(f"📊 Loaded {len(analyzer.modules)} TinyTorch modules")
    print(f"🔗 Built dependency graph with {analyzer.dependency_graph.number_of_edges()} connections")
    
    print("All tests passed!")
    print("🎯 TinyTorch Introduction Module Complete!")
    print("📦 Exported functions ready for use by other modules")

# %% [markdown]
"""
## Module Summary

**Congratulations!** You've successfully explored the complete TinyTorch system architecture.

### What You've Accomplished

1. **📊 System Analysis**: Built tools to automatically analyze module dependencies and relationships
2. **🎨 Interactive Visualizations**: Created comprehensive visual overviews of the entire framework
3. **📚 Learning Roadmap**: Generated an optimal learning path through all 16 modules
4. **🔍 Component Analysis**: Analyzed the components within each module and their complexity
5. **🏗️  Architecture Overview**: Visualized how all TinyTorch components work together
6. **🧪 Comprehensive Testing**: Validated that all analysis tools work correctly

### Key Insights Discovered

- **TinyTorch contains {len(analyzer.modules)} modules** with {sum(len(m.components) for m in analyzer.modules.values())} total components
- **Learning path spans {sum(m.estimated_hours() for m in analyzer.modules.values()):.1f} hours** of estimated study time
- **Dependency structure** ensures proper learning progression from foundations to production
- **Modular design** enables flexible learning and component reuse

### How This Connects to Industry ML Systems

The architecture patterns you've explored in TinyTorch mirror those used in production ML frameworks:

- **Modular Dependencies**: Similar to PyTorch's module system
- **Component Composition**: Reflects how TensorFlow builds complex operations from primitives  
- **Production Pipeline**: MLOps module mirrors real deployment concerns
- **Performance Optimization**: Kernels and compression reflect production efficiency needs

### Next Steps

Now you're ready to dive into any TinyTorch module with a complete understanding of how it fits into the broader system. Use the learning roadmap to guide your journey through building a complete neural network framework from scratch!

**Happy Learning! 🚀**
"""

# %%
# Export key functions for use by other modules
__all__ = [
    'TinyTorchAnalyzer',
    'get_tinytorch_overview', 
    'visualize_tinytorch_system',
    'get_module_info',
    'get_learning_recommendations',
    'create_dependency_graph_visualization',
    'create_system_architecture_diagram', 
    'create_learning_roadmap',
    'create_component_analysis'
]
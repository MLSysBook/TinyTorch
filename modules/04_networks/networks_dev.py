# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Networks - Building Intelligence Through Layer Composition

Welcome to Networks! You'll learn how to combine individual layers into complete neural networks that can solve complex problems.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 01 (Tensor): Multi-dimensional data structures for inputs and outputs
- Module 02 (Activations): Nonlinear functions that create intelligence
- Module 03 (Layers): Linear layers that transform data with learnable parameters

**What's Working**: You can transform data with individual layers and activations!

**The Gap**: Individual layers solve simple problems - real intelligence emerges when layers compose into networks.

**This Module's Solution**: Learn to manually compose layers into multi-layer networks with different architectures.

**Connection Map**:
```
Layers → Manual Composition → Complete Networks
(transforms)  (architecture)     (intelligence)
```

## Learning Objectives
1. **Manual Network Architecture**: Build networks by composing layers step-by-step
2. **Parameter Management**: Count and track parameters across multiple layers
3. **Forward Pass Logic**: Understand data flow through network architectures
4. **Network Architectures**: Create different network shapes (wide, deep, custom)
5. **Systems Understanding**: Analyze memory usage and computational complexity

## Build → Test → Use
1. **Build**: Manual network composition functions and parameter counting
2. **Test**: Validate networks with different architectures and input sizes
3. **Use**: Apply networks to solve problems requiring multiple transformations
"""

# %%
# Essential imports for network composition
import numpy as np
import sys
import os
from typing import List, Tuple, Union, Optional

# Import building blocks from previous modules - ONLY use concepts we've learned
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
    from tinytorch.core.layers import Linear, Module
except ImportError:
    # Development fallback
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_activations'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_layers'))
    from tensor_dev import Tensor
    from activations_dev import ReLU, Sigmoid, Tanh, Softmax
    from layers_dev import Linear, Module

# %% [markdown]
"""
## Part 1: Understanding Network Architecture

### What Makes a Neural Network?

A neural network is simply **multiple layers composed together** where each layer's output becomes the next layer's input.

```
Input → Layer1 → Activation → Layer2 → Activation → Output
 (4)      (8)       (8)       (3)       (3)       (3)
```

**Key Insights**:
- **Composition**: Networks = layers + activations in sequence
- **Data Flow**: Output shape of layer N must match input shape of layer N+1
- **Intelligence**: Nonlinearity from activations enables complex pattern learning
- **Architecture**: Layer sizes and arrangements determine network capability
"""

# %% [markdown]
"""
## Part 2: Manual Network Composition

Let's start by learning to compose networks manually before automation.
"""

# %% nbgrader={"grade": false, "grade_id": "network-composition", "solution": true}
def compose_two_layer_network(input_size: int, hidden_size: int, output_size: int,
                             activation=ReLU) -> Tuple[Linear, object, Linear]:
    """
    Create a 2-layer network manually: Input → Linear → Activation → Linear → Output

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden layer neurons
        output_size: Number of output features
        activation: Activation function class (default: ReLU)

    Returns:
        Tuple of (layer1, activation_instance, layer2)

    TODO: Create two Linear layers and one activation function

    APPROACH:
    1. Create first Linear layer: input_size → hidden_size
    2. Create activation function instance
    3. Create second Linear layer: hidden_size → output_size
    4. Return all three components as tuple

    EXAMPLE:
    >>> layer1, act, layer2 = compose_two_layer_network(4, 8, 3)
    >>> x = Tensor([[1, 2, 3, 4]])
    >>> h = layer1(x)      # (1, 4) → (1, 8)
    >>> h_act = act(h)     # (1, 8) → (1, 8)
    >>> y = layer2(h_act)  # (1, 8) → (1, 3)
    >>> print(y.shape)     # (1, 3)

    HINTS:
    - Use Linear(input_size, hidden_size) for first layer
    - Create activation instance with activation()
    - Use Linear(hidden_size, output_size) for second layer
    - Return as (layer1, activation_instance, layer2)
    """
    ### BEGIN SOLUTION
    # Create first layer: input → hidden
    layer1 = Linear(input_size, hidden_size)

    # Create activation function instance
    activation_instance = activation()

    # Create second layer: hidden → output
    layer2 = Linear(hidden_size, output_size)

    return layer1, activation_instance, layer2
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Two-Layer Network Composition
Test that we can manually compose a simple 2-layer network
"""

# %%
def test_unit_two_layer_composition():
    """Test two-layer network composition with different configurations"""
    print("🔬 Unit Test: Two-Layer Network Composition...")

    # Test 1: Basic composition
    layer1, activation, layer2 = compose_two_layer_network(4, 8, 3)

    assert isinstance(layer1, Linear), "First component should be Linear layer"
    assert isinstance(activation, ReLU), "Second component should be activation function"
    assert isinstance(layer2, Linear), "Third component should be Linear layer"

    assert layer1.input_size == 4, "First layer should have correct input size"
    assert layer1.output_size == 8, "First layer should have correct output size"
    assert layer2.input_size == 8, "Second layer should have correct input size"
    assert layer2.output_size == 3, "Second layer should have correct output size"

    # Test 2: Forward pass compatibility
    x = Tensor(np.random.randn(2, 4))
    h = layer1(x)
    h_activated = activation(h)
    y = layer2(h_activated)

    assert h.shape == (2, 8), "Hidden layer output should have correct shape"
    assert h_activated.shape == (2, 8), "Activated hidden should preserve shape"
    assert y.shape == (2, 3), "Final output should have correct shape"

    # Test 3: Different activation functions
    layer1_sig, sig_act, layer2_sig = compose_two_layer_network(3, 5, 2, Sigmoid)
    assert isinstance(sig_act, Sigmoid), "Should create Sigmoid activation when specified"

    print("✅ Two-layer network composition works correctly!")

test_unit_two_layer_composition()

# %% [markdown]
"""
## Part 3: Forward Pass Through Networks

Now let's implement the logic for running data through composed networks.
"""

# %% nbgrader={"grade": false, "grade_id": "forward-pass", "solution": true}
def forward_pass_two_layer(x: Tensor, layer1: Linear, activation, layer2: Linear) -> Tensor:
    """
    Execute forward pass through a 2-layer network.

    Args:
        x: Input tensor
        layer1: First Linear layer
        activation: Activation function
        layer2: Second Linear layer

    Returns:
        Output tensor after passing through the network

    TODO: Implement forward pass: x → layer1 → activation → layer2 → output

    APPROACH:
    1. Pass input through first layer
    2. Apply activation function to result
    3. Pass activated result through second layer
    4. Return final output

    EXAMPLE:
    >>> x = Tensor([[1, 2, 3, 4]])  # (1, 4)
    >>> y = forward_pass_two_layer(x, layer1, relu, layer2)
    >>> print(y.shape)  # (1, output_size)

    HINTS:
    - Call each component in sequence: layer1(x), activation(h), layer2(h_act)
    - Each output becomes input to next component
    - Return the final result
    """
    ### BEGIN SOLUTION
    # Step 1: First layer transformation
    hidden = layer1(x)

    # Step 2: Apply activation function
    hidden_activated = activation(hidden)

    # Step 3: Second layer transformation
    output = layer2(hidden_activated)

    return output
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Forward Pass Through Network
Test that data flows correctly through our manual network
"""

# %%
def test_unit_forward_pass():
    """Test forward pass through manually composed networks"""
    print("🔬 Unit Test: Forward Pass Through Networks...")

    # Create test network
    layer1, relu_act, layer2 = compose_two_layer_network(5, 10, 3)

    # Test 1: Single sample
    x_single = Tensor(np.random.randn(1, 5))
    y_single = forward_pass_two_layer(x_single, layer1, relu_act, layer2)

    assert y_single.shape == (1, 3), "Single sample should produce correct output shape"
    assert hasattr(y_single, 'shape') and hasattr(y_single, 'data'), "Output should be a Tensor-like object"

    # Test 2: Batch processing
    x_batch = Tensor(np.random.randn(4, 5))
    y_batch = forward_pass_two_layer(x_batch, layer1, relu_act, layer2)

    assert y_batch.shape == (4, 3), "Batch should produce correct output shape"

    # Test 3: Different network architectures
    wide_layer1, wide_act, wide_layer2 = compose_two_layer_network(2, 50, 1)
    x_wide = Tensor(np.random.randn(3, 2))
    y_wide = forward_pass_two_layer(x_wide, wide_layer1, wide_act, wide_layer2)

    assert y_wide.shape == (3, 1), "Wide network should work correctly"

    print("✅ Forward pass through networks works correctly!")

test_unit_forward_pass()

# %% [markdown]
"""
## Part 4: Deep Network Composition

Real neural networks often have more than 2 layers. Let's build deep networks manually.
"""

# %% nbgrader={"grade": false, "grade_id": "deep-network", "solution": true}
def compose_deep_network(layer_sizes: List[int], activation=ReLU) -> List:
    """
    Create a deep network with arbitrary number of layers.

    Args:
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation: Activation function class

    Returns:
        List of network components [layer1, activation1, layer2, activation2, ..., final_layer]

    TODO: Create alternating Linear layers and activations for each pair of sizes

    APPROACH:
    1. Iterate through pairs of consecutive sizes in layer_sizes
    2. For each pair, create Linear(size_i, size_i+1) and activation()
    3. Don't add activation after the final layer (output layer typically no activation)
    4. Return list of all components in order

    EXAMPLE:
    >>> components = compose_deep_network([4, 8, 6, 3])
    >>> # Creates: Linear(4,8), ReLU(), Linear(8,6), ReLU(), Linear(6,3)
    >>> len(components)  # 5 components

    HINTS:
    - Use zip(layer_sizes[:-1], layer_sizes[1:]) to get consecutive pairs
    - Add Linear layer, then activation for each pair (except last layer)
    - Last layer: only add Linear, no activation
    - Return list of all components
    """
    ### BEGIN SOLUTION
    components = []

    # Process all but the last layer (add Linear + Activation)
    for i in range(len(layer_sizes) - 2):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]

        # Add Linear layer
        components.append(Linear(input_size, output_size))
        # Add activation
        components.append(activation())

    # Add final layer (Linear only, no activation)
    if len(layer_sizes) >= 2:
        final_input = layer_sizes[-2]
        final_output = layer_sizes[-1]
        components.append(Linear(final_input, final_output))

    return components
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Deep Network Composition
Test that we can build networks with arbitrary depth
"""

# %%
def test_unit_deep_network():
    """Test deep network composition with various architectures"""
    print("🔬 Unit Test: Deep Network Composition...")

    # Test 1: 3-layer network
    components_3layer = compose_deep_network([4, 8, 6, 3])
    expected_components = 5  # Linear, ReLU, Linear, ReLU, Linear

    assert len(components_3layer) == expected_components, f"3-layer network should have {expected_components} components"

    # Verify component types and order
    assert isinstance(components_3layer[0], Linear), "First component should be Linear"
    assert isinstance(components_3layer[1], ReLU), "Second component should be ReLU"
    assert isinstance(components_3layer[2], Linear), "Third component should be Linear"
    assert isinstance(components_3layer[3], ReLU), "Fourth component should be ReLU"
    assert isinstance(components_3layer[4], Linear), "Fifth component should be Linear (final)"

    # Test 2: Verify layer sizes
    assert components_3layer[0].input_size == 4, "First layer should have correct input size"
    assert components_3layer[0].output_size == 8, "First layer should have correct output size"
    assert components_3layer[2].input_size == 8, "Second layer should have correct input size"
    assert components_3layer[2].output_size == 6, "Second layer should have correct output size"
    assert components_3layer[4].input_size == 6, "Final layer should have correct input size"
    assert components_3layer[4].output_size == 3, "Final layer should have correct output size"

    # Test 3: Different activation function
    components_sigmoid = compose_deep_network([2, 4, 1], Sigmoid)
    assert isinstance(components_sigmoid[1], Sigmoid), "Should use specified activation function"

    # Test 4: Single layer (edge case)
    components_single = compose_deep_network([5, 2])
    assert len(components_single) == 1, "Single layer should have 1 component"
    assert isinstance(components_single[0], Linear), "Single component should be Linear layer"

    print("✅ Deep network composition works correctly!")

test_unit_deep_network()

# %% [markdown]
"""
## Part 5: Forward Pass Through Deep Networks

Now implement forward pass logic for networks of arbitrary depth.
"""

# %% nbgrader={"grade": false, "grade_id": "deep-forward", "solution": true}
def forward_pass_deep(x: Tensor, components: List) -> Tensor:
    """
    Execute forward pass through a deep network with arbitrary components.

    Args:
        x: Input tensor
        components: List of network components (layers and activations)

    Returns:
        Output tensor after passing through all components

    TODO: Apply each component in sequence to transform the input

    APPROACH:
    1. Start with input tensor
    2. Apply each component in order: x = component(x)
    3. Each component's output becomes next component's input
    4. Return final result

    EXAMPLE:
    >>> components = [Linear(4,8), ReLU(), Linear(8,3)]
    >>> x = Tensor([[1, 2, 3, 4]])
    >>> y = forward_pass_deep(x, components)
    >>> print(y.shape)  # (1, 3)

    HINTS:
    - Use a for loop: for component in components:
    - Apply each component: x = component(x)
    - Return the final transformed x
    """
    ### BEGIN SOLUTION
    # Apply each component in sequence
    current_tensor = x
    for component in components:
        current_tensor = component(current_tensor)

    return current_tensor
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Deep Forward Pass
Test forward pass through networks of varying depth
"""

# %%
def test_unit_deep_forward():
    """Test forward pass through deep networks"""
    print("🔬 Unit Test: Deep Forward Pass...")

    # Test 1: 3-layer network
    components = compose_deep_network([5, 10, 8, 3])
    x = Tensor(np.random.randn(2, 5))
    y = forward_pass_deep(x, components)

    assert y.shape == (2, 3), "Deep network should produce correct output shape"
    assert hasattr(y, 'shape') and hasattr(y, 'data'), "Output should be a Tensor-like object"

    # Test 2: Very deep network
    deep_components = compose_deep_network([4, 16, 12, 8, 6, 2])
    x_deep = Tensor(np.random.randn(1, 4))
    y_deep = forward_pass_deep(x_deep, deep_components)

    assert y_deep.shape == (1, 2), "Very deep network should work correctly"

    # Test 3: Wide network
    wide_components = compose_deep_network([3, 100, 1])
    x_wide = Tensor(np.random.randn(5, 3))
    y_wide = forward_pass_deep(x_wide, wide_components)

    assert y_wide.shape == (5, 1), "Wide network should work correctly"

    # Test 4: Single layer
    single_components = compose_deep_network([6, 4])
    x_single = Tensor(np.random.randn(1, 6))
    y_single = forward_pass_deep(x_single, single_components)

    assert y_single.shape == (1, 4), "Single layer should work correctly"

    print("✅ Deep forward pass works correctly!")

test_unit_deep_forward()

# %% [markdown]
"""
## Part 6: Parameter Counting and Analysis

Understanding how many learnable parameters are in a network is crucial for memory management and computational complexity.
"""

# %% nbgrader={"grade": false, "grade_id": "parameter-counting", "solution": true}
def count_network_parameters(components: List) -> Tuple[int, dict]:
    """
    Count total parameters in a network and provide detailed breakdown.

    Args:
        components: List of network components

    Returns:
        Tuple of (total_parameters, parameter_breakdown)

    TODO: Count parameters in each Linear layer and provide breakdown

    APPROACH:
    1. Initialize total counter and breakdown dictionary
    2. Iterate through components looking for Linear layers
    3. For each Linear layer: count weights (input_size × output_size) + biases (output_size)
    4. Store breakdown by layer and return total + breakdown

    EXAMPLE:
    >>> components = [Linear(4,8), ReLU(), Linear(8,3)]
    >>> total, breakdown = count_network_parameters(components)
    >>> print(total)  # (4*8 + 8) + (8*3 + 3) = 32 + 8 + 24 + 3 = 67

    HINTS:
    - Only Linear layers have parameters (activations have none)
    - For Linear layer: parameters = input_size * output_size + output_size
    - Use isinstance(component, Linear) to identify Linear layers
    - Track breakdown with layer names/indices
    """
    ### BEGIN SOLUTION
    total_params = 0
    breakdown = {}

    layer_count = 0
    for i, component in enumerate(components):
        if isinstance(component, Linear):
            layer_count += 1

            # Count weights and biases
            weights = component.input_size * component.output_size
            biases = component.output_size
            layer_params = weights + biases

            # Add to total
            total_params += layer_params

            # Add to breakdown
            breakdown[f"Linear_Layer_{layer_count}"] = {
                "weights": weights,
                "biases": biases,
                "total": layer_params,
                "shape": f"({component.input_size}, {component.output_size})"
            }

    return total_params, breakdown
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Parameter Counting
Test that we correctly count parameters across network architectures
"""

# %%
def test_unit_parameter_counting():
    """Test parameter counting across different network architectures"""
    print("🔬 Unit Test: Parameter Counting...")

    # Test 1: Simple 2-layer network
    components = compose_deep_network([4, 8, 3])
    total, breakdown = count_network_parameters(components)

    # Expected: (4*8 + 8) + (8*3 + 3) = 40 + 27 = 67
    expected_total = (4*8 + 8) + (8*3 + 3)
    assert total == expected_total, f"Expected {expected_total} parameters, got {total}"

    # Verify breakdown structure
    assert "Linear_Layer_1" in breakdown, "Should have first layer in breakdown"
    assert "Linear_Layer_2" in breakdown, "Should have second layer in breakdown"
    assert breakdown["Linear_Layer_1"]["weights"] == 32, "First layer should have 32 weights"
    assert breakdown["Linear_Layer_1"]["biases"] == 8, "First layer should have 8 biases"

    # Test 2: Single layer
    single_components = compose_deep_network([10, 5])
    single_total, single_breakdown = count_network_parameters(single_components)

    expected_single = 10*5 + 5  # 55
    assert single_total == expected_single, f"Single layer should have {expected_single} parameters"

    # Test 3: Deep network
    deep_components = compose_deep_network([3, 6, 4, 2])
    deep_total, deep_breakdown = count_network_parameters(deep_components)

    # Expected: (3*6+6) + (6*4+4) + (4*2+2) = 24 + 28 + 10 = 62
    expected_deep = (3*6 + 6) + (6*4 + 4) + (4*2 + 2)
    assert deep_total == expected_deep, f"Deep network should have {expected_deep} parameters"
    assert len(deep_breakdown) == 3, "Deep network should have 3 Linear layers in breakdown"

    # Test 4: Network with activations (shouldn't count activation parameters)
    mixed_components = [Linear(5, 10), ReLU(), Linear(10, 2), Sigmoid()]
    mixed_total, mixed_breakdown = count_network_parameters(mixed_components)

    expected_mixed = (5*10 + 10) + (10*2 + 2)  # 60 + 22 = 82
    assert mixed_total == expected_mixed, "Should only count Linear layer parameters"
    assert len(mixed_breakdown) == 2, "Should only include Linear layers in breakdown"

    print("✅ Parameter counting works correctly!")

test_unit_parameter_counting()

# %% [markdown]
"""
## Part 7: Network Architecture Patterns

Let's implement common network architecture patterns used in practice.
"""

# %% nbgrader={"grade": false, "grade_id": "network-patterns", "solution": true}
def create_classifier_network(input_size: int, num_classes: int, hidden_sizes: List[int] = None) -> List:
    """
    Create a classification network with sigmoid output activation.

    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        hidden_sizes: List of hidden layer sizes (optional)

    Returns:
        List of network components with Sigmoid output for classification

    TODO: Create network ending with Sigmoid activation for classification

    APPROACH:
    1. Use provided hidden_sizes or default to [hidden_size] if None
    2. Create base network structure: input → hidden layers → output
    3. Add Sigmoid activation at the end for classification probabilities
    4. Return complete component list

    EXAMPLE:
    >>> components = create_classifier_network(784, 10, [128, 64])
    >>> # Creates: Linear(784,128), ReLU(), Linear(128,64), ReLU(), Linear(64,10), Sigmoid()

    HINTS:
    - If hidden_sizes is None, use a reasonable default like [input_size // 2]
    - Build layer_sizes list: [input_size] + hidden_sizes + [num_classes]
    - Use compose_deep_network to create base network
    - Add Sigmoid() activation at the end for classification
    """
    ### BEGIN SOLUTION
    # Handle default hidden sizes
    if hidden_sizes is None:
        hidden_sizes = [max(input_size // 2, num_classes * 2)]

    # Build complete layer sizes
    layer_sizes = [input_size] + hidden_sizes + [num_classes]

    # Create base network
    components = compose_deep_network(layer_sizes)

    # Add Sigmoid activation for classification
    components.append(Sigmoid())

    return components
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "regression-network", "solution": true}
def create_regression_network(input_size: int, output_size: int = 1, hidden_sizes: List[int] = None) -> List:
    """
    Create a regression network with no output activation.

    Args:
        input_size: Number of input features
        output_size: Number of output values (default: 1)
        hidden_sizes: List of hidden layer sizes (optional)

    Returns:
        List of network components with no output activation for regression

    TODO: Create network with no output activation for regression

    APPROACH:
    1. Use provided hidden_sizes or create reasonable default
    2. Build layer_sizes list and create network
    3. Do NOT add output activation (regression predicts raw values)
    4. Return component list

    EXAMPLE:
    >>> components = create_regression_network(4, 1, [8, 4])
    >>> # Creates: Linear(4,8), ReLU(), Linear(8,4), ReLU(), Linear(4,1)
    >>> # No output activation for regression

    HINTS:
    - Default hidden_sizes could be [input_size, input_size // 2]
    - Use compose_deep_network directly (it doesn't add output activation)
    - Don't add any activation after the final layer
    """
    ### BEGIN SOLUTION
    # Handle default hidden sizes
    if hidden_sizes is None:
        hidden_sizes = [input_size, max(input_size // 2, output_size * 2)]

    # Build complete layer sizes
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    # Create network (compose_deep_network doesn't add output activation)
    components = compose_deep_network(layer_sizes)

    return components
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Network Architecture Patterns
Test specialized network architectures for different tasks
"""

# %%
def test_unit_network_patterns():
    """Test different network architecture patterns"""
    print("🔬 Unit Test: Network Architecture Patterns...")

    # Test 1: Classification network
    classifier = create_classifier_network(784, 10, [128, 64])

    # Should end with Sigmoid for classification
    assert isinstance(classifier[-1], Sigmoid), "Classifier should end with Sigmoid"

    # Test forward pass
    x_class = Tensor(np.random.randn(1, 784))
    y_class = forward_pass_deep(x_class, classifier)

    assert y_class.shape == (1, 10), "Classifier should output correct number of classes"
    # Note: We can't easily test that output is in [0,1] without more sophisticated sigmoid implementation

    # Test 2: Regression network
    regressor = create_regression_network(4, 1, [8, 4])

    # Should NOT end with activation
    assert not isinstance(regressor[-1], (Sigmoid, ReLU, Tanh)), "Regressor should not end with activation"
    assert isinstance(regressor[-1], Linear), "Regressor should end with Linear layer"

    # Test forward pass
    x_reg = Tensor(np.random.randn(3, 4))
    y_reg = forward_pass_deep(x_reg, regressor)

    assert y_reg.shape == (3, 1), "Regressor should output correct shape"

    # Test 3: Multi-output regression
    multi_regressor = create_regression_network(6, 3, [10, 5])
    x_multi = Tensor(np.random.randn(2, 6))
    y_multi = forward_pass_deep(x_multi, multi_regressor)

    assert y_multi.shape == (2, 3), "Multi-output regressor should work"

    # Test 4: Default hidden sizes
    default_classifier = create_classifier_network(20, 5)  # No hidden_sizes specified
    x_default = Tensor(np.random.randn(1, 20))
    y_default = forward_pass_deep(x_default, default_classifier)

    assert y_default.shape == (1, 5), "Default classifier should work"

    print("✅ Network architecture patterns work correctly!")

test_unit_network_patterns()

# %%
def test_module():
    """Run all module tests to verify complete implementation"""
    print("🧪 Running all Network module tests...")

    test_unit_two_layer_composition()
    test_unit_forward_pass()
    test_unit_deep_network()
    test_unit_deep_forward()
    test_unit_parameter_counting()
    test_unit_network_patterns()

    print("✅ All Network module tests passed! Manual network composition complete.")

# %% [markdown]
"""
## 🔍 Systems Analysis

Now that your network implementations are complete and tested, let's analyze their systems behavior:

### Performance and Memory Characteristics

Understanding how networks scale with size and depth is crucial for building real ML systems.
"""

# %%
def measure_network_scaling():
    """
    📊 SYSTEMS MEASUREMENT: Network Scaling Analysis

    Measure how network complexity affects performance and memory usage.
    """
    print("📊 NETWORK SCALING MEASUREMENT")
    print("Testing how network depth and width affect computational complexity...")

    import time

    # Test different network architectures
    architectures = [
        ("Narrow-Deep", [10, 8, 6, 4, 2]),
        ("Wide-Shallow", [10, 50, 2]),
        ("Balanced", [10, 20, 10, 2]),
        ("Very Deep", [10, 8, 6, 5, 4, 3, 2])
    ]

    batch_size = 100
    num_trials = 10

    for name, layer_sizes in architectures:
        print(f"\n🔧 Testing {name} architecture: {layer_sizes}")

        # Create network
        components = compose_deep_network(layer_sizes)
        total_params, breakdown = count_network_parameters(components)

        # Measure forward pass time
        x = Tensor(np.random.randn(batch_size, layer_sizes[0]))

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            y = forward_pass_deep(x, components)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times) * 1000  # Convert to milliseconds

        print(f"  Parameters: {total_params:,}")
        print(f"  Layers: {len([c for c in components if isinstance(c, Linear)])}")
        print(f"  Forward pass: {avg_time:.2f}ms (batch={batch_size})")
        print(f"  Time per sample: {avg_time/batch_size:.3f}ms")

        # Memory analysis
        total_weights = sum(layer.weights.data.size for layer in components if isinstance(layer, Linear))
        total_biases = sum(layer.bias.data.size for layer in components if isinstance(layer, Linear))
        memory_mb = (total_weights + total_biases) * 4 / 1024 / 1024  # float32 = 4 bytes

        print(f"  Memory usage: {memory_mb:.2f} MB")

    print(f"\n💡 SCALING INSIGHTS:")
    print(f"   • Depth vs Width: More layers = more sequential computation")
    print(f"   • Parameter count dominates memory usage")
    print(f"   • Batch processing amortizes per-sample overhead")
    print(f"   • Network architecture significantly impacts performance")

# Run the measurement
measure_network_scaling()

# %%
def measure_parameter_scaling():
    """
    💾 SYSTEMS MEASUREMENT: Parameter Memory Analysis

    Understand how parameter count scales with network size.
    """
    print("💾 PARAMETER MEMORY MEASUREMENT")
    print("Analyzing parameter scaling patterns...")

    # Test parameter scaling with width
    print("\n📏 Width Scaling (2-layer networks):")
    widths = [10, 50, 100, 200, 500]

    for width in widths:
        components = compose_deep_network([10, width, 5])
        total_params, _ = count_network_parameters(components)
        memory_mb = total_params * 4 / 1024 / 1024

        print(f"  Width {width:3d}: {total_params:,} params, {memory_mb:.2f} MB")

    # Test parameter scaling with depth
    print("\n📏 Depth Scaling (constant width=20):")
    depths = [2, 4, 6, 8, 10]

    for depth in depths:
        layer_sizes = [20] * (depth + 1)  # depth+1 layer sizes for depth layers
        layer_sizes[-1] = 5  # Output size
        components = compose_deep_network(layer_sizes)
        total_params, _ = count_network_parameters(components)
        memory_mb = total_params * 4 / 1024 / 1024

        print(f"  Depth {depth:2d}: {total_params:,} params, {memory_mb:.2f} MB")

    print(f"\n💡 PARAMETER INSIGHTS:")
    print(f"   • Width scaling: Quadratic growth O(W²) for layer connections")
    print(f"   • Depth scaling: Linear growth O(D) for constant width")
    print(f"   • First and last layers often dominate parameter count")
    print(f"   • Memory grows linearly with parameter count")

# Run the measurement
measure_parameter_scaling()

# %%
def measure_batch_processing():
    """
    📦 SYSTEMS MEASUREMENT: Batch Processing Efficiency

    Analyze how batch size affects computational efficiency.
    """
    print("📦 BATCH PROCESSING MEASUREMENT")
    print("Testing computational efficiency across batch sizes...")

    import time

    # Create test network
    components = compose_deep_network([100, 50, 25, 10])

    batch_sizes = [1, 10, 50, 100, 500, 1000]
    num_trials = 5

    print("\nBatch Size | Total Time | Time/Sample | Throughput")
    print("-" * 55)

    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size, 100))

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            y = forward_pass_deep(x, components)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times) * 1000  # milliseconds
        time_per_sample = avg_time / batch_size
        throughput = 1000 / time_per_sample  # samples per second

        print(f"{batch_size:9d} | {avg_time:9.2f}ms | {time_per_sample:10.3f}ms | {throughput:8.0f} samples/s")

    print(f"\n💡 BATCH PROCESSING INSIGHTS:")
    print(f"   • Larger batches amortize per-batch overhead")
    print(f"   • Time per sample decreases with batch size")
    print(f"   • Throughput increases significantly with batching")
    print(f"   • Memory usage scales linearly with batch size")

# Run the measurement
measure_batch_processing()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented manual network composition, let's connect this to broader ML systems principles:
"""

# %% [markdown]
"""
### Question 1: Memory and Performance Analysis

In your `count_network_parameters()` function, you discovered that a 3-layer network with sizes [784, 128, 64, 10] has about 109,000 parameters.

When you tested this network with different batch sizes, you saw that processing time per sample decreased with larger batches. Analyze the memory and computational trade-offs:

**Your Implementation Analysis:**
- How does the parameter memory (109K parameters × 4 bytes = ~436KB) compare to activation memory for different batch sizes?
- Why does your `forward_pass_deep()` function become more efficient per sample with larger batches?
- At what batch size would activation memory exceed parameter memory for this network?

**Systems Engineering Question:**
If you needed to deploy this network on a device with only 1MB of available memory, what modifications to your network composition functions would you implement to stay within memory constraints while maintaining reasonable accuracy?

Think about: Parameter sharing strategies, layer width reduction, depth vs width trade-offs
"""

# %% [markdown]
"""
### Question 2: Architecture Scaling Analysis

Your `compose_deep_network()` function can create networks of arbitrary depth and width. You measured that very deep networks (10+ layers) have linear parameter growth but may suffer from other issues.

**Implementation Scaling Analysis:**
- In your deep network experiments, which architecture pattern (narrow-deep vs wide-shallow) was more computationally efficient?
- How would you modify your `forward_pass_deep()` function to handle networks with 100+ layers efficiently?
- What bottlenecks would emerge in your current manual composition approach for very large networks?

**Production Engineering Question:**
Design a modification to your current network composition system that could handle production-scale networks (1000+ layers, millions of parameters) while maintaining the educational clarity of manual composition.

Think about: Memory checkpointing, activation recomputation, gradient accumulation patterns
"""

# %% [markdown]
"""
### Question 3: Integration and Modularity Analysis

Your manual network composition approach gives you complete control over layer ordering and activation placement. However, you've seen that composing networks manually becomes complex for large architectures.

**Integration Analysis:**
- How would you extend your current `create_classifier_network()` and `create_regression_network()` functions to support more complex architectures like residual connections?
- What interface changes to your component system would be needed to handle branching network topologies?
- How does manual composition compare to automated composition in terms of debugging and understanding?

**Systems Architecture Question:**
Design a hybrid approach that maintains the educational benefits of your manual composition while providing the convenience of automated network building for complex architectures. What abstractions would you introduce?

Think about: Component interfaces, graph representations, debugging visibility
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Networks - Manual Composition Mastery

Congratulations! You've successfully implemented manual network composition that forms the foundation of all neural network architectures:

### What You've Accomplished
✅ **Manual Network Composition**: Built 150+ lines of network architecture code with step-by-step layer composition
✅ **Forward Pass Logic**: Implemented data flow through networks of arbitrary depth and complexity
✅ **Parameter Analysis**: Created comprehensive parameter counting and memory analysis systems
✅ **Architecture Patterns**: Built specialized networks for classification, regression, and custom tasks
✅ **Systems Understanding**: Analyzed scaling behavior, memory usage, and computational complexity

### Key Learning Outcomes
- **Network Architecture**: Understanding how layers compose into intelligent systems through manual control
- **Data Flow Principles**: Mastery of tensor shape transformations through network layers
- **Parameter Management**: Deep insight into memory requirements and computational complexity
- **Performance Characteristics**: Knowledge of how network depth and width affect efficiency

### Mathematical Foundations Mastered
- **Composition Functions**: f(g(h(x))) = network(x) through sequential application
- **Parameter Scaling**: O(input_size × output_size) per layer, O(depth) for network
- **Memory Complexity**: Linear scaling with parameters plus O(batch_size × max_layer_width) for activations

### Professional Skills Developed
- **Manual Architecture Design**: Building networks layer-by-layer with complete understanding
- **Systems Analysis**: Measuring and optimizing network performance characteristics
- **Memory Engineering**: Understanding parameter vs activation memory trade-offs
- **Performance Optimization**: Batch processing and computational efficiency analysis

### Ready for Advanced Applications
Your manual network composition now enables:
- **Custom Architectures**: Build any network topology with complete understanding
- **Performance Analysis**: Measure and optimize network computational characteristics
- **Memory Management**: Predict and control network memory requirements
- **Educational Foundation**: Deep understanding before automated composition tools

### Connection to Real ML Systems
Your implementation mirrors production patterns:
- **PyTorch**: Your manual composition matches nn.Sequential() internal behavior
- **TensorFlow**: Similar to tf.keras.Sequential() layer-by-layer construction
- **Industry Standard**: Manual composition used for custom architectures and research

### Next Steps
1. **Export your module**: `tito module complete 04_networks`
2. **Validate integration**: `tito test --module networks`
3. **Explore automated composition**: Your foundation enables understanding Sequential in Module 05
4. **Ready for Module 05**: Linear Networks with automated composition tools

**🚀 Achievement Unlocked**: Your manual network composition mastery provides the deep understanding needed for building automated ML frameworks. You've learned to think like a neural network architect!
"""

# %%
if __name__ == "__main__":
    # Run all tests to validate complete implementation
    test_module()

    # Display completion message
    print("\n" + "="*60)
    print("🎯 MODULE 04 (NETWORKS) COMPLETE!")
    print("📈 Progress: Manual Network Composition ✓")
    print("🔥 Next up: Module 05 - Automated Linear Networks!")
    print("💪 You're building real ML architecture understanding!")
    print("="*60)
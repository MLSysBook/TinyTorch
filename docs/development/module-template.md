# Module Template: "Where This Code Lives" Section

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/{module_name}/{module_name}_dev.py`  
**Building Side:** Code exports to `tinytorch.core.{destination}`

```python
# Final package structure:
from tinytorch.core.{destination} import {exported_classes}
from tinytorch.core.tensor import Tensor
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like industry frameworks
- **Consistency:** Related functionality grouped together

## Template Variables

Replace these placeholders in each module:

- `{module_name}`: The module directory name (e.g., "tensor", "layers", "cnn")
- `{destination}`: Where the code exports in the final package (e.g., "tensor", "layers", "activations")
- `{exported_classes}`: The main classes/functions being exported (e.g., "Tensor", "Dense, Conv2D", "ReLU, Sigmoid")

## Examples

### Tensor Module
```python
# Learning Side: modules/tensor/tensor_dev.py
# Building Side: tinytorch.core.tensor
from tinytorch.core.tensor import Tensor
```

### Layers Module  
```python
# Learning Side: modules/layers/layers_dev.py
# Building Side: tinytorch.core.layers
from tinytorch.core.layers import Dense, Conv2D
```

### CNN Module
```python
# Learning Side: modules/cnn/cnn_dev.py
# Building Side: tinytorch.core.layers (Conv2D lives with Dense)
from tinytorch.core.layers import Dense, Conv2D
```

## Usage Instructions

1. Copy this template section into each module's `*_dev.py` file
2. Replace the template variables with module-specific values
3. Update the `#| default_exp` directive to match the destination
4. Ensure the exported classes match what's actually being exported 
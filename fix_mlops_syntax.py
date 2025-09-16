#!/usr/bin/env python3
"""Fix syntax errors in mlops_dev.py"""

import re

# Read the file
with open('modules/source/15_mlops/mlops_dev.py', 'r') as f:
    content = f.read()

# Fix the malformed function definitions
# Pattern: def if __name__ == "__main__":\n    function_name():
pattern = r'def if __name__ == "__main__":\n    (\w+)\(\):'
replacement = r'def \1():'

content = re.sub(pattern, replacement, content)

# Write back
with open('modules/source/15_mlops/mlops_dev.py', 'w') as f:
    f.write(content)

print("✅ Fixed syntax errors in mlops_dev.py")
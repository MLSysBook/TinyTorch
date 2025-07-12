# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Assignment 0: Setup - TinyTorch Development Environment (INSTRUCTOR VERSION)

This is the instructor solution version showing how solutions are filled in.
"""

# %%
#| default_exp core.utils

# %%
#| export
# Required imports for TinyTorch utilities
import sys
import platform
from datetime import datetime
import os
from pathlib import Path

# %% [markdown]
"""
## Problem 1: Hello Function (10 points)

Write a function that displays a welcome message for TinyTorch.
"""

# %% nbgrader={"grade": false, "grade_id": "hello_function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def hello_tinytorch():
    """
    Display a welcome message for TinyTorch.
    
    This function should:
    1. Try to load ASCII art from 'tinytorch_flame.txt' if it exists
    2. If the file doesn't exist, display a simple text banner
    3. Print "TinyTorch" and "Build ML Systems from Scratch!"
    4. Handle any exceptions gracefully
    """
    ### BEGIN SOLUTION
    # YOUR CODE HERE
    raise NotImplementedError()
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "add_function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
        
    Returns:
        Sum of a and b
    """
    ### BEGIN SOLUTION
    # YOUR CODE HERE
    raise NotImplementedError()
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "systeminfo_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SystemInfo:
    """
    A class for collecting and displaying system information.
    """
    
    def __init__(self):
        """
        Initialize the SystemInfo object.
        Collect Python version, platform, and machine information.
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
    
    def __str__(self):
        """
        Return a formatted string representation of system information.
        Format: "Python X.Y.Z on Platform (Architecture)"
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
    
    def is_compatible(self):
        """
        Check if the Python version is compatible (>= 3.8).
        Returns True if compatible, False otherwise.
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "developer_profile_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class DeveloperProfile:
    """
    A class representing a developer profile.
    """
    
    def __init__(self, name="Student", email="student@example.com", affiliation="TinyTorch Community", specialization="ML Systems"):
        """
        Initialize a developer profile.
        
        Args:
            name: Developer's name
            email: Developer's email
            affiliation: Developer's affiliation or organization
            specialization: Developer's area of specialization
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
    
    def __str__(self):
        """
        Return a basic string representation of the developer.
        Format: "Name (email)"
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
    
    def get_signature(self):
        """
        Return a formatted signature for the developer.
        Should include name, affiliation, and specialization.
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
    
    def get_profile_info(self):
        """
        Return comprehensive profile information as a dictionary.
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION 
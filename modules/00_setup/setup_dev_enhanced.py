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
# Module 0: Setup - Tiny🔥Torch Development Workflow (Enhanced for NBGrader)

Welcome to TinyTorch! This module teaches you the development workflow you'll use throughout the course.

## Learning Goals
- Understand the nbdev notebook-to-Python workflow
- Write your first TinyTorch code
- Run tests and use the CLI tools
- Get comfortable with the development rhythm

## The TinyTorch Development Cycle

1. **Write code** in this notebook using `#| export` 
2. **Export code** with `python bin/tito.py sync --module setup`
3. **Run tests** with `python bin/tito.py test --module setup`
4. **Check progress** with `python bin/tito.py info`

## New: NBGrader Integration
This module is also configured for automated grading with **100 points total**:
- Basic Functions: 30 points
- SystemInfo Class: 35 points  
- DeveloperProfile Class: 35 points

Let's get started!
"""

# %%
#| default_exp core.utils

# %%
#| export
# Setup imports and environment
import sys
import platform
from datetime import datetime
import os
from pathlib import Path

print("🔥 TinyTorch Development Environment")
print(f"Python {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
"""
## Step 1: Basic Functions (30 Points)

Let's start with simple functions that form the foundation of TinyTorch.
"""

# %%
#| export
def hello_tinytorch():
    """
    A simple hello world function for TinyTorch.
    
    Display TinyTorch ASCII art and welcome message.
    Load the flame art from tinytorch_flame.txt file with graceful fallback.
    """
    #| exercise_start
    #| hint: Load ASCII art from tinytorch_flame.txt file with graceful fallback
    #| solution_test: Function should display ASCII art and welcome message
    #| difficulty: easy
    #| points: 10
    
    ### BEGIN SOLUTION
    try:
        # Get the directory containing this file
        current_dir = Path(__file__).parent
        art_file = current_dir / "tinytorch_flame.txt"
        
        if art_file.exists():
            with open(art_file, 'r') as f:
                ascii_art = f.read()
            print(ascii_art)
            print("Tiny🔥Torch")
            print("Build ML Systems from Scratch!")
        else:
            print("🔥 TinyTorch 🔥")
            print("Build ML Systems from Scratch!")
    except NameError:
        # Handle case when running in notebook where __file__ is not defined
        try:
            art_file = Path(os.getcwd()) / "tinytorch_flame.txt"
            if art_file.exists():
                with open(art_file, 'r') as f:
                    ascii_art = f.read()
                print(ascii_art)
                print("Tiny🔥Torch")
                print("Build ML Systems from Scratch!")
            else:
                print("🔥 TinyTorch 🔥")
                print("Build ML Systems from Scratch!")
        except:
            print("🔥 TinyTorch 🔥")
            print("Build ML Systems from Scratch!")
    ### END SOLUTION
    
    #| exercise_end

def add_numbers(a, b):
    """
    Add two numbers together.
    
    This is the foundation of all mathematical operations in ML.
    """
    #| exercise_start
    #| hint: Use the + operator to add two numbers
    #| solution_test: add_numbers(2, 3) should return 5
    #| difficulty: easy
    #| points: 10
    
    ### BEGIN SOLUTION
    return a + b
    ### END SOLUTION
    
    #| exercise_end

# %% [markdown]
"""
## Hidden Tests: Basic Functions (10 Points)

These tests verify the basic functionality and award points automatically.
"""

# %%
### BEGIN HIDDEN TESTS
def test_hello_tinytorch():
    """Test hello_tinytorch function (5 points)"""
    import io
    import sys
    
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        hello_tinytorch()
        output = captured_output.getvalue()
        
        # Check that some output was produced
        assert len(output) > 0, "Function should produce output"
        assert "TinyTorch" in output, "Output should contain 'TinyTorch'"
        
    finally:
        sys.stdout = sys.__stdout__

def test_add_numbers():
    """Test add_numbers function (5 points)"""
    # Test basic addition
    assert add_numbers(2, 3) == 5, "add_numbers(2, 3) should return 5"
    assert add_numbers(0, 0) == 0, "add_numbers(0, 0) should return 0"
    assert add_numbers(-1, 1) == 0, "add_numbers(-1, 1) should return 0"
    
    # Test with floats
    assert add_numbers(2.5, 3.5) == 6.0, "add_numbers(2.5, 3.5) should return 6.0"
    
    # Test with negative numbers
    assert add_numbers(-5, -3) == -8, "add_numbers(-5, -3) should return -8"
### END HIDDEN TESTS

# %% [markdown]
"""
## Step 2: SystemInfo Class (35 Points)

Let's create a class that collects and displays system information.
"""

# %%
#| export
class SystemInfo:
    """
    Simple system information class.
    
    Collects and displays Python version, platform, and machine information.
    """
    
    def __init__(self):
        """
        Initialize system information collection.
        
        Collect Python version, platform, and machine information.
        """
        #| exercise_start
        #| hint: Use sys.version_info, platform.system(), and platform.machine()
        #| solution_test: Should store Python version, platform, and machine info
        #| difficulty: medium
        #| points: 15
        
        ### BEGIN SOLUTION
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.machine = platform.machine()
        ### END SOLUTION
        
        #| exercise_end
    
    def __str__(self):
        """
        Return human-readable system information.
        
        Format system info as a readable string.
        """
        #| exercise_start
        #| hint: Format as "Python X.Y on Platform (Machine)"
        #| solution_test: Should return formatted string with version and platform
        #| difficulty: easy
        #| points: 10
        
        ### BEGIN SOLUTION
        return f"Python {self.python_version.major}.{self.python_version.minor} on {self.platform} ({self.machine})"
        ### END SOLUTION
        
        #| exercise_end
    
    def is_compatible(self):
        """
        Check if system meets minimum requirements.
        
        Check if Python version is >= 3.8
        """
        #| exercise_start
        #| hint: Compare self.python_version with (3, 8) tuple
        #| solution_test: Should return True for Python >= 3.8
        #| difficulty: medium
        #| points: 10
        
        ### BEGIN SOLUTION
        return self.python_version >= (3, 8)
        ### END SOLUTION
        
        #| exercise_end

# %% [markdown]
"""
## Hidden Tests: SystemInfo Class (35 Points)

These tests verify the SystemInfo class implementation.
"""

# %%
### BEGIN HIDDEN TESTS
def test_systeminfo_init():
    """Test SystemInfo initialization (15 points)"""
    info = SystemInfo()
    
    # Check that attributes are set
    assert hasattr(info, 'python_version'), "Should have python_version attribute"
    assert hasattr(info, 'platform'), "Should have platform attribute"
    assert hasattr(info, 'machine'), "Should have machine attribute"
    
    # Check types
    assert isinstance(info.python_version, tuple), "python_version should be tuple"
    assert isinstance(info.platform, str), "platform should be string"
    assert isinstance(info.machine, str), "machine should be string"
    
    # Check values are reasonable
    assert len(info.python_version) >= 2, "python_version should have at least major.minor"
    assert len(info.platform) > 0, "platform should not be empty"

def test_systeminfo_str():
    """Test SystemInfo string representation (10 points)"""
    info = SystemInfo()
    str_repr = str(info)
    
    # Check that the string contains expected elements
    assert "Python" in str_repr, "String should contain 'Python'"
    assert str(info.python_version.major) in str_repr, "String should contain major version"
    assert str(info.python_version.minor) in str_repr, "String should contain minor version"
    assert info.platform in str_repr, "String should contain platform"
    assert info.machine in str_repr, "String should contain machine"

def test_systeminfo_compatibility():
    """Test SystemInfo compatibility check (10 points)"""
    info = SystemInfo()
    compatibility = info.is_compatible()
    
    # Check that it returns a boolean
    assert isinstance(compatibility, bool), "is_compatible should return boolean"
    
    # Check that it's reasonable (we're running Python >= 3.8)
    assert compatibility == True, "Should return True for Python >= 3.8"
### END HIDDEN TESTS

# %% [markdown]
"""
## Step 3: DeveloperProfile Class (35 Points)

Let's create a personalized developer profile system.
"""

# %%
#| export
class DeveloperProfile:
    """
    Developer profile for personalizing TinyTorch experience.
    
    Stores and displays developer information with ASCII art.
    """
    
    @staticmethod
    def _load_default_flame():
        """
        Load the default TinyTorch flame ASCII art from file.
        
        Load from tinytorch_flame.txt with graceful fallback.
        """
        #| exercise_start
        #| hint: Use Path and file operations with try/except for fallback
        #| solution_test: Should load ASCII art from file or provide fallback
        #| difficulty: hard
        #| points: 5
        
        ### BEGIN SOLUTION
        try:
            # Try to get the directory of the current file
            try:
                current_dir = os.path.dirname(__file__)
            except NameError:
                current_dir = os.getcwd()
            
            flame_path = os.path.join(current_dir, 'tinytorch_flame.txt')
            
            with open(flame_path, 'r', encoding='utf-8') as f:
                flame_art = f.read()
            
            return f"""{flame_art}
                    
                    Tiny🔥Torch
            Build ML Systems from Scratch!
            """
        except (FileNotFoundError, IOError):
            # Fallback to simple flame if file not found
            return """
    🔥 TinyTorch Developer 🔥
         .  .  .  .  .  .
        .    .  .  .  .   .
       .  .    .  .  .  .  .
      .  .  .    .  .  .  .  .
     .  .  .  .    .  .  .  .  .
    .  .  .  .  .    .  .  .  .  .
   .  .  .  .  .  .    .  .  .  .  .
  .  .  .  .  .  .  .    .  .  .  .  .
 .  .  .  .  .  .  .  .    .  .  .  .  .
.  .  .  .  .  .  .  .  .    .  .  .  .  .
 \\  \\  \\  \\  \\  \\  \\  \\  \\  /  /  /  /  /  /
  \\  \\  \\  \\  \\  \\  \\  \\  /  /  /  /  /  /
   \\  \\  \\  \\  \\  \\  \\  /  /  /  /  /  /
    \\  \\  \\  \\  \\  \\  /  /  /  /  /  /
     \\  \\  \\  \\  \\  /  /  /  /  /  /
      \\  \\  \\  \\  /  /  /  /  /
       \\  \\  \\  /  /  /  /  /  /
        \\  \\  /  /  /  /  /  /
         \\  /  /  /  /  /  /
          \\/  /  /  /  /  /
           \\/  /  /  /  /
            \\/  /  /  /
             \\/  /  /
              \\/  /
               \\/
                    
                    Tiny🔥Torch
            Build ML Systems from Scratch!
            """
        ### END SOLUTION
        
        #| exercise_end
    
    def __init__(self, name="Vijay Janapa Reddi", affiliation="Harvard University", 
                 email="vj@eecs.harvard.edu", github_username="profvjreddi", ascii_art=None):
        """
        Initialize developer profile.
        
        Store developer information with sensible defaults.
        """
        #| exercise_start
        #| hint: Store all parameters as instance attributes, use _load_default_flame for ascii_art if None
        #| solution_test: Should store all developer information
        #| difficulty: medium
        #| points: 15
        
        ### BEGIN SOLUTION
        self.name = name
        self.affiliation = affiliation
        self.email = email
        self.github_username = github_username
        self.ascii_art = ascii_art or self._load_default_flame()
        ### END SOLUTION
        
        #| exercise_end
    
    def __str__(self):
        """
        Return formatted developer information.
        
        Format as professional signature.
        """
        #| exercise_start
        #| hint: Format as "👨‍💻 Name | Affiliation | @username"
        #| solution_test: Should return formatted string with name, affiliation, and username
        #| difficulty: easy
        #| points: 5
        
        ### BEGIN SOLUTION
        return f"👨‍💻 {self.name} | {self.affiliation} | @{self.github_username}"
        ### END SOLUTION
        
        #| exercise_end
    
    def get_signature(self):
        """
        Get a short signature for code headers.
        
        Return concise signature like "Built by Name (@github)"
        """
        #| exercise_start
        #| hint: Format as "Built by Name (@username)"
        #| solution_test: Should return signature with name and username
        #| difficulty: easy
        #| points: 5
        
        ### BEGIN SOLUTION
        return f"Built by {self.name} (@{self.github_username})"
        ### END SOLUTION
        
        #| exercise_end
    
    def get_ascii_art(self):
        """
        Get ASCII art for the profile.
        
        Return custom ASCII art or default flame.
        """
        #| exercise_start
        #| hint: Simply return self.ascii_art
        #| solution_test: Should return stored ASCII art
        #| difficulty: easy
        #| points: 5
        
        ### BEGIN SOLUTION
        return self.ascii_art
        ### END SOLUTION
        
        #| exercise_end

# %% [markdown]
"""
## Hidden Tests: DeveloperProfile Class (35 Points)

These tests verify the DeveloperProfile class implementation.
"""

# %%
### BEGIN HIDDEN TESTS
def test_developer_profile_init():
    """Test DeveloperProfile initialization (15 points)"""
    # Test with defaults
    profile = DeveloperProfile()
    
    assert hasattr(profile, 'name'), "Should have name attribute"
    assert hasattr(profile, 'affiliation'), "Should have affiliation attribute"
    assert hasattr(profile, 'email'), "Should have email attribute"
    assert hasattr(profile, 'github_username'), "Should have github_username attribute"
    assert hasattr(profile, 'ascii_art'), "Should have ascii_art attribute"
    
    # Check default values
    assert profile.name == "Vijay Janapa Reddi", "Should have default name"
    assert profile.affiliation == "Harvard University", "Should have default affiliation"
    assert profile.email == "vj@eecs.harvard.edu", "Should have default email"
    assert profile.github_username == "profvjreddi", "Should have default username"
    assert profile.ascii_art is not None, "Should have ASCII art"
    
    # Test with custom values
    custom_profile = DeveloperProfile(
        name="Test User",
        affiliation="Test University",
        email="test@test.com",
        github_username="testuser",
        ascii_art="Custom Art"
    )
    
    assert custom_profile.name == "Test User", "Should store custom name"
    assert custom_profile.affiliation == "Test University", "Should store custom affiliation"
    assert custom_profile.email == "test@test.com", "Should store custom email"
    assert custom_profile.github_username == "testuser", "Should store custom username"
    assert custom_profile.ascii_art == "Custom Art", "Should store custom ASCII art"

def test_developer_profile_str():
    """Test DeveloperProfile string representation (5 points)"""
    profile = DeveloperProfile()
    str_repr = str(profile)
    
    assert "👨‍💻" in str_repr, "Should contain developer emoji"
    assert profile.name in str_repr, "Should contain name"
    assert profile.affiliation in str_repr, "Should contain affiliation"
    assert f"@{profile.github_username}" in str_repr, "Should contain @username"

def test_developer_profile_signature():
    """Test DeveloperProfile signature (5 points)"""
    profile = DeveloperProfile()
    signature = profile.get_signature()
    
    assert "Built by" in signature, "Should contain 'Built by'"
    assert profile.name in signature, "Should contain name"
    assert f"@{profile.github_username}" in signature, "Should contain @username"

def test_developer_profile_ascii_art():
    """Test DeveloperProfile ASCII art (5 points)"""
    profile = DeveloperProfile()
    ascii_art = profile.get_ascii_art()
    
    assert isinstance(ascii_art, str), "ASCII art should be string"
    assert len(ascii_art) > 0, "ASCII art should not be empty"
    assert "TinyTorch" in ascii_art, "ASCII art should contain 'TinyTorch'"

def test_default_flame_loading():
    """Test default flame loading (5 points)"""
    flame_art = DeveloperProfile._load_default_flame()
    
    assert isinstance(flame_art, str), "Flame art should be string"
    assert len(flame_art) > 0, "Flame art should not be empty"
    assert "TinyTorch" in flame_art, "Flame art should contain 'TinyTorch'"
### END HIDDEN TESTS

# %% [markdown]
"""
## Test Your Implementation

Run these cells to test your implementation:
"""

# %%
# Test basic functions
print("Testing Basic Functions:")
try:
    hello_tinytorch()
    print(f"2 + 3 = {add_numbers(2, 3)}")
    print("✅ Basic functions working!")
except Exception as e:
    print(f"❌ Error: {e}")

# %%
# Test SystemInfo
print("\nTesting SystemInfo:")
try:
    info = SystemInfo()
    print(f"System: {info}")
    print(f"Compatible: {info.is_compatible()}")
    print("✅ SystemInfo working!")
except Exception as e:
    print(f"❌ Error: {e}")

# %%
# Test DeveloperProfile
print("\nTesting DeveloperProfile:")
try:
    profile = DeveloperProfile()
    print(f"Profile: {profile}")
    print(f"Signature: {profile.get_signature()}")
    print("✅ DeveloperProfile working!")
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
"""
## 🎉 Module Complete!

You've successfully implemented the setup module with **100 points total**:

### Point Breakdown:
- **hello_tinytorch()**: 10 points
- **add_numbers()**: 10 points  
- **Basic function tests**: 10 points
- **SystemInfo.__init__()**: 15 points
- **SystemInfo.__str__()**: 10 points
- **SystemInfo.is_compatible()**: 10 points
- **DeveloperProfile.__init__()**: 15 points
- **DeveloperProfile methods**: 20 points

### What's Next:
1. Export your code: `tito sync --module setup`
2. Run tests: `tito test --module setup`
3. Generate assignment: `tito nbgrader generate --module setup`
4. Move to Module 1: Tensor!

### NBGrader Features:
- ✅ Automatic grading with 100 points
- ✅ Partial credit for each component
- ✅ Hidden tests for comprehensive validation
- ✅ Immediate feedback for students
- ✅ Compatible with existing TinyTorch workflow

Happy building! 🔥
""" 
# %% [markdown]
# # TinyTorch Setup Module
# 
# Welcome to TinyTorch! This is your first module in the Machine Learning Systems course.
# 
# This module simply displays our beautiful ASCII art to get you started.

# %%
import os
from pathlib import Path

def hello_tinytorch():
    """Display the TinyTorch ASCII art"""
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

# %% [markdown]
# ## Test the Setup
# 
# Let's run our hello function to see the ASCII art:

# %%
if __name__ == "__main__":
    hello_tinytorch()

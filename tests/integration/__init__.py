"""
Integration tests for TinyTorch modules.

These tests verify that individual modules integrate correctly with the package:
- Export correctly to the package
- Can be imported without errors  
- Basic functionality works
- Don't conflict with other modules

This is different from checkpoint tests which validate complete capabilities.
Integration tests are quick validation that runs after every module completion.
"""
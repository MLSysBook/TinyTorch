"""
Setup Module Reference Solution

This file contains the reference implementation for the setup module.
Instructors can use this to verify student work or provide hints.

Student Task: Implement hello_tinytorch() function in tinytorch/core/utils.py
"""

def hello_tinytorch() -> str:
    """
    Return a greeting message for new TinyTorch users.
    
    This function serves as the "hello world" for the TinyTorch system.
    It introduces students to the development workflow and testing process.
    
    Returns:
        A welcoming message string that includes:
        - Welcoming content
        - TinyTorch branding (🔥 emoji)
        - Encouraging message about building ML systems
    """
    return "🔥 Welcome to TinyTorch! Ready to build ML systems from scratch! 🔥"


# Example usage and testing
if __name__ == "__main__":
    # Test the function
    result = hello_tinytorch()
    print(f"Function result: {result}")
    
    # Verify requirements
    checks = []
    checks.append(isinstance(result, str))
    checks.append(len(result.strip()) > 0)
    checks.append('🔥' in result)
    checks.append(any(word in result.lower() for word in ['welcome', 'hello', 'tinytorch', 'ready']))
    
    print("\nRequirements check:")
    print(f"  ✅ Returns string: {checks[0]}")
    print(f"  ✅ Non-empty: {checks[1]}")
    print(f"  ✅ Contains 🔥: {checks[2]}")
    print(f"  ✅ Welcoming content: {checks[3]}")
    
    if all(checks):
        print("\n🎉 All requirements met!")
    else:
        print("\n❌ Some requirements not met.") 
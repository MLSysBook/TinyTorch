# Development Workflow Rules

## Branch-First Development
- **Always create a branch** for any work - never work directly on main
- **Branch naming**: `feature/description`, `fix/issue`, `refactor/component`
- **Remind user** to create branches if they forget

## Work Process
1. **Plan**: Define what changes are needed and why
2. **Reason**: Think through the approach and potential issues  
3. **Test**: Write tests to verify success before implementing
4. **Execute**: Implement changes in a new Git branch
5. **Verify**: Run all tests and ensure everything works
6. **Merge**: Only merge when fully tested and verified

## Testing Standards
- **Always use pytest** for all tests
- **Test before implementing** - write tests that define success
- **Test after implementing** - verify everything works
- **Test edge cases** and error conditions

## Documentation
- **Prefer Quarto** for documentation generation
- **Keep rules short** and actionable
- **Update rules** as patterns emerge

This ensures quality, traceability, and prevents breaking main branch. 
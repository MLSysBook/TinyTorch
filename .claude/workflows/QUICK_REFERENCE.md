# 🚀 TinyTorch Agent Workflow Quick Reference

## Website Updates
**Triggers**: "update website", "fix docs", "add content", "change page"
```
Content Strategist → Website Designer
```

## Module Development  
**Triggers**: "implement module", "build module", "fix module"
```
Module Developer → QA Agent → Package Manager
```

## Testing
**Triggers**: "test", "verify", "check if", "validate"
```
QA Agent → Report → Fix if Needed
```

## Educational Content
**Triggers**: "improve learning", "educational content"
```
Education Architect → Module Developer → QA
```

## Complete Module Update
**Triggers**: "complete module overhaul"
```
Education Architect → Module Developer → QA → Package Manager → Documentation
```

## Quick Commands

### Website Content Update
```
"Update the website to [describe change]"
→ Automatically triggers Content Strategist → Designer
```

### Module Implementation
```
"Implement [module name] with [requirements]"
→ Automatically triggers Module Developer → QA → Package
```

### Quality Check
```
"Test that [feature] works correctly"
→ Automatically triggers QA Agent
```

## Workflow Rules

1. **NEVER skip agents** in the defined sequences
2. **ALWAYS use workflows** when trigger words appear
3. **Content Strategist ALWAYS before Designer** for website
4. **QA ALWAYS after Module Developer** for code
5. **Package Manager ALWAYS after QA** for integration

## Common Patterns

### Adding New Website Page
```
User: "Add a new page about debugging"
Claude: [Uses Content Strategist → Designer automatically]
```

### Fixing Documentation
```
User: "Fix the confusing parts in the setup guide"
Claude: [Uses Content Strategist → Designer automatically]
```

### Module Enhancement
```
User: "Improve the tensor module with better tests"
Claude: [Uses Module Developer → QA → Package automatically]
```

## DO NOT

- ❌ Ask "Should I use agents?"
- ❌ Skip workflow steps
- ❌ Change agent order
- ❌ Create content without Content Strategist
- ❌ Implement without QA testing

## ALWAYS

- ✅ Follow the workflow sequences
- ✅ Use agents for their defined purposes
- ✅ Complete all steps in order
- ✅ Let Content Strategist audit for duplicates
- ✅ Let QA validate before committing
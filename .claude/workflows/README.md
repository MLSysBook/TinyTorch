# 🎯 TinyTorch Workflow System

## Quick Command Reference

| Command | Purpose | Workflow |
|---------|---------|----------|
| `/website` | Website updates | Content Strategist → Designer |
| `/module` | Module development | Developer → QA → Package |
| `/test` | Testing | QA → Report |
| `/tito` | CLI updates | CLI Developer → QA |
| `/education` | Educational content | Architect → Developer → QA |
| `/full-module` | Complete overhaul | Education → Dev → QA → Package → Docs |

## Using Workflows

### Basic Usage
```
/[workflow] [your request]
```

### Examples
```
/website add debugging guide
/module fix tensor operations
/test all checkpoints
/tito add new command
/education improve autograd learning
```

## Workflow Files

Each workflow has a detailed specification:
- `website.md` - Website content updates
- `module.md` - Module development
- `test.md` - Testing procedures
- `tito.md` - CLI development
- `education.md` - Educational improvements

## Workflow Principles

1. **Sequential Execution** - Agents work in defined order
2. **No Skipping** - Every step must complete
3. **Quality Gates** - Each agent validates before handoff
4. **Clear Deliverables** - Each agent produces specific outputs
5. **Error Handling** - Failures loop back for fixes

## Custom Workflows

To create a new workflow:
1. Create file in `.claude/workflows/[name].md`
2. Define sequence with mermaid diagram
3. Specify agent responsibilities
4. Add quality checks
5. Document use cases

## Workflow vs Direct Agent

**Use Workflows When:**
- Multiple agents needed
- Specific sequence required
- Quality gates important
- Standard process exists

**Use Direct Agent When:**
- Single agent sufficient
- Exploratory work
- Quick questions
- Non-standard request

## Advanced Usage

### Chaining Workflows
```
/module implement tensor → /test tensor → /website document tensor
```

### Conditional Workflows
```
/test all → [if fails] → /module fix → /test again
```

### Parallel Workflows
```
/website update docs + /module fix bugs (run simultaneously)
```

## Best Practices

1. **Always use workflows for:**
   - Website updates (prevent duplicates)
   - Module changes (ensure testing)
   - CLI additions (maintain consistency)

2. **Start with slash command:**
   - Makes intent clear
   - Triggers correct workflow
   - Ensures proper sequence

3. **Be specific in requests:**
   - Include what you want done
   - Specify constraints if any
   - Mention related components

## Troubleshooting

**If workflow fails:**
- Check agent outputs for errors
- Verify prerequisites met
- Ensure files exist
- Review quality gate failures

**Common issues:**
- Missing dependencies
- File not found
- Test failures
- Integration conflicts

## Workflow Metrics

Track workflow effectiveness:
- Time to complete
- Quality gate pass rate
- Error frequency
- User satisfaction

---

*For detailed workflow specifications, see individual workflow files in this directory.*
# Community Configuration Setup

## Storage Location

All community data is stored **project-locally** in `.tinytorch/` directory (not in home directory):

```
.tinytorch/
├── config.json          # Configuration (website URLs, settings)
├── community/
│   └── profile.json     # User's community profile
└── submissions/         # Benchmark submissions (ready for website)
```

## Configuration File (`.tinytorch/config.json`)

The configuration file is automatically created on first use with these defaults:

```json
{
  "website": {
    "base_url": "https://tinytorch.ai",
    "community_map_url": "https://tinytorch.ai/community",
    "api_url": null,
    "enabled": false
  },
  "local": {
    "enabled": true,
    "auto_sync": false
  }
}
```

### Configuration Fields

**Website Settings:**
- `base_url`: Base URL for TinyTorch website
- `community_map_url`: URL to community map page
- `api_url`: API endpoint URL (set when API is ready)
- `enabled`: Enable website integration (set to `true` when ready)

**Local Settings:**
- `enabled`: Always `true` - local storage is always enabled
- `auto_sync`: Auto-sync to website when enabled (future feature)

## Website Integration Stubs

All commands have stubs for website integration that are currently disabled:

### Join Command
```python
def _notify_website_join(self, profile: Dict[str, Any]) -> None:
    """Stub: Notify website when user joins."""
    config = self._get_config()
    if not config.get("website", {}).get("enabled", False):
        return
    
    api_url = config.get("website", {}).get("api_url")
    if api_url:
        # TODO: Implement API call when website is ready
        # import requests
        # response = requests.post(f"{api_url}/api/community/join", json=profile)
        pass
```

### Leave Command
```python
def _notify_website_leave(self, anonymous_id: Optional[str]) -> None:
    """Stub: Notify website when user leaves."""
    # Similar structure to join
```

### Benchmark Submission
```python
def _submit_to_website(self, submission: Dict[str, Any]) -> None:
    """Stub: Submit benchmark results to website."""
    # Similar structure
```

## Enabling Website Integration

When the website API is ready:

1. **Update configuration:**
   ```json
   {
     "website": {
       "api_url": "https://api.tinytorch.ai",
       "enabled": true
     }
   }
   ```

2. **Implement API calls:**
   - Uncomment TODO sections in `community.py` and `benchmark.py`
   - Add `requests` dependency if needed
   - Implement error handling

3. **Test integration:**
   - Test join/leave notifications
   - Test benchmark submission
   - Verify data sync

## Current Behavior (Local-Only)

**All commands work locally:**
- ✅ `tito community join` - Saves profile to `.tinytorch/community/profile.json`
- ✅ `tito community update` - Updates local profile
- ✅ `tito community leave` - Removes local profile
- ✅ `tito benchmark baseline` - Saves to `.tito/benchmarks/`
- ✅ `tito benchmark capstone` - Saves to `.tito/benchmarks/`

**Website stubs are present but disabled:**
- Stubs call `_get_config()` to check if website is enabled
- If disabled (default), commands work purely locally
- When enabled, stubs will make API calls

## Benefits of Project-Local Storage

1. **Version Control Friendly**: `.tinytorch/` can be gitignored or committed
2. **Project-Specific**: Each TinyTorch project has its own community profile
3. **Portable**: Easy to move/share projects with their data
4. **Privacy**: Data stays in project, not in home directory

## Migration Notes

If you had data in `~/.tinytorch/`, you can migrate:

```bash
# Copy old data to new location
cp -r ~/.tinytorch/community .tinytorch/
cp ~/.tinytorch/config.json .tinytorch/config.json  # if exists
```

The new system will automatically use `.tinytorch/` in the project root.


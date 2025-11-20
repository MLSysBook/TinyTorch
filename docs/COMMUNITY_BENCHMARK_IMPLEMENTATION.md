# Community & Benchmark Commands - Implementation Document

## Overview

This document describes the implementation of community and benchmark commands for TinyTorch, an educational ML systems framework. The goal is to create a "Hello World" user journey where students feel part of a global cohort after completing setup and initial milestones.

## Design Philosophy

**Educational Focus**: TinyTorch is an educational framework. Community features should:
- Encourage learning and progress, not competition
- Create cohort feeling (students see peers, not rankings)
- Be privacy-friendly (all data optional, anonymous IDs)
- Work locally first, sync to website later

**Local-First Approach**: 
- All data stored project-locally in `.tinytorch/` directory
- Website integration via stubs (ready for future API)
- No external dependencies required for core functionality

## Implementation

### 1. Benchmark Commands (`tito benchmark`)

#### Baseline Benchmark (`tito benchmark baseline`)
**Purpose**: Quick setup validation - "Hello World" moment

**What it does**:
- Runs lightweight benchmarks (tensor ops, matrix multiply, forward pass)
- Calculates score (0-100) based on performance
- Saves results to `.tito/benchmarks/baseline_TIMESTAMP.json`
- Auto-prompts for submission after completion

**When to run**: After setup, anytime

**Output Example**:
```
🎯 Baseline Benchmark

📊 Your Baseline Performance:
   • Tensor Operations: ⚡ 0.5ms
   • Matrix Multiply: ⚡ 2.3ms
   • Forward Pass: ⚡ 5.2ms
   • Score: 85/100

✅ Setup verified and working!
```

#### Capstone Benchmark (`tito benchmark capstone`)
**Purpose**: Full performance evaluation after Module 20

**What it does**:
- Runs full benchmark suite from Module 20
- Supports tracks: speed, compression, accuracy, efficiency, all
- Uses Module 19's Benchmark class (when available)
- Falls back gracefully if Module 20 not complete
- Auto-prompts for submission after completion

**When to run**: After Module 20 (Capstone)

**Output Example**:
```
🏆 Capstone Benchmark Results

📊 Speed Track:
   • Latency: 45.2ms
   • Throughput: 22.1 ops/sec
   • Score: 92/100

📊 Overall Score: 90/100
```

### 2. Community Commands (`tito community`)

#### Join (`tito community join`)
**Purpose**: Join the global TinyTorch community

**What it does**:
- Collects: country, institution, course type, experience level (all optional)
- Generates anonymous UUID
- Auto-detects cohort (Fall 2024, Spring 2025, etc.)
- Saves profile to `.tinytorch/community/profile.json`
- Shows welcome message with cohort info

**Privacy**: All fields optional, anonymous IDs, local storage

#### Update (`tito community update`)
**Purpose**: Update community profile

**What it does**:
- Updates profile fields (country, institution, course type, experience)
- Auto-updates progress from `.tito/milestones.json` and `.tito/progress.json`
- Interactive or command-line updates

#### Leave (`tito community leave`)
**Purpose**: Remove community profile

**What it does**:
- Removes profile file
- Confirmation prompt (can skip with `--force`)
- Preserves benchmark submissions

#### Stats & Profile (`tito community stats`, `tito community profile`)
**Purpose**: View community information

**What it does**:
- Shows community statistics
- Displays full profile in table format
- Shows progress: milestones, modules, capstone score

## Data Storage

### Project-Local Storage (`.tinytorch/`)

All data stored in project root, not home directory:

```
.tinytorch/
├── config.json          # Configuration (website URLs, settings)
├── community/
│   └── profile.json     # User's community profile
└── submissions/         # Benchmark submissions (ready for website)
```

### Profile Structure (`profile.json`)

```json
{
  "anonymous_id": "uuid",
  "joined_at": "2024-11-20T10:30:00",
  "location": {
    "country": "United States"
  },
  "institution": {
    "name": "Harvard University",
    "type": null
  },
  "context": {
    "course_type": "university",
    "experience_level": "intermediate",
    "cohort": "Fall 2024"
  },
  "progress": {
    "setup_verified": false,
    "milestones_passed": 0,
    "modules_completed": 0,
    "capstone_score": null
  }
}
```

### Configuration (`config.json`)

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

## Website Integration Stubs

All commands have stubs for future website integration:

### Join Notification
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

### Leave Notification
```python
def _notify_website_leave(self, anonymous_id: Optional[str]) -> None:
    """Stub: Notify website when user leaves."""
    # Similar structure
```

### Benchmark Submission
```python
def _submit_to_website(self, submission: Dict[str, Any]) -> None:
    """Stub: Submit benchmark results to website."""
    # Similar structure
```

**Current Behavior**: Stubs check configuration. If website integration disabled (default), commands work purely locally. When enabled, stubs will make API calls.

## User Journey

### 1. Setup & Join
```bash
# After setup
tito community join
# → Collects info, saves profile, shows welcome

# Run baseline benchmark
tito benchmark baseline
# → Runs benchmarks, shows results, prompts for submission
```

### 2. Progress Updates
```bash
# Update profile
tito community update
# → Updates fields, auto-updates progress

# View profile
tito community profile
# → Shows full profile with progress
```

### 3. Capstone Completion
```bash
# After Module 20
tito benchmark capstone
# → Runs full benchmarks, prompts for submission
```

## Privacy & Security

**Privacy Features**:
- ✅ All fields optional
- ✅ Anonymous UUIDs (no personal identifiers)
- ✅ Local storage (user controls sharing)
- ✅ No auto-detection (country detection disabled)
- ✅ Explicit consent for sharing

**Security Considerations**:
- Profile data stored locally (not transmitted unless user opts in)
- Anonymous IDs prevent tracking
- Website integration opt-in only

## Educational Benefits

**Cohort Feeling**:
- Students see they're part of a global community
- Cohort identification (Fall 2024, Spring 2025, etc.)
- Institution-based cohorts (Harvard, Stanford, etc.)
- Progress comparisons (milestones, modules completed)

**Motivation**:
- "Hello World" moment after setup
- Progress tracking and celebration
- Community map visualization (future)
- Peer visibility (future)

**Learning Support**:
- Not competitive (no rankings)
- Encourages sharing and learning
- Privacy-friendly (students control data)

## Technical Implementation

### Files Created
- `tito/commands/benchmark.py` - Benchmark commands
- `tito/commands/community.py` - Community commands

### Files Modified
- `tito/commands/__init__.py` - Added command exports
- `tito/main.py` - Registered new commands

### Dependencies
- `rich` - Beautiful terminal output (already in requirements)
- `numpy` - Benchmark calculations (already in requirements)
- No external API dependencies (local-first)

## Future Enhancements

**Phase 1 (Current)**: ✅
- Local storage
- Basic commands
- Website stubs

**Phase 2 (Future)**:
- Website API integration
- Community map visualization
- Cohort filtering and comparisons
- Progress rankings (optional, opt-in)

**Phase 3 (Future)**:
- Real-time updates
- Peer connections
- Study groups
- Mentorship matching

## Testing

Commands are ready to test:
```bash
# Test benchmark
tito benchmark baseline
tito benchmark capstone

# Test community
tito community join
tito community profile
tito community update
tito community stats
tito community leave
```

## Questions for Expert Review

1. **Storage Approach**: Is project-local storage (`.tinytorch/`) the right approach for an educational framework? Should we consider home directory instead?

2. **Privacy Model**: Is the anonymous UUID + optional fields approach appropriate for students? Any privacy concerns?

3. **Website Integration**: Are the stubs structured correctly? Should we use a different pattern for future API integration?

4. **Educational Focus**: Does this design support learning without creating unhealthy competition? Are there features we should add/remove?

5. **Cohort Features**: Is cohort identification (Fall 2024, institution-based) the right approach? Should we add more cohort types?

6. **Benchmark Design**: Are baseline and capstone benchmarks appropriate? Should we add more benchmark types?

7. **Data Collection**: What data should we collect? What should we avoid?

8. **Community Map**: Is a global map visualization appropriate for an educational framework? Privacy concerns?

9. **Integration Points**: Should we integrate with existing systems (GitHub, LMS, etc.)?

10. **Scalability**: Will this design scale to thousands of students? What bottlenecks should we anticipate?


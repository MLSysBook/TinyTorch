# TinyTorch JSON Data Formats

This document describes all JSON formats used for tracking progress, milestones, and system status in TinyTorch. These can be used for syncing data to external websites or dashboards.

---

## 1. Simple Module Progress (`progress.json` in root)

**Location**: `TinyTorch/progress.json`

**Purpose**: Lightweight module tracking for workflow commands

**Format**:
```json
{
  "started_modules": [
    "02",
    "03"
  ],
  "completed_modules": [
    "01",
    "02",
    "03"
  ],
  "last_worked": "03",
  "last_completed": "03",
  "last_updated": "2025-11-22T11:47:59.625329"
}
```

**Fields**:
- `started_modules`: Array of module numbers that have been started (e.g., ["01", "02"])
- `completed_modules`: Array of module numbers that have been completed
- `last_worked`: The module number most recently worked on
- `last_completed`: The module number most recently completed
- `last_updated`: ISO 8601 timestamp of last update

---

## 2. Comprehensive Module Progress (`.tito/progress.json`)

**Location**: `TinyTorch/.tito/progress.json`

**Purpose**: Full module tracking with dates and versions

**Format**:
```json
{
  "version": "1.0",
  "completed_modules": [1, 2, 3, 4, 5, 6, 7],
  "completion_dates": {
    "1": "2025-11-16T10:00:00",
    "2": "2025-11-16T11:00:00",
    "3": "2025-11-16T12:00:00",
    "4": "2025-11-16T13:00:00",
    "5": "2025-11-16T14:00:00",
    "6": "2025-11-16T15:00:00",
    "7": "2025-11-16T16:00:00"
  }
}
```

**Fields**:
- `version`: Schema version (currently "1.0")
- `completed_modules`: Array of module numbers (as integers) that have been completed
- `completion_dates`: Object mapping module numbers to ISO 8601 completion timestamps

---

## 3. Milestone Progress (`.tito/milestones.json`)

**Location**: `TinyTorch/.tito/milestones.json`

**Purpose**: Track historical ML milestone achievements

**Format**:
```json
{
  "version": "1.0",
  "completed_milestones": ["03", "04"],
  "completion_dates": {
    "03": "2025-11-16T15:00:00",
    "04": "2025-11-16T17:30:00"
  },
  "unlocked_milestones": ["01", "02", "03", "04", "05"],
  "unlock_dates": {
    "01": "2025-11-15T10:00:00",
    "02": "2025-11-15T14:00:00",
    "03": "2025-11-16T12:00:00",
    "04": "2025-11-16T16:00:00",
    "05": "2025-11-16T18:00:00"
  },
  "total_unlocked": 5,
  "achievements": []
}
```

**Fields**:
- `version`: Schema version (currently "1.0")
- `completed_milestones`: Array of milestone IDs that have been successfully completed
- `completion_dates`: Object mapping milestone IDs to ISO 8601 completion timestamps
- `unlocked_milestones`: Array of milestone IDs that are available to attempt
- `unlock_dates`: Object mapping milestone IDs to ISO 8601 unlock timestamps
- `total_unlocked`: Total count of unlocked milestones
- `achievements`: Array for additional achievement tracking (extensible)

**Milestone IDs**:
- `"01"` - 1957: Perceptron
- `"02"` - 1969: XOR Problem (Backpropagation)
- `"03"` - 1986: MLP Revival
- `"04"` - 1998: CNN Revolution (LeNet)
- `"05"` - 2017: Transformer Era (Attention)
- `"06"` - 2018: MLPerf Benchmarking

---

## 4. User Configuration (`.tito/config.json`)

**Location**: `TinyTorch/.tito/config.json`

**Purpose**: User preferences and settings

**Format**:
```json
{
  "logo_theme": "standard"
}
```

**Fields**:
- `logo_theme`: UI theme preference ("standard", "pride", "retro", etc.)

---

## 5. Module Report Card (`instructor/reports/`)

**Location**: `TinyTorch/instructor/reports/{module_name}_report_card_{timestamp}.json`

**Purpose**: Detailed pedagogical analysis of module quality

**Format** (abbreviated):
```json
{
  "module_name": "02_activations",
  "module_path": "modules/source/02_activations",
  "analysis_date": "2025-07-12T22:48:40.235285",
  "total_lines": 1417,
  "total_cells": 17,
  "avg_cell_length": 65.29,
  "scaffolding_quality": 3,
  "complexity_distribution": {
    "1": 2,
    "2": 2,
    "3": 10,
    "4": 2,
    "5": 1
  },
  "learning_progression_quality": 4,
  "concepts_covered": [
    "Sigmoid",
    "Tanh",
    "Softmax",
    "...more concepts..."
  ],
  "todo_count": 4,
  "hint_count": 5,
  "test_count": 1,
  "critical_issues": [
    "Module too long (1417 lines) - students will be overwhelmed",
    "8 cells are too long (>50 lines)"
  ],
  "overwhelm_points": [
    "Cell 1: Too many concepts (5)",
    "Cell 2: Very long cell (86 lines)",
    "...more issues..."
  ],
  "recommendations": [
    "Break module into smaller sections or multiple modules",
    "Split 12 long cells into smaller, focused cells"
  ],
  "overall_grade": "C",
  "category_grades": {
    "Scaffolding": "C",
    "Complexity": "B",
    "Cell_Length": "D"
  }
}
```

---

## 6. Performance Results (`tests/performance/`)

**Location**: `TinyTorch/tests/performance/performance_results/{module_name}_performance_results.json`

**Purpose**: Performance benchmarking results

**Format**:
```json
{
  "timer_accuracy": "{'timer_accuracy': False, 'measurement_consistency': False, 'fast_operation_time_ms': 0.0011, 'slow_operation_time_ms': 11.936, 'ratio_actual': 10436.67, 'ratio_expected': 100}",
  "memory_profiler_accuracy": "{'memory_accuracy': True, 'small_allocation_mb': 1.001, 'large_allocation_mb': 10.001}",
  "flop_counter_accuracy": "{'linear_flop_accuracy': True, 'conv_flop_accuracy': True, 'linear_calculated': 264192, 'conv_calculated': 133632000}",
  "profiler_overhead": "{'overhead_acceptable': True, 'overhead_factor': 1.029}",
  "simple_profiler_interface": "{'has_required_fields': True, 'reasonable_timing': False, 'wall_time': 0.0000370}",
  "real_world_scenario": "Error: integer modulo by zero"
}
```

---

## Combined Export Format for Website Sync

**Recommended combined format for sending to external dashboards**:

```json
{
  "user_id": "student_identifier",
  "timestamp": "2025-11-22T11:47:59.625329",
  "version": "1.0",
  
  "module_progress": {
    "total_modules": 20,
    "completed_count": 7,
    "completed_modules": [1, 2, 3, 4, 5, 6, 7],
    "completion_dates": {
      "1": "2025-11-16T10:00:00",
      "2": "2025-11-16T11:00:00",
      "3": "2025-11-16T12:00:00",
      "4": "2025-11-16T13:00:00",
      "5": "2025-11-16T14:00:00",
      "6": "2025-11-16T15:00:00",
      "7": "2025-11-16T16:00:00"
    },
    "last_worked": "07",
    "last_completed": "07",
    "completion_percentage": 35
  },
  
  "milestone_progress": {
    "total_milestones": 6,
    "unlocked_count": 5,
    "completed_count": 2,
    "unlocked_milestones": [
      {
        "id": "01",
        "name": "1957: Perceptron",
        "unlocked_at": "2025-11-15T10:00:00",
        "completed": true,
        "completed_at": "2025-11-16T15:00:00"
      },
      {
        "id": "02",
        "name": "1969: XOR Problem",
        "unlocked_at": "2025-11-15T14:00:00",
        "completed": false,
        "completed_at": null
      },
      {
        "id": "03",
        "name": "1986: MLP Revival",
        "unlocked_at": "2025-11-16T12:00:00",
        "completed": true,
        "completed_at": "2025-11-16T17:30:00"
      },
      {
        "id": "04",
        "name": "1998: CNN Revolution",
        "unlocked_at": "2025-11-16T16:00:00",
        "completed": false,
        "completed_at": null
      },
      {
        "id": "05",
        "name": "2017: Transformer Era",
        "unlocked_at": "2025-11-16T18:00:00",
        "completed": false,
        "completed_at": null
      }
    ],
    "locked_milestones": [
      {
        "id": "06",
        "name": "2018: MLPerf Benchmarking"
      }
    ]
  },
  
  "statistics": {
    "total_study_time_hours": 12.5,
    "modules_per_day": 1.2,
    "current_streak_days": 5,
    "longest_streak_days": 7
  },
  
  "achievements": {
    "first_module": "2025-11-16T10:00:00",
    "first_milestone": "2025-11-16T15:00:00",
    "halfway_point": "2025-11-20T14:00:00"
  }
}
```

---

## API Endpoints (Suggested)

For a TinyTorch progress tracking website, consider these endpoints:

### Upload Progress
```
POST /api/progress/upload
Content-Type: application/json

{
  "user_id": "...",
  "module_progress": {...},
  "milestone_progress": {...}
}
```

### Get Current Progress
```
GET /api/progress/:user_id

Response: Combined JSON format above
```

### Get Leaderboard
```
GET /api/leaderboard?metric=modules_completed

Response:
{
  "leaderboard": [
    {
      "user_id": "...",
      "username": "...",
      "modules_completed": 15,
      "milestones_completed": 4,
      "rank": 1
    },
    ...
  ]
}
```

---

## Implementation Notes

1. **Privacy**: All JSON files are stored locally in `.tito/` directory (gitignored)
2. **Timestamps**: Use ISO 8601 format (`YYYY-MM-DDTHH:MM:SS`)
3. **Module Numbers**: Can be strings ("01") or integers (1) depending on context
4. **Version Field**: Allows schema evolution without breaking changes
5. **Backups**: Automatic backups stored in `.tito/backups/`

## Accessing from CLI

```bash
# View current progress
tito status --progress

# Check specific progress file
cat .tito/progress.json
cat .tito/milestones.json

# View milestone progress
tito milestone progress
```


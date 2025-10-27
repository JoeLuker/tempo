# TEMPO Configuration System

## Overview

TEMPO supports elegant YAML-based configuration for reproducible runs.

**Workflow**: `YAML config → TEMPO → JSON output`

## Quick Start

```bash
# Run from config file
python3 run_from_config.py configs/examples/simple.yaml

# Override output location
python3 run_from_config.py configs/examples/two-phase.yaml --output my_results.json

# Dry run (validate config without running)
python3 run_from_config.py configs/examples/multi-phase.yaml --dry-run
```

## Configuration Structure

### Basic Config

```yaml
name: "my-experiment"
description: "What this run does"
tags: ["experiment", "code-gen"]

model:
  name: "deepcogito/cogito-v1-preview-llama-3B"
  revision: "main"

generation:
  prompt: "Your generation prompt"
  max_tokens: 150
  selection_threshold: 0.12
  seed: 42

output:
  json_output: true
  json_file: "output/my-results.json"
```

### Two-Phase Generation

```yaml
extensions:
  two_phase: true
  dynamic_phase: true
  max_positions: 100      # Switch at 100 total positions
  phase2_threshold: 1.0   # No branching in phase 2
```

### Multi-Phase Generation

```yaml
multi_phase:
  enabled: true
  phases:
    - name: "exploration"
      max_positions: 50
      threshold: 0.08
      description: "Aggressive exploration"

    - name: "refinement"
      max_positions: 100
      threshold: 0.15
      description: "Moderate exploration"

    - name: "commitment"
      max_positions: 999
      threshold: 1.0
      description: "Final clean output"
```

### Extensions

```yaml
extensions:
  # Built-in extensions
  confidence_surfing: true      # Adaptive threshold based on entropy
  genealogy_tracking: true      # Track token lineage
  entropy_watching: true        # Monitor entropy patterns

  # Two-phase (alternative to multi-phase)
  two_phase: true
  dynamic_phase: true
  max_positions: 100
  phase2_threshold: 1.0
```

### Pruning

```yaml
pruning:
  enabled: true
  attention_threshold: 0.02   # Remove tokens with attention below this
```

### Debug Options

```yaml
debug:
  debug_mode: false
  show_token_ids: false
  profile: false
  verbose: false
```

## Example Configs

### `examples/simple.yaml`
Basic code generation with default settings.

### `examples/two-phase.yaml`
Two-phase generation: explore then commit.

### `examples/multi-phase.yaml`
Complex multi-phase with 4 distinct phases.

## Creating Your Own Configs

1. Copy an example config
2. Modify the prompt and parameters
3. Run with `python3 run_from_config.py your_config.yaml`
4. Results saved to JSON

## Output Format

All runs output JSON with:

```json
{
  "prompt": "...",
  "final_text": "...",
  "steps": [
    {
      "step": 0,
      "branching_factor": 2,
      "entropy": 2.5,
      "tokens": [{"id": 1, "prob": 0.5}, ...],
      ...
    }
  ],
  "statistics": {
    "total_steps": 100,
    "generation_time_seconds": 10.5,
    ...
  }
}
```

## Advanced: Programmatic Usage

```python
from src.config.schema import TEMPOConfig

# Load config
config = TEMPOConfig.from_yaml("my_config.yaml")

# Modify
config.generation.max_tokens = 200

# Save
config.to_yaml("modified_config.yaml")

# Convert to args dict for TEMPO
args = config.to_args_dict()
```

## Tips

- Use `--dry-run` to validate configs before running
- Use `--save-config` to save modified configs
- JSON output is perfect for analysis and visualization
- Create config templates for common tasks

## Benefits

✅ **Reproducible**: Same YAML = same results
✅ **Shareable**: Share configs with collaborators
✅ **Versioned**: Track configs in git
✅ **Composable**: Build complex configs from simple templates
✅ **Type-safe**: Schema validation prevents errors
✅ **Self-documenting**: YAML is readable

**YAML in → JSON out → Science!** 🎯

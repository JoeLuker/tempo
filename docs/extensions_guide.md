# TEMPO Extension System Guide

## Overview

TEMPO has a dual-layer extension system:

1. **Ultra-Simple** (`src/extensions/ultra_simple.py`) - The foundation
2. **Composable** (`src/extensions/composable.py`) - Functional composition layer

Both are built on the same principle: **Extensions are just functions** `state → state`.

## Ultra-Simple Extensions

### The Core Concept

```python
# An extension is just a function
def my_extension(state: GenState) -> GenState:
    # Do something with state
    return replace(state, threshold=0.2)

# Use it in a list
extensions = [my_extension, another_extension]
```

### Built-in Extensions

```python
from src.extensions import confidence_surf, track_genealogy, watch_entropy

# Confidence surfing - adjusts threshold based on entropy
confidence_surf(state)

# Genealogy tracking - records token lineage
track_genealogy(state)

# Entropy watching - monitors entropy patterns
watch_entropy(state)
```

### Creating Custom Extensions

```python
from dataclasses import replace
from src.extensions.ultra_simple import GenState

def my_custom_extension(state: GenState) -> GenState:
    """My custom logic."""
    if state.branching_factor > 3:
        # Too many branches, increase threshold
        return replace(state, threshold=min(0.5, state.threshold * 1.5))
    return state
```

### Configuration via Closures

```python
def make_custom_threshold(target: float) -> Extension:
    """Factory function for configured extension."""
    def extension(state: GenState) -> GenState:
        if state.step == 50:
            return replace(state, threshold=target)
        return state
    return extension

# Create configured instances
phase2 = make_custom_threshold(1.0)
conservative = make_custom_threshold(0.3)
```

## Composable Extensions

### Core Combinators

#### `pipe(*extensions)` - Left-to-right composition

```python
from src.extensions.composable import pipe

# Unix pipe style: first → second → third
ext = pipe(
    watch_entropy,
    confidence_surf,
    track_genealogy
)
```

#### `compose(*extensions)` - Right-to-left composition

```python
from src.extensions.composable import compose

# Mathematical composition: last → middle → first
ext = compose(
    track_genealogy,  # Runs last
    confidence_surf,  # Runs second
    watch_entropy     # Runs first
)
```

### Conditional Execution

#### `when(predicate, extension)` - Run if true

```python
from src.extensions.composable import when

# Only run when branching
ext = when(
    lambda s: s.branching_factor > 1,
    track_genealogy
)
```

#### `unless(predicate, extension)` - Run if false

```python
from src.extensions.composable import unless

# Skip first step
ext = unless(
    lambda s: s.step == 0,
    confidence_surf
)
```

#### `branch(predicate, true_ext, false_ext)` - If-else branching

```python
from src.extensions.composable import branch, modify_threshold

ext = branch(
    lambda s: s.entropy > 2.5,
    modify_threshold(2.0),  # High entropy: explore more
    modify_threshold(0.5)   # Low entropy: commit
)
```

### Temporal Control

```python
from src.extensions.composable import (
    before_step, after_step, between_steps, every_n_steps, once_at
)

# Run before step 100
early_exploration = before_step(100, set_threshold(0.12))

# Run after step 100
late_commitment = after_step(100, set_threshold(1.0))

# Run between steps 50-150
middle_phase = between_steps(50, 150, confidence_surf)

# Run every 10 steps
periodic_logging = every_n_steps(10, log_statistics)

# Run exactly once at step 75
inject_at_75 = once_at(75, inject_prompt("Now refine:"))
```

### State Transformers

```python
from src.extensions.composable import (
    modify_threshold, set_threshold, clamp_threshold, with_metadata
)

# Multiply threshold
double = modify_threshold(2.0)
halve = modify_threshold(0.5)

# Set exact value
commit_mode = set_threshold(1.0)

# Clamp to range
bounded = clamp_threshold(0.05, 0.30)

# Set metadata
mark_phase = with_metadata('phase', 2)
```

## Real-World Examples

### Example 1: Simple Two-Phase Generation

```python
from src.extensions.composable import pipe, before_step, after_step, set_threshold

# Phase 1: Explore until step 100
# Phase 2: Commit after step 100
two_phase = pipe(
    before_step(100, set_threshold(0.12)),
    after_step(100, set_threshold(1.0))
)

# Use it
extensions = [two_phase]
```

### Example 2: Adaptive Multi-Phase Pipeline

```python
from src.extensions.composable import *
from src.extensions import watch_entropy, confidence_surf, track_genealogy

# Complex adaptive pipeline
ext = pipe(
    # Always monitor entropy
    watch_entropy,

    # Phase 1: Aggressive exploration (0-50)
    before_step(50, pipe(
        set_threshold(0.10),
        when(lambda s: s.branching_factor > 2, track_genealogy),
        adaptive_threshold(low_entropy=1.0, high_entropy=4.0)
    )),

    # Phase 2: Moderate exploration (50-100)
    between_steps(50, 100, pipe(
        set_threshold(0.15),
        confidence_surf
    )),

    # Phase 3: Refinement (100-150)
    between_steps(100, 150, pipe(
        set_threshold(0.25),
        every_n_steps(10, with_metadata('checkpoint', True))
    )),

    # Phase 4: Commitment (150+)
    after_step(150, set_threshold(1.0)),

    # Always keep threshold in bounds
    clamp_threshold(0.05, 0.95)
)
```

### Example 3: Conditional Branching Based on State

```python
from src.extensions.composable import *

# Different strategies based on entropy and branching
adaptive_ext = branch(
    lambda s: s.entropy > 3.0,
    # High uncertainty: be conservative
    pipe(
        modify_threshold(0.5),
        when(lambda s: s.branching_factor > 5, modify_threshold(0.3))
    ),
    # Low uncertainty: explore more
    branch(
        lambda s: s.branching_factor < 2,
        modify_threshold(2.0),  # Not branching: encourage it
        modify_threshold(1.2)   # Already branching: slight boost
    )
)
```

### Example 4: Position-Based Phase Switching

```python
from src.extensions.composable import phase_switcher

# Built-in convenience function
ext = phase_switcher(
    phase1_positions=100,  # Switch at 100 total positions
    phase1_threshold=0.12,  # Exploration threshold
    phase2_threshold=1.0    # Commitment threshold
)
```

## Usage in TEMPO

### Command-Line (Two-Phase)

```bash
python3 run_tempo.py \
  --prompt "Your prompt" \
  --selection-threshold 0.12 \
  --two-phase \
  --dynamic-phase \
  --max-positions 100 \
  --phase2-threshold 1.0
```

### Programmatic Usage

```python
from src.experiments import ArgumentParser, ExperimentRunner
from src.extensions.composable import *

# Parse args
args = ArgumentParser.parse_args()

# Create custom extension pipeline
my_extensions = pipe(
    watch_entropy,
    before_step(50, set_threshold(0.10)),
    between_steps(50, 100, adaptive_threshold()),
    after_step(100, set_threshold(1.0)),
    clamp_threshold(0.05, 0.95)
)

# Add to args
args['extensions'] = [my_extensions]

# Run experiment
ExperimentRunner.run_experiment(args)
```

## Design Principles

### 1. Simplicity

Extensions are just functions. No classes, no decorators, no magic.

```python
def my_ext(state): return state  # That's it!
```

### 2. Composability

Build complex behaviors from simple primitives.

```python
complex = pipe(simple1, simple2, when(pred, simple3))
```

### 3. No Side Effects (except metadata)

Extensions return new state via `replace()`. Immutability by default.

```python
# Good: Returns new state
return replace(state, threshold=0.5)

# Bad: Mutates state
state.threshold = 0.5
return state
```

### 4. Testability

Pure functions are easy to test.

```python
def test_my_extension():
    state = GenState(step=1, ...)
    result = my_extension(state)
    assert result.threshold == 0.5
```

### 5. Discoverability

Everything is explicit. No hidden behavior.

```python
# You can see exactly what runs
extensions = [ext1, ext2, ext3]

# And exactly when
ext = when(lambda s: s.step > 10, my_ext)
```

## Performance Considerations

Extensions run on EVERY generation step. Keep them fast:

- ✅ Simple conditionals
- ✅ Threshold math
- ✅ Metadata updates
- ❌ Heavy computation
- ❌ I/O operations
- ❌ External API calls

## Advanced Patterns

### Extension Factories with Configuration

```python
def make_phase_controller(phases: list[tuple[int, float]]):
    """Create multi-phase extension from config."""
    def controller(state: GenState) -> GenState:
        pos = state.prompt_length + state.step
        for max_pos, threshold in phases:
            if pos < max_pos:
                return replace(state, threshold=threshold)
        return state
    return controller

# Use it
ext = make_phase_controller([
    (50, 0.10),   # 0-50: aggressive
    (100, 0.15),  # 50-100: moderate
    (150, 0.25),  # 100-150: conservative
    (999, 1.0)    # 150+: commit
])
```

### Stateful Extensions (via Closure)

```python
def make_decay_threshold(initial: float, decay_rate: float = 0.99):
    """Gradually decay threshold over time."""
    current = {'value': initial}

    def extension(state: GenState) -> GenState:
        if state.step > 0:
            current['value'] *= decay_rate
        return replace(state, threshold=current['value'])

    return extension
```

### Logging Extensions

```python
def make_logger(log_every: int = 10):
    """Log state periodically."""
    def logger(state: GenState) -> GenState:
        if state.step % log_every == 0:
            print(f"Step {state.step}: threshold={state.threshold:.3f}, "
                  f"entropy={state.entropy:.2f}, branches={state.branching_factor}")
        return state
    return logger
```

## Comparison with Other Systems

### vs. Registry Pattern

**Registry-based** (complex):
```python
@register_extension("my_ext")
class MyExtension(Extension):
    def __init__(self, config):
        ...
    def execute(self, state):
        ...

# Use it
ext = ExtensionRegistry.get("my_ext", config={...})
```

**TEMPO** (simple):
```python
def my_ext(state):
    return replace(state, threshold=0.5)

# Use it
extensions = [my_ext]
```

### vs. Decorator Pattern

**Decorator-based** (magic):
```python
@extension(priority=10, enabled_when="branching")
def my_ext(state):
    ...
```

**TEMPO** (explicit):
```python
def my_ext(state):
    ...

ext = when(lambda s: s.branching_factor > 1, my_ext)
```

## Summary

TEMPO's extension system gives you:

- ✅ **Simplicity**: Just functions
- ✅ **Composability**: Combine primitives into complex behaviors
- ✅ **Transparency**: No hidden magic
- ✅ **Testability**: Pure functions are easy to test
- ✅ **Flexibility**: Build anything from simple to complex
- ✅ **Performance**: Minimal overhead

**Total core code**: ~150 lines (ultra_simple.py + composable.py)

**Power**: Unlimited compositional complexity

That's the beauty of functional programming! 🚀

"""Composable extension system with functional composition patterns.

Makes extensions truly composable with:
- Combinator functions (compose, pipe, when, unless, etc.)
- Extension chaining and branching
- Conditional execution
- Extension transformation
"""

from dataclasses import replace
from typing import Callable, Optional
from .ultra_simple import GenState, Extension


# ============================================================================
# Core Combinators
# ============================================================================

def compose(*extensions: Extension) -> Extension:
    """Compose extensions right-to-left (mathematical composition).

    compose(f, g, h)(x) = f(g(h(x)))

    Args:
        *extensions: Extensions to compose

    Returns:
        Single composed extension

    Example:
        ext = compose(track_genealogy, confidence_surf, watch_entropy)
        # Runs: watch_entropy -> confidence_surf -> track_genealogy
    """
    def composed(state: GenState) -> GenState:
        for ext in reversed(extensions):
            state = ext(state)
        return state
    return composed


def pipe(*extensions: Extension) -> Extension:
    """Pipe extensions left-to-right (Unix pipe style).

    pipe(f, g, h)(x) = h(g(f(x)))

    Args:
        *extensions: Extensions to pipe

    Returns:
        Single piped extension

    Example:
        ext = pipe(watch_entropy, confidence_surf, track_genealogy)
        # Runs: watch_entropy -> confidence_surf -> track_genealogy
    """
    def piped(state: GenState) -> GenState:
        for ext in extensions:
            state = ext(state)
        return state
    return piped


def when(predicate: Callable[[GenState], bool], extension: Extension) -> Extension:
    """Run extension only when predicate is true.

    Args:
        predicate: Function that takes state and returns bool
        extension: Extension to run if predicate is true

    Returns:
        Conditional extension

    Example:
        # Only surf confidence when branching
        ext = when(lambda s: s.branching_factor > 1, confidence_surf)
    """
    def conditional(state: GenState) -> GenState:
        if predicate(state):
            return extension(state)
        return state
    return conditional


def unless(predicate: Callable[[GenState], bool], extension: Extension) -> Extension:
    """Run extension unless predicate is true.

    Args:
        predicate: Function that takes state and returns bool
        extension: Extension to run if predicate is false

    Returns:
        Conditional extension

    Example:
        # Don't surf confidence on first step
        ext = unless(lambda s: s.step == 0, confidence_surf)
    """
    def conditional(state: GenState) -> GenState:
        if not predicate(state):
            return extension(state)
        return state
    return conditional


def branch(
    predicate: Callable[[GenState], bool],
    true_ext: Extension,
    false_ext: Optional[Extension] = None
) -> Extension:
    """Branch execution based on predicate.

    Args:
        predicate: Function that takes state and returns bool
        true_ext: Extension to run if predicate is true
        false_ext: Extension to run if predicate is false (optional)

    Returns:
        Branching extension

    Example:
        ext = branch(
            lambda s: s.entropy > 2.0,
            make_confidence_surf(explore_mult=3.0),  # High entropy
            make_confidence_surf(conservative_mult=0.3)  # Low entropy
        )
    """
    def branching(state: GenState) -> GenState:
        if predicate(state):
            return true_ext(state)
        elif false_ext:
            return false_ext(state)
        return state
    return branching


def after_step(n: int, extension: Extension) -> Extension:
    """Run extension only after step n.

    Args:
        n: Step number
        extension: Extension to run

    Returns:
        Conditional extension

    Example:
        ext = after_step(10, confidence_surf)  # Only after step 10
    """
    return when(lambda s: s.step > n, extension)


def before_step(n: int, extension: Extension) -> Extension:
    """Run extension only before step n.

    Args:
        n: Step number
        extension: Extension to run

    Returns:
        Conditional extension

    Example:
        ext = before_step(50, aggressive_exploration)  # Only before step 50
    """
    return when(lambda s: s.step < n, extension)


def between_steps(start: int, end: int, extension: Extension) -> Extension:
    """Run extension only between steps.

    Args:
        start: Start step (inclusive)
        end: End step (inclusive)
        extension: Extension to run

    Returns:
        Conditional extension

    Example:
        ext = between_steps(10, 50, confidence_surf)  # Only steps 10-50
    """
    return when(lambda s: start <= s.step <= end, extension)


def every_n_steps(n: int, extension: Extension) -> Extension:
    """Run extension every n steps.

    Args:
        n: Step interval
        extension: Extension to run

    Returns:
        Conditional extension

    Example:
        ext = every_n_steps(10, log_statistics)  # Every 10 steps
    """
    return when(lambda s: s.step % n == 0, extension)


def once_at(step: int, extension: Extension) -> Extension:
    """Run extension exactly once at specified step.

    Args:
        step: Step number to run at
        extension: Extension to run

    Returns:
        Conditional extension

    Example:
        ext = once_at(25, inject_prompt("Now rewrite:"))
    """
    return when(lambda s: s.step == step, extension)


# ============================================================================
# State Transformers
# ============================================================================

def modify_threshold(multiplier: float) -> Extension:
    """Create extension that multiplies threshold.

    Args:
        multiplier: Factor to multiply threshold by

    Returns:
        Extension that modifies threshold

    Example:
        double_threshold = modify_threshold(2.0)
        halve_threshold = modify_threshold(0.5)
    """
    def modifier(state: GenState) -> GenState:
        return replace(state, threshold=state.threshold * multiplier)
    return modifier


def set_threshold(value: float) -> Extension:
    """Create extension that sets threshold to exact value.

    Args:
        value: Threshold value

    Returns:
        Extension that sets threshold

    Example:
        commit_phase = set_threshold(1.0)
    """
    def setter(state: GenState) -> GenState:
        return replace(state, threshold=value)
    return setter


def clamp_threshold(min_val: float = 0.05, max_val: float = 0.95) -> Extension:
    """Create extension that clamps threshold to range.

    Args:
        min_val: Minimum threshold
        max_val: Maximum threshold

    Returns:
        Extension that clamps threshold

    Example:
        bounded = clamp_threshold(0.05, 0.30)
    """
    def clamper(state: GenState) -> GenState:
        clamped = max(min_val, min(max_val, state.threshold))
        return replace(state, threshold=clamped)
    return clamper


def with_metadata(key: str, value: any) -> Extension:
    """Create extension that sets metadata key.

    Args:
        key: Metadata key
        value: Value to set

    Returns:
        Extension that sets metadata

    Example:
        mark_phase1 = with_metadata('phase', 1)
    """
    def setter(state: GenState) -> GenState:
        state.metadata[key] = value
        return state
    return setter


# ============================================================================
# Composition Patterns
# ============================================================================

def phase_switcher(
    phase1_positions: int,
    phase1_threshold: float,
    phase2_threshold: float
) -> Extension:
    """Create two-phase extension using combinators.

    Args:
        phase1_positions: When to switch phases
        phase1_threshold: Threshold for phase 1
        phase2_threshold: Threshold for phase 2

    Returns:
        Two-phase extension

    Example:
        ext = phase_switcher(100, 0.12, 1.0)
    """
    return pipe(
        # Phase 1: exploration
        when(
            lambda s: s.prompt_length + s.step < phase1_positions,
            set_threshold(phase1_threshold)
        ),
        # Phase 2: commitment
        when(
            lambda s: s.prompt_length + s.step >= phase1_positions,
            pipe(
                set_threshold(phase2_threshold),
                with_metadata('phase', 2)
            )
        )
    )


def adaptive_threshold(
    low_entropy: float = 1.5,
    high_entropy: float = 3.0,
    explore_mult: float = 2.0,
    conservative_mult: float = 0.5
) -> Extension:
    """Create adaptive threshold extension using combinators.

    Args:
        low_entropy: Entropy threshold for exploration
        high_entropy: Entropy threshold for conservation
        explore_mult: Multiplier when entropy is low
        conservative_mult: Multiplier when entropy is high

    Returns:
        Adaptive threshold extension

    Example:
        ext = adaptive_threshold(1.5, 3.0, 2.0, 0.5)
    """
    return pipe(
        branch(
            lambda s: s.entropy < low_entropy,
            pipe(modify_threshold(explore_mult), clamp_threshold(0.05, 0.30)),
            branch(
                lambda s: s.entropy > high_entropy,
                pipe(modify_threshold(conservative_mult), clamp_threshold(0.05, 0.30))
            )
        )
    )


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_composition():
    """Basic composition examples."""
    from .ultra_simple import confidence_surf, track_genealogy, watch_entropy

    print("Example 1: Basic Composition")
    print("-" * 60)

    # Pipe style (left to right)
    ext1 = pipe(watch_entropy, confidence_surf, track_genealogy)

    # Compose style (right to left)
    ext2 = compose(track_genealogy, confidence_surf, watch_entropy)

    # Both are equivalent!
    print("Created two equivalent composed extensions")
    print()


def example_conditional():
    """Conditional execution examples."""
    from .ultra_simple import confidence_surf, track_genealogy

    print("Example 2: Conditional Execution")
    print("-" * 60)

    # Only surf when branching
    ext = when(lambda s: s.branching_factor > 1, confidence_surf)

    # Track genealogy except first step
    ext = unless(lambda s: s.step == 0, track_genealogy)

    # Branch based on entropy
    ext = branch(
        lambda s: s.entropy > 2.0,
        modify_threshold(2.0),  # High entropy: explore
        modify_threshold(0.5)   # Low entropy: commit
    )

    print("Created conditional extensions")
    print()


def example_temporal():
    """Temporal control examples."""
    from .ultra_simple import confidence_surf

    print("Example 3: Temporal Control")
    print("-" * 60)

    # Phase-based control
    phase1 = before_step(50, set_threshold(0.12))
    phase2 = after_step(50, set_threshold(1.0))

    # Combine phases
    two_phase = pipe(phase1, phase2)

    # Window-based control
    exploration_window = between_steps(10, 40, confidence_surf)

    # Periodic execution
    log_every_10 = every_n_steps(10, with_metadata('checkpoint', True))

    print("Created temporal control extensions")
    print()


def example_complex_pipeline():
    """Complex pipeline example."""
    from .ultra_simple import confidence_surf, track_genealogy, watch_entropy

    print("Example 4: Complex Pipeline")
    print("-" * 60)

    # Build a sophisticated extension pipeline
    ext = pipe(
        # Always watch entropy
        watch_entropy,

        # Phase 1: Exploration (steps 0-100)
        before_step(100, pipe(
            set_threshold(0.12),
            when(lambda s: s.branching_factor > 1, track_genealogy),
            confidence_surf
        )),

        # Phase 2: Refinement (steps 100-150)
        between_steps(100, 150, pipe(
            set_threshold(0.20),
            confidence_surf
        )),

        # Phase 3: Commitment (steps 150+)
        after_step(150, set_threshold(1.0)),

        # Always clamp threshold
        clamp_threshold(0.05, 0.95)
    )

    print("Created complex multi-phase pipeline")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("COMPOSABLE EXTENSION SYSTEM")
    print("=" * 60)
    print()

    example_basic_composition()
    example_conditional()
    example_temporal()
    example_complex_pipeline()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ Functional composition (pipe, compose)
✅ Conditional execution (when, unless, branch)
✅ Temporal control (before_step, after_step, between_steps)
✅ State transformers (modify_threshold, set_threshold, clamp_threshold)
✅ Higher-order combinators
✅ No side effects (except metadata)
✅ Fully composable

Build complex behaviors from simple primitives!

Example:
    ext = pipe(
        watch_entropy,
        before_step(100, set_threshold(0.12)),
        after_step(100, set_threshold(1.0)),
        when(lambda s: s.branching_factor > 1, track_genealogy)
    )
""")

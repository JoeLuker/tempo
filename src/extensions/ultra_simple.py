"""Ultra-simple extension system - NO REGISTRY.

Just a list of functions. That's it.
"""

from dataclasses import dataclass, replace, field
from typing import Dict, Any, Tuple, Callable


@dataclass(frozen=True)
class GenState:
    """Immutable generation state."""
    step: int
    entropy: float
    threshold: float
    selected_tokens: Tuple[Tuple[int, float], ...]
    branching_factor: int
    prompt_length: int

    # Mutable metadata (only place we allow mutation)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


# Extension signature: state -> state
Extension = Callable[[GenState], GenState]


def run_extensions(state: GenState, extensions: list[Extension]) -> GenState:
    """Run a list of extensions.

    Args:
        state: Current state
        extensions: List of extension functions

    Returns:
        Modified state
    """
    for ext in extensions:
        state = ext(state)
    return state


# ============================================================================
# Built-in Extensions - Just functions, no decorators
# ============================================================================

def confidence_surf(state: GenState) -> GenState:
    """Adjust threshold based on entropy."""
    if state.entropy < 1.5:
        # Confident → explore more
        return replace(state, threshold=min(0.3, state.threshold * 2.0))
    elif state.entropy > 3.0:
        # Confused → be conservative
        return replace(state, threshold=max(0.05, state.threshold * 0.5))
    return state


def track_genealogy(state: GenState) -> GenState:
    """Track token genealogy."""
    if 'genealogy' not in state.metadata:
        state.metadata['genealogy'] = []

    state.metadata['genealogy'].append({
        'step': state.step,
        'tokens': list(state.selected_tokens),
        'branching': state.branching_factor,
    })

    return state


def watch_entropy(state: GenState) -> GenState:
    """Watch for entropy patterns."""
    if 'entropy_history' not in state.metadata:
        state.metadata['entropy_history'] = []

    history = state.metadata['entropy_history']
    history.append(state.entropy)

    # Detect spike
    if len(history) >= 2 and history[-1] - history[-2] > 2.0:
        state.metadata['capture_attention'] = True

    return state


def collect_pruned(state: GenState) -> GenState:
    """Collect pruned tokens for recovery."""
    if 'graveyard' not in state.metadata:
        state.metadata['graveyard'] = {}

    # Would get pruned tokens from state if available
    # For now, just maintain the structure

    return state


# ============================================================================
# Configuration Factories (closures for config)
# ============================================================================

def make_confidence_surf(low_thresh: float = 1.5, high_thresh: float = 3.0,
                        explore_mult: float = 2.0, conservative_mult: float = 0.5):
    """Create configured confidence surfing extension.

    Args:
        low_thresh: Entropy below this = confident
        high_thresh: Entropy above this = confused
        explore_mult: Multiply threshold by this when exploring
        conservative_mult: Multiply threshold by this when conservative

    Returns:
        Configured extension function
    """
    def extension(state: GenState) -> GenState:
        if state.entropy < low_thresh:
            return replace(state, threshold=min(0.3, state.threshold * explore_mult))
        elif state.entropy > high_thresh:
            return replace(state, threshold=max(0.05, state.threshold * conservative_mult))
        return state

    return extension


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic():
    """Basic usage - just a list."""
    print("Example 1: Basic usage")
    print("-" * 40)

    # Define which extensions to use
    extensions = [
        confidence_surf,
        track_genealogy,
        watch_entropy,
    ]

    # Run generation
    state = GenState(
        step=0,
        entropy=3.2,
        threshold=0.1,
        selected_tokens=((1, 0.5), (2, 0.3)),
        branching_factor=2,
        prompt_length=6,
    )

    state = run_extensions(state, extensions)

    print(f"Threshold: {state.threshold}")
    print(f"Genealogy entries: {len(state.metadata.get('genealogy', []))}")
    print()


def example_configured():
    """Using configured extensions."""
    print("Example 2: Configured extensions")
    print("-" * 40)

    # Create configured extensions
    aggressive_surf = make_confidence_surf(low_thresh=2.0, explore_mult=3.0)
    conservative_surf = make_confidence_surf(high_thresh=2.5, conservative_mult=0.3)

    # Pick which config to use
    extensions = [
        aggressive_surf,  # or conservative_surf
        track_genealogy,
    ]

    state = GenState(
        step=0, entropy=1.8, threshold=0.1,
        selected_tokens=((1, 0.5),), branching_factor=1, prompt_length=6
    )

    state = run_extensions(state, extensions)
    print(f"Threshold after aggressive: {state.threshold}")
    print()


def example_inline():
    """Inline - no extensions at all."""
    print("Example 3: Inline (no extensions)")
    print("-" * 40)

    state = GenState(
        step=0, entropy=3.5, threshold=0.1,
        selected_tokens=((1, 0.5),), branching_factor=1, prompt_length=6
    )

    # Just do it inline
    if state.entropy > 3.0:
        state = replace(state, threshold=0.05)

    # Track stuff
    state.metadata['tracked'] = True

    print(f"Threshold: {state.threshold}")
    print(f"Tracked: {state.metadata.get('tracked')}")
    print()


def example_conditional():
    """Conditional execution."""
    print("Example 4: Conditional execution")
    print("-" * 40)

    # Define extensions
    extensions = [confidence_surf, track_genealogy]

    state = GenState(
        step=0, entropy=2.5, threshold=0.1,
        selected_tokens=((1, 0.5),), branching_factor=1, prompt_length=6
    )

    # Only run some extensions conditionally
    active_extensions = []
    if state.step > 0:  # Skip first step
        active_extensions.append(confidence_surf)
    if state.branching_factor > 1:  # Only for parallel tokens
        active_extensions.append(track_genealogy)

    state = run_extensions(state, active_extensions)
    print(f"Extensions run: {len(active_extensions)}")
    print()


def example_multi_step():
    """Multi-step generation."""
    print("Example 5: Multi-step generation")
    print("-" * 40)

    extensions = [confidence_surf, track_genealogy, watch_entropy]

    state = GenState(
        step=0, entropy=3.0, threshold=0.1,
        selected_tokens=((1, 0.5),), branching_factor=1, prompt_length=6
    )

    # Simulate 5 steps
    for i in range(5):
        state = run_extensions(state, extensions)

        # Update for next step
        state = replace(
            state,
            step=i + 1,
            entropy=max(0.5, state.entropy - 0.4),
            selected_tokens=tuple((j, 0.2) for j in range((i % 2) + 1)),
            branching_factor=(i % 2) + 1,
        )

    print(f"Final threshold: {state.threshold}")
    print(f"Genealogy entries: {len(state.metadata.get('genealogy', []))}")
    print(f"Entropy history: {len(state.metadata.get('entropy_history', []))}")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("ULTRA-SIMPLE EXTENSION SYSTEM")
    print("=" * 60)
    print()

    example_basic()
    example_configured()
    example_inline()
    example_conditional()
    example_multi_step()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ No registry
✅ No decorators
✅ No magic
✅ Just a list of functions
✅ Extensions = [func1, func2, func3]
✅ run_extensions(state, extensions)

That's literally it.

To add an extension: Define a function, add to list
To configure: Use closures (factories)
To disable: Remove from list
To debug: Print in the function

TOTAL CODE: ~80 lines including examples
""")

"""Two-phase generation extension.

Phase 1: Explore with supertokens (threshold ~0.12)
Phase 2: Commit to single path (threshold 1.0)
"""

from dataclasses import replace
from src.extensions.ultra_simple import GenState, Extension


def make_two_phase(phase1_steps: int = 25, phase2_threshold: float = 1.0) -> Extension:
    """Create two-phase generation extension.

    Args:
        phase1_steps: Number of steps in Phase 1 (exploration)
        phase2_threshold: Threshold for Phase 2 (commitment)

    Returns:
        Extension function that switches phases at boundary

    Usage:
        extensions = [make_two_phase(phase1_steps=25, phase2_threshold=1.0)]
    """
    def extension(state: GenState) -> GenState:
        if state.step == phase1_steps:
            # Switch to Phase 2: force single-token selection
            return replace(state, threshold=phase2_threshold)
        return state

    return extension


def make_dynamic_two_phase(max_positions: int = 100, phase2_threshold: float = 1.0) -> Extension:
    """Create two-phase generation with dynamic position-based switching.

    Phase 1 continues until total position count reaches max_positions,
    then switches to Phase 2 with single-token selection.

    Args:
        max_positions: Maximum total positions (tokens including supertokens) in Phase 1
        phase2_threshold: Threshold for Phase 2 (commitment, default: 1.0)

    Returns:
        Extension function that switches based on position count

    Usage:
        # Allow up to 100 tokens in Phase 1, then commit
        extensions = [make_dynamic_two_phase(max_positions=100, phase2_threshold=1.0)]

    Note:
        Position count = prompt_length + step because:
        - prompt_length = initial tokens
        - Each generation step adds 1 position (even if multiple parallel tokens)
    """
    phase_switched = {'switched': False}  # Mutable state to track transition

    def extension(state: GenState) -> GenState:
        # Calculate current total positions
        # Position count = prompt + generated positions
        # Each step adds 1 position (supertokens share position)
        current_positions = state.prompt_length + state.step

        # Check if we've hit the limit and haven't switched yet
        if current_positions >= max_positions and not phase_switched['switched']:
            phase_switched['switched'] = True
            state.metadata['phase1_ended_at_step'] = state.step
            state.metadata['phase1_total_positions'] = current_positions
            return replace(state, threshold=phase2_threshold)

        return state

    return extension

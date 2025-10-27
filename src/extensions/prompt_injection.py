"""Prompt injection extension for interactive generation.

Allows injecting new prompt text at specified steps or conditions during generation.
This enables true multi-phase generation where the model can be given new instructions
mid-generation.
"""

from dataclasses import replace
from typing import Callable
from src.extensions.ultra_simple import GenState, Extension


def make_step_triggered_injection(
    trigger_step: int,
    prompt_text: str
) -> Extension:
    """Inject prompt text at a specific step.

    Args:
        trigger_step: Step number when prompt should be injected
        prompt_text: Text to inject (will be tokenized and appended to sequence)

    Returns:
        Extension function that signals prompt injection

    Usage:
        # At step 50, inject "Now rewrite the above code cleanly:"
        ext = make_step_triggered_injection(50, "Now rewrite the above code cleanly:")
    """
    def extension(state: GenState) -> GenState:
        if state.step == trigger_step:
            # Signal prompt injection via metadata
            state.metadata['inject_prompt'] = prompt_text
            state.metadata['injection_step'] = trigger_step
        return state

    return extension


def make_position_triggered_injection(
    trigger_positions: int,
    prompt_text: str
) -> Extension:
    """Inject prompt text when total positions reach a threshold.

    Args:
        trigger_positions: Position count when prompt should be injected
        prompt_text: Text to inject

    Returns:
        Extension function that signals prompt injection

    Usage:
        # At 100 positions, inject "Now complete the implementation:"
        ext = make_position_triggered_injection(100, "Now complete the implementation:")
    """
    injected = {'done': False}

    def extension(state: GenState) -> GenState:
        current_positions = state.prompt_length + state.step

        if current_positions >= trigger_positions and not injected['done']:
            injected['done'] = True
            state.metadata['inject_prompt'] = prompt_text
            state.metadata['injection_step'] = state.step
            state.metadata['injection_at_position'] = current_positions

        return state

    return extension


def make_condition_triggered_injection(
    condition: Callable[[GenState], bool],
    prompt_text: str,
    one_shot: bool = True
) -> Extension:
    """Inject prompt text when a condition is met.

    Args:
        condition: Function that takes GenState and returns bool
        prompt_text: Text to inject when condition is True
        one_shot: If True, only inject once (default: True)

    Returns:
        Extension function that signals prompt injection

    Usage:
        # When entropy drops below 1.0, inject refinement prompt
        def low_entropy(state):
            return state.entropy < 1.0

        ext = make_condition_triggered_injection(
            low_entropy,
            "The exploration phase is complete. Now write the final version:"
        )
    """
    injected = {'done': False}

    def extension(state: GenState) -> GenState:
        if one_shot and injected['done']:
            return state

        if condition(state):
            if one_shot:
                injected['done'] = True
            state.metadata['inject_prompt'] = prompt_text
            state.metadata['injection_step'] = state.step

        return state

    return extension


def make_interactive_phase(
    phase_transitions: list[tuple[int, str]]
) -> Extension:
    """Create multi-phase generation with automatic prompt injection.

    Args:
        phase_transitions: List of (position, prompt) tuples defining phase boundaries

    Returns:
        Extension that injects prompts at specified positions

    Usage:
        ext = make_interactive_phase([
            (50, "Now outline the solution:"),
            (100, "Now write the complete implementation:"),
            (150, "Now add error handling and edge cases:")
        ])
    """
    phase_index = {'current': 0}

    def extension(state: GenState) -> GenState:
        current_positions = state.prompt_length + state.step
        current_phase = phase_index['current']

        # Check if we should transition to next phase
        if current_phase < len(phase_transitions):
            trigger_pos, prompt = phase_transitions[current_phase]
            if current_positions >= trigger_pos:
                phase_index['current'] += 1
                state.metadata['inject_prompt'] = prompt
                state.metadata['injection_step'] = state.step
                state.metadata['phase_number'] = phase_index['current']

        return state

    return extension


# Common prompt templates
REWRITE_PROMPT = "\n\nNow rewrite the above code cleanly without any exploration:\n\n"
REFINE_PROMPT = "\n\nNow refine and complete the implementation:\n\n"
ADD_TESTS_PROMPT = "\n\nNow add comprehensive tests for the above code:\n\n"
ADD_DOCS_PROMPT = "\n\nNow add documentation and docstrings:\n\n"
FIX_BUGS_PROMPT = "\n\nNow review and fix any bugs in the above code:\n\n"


def make_two_phase_rewrite(trigger_positions: int = 100) -> Extension:
    """Convenience function for two-phase: explore then rewrite.

    Args:
        trigger_positions: When to switch from exploration to rewrite phase

    Returns:
        Extension that injects rewrite prompt at position threshold
    """
    return make_position_triggered_injection(trigger_positions, REWRITE_PROMPT)

"""Interactive multi-phase generation with automatic prompt injection.

This module provides a higher-level wrapper around TEMPO generation
that enables true interactive multi-phase generation by running multiple
generation passes with prompt injection between phases.
"""

from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class PhaseConfig:
    """Configuration for a generation phase."""
    max_positions: int  # How many positions before switching to next phase
    threshold: float  # Selection threshold for this phase
    injection_prompt: Optional[str] = None  # Prompt to inject at start of NEXT phase


def run_interactive_generation(
    use_case: 'GenerateTextUseCase',
    initial_prompt: str,
    phases: list[PhaseConfig],
    base_config: 'GenerationConfig'
) -> 'GenerationResult':
    """Run multi-phase interactive generation.

    Args:
        use_case: The GenerateTextUseCase instance
        initial_prompt: Starting prompt
        phases: List of phase configurations
        base_config: Base generation config to modify per phase

    Returns:
        Final generation result

    Example:
        phases = [
            PhaseConfig(
                max_positions=100,
                threshold=0.12,
                injection_prompt="\\n\\nNow rewrite the above code cleanly:\\n\\n"
            ),
            PhaseConfig(
                max_positions=150,
                threshold=1.0,
                injection_prompt=None
            )
        ]
        result = run_interactive_generation(use_case, prompt, phases, config)
    """
    current_prompt = initial_prompt
    cumulative_result = None

    for phase_idx, phase in enumerate(phases):
        print(f"\\n=== Phase {phase_idx + 1}/{len(phases)} ===")
        print(f"Threshold: {phase.threshold}, Max positions: {phase.max_positions}")

        # Create config for this phase
        from ..domain.entities.parallel_generation import GenerationConfig
        phase_config = GenerationConfig(
            max_tokens=phase.max_positions,
            selection_threshold=phase.threshold,
            min_steps=base_config.min_steps,
            use_retroactive_removal=base_config.use_retroactive_removal,
            disable_kv_cache=base_config.disable_kv_cache,
            isolate_parallel_tokens=base_config.isolate_parallel_tokens,
            show_token_ids=base_config.show_token_ids,
            system_content=base_config.system_content,
            return_parallel_sets=base_config.return_parallel_sets,
            sequence_callback=base_config.sequence_callback
        )

        # Run generation for this phase
        result = use_case.execute(
            prompt=current_prompt,
            config=phase_config
        )

        # Update cumulative result
        if cumulative_result is None:
            cumulative_result = result
        else:
            # Merge results
            cumulative_result.generated_text += result.raw_generated_text
            cumulative_result.raw_generated_text += result.raw_generated_text
            cumulative_result.generation_time += result.generation_time

        # Prepare prompt for next phase
        if phase_idx < len(phases) - 1 and phase.injection_prompt:
            # Inject prompt for next phase
            current_prompt = result.generated_text + phase.injection_prompt
            print(f"Injecting prompt: '{phase.injection_prompt[:50]}...'")

    return cumulative_result

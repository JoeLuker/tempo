#!/usr/bin/env python3
"""Interactive multi-phase TEMPO generation.

This script demonstrates true interactive generation where we programmatically
control phase transitions and prompt injection.
"""

import sys
from src.experiments import ArgumentParser
from src.utils.model_utils import load_tempo_components
from src.application.use_cases.generate_text import GenerateTextUseCase
from src.domain.entities.parallel_generation import GenerationConfig
from src.extensions.two_phase import make_dynamic_two_phase


def run_interactive_generation(args_dict):
    """Run interactive multi-phase generation."""

    # Extract args
    prompt = args_dict['prompt']
    selection_threshold = args_dict['selection_threshold']
    phase1_positions = args_dict.get('max_positions', 100)
    phase2_threshold = args_dict.get('phase2_threshold', 1.0)
    max_tokens_total = args_dict.get('max_tokens', 300)

    # Load model components
    print("Loading TEMPO components...")
    components = load_tempo_components(args_dict)

    # Create use case
    use_case = GenerateTextUseCase(
        token_generator=components['token_generator'],
        tokenizer=components['tokenizer_adapter'],
        generation_strategy=components['strategy'],
        sequence_manager=components['sequence_manager'],
        rope_modifier=components.get('rope_modifier'),
        attention_manager=components.get('attention_manager'),
        formatter=components.get('formatter'),
        debug_mode=args_dict.get('debug_mode', False),
        extensions=None  # We'll handle phase switching ourselves
    )

    print("\n" + "="*60)
    print("INTERACTIVE MULTI-PHASE GENERATION")
    print("="*60)

    # PHASE 1: Exploration with supertokens
    print(f"\n🔍 Phase 1: Exploration (threshold={selection_threshold})")
    print(f"   Generating up to {phase1_positions} positions...")

    phase1_config = GenerationConfig(
        max_tokens=phase1_positions,
        selection_threshold=selection_threshold,
        min_steps=0,
        use_retroactive_removal=args_dict.get('use_retroactive_pruning', False),
        disable_kv_cache=args_dict.get('disable_kv_cache', False),
        isolate_parallel_tokens=args_dict.get('isolate_parallel_tokens', True),
        show_token_ids=False,
        system_content=None,
        return_parallel_sets=False
    )

    phase1_result = use_case.execute(
        prompt=prompt,
        config=phase1_config
    )

    print(f"✓ Phase 1 complete ({phase1_result.generation_time:.2f}s)")
    print(f"   Generated: {len(phase1_result.raw_generated_text)} chars")

    # Show Phase 1 output
    print("\n" + "-"*60)
    print("Phase 1 Output (with supertoken exploration):")
    print("-"*60)
    print(phase1_result.generated_text[:500])
    if len(phase1_result.generated_text) > 500:
        print("... (truncated)")

    # PHASE 2: Rewrite cleanly
    print(f"\n✍️  Phase 2: Clean Rewrite (threshold={phase2_threshold})")

    # Inject rewrite prompt
    rewrite_prompt = phase1_result.generated_text + "\n\nNow rewrite the above code in final clean form without exploration:\n\n"

    phase2_config = GenerationConfig(
        max_tokens=max_tokens_total - phase1_positions,
        selection_threshold=phase2_threshold,
        min_steps=0,
        use_retroactive_removal=False,
        disable_kv_cache=args_dict.get('disable_kv_cache', False),
        isolate_parallel_tokens=False,  # No isolation needed, single tokens only
        show_token_ids=False,
        system_content=None,
        return_parallel_sets=False
    )

    phase2_result = use_case.execute(
        prompt=rewrite_prompt,
        config=phase2_config
    )

    print(f"✓ Phase 2 complete ({phase2_result.generation_time:.2f}s)")
    print(f"   Generated: {len(phase2_result.raw_generated_text)} chars")

    # Show Phase 2 output
    print("\n" + "-"*60)
    print("Phase 2 Output (clean final version):")
    print("-"*60)
    print(phase2_result.raw_generated_text[:500])
    if len(phase2_result.raw_generated_text) > 500:
        print("... (truncated)")

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total time: {phase1_result.generation_time + phase2_result.generation_time:.2f}s")
    print(f"Phase 1: {phase1_positions} positions with threshold {selection_threshold}")
    print(f"Phase 2: Rewrite with threshold {phase2_threshold}")


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args_dict = ArgumentParser.parse_args()

        # Run interactive generation
        run_interactive_generation(args_dict)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

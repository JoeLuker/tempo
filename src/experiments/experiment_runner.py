"""Experiment runner for TEMPO text generation.

This module provides a simplified runner that uses the new domain-driven architecture.
"""

import torch
import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

# Domain imports
from src.domain.entities.parallel_generation import GenerationConfig
from src.domain.services.generation_orchestrator import GenerationOrchestrator

# Infrastructure imports
from src.infrastructure.model import ModelAdapter
from src.infrastructure.cache import CacheManager
from src.infrastructure.generation.token_generator_impl import TokenGeneratorImpl
from src.infrastructure.generation.standard_generation_strategy import StandardGenerationStrategy
from src.infrastructure.tokenization import TokenizerAdapter
from src.infrastructure.performance import PerformanceTracker
from src.infrastructure.selection import ThresholdTokenSelector

# Application imports
from src.application.use_cases.generate_text import GenerateTextUseCase
from src.application.services.sequence_manager import SequenceManager
from src.application.services.rope_service import RoPEService
from src.application.services.attention_service import AttentionService
from src.application.services.pruning_service import PruningService

# Model imports
from src.modeling.model_wrapper import TEMPOModelWrapper

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Responsible for running text generation experiments using TEMPO."""

    def __init__(self, model, tokenizer, device: str = "mps", skip_wrapping: bool = False):
        """Initialize the experiment runner.

        Args:
            model: The language model (wrapped or unwrapped)
            tokenizer: HuggingFace tokenizer
            device: Device to use for computation
            skip_wrapping: If True, don't auto-wrap the model in TEMPOModelWrapper
        """
        # Ensure model is wrapped in TEMPOModelWrapper if not skipping wrapping
        if not skip_wrapping and not isinstance(model, TEMPOModelWrapper):
            logger.warning("Model not wrapped with TEMPOModelWrapper. Wrapping now...")
            self.model = TEMPOModelWrapper(model)
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device
        self.debug_mode = False

    def run_experiment(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run a generation experiment with the given parameters.

        Args:
            args: Dictionary of experiment parameters

        Returns:
            dict[str, Any]: Results dictionary
        """
        # Extract parameters
        prompt = args.get("prompt", "")
        max_tokens = args.get("max_tokens", 100)
        selection_threshold = args.get("selection_threshold", 0.1)
        use_retroactive_removal = args.get("use_retroactive_removal", False)
        output_dir = args.get("output_dir", "./output")
        min_steps = args.get("min_steps", 0)
        show_token_ids = args.get("show_token_ids", False)
        debug_mode = args.get("debug_mode", False)
        disable_kv_cache = args.get("disable_kv_cache", False)
        enable_thinking = args.get("enable_thinking", False)
        isolate_parallel_tokens = not args.get("allow_intraset_token_visibility", False)
        use_mcts = args.get("use_mcts", False)

        # Set debug mode
        self.debug_mode = debug_mode
        if debug_mode:
            logger.info("Debug mode enabled for experiment runner")
            if hasattr(self.model, "set_debug_mode"):
                self.model.set_debug_mode(True)
                logger.info("Model debug mode ENABLED")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup progress tracking
        print(f"\n{'='*60}")
        print(f"TEMPO Text Generation")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:50]}...")
        print(f"Selection Threshold: {selection_threshold}")
        print(f"Max Tokens: {max_tokens}")
        print(f"Debug Mode: {debug_mode}")
        print(f"{'='*60}\n")

        # Create infrastructure components
        logger.info("Initializing components...")

        # 1. Model Adapter
        model_adapter = ModelAdapter(model=self.model, device=self.device)

        # 2. Cache Manager
        cache_manager = CacheManager()

        # 3. Performance Tracker
        performance_tracker = PerformanceTracker()

        # 4. Token Generator
        token_generator = TokenGeneratorImpl(
            model_adapter=model_adapter,
            cache_manager=cache_manager,
            performance_tracker=performance_tracker,
            debug_mode=debug_mode
        )

        # 5. Tokenizer Adapter
        tokenizer_adapter = TokenizerAdapter(tokenizer=self.tokenizer)

        # 6. Token Selector
        token_selector = ThresholdTokenSelector(debug_mode=debug_mode)

        # 7. Generation Strategy
        generation_strategy = StandardGenerationStrategy(
            token_selector=token_selector,
            threshold_strategy=None,  # TODO: Wire up dynamic thresholding
            tokenizer=tokenizer_adapter,
            debug_mode=debug_mode
        )

        # 8. Sequence Manager
        sequence_manager = SequenceManager(debug_mode=debug_mode)

        # 9. RoPE Service (for parallel token positioning)
        rope_service = RoPEService(device=self.device, debug_mode=debug_mode)

        # 10. Attention Service (for controlling parallel token visibility)
        attention_service = AttentionService(
            isolate_parallel_tokens=isolate_parallel_tokens,
            device=self.device,
            debug_mode=debug_mode
        )

        # 11. Pruning Service (for retroactive pruning)
        pruning_service = None
        if use_retroactive_removal:
            attention_threshold = args.get("attention_threshold", 0.01)
            pruning_service = PruningService(
                attention_threshold=attention_threshold,
                use_relative_attention=not args.get("no_relative_attention", False),
                relative_threshold=args.get("relative_threshold", 0.5),
                use_multi_scale=not args.get("no_multi_scale_attention", False),
                num_layers_to_use=args.get("num_layers_to_use", None),
                use_sigmoid_threshold=not args.get("no_sigmoid_threshold", False),
                sigmoid_steepness=args.get("sigmoid_steepness", 10.0),
                debug_mode=debug_mode
            )
            logger.info(f"Retroactive pruning enabled with attention threshold: {attention_threshold}")

        # 12. Text Formatter (for beautiful output with colored brackets)
        from src.application.services.text_formatter import TextFormatter
        formatter = TextFormatter(tokenizer=tokenizer_adapter, debug_mode=debug_mode)

        # Create generation config
        system_content = "Enable deep thinking subroutine." if enable_thinking else None

        config = GenerationConfig(
            max_tokens=max_tokens,
            selection_threshold=selection_threshold,
            min_steps=min_steps,
            use_retroactive_removal=use_retroactive_removal,
            disable_kv_cache=disable_kv_cache,
            isolate_parallel_tokens=isolate_parallel_tokens,
            show_token_ids=show_token_ids,
            system_content=system_content,
            return_parallel_sets=False
        )

        # Create use case
        use_case = GenerateTextUseCase(
            token_generator=token_generator,
            tokenizer=tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            rope_modifier=rope_service,
            attention_manager=attention_service,
            formatter=formatter,
            debug_mode=debug_mode
        )

        # Handle MCTS if requested
        if use_mcts:
            logger.warning("MCTS mode requested but not yet integrated with new architecture. Using standard generation.")

        # Run generation
        logger.info(f"Starting generation...")
        generation_start = time.time()

        try:
            result = use_case.execute(
                prompt=prompt,
                config=config,
                retroactive_remover=pruning_service
            )

            generation_time = time.time() - generation_start

            # Build results dictionary
            results = {
                "generated_text": result.generated_text,
                "raw_generated_text": result.raw_generated_text,
                "clean_text": getattr(result, "clean_text", result.raw_generated_text),
                "prompt": prompt,
                "selection_threshold": selection_threshold,
                "use_retroactive_removal": use_retroactive_removal,
                "min_steps": min_steps,
                "generation_time": result.generation_time,
                "removal_time": result.removal_time,
                "removal_steps": result.removal_steps,
                "isolate_parallel_tokens": isolate_parallel_tokens,
                "enable_thinking": enable_thinking,
                "use_mcts": use_mcts,
            }

            # Print results
            print(f"\n{'='*60}")
            print(f"Generation Results")
            print(f"{'='*60}")
            print(f"Generated Text:\n{result.generated_text}")
            print(f"{'='*60}")
            print(f"Generation Time: {generation_time:.2f}s")
            if max_tokens > 0:
                print(f"Tokens/Second: {max_tokens/generation_time:.2f}")
            print(f"{'='*60}\n")

            # Save results
            logger.info(f"Saving results to {output_path / 'results.json'}")
            with open(output_path / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            return results

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

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
        isolate_parallel_tokens = args.get("isolate", False)
        use_mcts = args.get("use_mcts", False)

        # Memory control parameters
        max_memory_gb = args.get("max_memory_gb", 36.0)
        max_parallel_tokens = args.get("max_parallel_tokens", None)
        max_cache_tokens = args.get("max_cache_tokens", None)

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

        # 2. Cache Manager with memory limits
        cache_manager = CacheManager()
        if max_cache_tokens is not None:
            logger.info(f"Setting cache token limit to {max_cache_tokens}")
            # Cache manager will apply limits when initializing KV cache

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

        # 13. Experiment Data Capture (if requested in config)
        data_capture = None
        if args.get('capture_attention') or args.get('capture_logits') or args.get('capture_kv_cache'):
            from src.experiments.data_capture import ExperimentDataCapture

            data_capture = ExperimentDataCapture(
                experiment_config=args,
                output_dir=output_path
            )
            logger.info(f"Experiment data capture enabled - output to: {output_path}")

        # Create generation config with memory controls
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
            return_parallel_sets=True,  # Enable to get real probabilities
            max_parallel_tokens=max_parallel_tokens,  # Memory control
            max_cache_tokens=max_cache_tokens  # Memory control
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
            data_capture=data_capture,
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

            # Extract attention weights from cache if available
            attention_matrix = None
            try:
                cached_attention = cache_manager.get_cached_attention()
                if cached_attention is not None:
                    attention_pattern, seq_len = cached_attention
                    attention_layers = attention_pattern.layers
                    if attention_layers and len(attention_layers) > 0:
                        # Stack all layers: [num_layers, batch, heads, seq, seq]
                        stacked = torch.stack(attention_layers)
                        # Take first batch, then average across layers and heads: [seq, seq]
                        attention_matrix = stacked[:, 0, :, :, :].mean(dim=[0, 1]).cpu().numpy().tolist()
                        logger.info(f"Extracted {len(attention_matrix)}x{len(attention_matrix[0])} attention matrix averaged across {len(attention_layers)} layers")
            except Exception as e:
                logger.warning(f"Failed to extract attention weights: {e}", exc_info=True)

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
                "all_original_token_sets": result.all_original_token_sets,  # Include real token probabilities
                "all_surviving_token_sets": result.all_surviving_token_sets,
                "attention_matrix": attention_matrix,  # Attention weights for visualization
            }

            # Handle output format
            output_json = args.get("output_json", False)
            json_output_file = args.get("json_output_file")

            if output_json:
                # Build structured parallel token data
                parallel_tokens = []
                if result.all_surviving_token_sets:
                    for step in sorted(result.all_surviving_token_sets.keys()):
                        token_set = result.all_surviving_token_sets[step]

                        # Decode tokens
                        token_texts = []
                        for token_id, prob in token_set:
                            decoded = tokenizer_adapter.decode_tokens([token_id])
                            text = decoded[0] if isinstance(decoded, list) else decoded
                            token_texts.append({
                                "token_id": token_id,
                                "text": text,
                                "probability": prob
                            })

                        # Check if this was pruned
                        was_pruned = False
                        original_count = 0
                        if result.all_original_token_sets and step in result.all_original_token_sets:
                            original_count = len(result.all_original_token_sets[step])
                            was_pruned = original_count > len(token_set)

                        parallel_tokens.append({
                            "step": step,
                            "tokens": token_texts,
                            "was_pruned": was_pruned,
                            "original_count": original_count
                        })

                # Prepare JSON output
                json_output = {
                    "prompt": prompt,
                    "tokens": parallel_tokens,
                    "config": {
                        "selection_threshold": selection_threshold,
                        "max_tokens": max_tokens,
                        "use_retroactive_removal": use_retroactive_removal,
                        "attention_threshold": args.get("attention_threshold") if use_retroactive_removal else None,
                        "use_mcts": use_mcts,
                        "isolate_parallel_tokens": isolate_parallel_tokens,
                        "enable_thinking": enable_thinking,
                        "seed": args.get("seed", 42),
                    },
                    "metrics": {
                        "generation_time": result.generation_time,
                        "removal_time": result.removal_time,
                        "removal_steps": result.removal_steps,
                        "total_steps": len(parallel_tokens),
                        "parallel_steps": sum(1 for pt in parallel_tokens if len(pt["tokens"]) > 1),
                        "pruned_steps": sum(1 for pt in parallel_tokens if pt["was_pruned"]),
                        "tokens_per_second": max_tokens / generation_time if generation_time > 0 else 0,
                    }
                }

                # Output JSON
                if json_output_file:
                    json_path = output_path / json_output_file
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_output, f, indent=2, ensure_ascii=False)
                    print(f"JSON output saved to: {json_path}")
                else:
                    print(json.dumps(json_output, indent=2, ensure_ascii=False))
            else:
                # Print formatted results
                print(f"\n{'='*60}")
                print(f"Generation Results")
                print(f"{'='*60}")
                print(f"Generated Text:\n{result.generated_text}")
                print(f"{'='*60}")
                print(f"Generation Time: {generation_time:.2f}s")
                if max_tokens > 0:
                    print(f"Tokens/Second: {max_tokens/generation_time:.2f}")
                print(f"{'='*60}\n")

            # Always save results to results.json for internal tracking
            logger.info(f"Saving results to {output_path / 'results.json'}")
            with open(output_path / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            return results

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

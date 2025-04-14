import torch
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..generation.parallel_generator import ParallelGenerator
from ..generation.token_generator import TokenGenerator
from ..pruning.pruner import Pruner
from ..visualization.token_visualizer import TokenVisualizer
from ..visualization.position_visualizer import PositionVisualizer
from ..modeling.model_wrapper import TEMPOModelWrapper
import time
from tqdm import tqdm


class ExperimentRunner:
    """
    Responsible for running experiments with the parallel generator.
    """

    def __init__(
        self, model, tokenizer, device: str = "mps", skip_wrapping: bool = False
    ):
        """
        Initialize the experiment runner.

        Args:
            model: The language model (wrapped or unwrapped)
            tokenizer: HuggingFace tokenizer
            device: Device to use for computation
            skip_wrapping: If True, don't auto-wrap the model in TEMPOModelWrapper
        """
        # Ensure model is wrapped in TEMPOModelWrapper if not skipping wrapping
        if not skip_wrapping and not isinstance(model, TEMPOModelWrapper):
            print("Warning: Model not wrapped with TEMPOModelWrapper. Wrapping now...")
            self.model = TEMPOModelWrapper(model)
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device

        # Initialize visualizers
        self.token_visualizer = TokenVisualizer()
        self.position_visualizer = PositionVisualizer()

        # Debug mode flag
        self.debug_mode = False

    def run_experiment(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a generation experiment with the given parameters.

        Args:
            args: Dictionary of experiment parameters

        Returns:
            Dict[str, Any]: Results dictionary
        """
        # Extract parameters from args
        prompt = args.get("prompt", "")
        max_tokens = args.get("max_tokens", 100)
        threshold = args.get("threshold", 0.1)
        use_pruning = args.get("use_pruning", False)
        save_visualization = args.get("save_visualization", True)
        output_dir = args.get("output_dir", "./output")
        bezier_points = args.get("bezier_points", [0.2, 0.8])
        min_steps = args.get("min_steps", 0)
        show_token_ids = args.get("show_token_ids", False)
        use_custom_rope = args.get("use_custom_rope", True)
        debug_mode = args.get("debug_mode", False)
        disable_kv_cache_consistency = args.get("disable_kv_cache_consistency", False)
        disable_kv_cache = args.get("disable_kv_cache", False)
        enable_thinking = args.get("enable_thinking", False)
        allow_intraset_token_visibility = args.get(
            "allow_intraset_token_visibility", False
        )
        no_preserve_isolated_tokens = args.get("no_preserve_isolated_tokens", False)
        use_mcts = args.get("use_mcts", False)

        # Convert flags for backward compatibility
        isolate_parallel_tokens = not allow_intraset_token_visibility

        # Only preserve isolated tokens by default if the user didn't explicitly request pruning
        if (
            use_pruning and no_preserve_isolated_tokens is False
        ):  # If pruning is requested and preservation wasn't explicitly disabled
            # Don't automatically preserve - user wants pruning
            preserve_all_isolated_tokens = False
        else:
            # Use default behavior - preserve isolated tokens
            preserve_all_isolated_tokens = (
                not no_preserve_isolated_tokens if isolate_parallel_tokens else None
            )

        # Print initialization progress
        setup_steps = [
            "Setting up experiment",
            "Configuring pruning",
            "Creating generator",
            "Starting generation",
        ]
        setup_progress = tqdm(setup_steps, desc="Experiment setup", unit="step")

        # Set debug mode
        self.debug_mode = debug_mode
        if debug_mode:
            print("Debug mode enabled for experiment runner")
            self.model.set_debug_mode(True)

        # Create output directory if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_progress.update(1)  # First step complete

        # Modify pruning setup
        pruner = None
        retroactive_pruner = None

        if use_pruning:
            attention_threshold = args.get("attention_threshold", 0.01)

            # Create regular pruner with coherence strategy
            from src.pruning import Pruner, RetroactivePruner

            pruner = Pruner(
                model=self.model,
                tokenizer=self.tokenizer,
                strategy="coherence",  # Use coherence strategy
                coherence_threshold=0.1,  # Lower base threshold
                device=self.device,
                use_dynamic_threshold=args.get("dynamic_threshold", False),
                max_steps=args.get("max_tokens", None),
                bezier_points=args.get("bezier_points", [0.2, 0.8]),
                final_threshold=args.get("final_threshold", 1.0),
            )

            # Create retroactive pruner
            retroactive_pruner = RetroactivePruner(
                model=self.model,
                tokenizer=self.tokenizer,
                attention_threshold=attention_threshold,
                device=self.device,
                debug_mode=True,  # Enable debug mode
                dynamic_threshold_manager=pruner.threshold_manager if args.get("dynamic_threshold", False) else None
            )

            print(
                f"Using coherence-based pruning with retroactive pruning (attention threshold: {attention_threshold})"
                + (", dynamic threshold enabled" if args.get("dynamic_threshold", False) else "")
            )

        setup_progress.update(1)  # Pruning setup complete

        # Create token generator for MCTS (if needed)
        token_generator = None
        if use_mcts or use_pruning:  # Create token generator for both MCTS and pruning
            token_generator = TokenGenerator(
                model=self.model, tokenizer=self.tokenizer, device=self.device
            )

            # Set debug mode after creation if needed
            if debug_mode:
                token_generator.set_debug_mode(True)

            # Initialize token generator with the prompt to ensure it has cached attention
            if use_pruning and pruner is not None:
                # Get input tensors from the prompt
                input_ids, attention_mask = token_generator.prepare_input_from_prompt(prompt)
                
                # Run the model once to get cached attention
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                    )
                    if hasattr(outputs, "attentions") and outputs.attentions:
                        token_generator.cached_attention = outputs.attentions
                        token_generator.cached_attention_seq_len = [input_ids.size(1)]

                # Set token generator for pruning
                pruner.strategy.set_token_generator(token_generator)

        # Create the appropriate generator based on the mode
        if use_mcts:
            # Get MCTS parameters
            mcts_simulations = args.get("mcts_simulations", 10)
            mcts_c_puct = args.get("mcts_c_puct", 1.0)
            mcts_depth = args.get("mcts_depth", 5)

            print(
                f"Using Monte Carlo Tree Search with {mcts_simulations} simulations per step"
            )

            # Import the MCTS generator
            from src.search import MCTSGenerator

            generator = MCTSGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                token_generator=token_generator,
                retroactive_pruner=retroactive_pruner,
                c_puct=mcts_c_puct,
                num_simulations=mcts_simulations,
                attention_threshold=args.get("attention_threshold", 0.01),
                device=self.device,
                debug_mode=debug_mode,
            )
        else:
            # Use the standard ParallelGenerator
            generator = ParallelGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                pruner=pruner,
                device=self.device,
                has_custom_attention=True,  # Assuming model supports custom attention
                use_custom_rope=use_custom_rope,
                debug_mode=debug_mode,
            )

            # Set token generator for pruning if available
            if token_generator is not None and pruner is not None:
                pruner.strategy.set_token_generator(token_generator)

        setup_progress.update(1)  # Generator created

        # Configure generator (for ParallelGenerator)
        if not use_mcts:
            if use_custom_rope:
                print("Using custom RoPE modifications for parallel token positioning")

            # Configure RoPE modifier if available and debug options are set
            if (
                use_custom_rope
                and hasattr(generator, "rope_modifier")
                and generator.rope_modifier is not None
            ):
                if debug_mode:
                    generator.rope_modifier.set_debug_mode(True)
                    print("RoPE modifier debug mode enabled")

                if disable_kv_cache_consistency:
                    generator.rope_modifier.enable_kv_cache_consistency(False)
                    print("RoPE modifier KV cache consistency disabled")

            if disable_kv_cache:
                print("KV caching disabled for more consistent attention patterns")

            if allow_intraset_token_visibility:
                print(
                    f"Parallel tokens visibility mode enabled (tokens can see each other within the same set)"
                )
            else:
                print("Parallel tokens are isolated (default: can't see each other)")

                # Indicate pruning status based on use_pruning and preserve_all_isolated_tokens
                if use_pruning:
                    if not preserve_all_isolated_tokens:
                        print(
                            "Pruning will evaluate isolated tokens (explicitly requested)"
                        )
                    else:
                        print(
                            "Isolated tokens will be preserved (pruning only evaluates non-isolated tokens)"
                        )
                elif no_preserve_isolated_tokens:
                    print(
                        "Warning: No-preserve flag set but pruning is disabled, has no effect"
                    )

        # Prepare messages format for Cogito model if thinking is enabled
        system_content = None
        if enable_thinking:
            print("Enabling Cogito's deep thinking mode")
            system_content = "Enable deep thinking subroutine."

        setup_progress.update(1)  # Setup complete, starting generation
        setup_progress.close()

        # Run generation with timing
        generation_start = time.time()
        print(f"Starting token generation with threshold={threshold}...")

        if use_mcts:
            # Generate with MCTS generator
            generated_text = generator.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=1.0, top_p=0.9
            )

            # Create results structure similar to ParallelGenerator
            results = {
                "generated_text": generated_text,
                "raw_generated_text": generated_text,
                "prompt": prompt,
                "threshold": threshold,
                "use_pruning": use_pruning,
                "min_steps": min_steps,
                "generation_time": time.time() - generation_start,
                "use_mcts": True,
                "mcts_simulations": mcts_simulations,
                "mcts_c_puct": mcts_c_puct,
            }
        else:
            # Generate with ParallelGenerator
            results = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                threshold=threshold,
                return_parallel_sets=save_visualization,
                use_pruning=use_pruning,
                min_steps=min_steps,
                show_token_ids=show_token_ids,
                debug_mode=debug_mode,
                disable_kv_cache=disable_kv_cache,
                system_content=system_content,
                isolate_parallel_tokens=isolate_parallel_tokens,
                preserve_all_isolated_tokens=preserve_all_isolated_tokens,
                retroactive_pruner=retroactive_pruner,
            )

        generation_time = time.time() - generation_start

        # Add experiment parameters to results
        if use_pruning:
            results["pruning_strategy"] = args.get("pruning_strategy", "coherence")
            results["coherence_threshold"] = args.get("coherence_threshold", 0.3)
            if args.get("dynamic_threshold", False):
                results["dynamic_threshold"] = True
                results["bezier_points"] = bezier_points
            if args.get("pruning_strategy", "") == "hybrid":
                results["diversity_steps"] = args.get("diversity_steps", 0)

        # Add Cogito-specific parameters
        if enable_thinking:
            results["enable_thinking"] = True

        # Add isolation mode to results
        if not use_mcts and isolate_parallel_tokens:
            results["isolate_parallel_tokens"] = True

        # Add MCTS parameters to results
        if use_mcts:
            results["use_mcts"] = True
            results["mcts_simulations"] = mcts_simulations
            results["mcts_c_puct"] = mcts_c_puct
            results["mcts_depth"] = mcts_depth

        # Add model wrapper information if debug mode is enabled
        if debug_mode:
            intermediate_values = getattr(self.model, "intermediate_values", {})
            results["captured_intermediate_values"] = list(intermediate_values.keys())

        # Save visualizations if requested and if not using MCTS
        if save_visualization and not use_mcts and "parallel_sets" in results:
            # Visualize token sets
            visualization_path = output_path / "token_sets.png"
            self.token_visualizer.visualize_token_sets(results, visualization_path)

            # Visualize positions
            self.position_visualizer.visualize_position_tokens(results, output_path)
            self.position_visualizer.visualize_token_probabilities(results, output_path)
            self.position_visualizer.visualize_parallel_sets(results, output_path)

        # Print generated text and statistics
        print("\nGenerated Text:")
        print("-" * 50)
        print(results["generated_text"])
        print("-" * 50)

        # Print statistics if not using MCTS
        if not use_mcts:
            self.token_visualizer.print_statistics(results)
        else:
            print(f"Generation completed in {generation_time:.2f} seconds")
            print(f"Average tokens/second: {max_tokens/generation_time:.2f}")

        # Save results to JSON - invariant: results must be savable
        with open(output_path / "results.json", "w") as f:
            # Create a copy of results with only serializable data
            serializable_results = {
                "generated_text": results["generated_text"],
                "raw_generated_text": results.get(
                    "raw_generated_text", results["generated_text"]
                ),
                "prompt": results["prompt"],
                "threshold": results["threshold"],
                "use_pruning": results["use_pruning"],
                "enable_thinking": enable_thinking,
                "use_mcts": use_mcts,
            }

            # Add MCTS params if available
            if use_mcts:
                serializable_results["mcts_simulations"] = mcts_simulations
                serializable_results["mcts_c_puct"] = mcts_c_puct
                serializable_results["mcts_depth"] = mcts_depth

            # Add pruning params if available
            if "pruning_strategy" in results:
                serializable_results["pruning_strategy"] = results["pruning_strategy"]
            if "coherence_threshold" in results:
                serializable_results["coherence_threshold"] = results[
                    "coherence_threshold"
                ]
            if "dynamic_threshold" in results:
                serializable_results["dynamic_threshold"] = results["dynamic_threshold"]
            if "bezier_points" in results:
                serializable_results["bezier_points"] = results["bezier_points"]
            if "diversity_steps" in results:
                serializable_results["diversity_steps"] = results["diversity_steps"]

            json.dump(serializable_results, f, indent=2)

        return results

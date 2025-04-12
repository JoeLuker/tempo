import torch
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..generation.parallel_generator import ParallelGenerator
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
    
    def __init__(self, model, tokenizer, device: str = "mps", skip_wrapping: bool = False):
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
        allow_parallel_token_visibility = args.get("allow_parallel_token_visibility", False)
        no_preserve_isolated_tokens = args.get("no_preserve_isolated_tokens", False)
        
        # Convert flags for backward compatibility
        isolate_parallel_tokens = not allow_parallel_token_visibility
        
        # Only preserve isolated tokens by default if the user didn't explicitly request pruning
        if use_pruning and no_preserve_isolated_tokens is False:  # If pruning is requested and preservation wasn't explicitly disabled
            # Don't automatically preserve - user wants pruning
            preserve_all_isolated_tokens = False
        else:
            # Use default behavior - preserve isolated tokens
            preserve_all_isolated_tokens = not no_preserve_isolated_tokens if isolate_parallel_tokens else None
        
        # Print initialization progress
        setup_steps = ["Setting up experiment", "Configuring pruning", "Creating generator", "Starting generation"]
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
            
            # Create retroactive pruner for pruning previous parallel sets
            from src.pruning import RetroactivePruner
            retroactive_pruner = RetroactivePruner(
                model=self.model,
                tokenizer=self.tokenizer,
                attention_threshold=attention_threshold,
                device=self.device,
                debug_mode=debug_mode
            )
            
            # Also create regular pruner if strategy is specified
            pruning_strategy = args.get("pruning_strategy", "attention")
            if pruning_strategy == "diversity":
                diversity_clusters = args.get("diversity_clusters", 3)
                from src.pruning import Pruner
                pruner = Pruner(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    strategy="diversity",  # Use diversity for initial selection
                    diversity_clusters=diversity_clusters,
                    device=self.device
                )
                
                print(f"Using diversity selection with {diversity_clusters} clusters")
            elif pruning_strategy == "hybrid":
                diversity_clusters = args.get("diversity_clusters", 3)
                diversity_steps = args.get("diversity_steps", 5)
                from src.pruning import Pruner
                pruner = Pruner(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    strategy="hybrid",
                    diversity_clusters=diversity_clusters,
                    device=self.device,
                    diversity_steps=diversity_steps
                )
                
                print(f"Using hybrid selection: diversity for {diversity_steps} steps, then attention")
            else:
                # Just use retroactive pruning
                print(f"Using retroactive pruning with attention threshold {attention_threshold}")
        
        setup_progress.update(1)  # Pruning setup complete
        
        # Create the generator
        generator = ParallelGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            pruner=pruner,
            device=self.device,
            has_custom_attention=True,  # Assuming model supports custom attention
            use_custom_rope=use_custom_rope,
            debug_mode=debug_mode
        )
        
        setup_progress.update(1)  # Generator created
        
        # Configure generator
        if use_custom_rope:
            print("Using custom RoPE modifications for parallel token positioning")
            
        # Configure RoPE modifier if available and debug options are set
        if use_custom_rope and generator.rope_modifier is not None:
            if debug_mode:
                generator.rope_modifier.set_debug_mode(True)
                print("RoPE modifier debug mode enabled")
                
            if disable_kv_cache_consistency:
                generator.rope_modifier.enable_kv_cache_consistency(False)
                print("RoPE modifier KV cache consistency disabled")
        
        if disable_kv_cache:
            print("KV caching disabled for more consistent attention patterns")
            
        if allow_parallel_token_visibility:
            print("Parallel tokens will be able to see each other (visibility enabled)")
        else:
            print("Parallel tokens are isolated (default: can't see each other)")
            
            # Indicate pruning status based on use_pruning and preserve_all_isolated_tokens
            if use_pruning:
                if not preserve_all_isolated_tokens:
                    print("Pruning will evaluate isolated tokens (explicitly requested)")
                else:
                    print("Isolated tokens will be preserved (pruning only evaluates non-isolated tokens)")
            elif no_preserve_isolated_tokens:
                print("Warning: No-preserve flag set but pruning is disabled, has no effect")
        
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
            retroactive_pruner=retroactive_pruner
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
        if isolate_parallel_tokens:
            results["isolate_parallel_tokens"] = True
        
        # Add model wrapper information if debug mode is enabled
        if debug_mode:
            intermediate_values = getattr(self.model, "intermediate_values", {})
            results["captured_intermediate_values"] = list(intermediate_values.keys())
        
        # Save visualizations if requested
        if save_visualization and "parallel_sets" in results:
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
        
        # Print statistics
        self.token_visualizer.print_statistics(results)
        
        # Save results to JSON - invariant: results must be savable
        with open(output_path / "results.json", "w") as f:
            # Create a copy of results with only serializable data
            serializable_results = {
                "generated_text": results["generated_text"],
                "raw_generated_text": results["raw_generated_text"],
                "prompt": results["prompt"],
                "threshold": results["threshold"],
                "use_pruning": results["use_pruning"],
                "enable_thinking": enable_thinking
            }
            
            # Add pruning params if available
            if "pruning_strategy" in results:
                serializable_results["pruning_strategy"] = results["pruning_strategy"]
            if "coherence_threshold" in results:
                serializable_results["coherence_threshold"] = results["coherence_threshold"]
            if "dynamic_threshold" in results:
                serializable_results["dynamic_threshold"] = results["dynamic_threshold"]
            if "bezier_points" in results:
                serializable_results["bezier_points"] = results["bezier_points"]
            if "diversity_steps" in results:
                serializable_results["diversity_steps"] = results["diversity_steps"]
            
            json.dump(serializable_results, f, indent=2)
        
        return results 
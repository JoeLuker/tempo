#!/usr/bin/env python3
import argparse
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys

# Add this for handling imports when run as a script
if __name__ == "__main__":
    # Add the parent directory to the Python path so we can import from src
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)
    from src.model_loader import load_model
    from src.parallel_generator import ParallelThresholdGenerator
    from src.retroactive_pruning import RetroactivePruner
else:
    # Regular imports when imported as a module
    from src.model_loader import load_model
    from src.parallel_generator import ParallelThresholdGenerator
    from src.retroactive_pruning import RetroactivePruner

def parse_args():
    parser = argparse.ArgumentParser(description="Run Parallel Threshold Output generation experiments")
    
    parser.add_argument("--prompt", type=str, default="In a surprising turn of events, scientists discovered that",
                        help="Text prompt to start generation")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="HuggingFace model name/path")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Probability threshold for token selection")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="Minimum number of generation steps before stopping for EOS tokens")
    parser.add_argument("--use-pruning", action="store_true",
                        help="Enable retroactive pruning")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--coherence-threshold", type=float, default=0.7,
                        help="Threshold for pruning tokens based on coherence")
    parser.add_argument("--diversity-clusters", type=int, default=3,
                        help="Number of clusters for diversity-optimized pruning")
    parser.add_argument("--pruning-strategy", type=str, default="coherence", choices=["coherence", "diversity", "hybrid"],
                        help="Pruning strategy to use: coherence, diversity, or hybrid")
    parser.add_argument("--diversity-steps", type=int, default=10,
                        help="Number of steps to use diversity pruning before switching to coherence (for hybrid strategy)")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Run experiments with various threshold values")
    parser.add_argument("--thresholds", type=str, default="0.01,0.05,0.1,0.2,0.3",
                        help="Comma-separated list of thresholds to try")
    parser.add_argument("--dynamic-threshold", action="store_true",
                        help="Use dynamic coherence threshold that increases over steps")
    parser.add_argument("--final-threshold", type=float, default=1.0,
                        help="Final value for dynamic threshold (default 1.0, sets collapse if 1.0)")
    parser.add_argument("--bezier-points", type=str, default="0.2,0.8",
                        help="Bezier curve control points (comma-separated, values between 0-1)")
    parser.add_argument("--bezier-preset", type=str, choices=["slow-start", "linear", "fast-start", "s-curve"], 
                        help="Preset Bezier curves: slow-start [0.2,0.8], linear [0.5,0.5], fast-start [0.8,0.2], s-curve [0.2,0.2]")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU execution (no MPS)")
    
    # New parallel generation options
    parser.add_argument("--standard-generation", action="store_true",
                        help="Use standard single-token generation instead of parallel generation")
    parser.add_argument("--require-custom-attention", action="store_true",
                        help="Require custom attention support for parallel generation")
    parser.add_argument("--custom-attention", action="store_true", default=True,
                        help="Enable custom attention masking for parallel generation")
    parser.add_argument("--no-custom-attention", action="store_false", dest="custom_attention",
                        help="Disable custom attention masking for parallel generation")
    
    return parser.parse_args()

def visualize_token_sets(results, output_path):
    """Visualize the parallel token sets."""
    if "parallel_sets" not in results:
        return
        
    token_sets = results["parallel_sets"]
    pruned_sets = results.get("pruned_sets", None)
    use_pruning = results.get("use_pruning", False)
    dynamic_threshold = results.get("dynamic_threshold", False)
    bezier_points = results.get("bezier_points", [0.2, 0.8])
    pruning_strategy = results.get("pruning_strategy", "coherence") 
    diversity_steps = results.get("diversity_steps", 0)
    min_steps = results.get("min_steps", 0)
    
    # Count number of tokens per step
    steps = list(range(len(token_sets)))
    token_counts = [len(s) for s in token_sets]
    
    if pruned_sets:
        pruned_counts = [len(s) for s in pruned_sets]
    
    plt.figure(figsize=(12, 6 if not dynamic_threshold else 9))
    
    # Plot token counts
    plt.subplot(1 if not dynamic_threshold else 3, 2, 1)
    plt.bar(steps, token_counts, alpha=0.7, label="Original")
    if pruned_sets:
        plt.bar(steps, pruned_counts, alpha=0.5, label="After Pruning")
        
        # If using hybrid strategy, show where the switch happens
        if pruning_strategy == "hybrid" and diversity_steps > 0 and diversity_steps < len(steps):
            plt.axvline(x=diversity_steps, color='red', linestyle='--', 
                        label=f"Switch from Diversity to Coherence")
        
        # If min_steps is set, show a line indicating where the model can start stopping for EOS
        if min_steps > 0 and min_steps < len(steps):
            plt.axvline(x=min_steps, color='green', linestyle='-.',
                       label=f"Min. Steps ({min_steps})")
            
        plt.legend()
    plt.xlabel("Generation Step")
    plt.ylabel("Number of Parallel Tokens")
    
    strategy_text = f"Threshold={results['threshold']}"
    if pruning_strategy == "hybrid":
        strategy_text += f", Hybrid (Diversity→Coherence at step {diversity_steps})"
    elif pruning_strategy:
        strategy_text += f", {pruning_strategy.capitalize()} pruning"
    
    if min_steps > 0:
        strategy_text += f", Min Steps={min_steps}"
        
    plt.title(f"Tokens per Step ({strategy_text})")
    
    # Plot token probabilities for each step
    plt.subplot(1 if not dynamic_threshold else 3, 2, 2)
    for i, token_set in enumerate(token_sets[:20]):  # Limit to first 20 steps for clarity
        probs = [t[1] for t in token_set]
        plt.scatter([i] * len(probs), probs, alpha=0.6)
    
    # If using hybrid strategy, show where the switch happens
    if pruning_strategy == "hybrid" and diversity_steps > 0 and diversity_steps < 20:
        plt.axvline(x=diversity_steps, color='red', linestyle='--')
    
    # If min_steps is set, show a line indicating where the model can start stopping for EOS
    if min_steps > 0 and min_steps < 20:
        plt.axvline(x=min_steps, color='green', linestyle='-.')
        
    plt.xlabel("Generation Step")
    plt.ylabel("Token Probability")
    plt.title("Token Probabilities by Step")
    
    # If dynamic threshold is used, plot the threshold progression
    if dynamic_threshold and pruned_sets:
        plt.subplot(3, 2, (3, 5))
        
        # Create estimated threshold values based on pruned tokens
        thresholds = []
        for i, (orig_set, pruned_set) in enumerate(zip(token_sets, pruned_sets)):
            if len(orig_set) <= 1 or len(pruned_set) <= 0:
                # Skip steps with no pruning effect
                continue
                
            # Estimate threshold as the lowest probability in pruned set
            min_prob_kept = min([t[1] for t in pruned_set])
            thresholds.append((i, min_prob_kept))
        
        if thresholds:
            steps_with_threshold, threshold_values = zip(*thresholds)
            plt.plot(steps_with_threshold, threshold_values, 'r-', marker='o', alpha=0.7, label="Observed")
            plt.xlabel("Generation Step")
            plt.ylabel("Threshold")
            plt.title("Dynamic Threshold Progression")
            
            # Plot the theoretical Bezier curve
            def cubic_bezier(t, p0, p1, p2, p3):
                return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                
            # Generate the theoretical curve
            t_values = np.linspace(0, 1, len(steps))
            base_threshold = results.get("coherence_threshold", 0.3)
            
            # Extract Bezier control points
            if bezier_points and len(bezier_points) == 2:
                p1, p2 = bezier_points
            else:
                p1, p2 = 0.2, 0.8  # default
                
            # Calculate curve shape
            bezier_shape = [cubic_bezier(t, 0.0, p1, p2, 1.0) for t in t_values]
            bezier_curve = [base_threshold + (1.0 - base_threshold) * shape for shape in bezier_shape]
            
            # Plot the theoretical curve
            plt.plot(steps, bezier_curve, 'b--', alpha=0.7, label=f"Bezier [{p1:.1f},{p2:.1f}]")
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)

def run_experiment(args):
    """Run a single experiment with the specified parameters."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse Bezier control points if using dynamic threshold
    bezier_points = None
    if args.dynamic_threshold:
        bezier_points = [float(p) for p in args.bezier_points.split(",")]
        if len(bezier_points) != 2:
            print("Warning: Bezier points should be exactly 2 values. Using default [0.2, 0.8]")
            bezier_points = [0.2, 0.8]  # Default to exponential-like curve
            
        # Apply presets if specified
        if args.bezier_preset:
            if args.bezier_preset == "slow-start":
                bezier_points = [0.2, 0.8]  # Starts slow, accelerates
                print("Using 'slow-start' Bezier preset: [0.2, 0.8]")
            elif args.bezier_preset == "linear":
                bezier_points = [0.5, 0.5]  # Approximates linear growth
                print("Using 'linear' Bezier preset: [0.5, 0.5]")
            elif args.bezier_preset == "fast-start":
                bezier_points = [0.8, 0.2]  # Starts fast, decelerates
                print("Using 'fast-start' Bezier preset: [0.8, 0.2]")
            elif args.bezier_preset == "s-curve":
                bezier_points = [0.2, 0.2]  # S-shaped curve with slow start and end
                print("Using 's-curve' Bezier preset: [0.2, 0.2]")
    
    # Load model and tokenizer with custom attention if requested
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(
        args.model, 
        use_mps=not args.cpu,
        use_custom_attention=args.custom_attention
    )
    
    # Determine generation mode
    if args.standard_generation:
        generation_mode = "standard"
        print("Using standard (single-token) generation mode")
    else:
        generation_mode = "parallel"
        if args.custom_attention:
            print("Using parallel generation with custom attention masking")
        else:
            print("Using parallel generation without custom attention masking")
    
    # Initialize pruner if needed
    pruner = None
    if args.use_pruning:
        print(f"Initializing pruner with strategy: {args.pruning_strategy}")
        if args.pruning_strategy == "coherence":
            print(f"Coherence threshold: {args.coherence_threshold}")
            if args.dynamic_threshold:
                print(f"Using comprehensive dynamic threshold")
                print(f"  - Starting at {args.coherence_threshold}")
                print(f"  - Gradually increasing to {args.final_threshold} over {args.max_tokens} steps")
                print(f"  - Using Bezier curve with control points: {bezier_points}")
                print(f"  - Reapplying to ALL token sets as threshold increases")
                if args.final_threshold >= 0.999:
                    print(f"  - All sets will collapse to single tokens by completion (final threshold ≈ 1.0)")
                else:
                    print(f"  - Sets will maintain multiple tokens based on threshold (final threshold < 1.0)")
        elif args.pruning_strategy == "diversity":
            print(f"Diversity clusters: {args.diversity_clusters}")
        elif args.pruning_strategy == "hybrid":
            print(f"Hybrid strategy using diversity for {args.diversity_steps} steps, then coherence")
            print(f"Diversity clusters: {args.diversity_clusters}")
            print(f"Coherence threshold: {args.coherence_threshold}")
            
        pruner = RetroactivePruner(
            model=model,
            tokenizer=tokenizer,
            coherence_threshold=args.coherence_threshold,
            diversity_clusters=args.diversity_clusters,
            pruning_strategy=args.pruning_strategy,
            device="mps" if not args.cpu and torch.backends.mps.is_available() else "cpu",
            use_dynamic_threshold=args.dynamic_threshold,
            max_steps=args.max_tokens,
            bezier_points=bezier_points,
            final_threshold=args.final_threshold,
            diversity_steps=args.diversity_steps
        )
    
    # Initialize generator
    print(f"Initializing generator with threshold: {args.threshold}")
    generator = ParallelThresholdGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
        device="mps" if not args.cpu and torch.backends.mps.is_available() else "cpu",
        pruner=pruner
    )
    
    # Generate text
    print(f"Generating with prompt: '{args.prompt}'")
    print(f"Using min_steps={args.min_steps}, max_tokens={args.max_tokens}")
    
    if generation_mode == "standard" and not args.custom_attention:
        # For standard generation without custom attention, use standard methods
        with torch.no_grad():
            input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to("mps" if not args.cpu and torch.backends.mps.is_available() else "cpu")
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + args.max_tokens, 
                do_sample=True,
                min_length=input_ids.shape[1] + args.min_steps  # Ensure min_steps tokens are generated
            )
            results = {
                "generated_text": tokenizer.decode(output[0], skip_special_tokens=True),
                "raw_generated_text": tokenizer.decode(output[0], skip_special_tokens=True),
                "prompt": args.prompt,
                "threshold": args.threshold,
                "use_pruning": args.use_pruning,
                "min_steps": args.min_steps
            }
    else:
        # Use parallel generation
        results = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            threshold=args.threshold,
            return_parallel_sets=True,
            use_pruning=args.use_pruning,
            require_custom_attention=args.require_custom_attention,
            min_steps=args.min_steps
        )
    
    # Add dynamic threshold info to results for visualization
    if args.use_pruning and args.dynamic_threshold:
        results["dynamic_threshold"] = True
        results["bezier_points"] = bezier_points
        
        # Check that all sets collapsed to a single token at the end
        if "pruned_sets" in results and results["pruned_sets"]:
            last_set = results["pruned_sets"][-1]
            print(f"\nDynamic threshold final pruning result:")
            print(f"Number of tokens in final step: {len(last_set)}")
            if len(last_set) == 1:
                print(f"SUCCESS: Final token set collapsed to a single token as expected")
            else:
                print(f"WARNING: Final token set did not collapse to a single token!")
    
    # Add pruning strategy info to results
    if args.use_pruning:
        results["pruning_strategy"] = args.pruning_strategy
        if args.pruning_strategy == "hybrid":
            results["diversity_steps"] = args.diversity_steps
    
    # Print generated text
    print("\nGenerated Text:")
    print("-" * 50)
    print(results["generated_text"])
    print("-" * 50)
    
    # Print statistics
    if "parallel_sets" in results:
        token_sets = results["parallel_sets"]
        token_counts = [len(s) for s in token_sets]
        
        print("\nParallel Generation Statistics:")
        print(f"Total steps: {len(token_sets)}")
        print(f"Average tokens per step: {np.mean(token_counts):.2f}")
        print(f"Max tokens in a step: {max(token_counts)}")
        
        if args.use_pruning and "pruned_sets" in results:
            pruned_sets = results["pruned_sets"]
            pruned_counts = [len(s) for s in pruned_sets]
            
            print(f"\nPruning Statistics ({args.pruning_strategy} strategy):")
            if args.pruning_strategy == "hybrid":
                print(f"Strategy: Diversity for {args.diversity_steps} steps, then Coherence")
            print(f"Average tokens before pruning: {np.mean(token_counts):.2f}")
            print(f"Average tokens after pruning: {np.mean(pruned_counts):.2f}")
            print(f"Max tokens before pruning: {max(token_counts)}")
            print(f"Max tokens after pruning: {max(pruned_counts)}")
            
            # Calculate the average reduction percentage
            # Avoid division by zero
            reductions = []
            for orig, pruned in zip(token_counts, pruned_counts):
                if orig > 0:
                    reduction = (1 - pruned / orig) * 100
                    reductions.append(reduction)
            
            if reductions:
                avg_reduction = np.mean(reductions)
                print(f"Average reduction: {avg_reduction:.1f}%")
                
                # Count how many sets had any pruning applied
                sets_pruned = sum(1 for orig, pruned in zip(token_counts, pruned_counts) if orig > pruned)
                print(f"Sets with pruning applied: {sets_pruned}/{len(token_sets)} ({(sets_pruned/len(token_sets))*100:.1f}%)")
            else:
                print("No reduction data available")
    
    # Save results
    strategy_name = "standard" if generation_mode == "standard" else args.pruning_strategy if args.use_pruning else "parallel"
    output_file = output_dir / f"results_{strategy_name}_thresh{args.threshold}.json"
    with open(output_file, "w") as f:
        # Convert any numpy values to Python scalars for JSON serialization
        json_serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                json_serializable_results[k] = v.item()
            else:
                json_serializable_results[k] = v
                
        json.dump(json_serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return results

def run_threshold_sweep(args):
    """Run experiments with multiple threshold values."""
    thresholds = [float(t) for t in args.thresholds.split(",")]
    
    # Determine generation mode for display
    generation_mode = "standard" if args.standard_generation else "parallel"
    attention_mode = "with custom attention" if args.custom_attention else "without custom attention"
    
    print(f"Running {generation_mode} threshold sweep {attention_mode} with values: {thresholds}")
    
    # Print pruning strategy info
    if args.use_pruning:
        if args.pruning_strategy == "hybrid":
            print(f"Using hybrid pruning strategy: diversity for {args.diversity_steps} steps, then coherence")
        else:
            print(f"Using {args.pruning_strategy} pruning strategy")
            
    all_results = []
    
    for threshold in tqdm(thresholds):
        args.threshold = threshold
        results = run_experiment(args)
        all_results.append(results)
    
    # Create comparison visualizations
    output_dir = Path(args.output_dir)
    
    # Compare number of tokens per threshold
    plt.figure(figsize=(10, 6))
    
    # For parallel generation, get average tokens per step
    if not args.standard_generation and "parallel_sets" in all_results[0]:
        avg_tokens = [
            np.mean([len(s) for s in result["parallel_sets"]])
            for result in all_results
        ]
        
        plt.plot(thresholds, avg_tokens, marker='o')
        plt.xlabel("Threshold Value")
        plt.ylabel("Average Tokens per Step")
        plt.title(f"Effect of Threshold on {generation_mode.capitalize()} Token Generation")
        plt.grid(True, alpha=0.3)
        
        # Also calculate and plot average token probabilities
        if len(all_results) > 0 and "parallel_sets" in all_results[0]:
            plt.figure(figsize=(10, 6))
            avg_probs = []
            
            for result in all_results:
                # Calculate average probability of all tokens across all sets
                all_probs = []
                for token_set in result["parallel_sets"]:
                    all_probs.extend([prob for _, prob in token_set])
                
                avg_probs.append(np.mean(all_probs) if all_probs else 0)
            
            plt.plot(thresholds, avg_probs, marker='o', color='orange')
            plt.xlabel("Threshold Value")
            plt.ylabel("Average Token Probability")
            plt.title(f"Average Token Probability by Threshold ({generation_mode.capitalize()})")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / "probability_by_threshold.png")
    
    # Save main comparison plot
    plt.savefig(output_dir / "threshold_comparison.png")
    
    # Save sweep results
    with open(output_dir / "threshold_sweep_summary.json", "w") as f:
        summary = {
            "thresholds": thresholds,
            "generation_mode": generation_mode,
            "custom_attention": args.custom_attention,
            "prompt": args.prompt,
            "min_steps": args.min_steps
        }
        
        # Add mode-specific metrics
        if not args.standard_generation and "parallel_sets" in all_results[0]:
            summary["avg_tokens_per_step"] = avg_tokens
            if 'avg_probs' in locals():
                summary["avg_token_probabilities"] = avg_probs
        
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    # Show generation mode based on args
    if args.standard_generation:
        print("Running in standard generation mode")
    else:
        if args.custom_attention:
            print("Running in parallel generation mode WITH custom attention (default)")
        else:
            print("Running in parallel generation mode WITHOUT custom attention")
    
    if args.threshold_sweep:
        run_threshold_sweep(args)
    else:
        run_experiment(args)
    
    print("\nGeneration Complete!")
    if args.use_pruning and args.pruning_strategy == "hybrid":
        print(f"Used hybrid pruning: diversity for {args.diversity_steps} steps, then coherence")
    if args.min_steps > 0:
        print(f"Enforced minimum of {args.min_steps} steps before stopping for EOS tokens")
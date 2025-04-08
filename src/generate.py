#!/usr/bin/env python3
import argparse
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model_loader import load_model
from parallel_generator import ParallelThresholdGenerator
from retroactive_pruning import RetroactivePruner

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
    parser.add_argument("--use-pruning", action="store_true",
                        help="Enable retroactive pruning")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--coherence-threshold", type=float, default=0.3,
                        help="Threshold for pruning tokens based on coherence")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Run experiments with various threshold values")
    parser.add_argument("--thresholds", type=str, default="0.01,0.05,0.1,0.2,0.3",
                        help="Comma-separated list of thresholds to try")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU execution (no MPS)")
    
    return parser.parse_args()

def visualize_token_sets(results, output_path):
    """Visualize the parallel token sets."""
    if "parallel_sets" not in results:
        return
        
    token_sets = results["parallel_sets"]
    pruned_sets = results.get("pruned_sets", None)
    
    # Count number of tokens per step
    steps = list(range(len(token_sets)))
    token_counts = [len(s) for s in token_sets]
    
    plt.figure(figsize=(12, 6))
    
    # Plot token counts
    plt.subplot(1, 2, 1)
    plt.bar(steps, token_counts)
    plt.xlabel("Generation Step")
    plt.ylabel("Number of Parallel Tokens")
    plt.title(f"Tokens per Step (Threshold={results['threshold']})")
    
    # Plot token probabilities for each step
    plt.subplot(1, 2, 2)
    for i, token_set in enumerate(token_sets[:20]):  # Limit to first 20 steps for clarity
        probs = [t[1] for t in token_set]
        plt.scatter([i] * len(probs), probs, alpha=0.6)
    
    plt.xlabel("Generation Step")
    plt.ylabel("Token Probability")
    plt.title("Token Probabilities by Step")
    
    plt.tight_layout()
    plt.savefig(output_path)
    
def run_experiment(args):
    """Run a single experiment with the specified parameters."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, use_mps=not args.cpu)
    
    # Initialize pruner if needed
    pruner = None
    if args.use_pruning:
        print(f"Initializing pruner with coherence threshold: {args.coherence_threshold}")
        pruner = RetroactivePruner(
            model=model,
            tokenizer=tokenizer,
            coherence_threshold=args.coherence_threshold,
            device="mps" if not args.cpu and torch.backends.mps.is_available() else "cpu"
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
    results = generator.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        return_parallel_sets=True,
        use_pruning=args.use_pruning
    )
    
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
        
    
    # Save results
    output_file = output_dir / f"results_thresh{args.threshold}.json"
    with open(output_file, "w") as f:
        # Convert any non-serializable objects to strings
        serializable_results = {
            k: (str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v)
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    # Visualize token sets
    visualize_token_sets(results, output_dir / f"parallel_tokens_thresh{args.threshold}.png")
    
    return results

def run_threshold_sweep(args):
    """Run experiments with multiple threshold values."""
    thresholds = [float(t) for t in args.thresholds.split(",")]
    
    print(f"Running threshold sweep with values: {thresholds}")
    all_results = []
    
    for threshold in tqdm(thresholds):
        args.threshold = threshold
        results = run_experiment(args)
        all_results.append(results)
    
    # Create comparison visualizations
    output_dir = Path(args.output_dir)
    
    # Compare number of tokens per threshold
    plt.figure(figsize=(10, 6))
    avg_tokens = [
        np.mean([len(s) for s in result["parallel_sets"]])
        for result in all_results
    ]
    
    plt.plot(thresholds, avg_tokens, marker='o')
    plt.xlabel("Threshold Value")
    plt.ylabel("Average Tokens per Step")
    plt.title("Effect of Threshold on Parallel Token Generation")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "threshold_comparison.png")
    
    # Save sweep results
    with open(output_dir / "threshold_sweep_summary.json", "w") as f:
        summary = {
            "thresholds": thresholds,
            "avg_tokens_per_step": avg_tokens,
            "prompt": args.prompt
        }
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    if args.threshold_sweep:
        run_threshold_sweep(args)
    else:
        run_experiment(args) 
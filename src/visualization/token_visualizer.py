import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class TokenVisualizer:
    """
    Responsible for visualizing token sets and generation statistics.
    """

    def __init__(self):
        """Initialize the token visualizer."""
        pass

    def visualize_token_sets(self, results: Dict[str, Any], output_path: str):
        """
        Visualize the parallel token sets.

        Args:
            results: Dictionary of generation results
            output_path: Path to save the visualization
        """
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

        # Initialize pruned_counts as empty list to avoid undefined variable error
        pruned_counts = []
        if pruned_sets:
            pruned_counts = [len(s) for s in pruned_sets]

        plt.figure(figsize=(12, 6 if not dynamic_threshold else 9))

        # Plot token counts
        plt.subplot(1 if not dynamic_threshold else 3, 2, 1)
        plt.bar(steps, token_counts, alpha=0.7, label="Original")

        if pruned_sets:
            plt.bar(steps, pruned_counts, alpha=0.5, label="After Pruning")

            # If using hybrid strategy, show where the switch happens
            if (
                pruning_strategy == "hybrid"
                and diversity_steps > 0
                and diversity_steps < len(steps)
            ):
                plt.axvline(
                    x=diversity_steps,
                    color="red",
                    linestyle="--",
                    label=f"Switch from Diversity to Coherence",
                )

            # If min_steps is set, show a line indicating where the model can start stopping for EOS
            if min_steps > 0 and min_steps < len(steps):
                plt.axvline(
                    x=min_steps,
                    color="green",
                    linestyle="-.",
                    label=f"Min. Steps ({min_steps})",
                )

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
        for i, token_set in enumerate(
            token_sets[:20]
        ):  # Limit to first 20 steps for clarity
            probs = [t[1] for t in token_set]
            plt.scatter([i] * len(probs), probs, alpha=0.6)

        # If using hybrid strategy, show where the switch happens
        if (
            pruning_strategy == "hybrid"
            and diversity_steps > 0
            and diversity_steps < 20
        ):
            plt.axvline(x=diversity_steps, color="red", linestyle="--")

        # If min_steps is set, show a line indicating where the model can start stopping for EOS
        if min_steps > 0 and min_steps < 20:
            plt.axvline(x=min_steps, color="green", linestyle="-.")

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
                plt.plot(
                    steps_with_threshold,
                    threshold_values,
                    "r-",
                    marker="o",
                    alpha=0.7,
                    label="Observed",
                )
                plt.xlabel("Generation Step")
                plt.ylabel("Threshold")
                plt.title("Dynamic Threshold Progression")

                # Plot the theoretical Bezier curve
                def cubic_bezier(t, p0, p1, p2, p3):
                    return (
                        (1 - t) ** 3 * p0
                        + 3 * (1 - t) ** 2 * t * p1
                        + 3 * (1 - t) * t**2 * p2
                        + t**3 * p3
                    )

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
                bezier_curve = [
                    base_threshold + (1.0 - base_threshold) * shape
                    for shape in bezier_shape
                ]

                # Plot the theoretical curve
                plt.plot(
                    steps,
                    bezier_curve,
                    "b--",
                    alpha=0.7,
                    label=f"Bezier [{p1:.1f},{p2:.1f}]",
                )
                plt.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def print_statistics(self, results: Dict[str, Any]):
        """
        Print statistics about the generation.

        Args:
            results: Dictionary of generation results
        """
        if "parallel_sets" not in results:
            return

        token_sets = results["parallel_sets"]
        token_counts = [len(s) for s in token_sets]

        print("\nParallel Generation Statistics:")
        print(f"Total steps: {len(token_sets)}")
        print(f"Average tokens per step: {np.mean(token_counts):.2f}")
        print(f"Max tokens in a step: {max(token_counts)}")

        if results.get("use_pruning") and "pruned_sets" in results:
            pruned_sets = results["pruned_sets"]
            pruned_counts = [len(s) for s in pruned_sets]
            pruning_strategy = results.get("pruning_strategy", "coherence")

            print(f"\nPruning Statistics ({pruning_strategy} strategy):")
            if pruning_strategy == "hybrid":
                print(
                    f"Strategy: Diversity for {results.get('diversity_steps', 0)} steps, then Coherence"
                )
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
                sets_pruned = sum(
                    1
                    for orig, pruned in zip(token_counts, pruned_counts)
                    if orig > pruned
                )
                print(
                    f"Sets with pruning applied: {sets_pruned}/{len(token_sets)} ({(sets_pruned/len(token_sets))*100:.1f}%)"
                )
            else:
                print("No reduction data available")

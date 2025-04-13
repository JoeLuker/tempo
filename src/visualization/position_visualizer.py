import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any
from pathlib import Path


class PositionVisualizer:
    """
    Responsible for visualizing tokens at different positions.
    """

    def __init__(self):
        """Initialize the position visualizer."""
        pass

    def visualize_position_tokens(self, results: Dict[str, Any], output_dir: Path):
        """
        Visualize tokens at each position.

        Args:
            results: Dictionary of generation results
            output_dir: Directory to save visualizations
        """
        if "position_to_tokens" not in results:
            print("Position information not found in results.")
            return

        position_to_tokens = results["position_to_tokens"]

        # Count tokens per position
        positions = sorted([int(p) for p in position_to_tokens.keys()])
        token_counts = [len(position_to_tokens[str(p)]) for p in positions]

        # Calculate position indices (starting after the prompt)
        prompt_len = len(results["prompt"].split())
        generation_positions = [i for i in range(len(positions) - prompt_len)]

        # Skip visualization if no generation positions
        if not generation_positions or len(token_counts) <= prompt_len:
            print("No generation positions to visualize.")
            return

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot number of tokens at each position
        plt.subplot(2, 1, 1)
        plt.bar(generation_positions, token_counts[prompt_len:], color="skyblue")
        plt.xlabel("Generation Position")
        plt.ylabel("Number of Parallel Tokens")
        plt.title(f"Number of Tokens per Position (threshold={results['threshold']})")

        # Create a visualization of tokens at each position
        plt.subplot(2, 1, 2)

        # Create a matrix representation
        max_tokens = max(token_counts[prompt_len:]) if token_counts[prompt_len:] else 0
        if max_tokens == 0:
            print("No tokens to visualize in the heatmap.")
            plt.text(0.5, 0.5, "No tokens to visualize", ha="center", va="center")
        else:
            token_matrix = np.zeros((len(generation_positions), max_tokens))
            token_labels = []

            for i, pos in enumerate(positions[prompt_len:]):
                tokens = position_to_tokens[str(pos)]
                for j, token in enumerate(tokens):
                    # Use probability as color intensity if available
                    token_matrix[i, j] = 1
                token_labels.append([t.strip() for t in tokens])

            # Plot heatmap
            ax = plt.gca()
            sns.heatmap(token_matrix, cmap="Blues", cbar=False, ax=ax)

            # Add token text as annotations
            for i in range(token_matrix.shape[0]):
                for j in range(len(token_labels[i])):
                    if token_matrix[i, j] > 0:
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            token_labels[i][j],
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        plt.tight_layout()
        plt.savefig(output_dir / "position_tokens.png")
        plt.close()

    def visualize_token_probabilities(self, results: Dict[str, Any], output_dir: Path):
        """
        Visualize token probabilities across positions.

        Args:
            results: Dictionary of generation results
            output_dir: Directory to save visualizations
        """
        if "parallel_sets" not in results:
            print("Parallel sets information not found in results.")
            return

        parallel_sets = results["parallel_sets"]
        pruned_sets = results.get("pruned_sets", None)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot token probabilities across positions
        for i, token_set in enumerate(parallel_sets):
            # Extract probabilities
            probs = [t[1] for t in token_set]
            # Plot points
            plt.scatter(
                [i] * len(probs),
                probs,
                alpha=0.7,
                c="blue",
                label="All tokens" if i == 0 else "",
            )

            # If pruning was used, highlight pruned tokens
            if pruned_sets:
                pruned_probs = [t[1] for t in pruned_sets[i]]
                if len(pruned_probs) < len(probs):
                    plt.scatter(
                        [i] * len(pruned_probs),
                        pruned_probs,
                        alpha=1.0,
                        c="red",
                        marker="x",
                        s=100,
                        label="Kept after pruning" if i == 0 else "",
                    )

        plt.xlabel("Generation Position")
        plt.ylabel("Token Probability")
        plt.title(f"Token Probabilities by Position (threshold={results['threshold']})")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "token_probabilities.png")
        plt.close()

    def visualize_parallel_sets(self, results: Dict[str, Any], output_dir: Path):
        """
        Visualize the parallel sets of tokens.

        Args:
            results: Dictionary of generation results
            output_dir: Directory to save visualizations
        """
        if "parallel_sets" not in results:
            print("Parallel sets information not found in results.")
            return

        parallel_sets = results["parallel_sets"]

        # Create figure for parallel sets visualization (show first 10 positions)
        num_pos = min(10, len(parallel_sets))
        fig, axes = plt.subplots(num_pos, 1, figsize=(12, 2 * num_pos))

        for i in range(num_pos):
            tokens = parallel_sets[i]
            ax = axes[i] if num_pos > 1 else axes

            # Sort by probability
            tokens.sort(key=lambda x: x[1], reverse=True)

            # Plot probabilities
            token_texts = [t[0].strip() for t in tokens]
            probs = [t[1] for t in tokens]

            ax.barh(token_texts, probs)
            ax.set_title(f"Position {i}")
            ax.set_xlabel("Probability")

            # Add values to bars
            for j, v in enumerate(probs):
                ax.text(v + 0.01, j, f"{v:.3f}", va="center")

        plt.tight_layout()
        plt.savefig(output_dir / "parallel_sets.png")
        plt.close()

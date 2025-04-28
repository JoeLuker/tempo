import argparse
from typing import Dict, Any


class ArgumentParser:
    """
    Responsible for parsing command line arguments for experiments.
    """

    @staticmethod
    def parse_args() -> Dict[str, Any]:
        """
        Parse command line arguments.

        Returns:
            Dict[str, Any]: Dictionary of parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Run parallel text generation with TEMPO"
        )

        # Basic generation parameters
        parser.add_argument(
            "--prompt",
            type=str,
            default="Once upon a time",
            help="Text prompt to start generation",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=100,
            help="Maximum number of tokens to generate",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Probability threshold for token selection",
        )
        parser.add_argument(
            "--min-steps",
            type=int,
            default=0,
            help="Minimum steps to generate before considering EOS tokens",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./output",
            help="Directory to save output",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="deepcogito/cogito-v1-preview-llama-3B",
            help="Model name or path to use",
        )

        # Pruning parameters
        parser.add_argument(
            "--use-pruning",
            action="store_true",
            help="Use retroactive pruning to refine token sets based on future token attention",
        )
        parser.add_argument(
            "--pruning-strategy",
            type=str,
            default="coherence",
            choices=["coherence", "diversity", "hybrid"],
            help="Pruning strategy to use",
        )
        parser.add_argument(
            "--attention-threshold",
            type=float,
            default=0.01,
            help="Attention threshold for retroactive pruning (lower means more tokens kept)",
        )
        parser.add_argument(
            "--diversity-clusters",
            type=int,
            default=3,
            help="Number of clusters for diversity selection",
        )
        parser.add_argument(
            "--diversity-steps",
            type=int,
            default=5,
            help="Number of steps to use diversity selection before switching to attention",
        )

        # MCTS parameters
        parser.add_argument(
            "--use-mcts",
            action="store_true",
            help="Use Monte Carlo Tree Search for text generation",
        )
        parser.add_argument(
            "--mcts-simulations",
            type=int,
            default=10,
            help="Number of MCTS simulations per step",
        )
        parser.add_argument(
            "--mcts-c-puct",
            type=float,
            default=1.0,
            help="Exploration constant for MCTS",
        )
        parser.add_argument(
            "--mcts-depth",
            type=int,
            default=5,
            help="Maximum depth for MCTS simulations",
        )

        # Dynamic threshold parameters
        parser.add_argument(
            "--dynamic-threshold",
            action="store_true",
            help="Use dynamic threshold that increases over steps",
        )
        parser.add_argument(
            "--final-threshold",
            type=float,
            default=1.0,
            help="Final threshold value for dynamic thresholding (default: 1.0)",
        )
        parser.add_argument(
            "--bezier-p1",
            type=float,
            default=0.2,
            help="First Bezier control point for dynamic threshold",
        )
        parser.add_argument(
            "--bezier-p2",
            type=float,
            default=0.8,
            help="Second Bezier control point for dynamic threshold",
        )

        # ReLU transition parameters
        parser.add_argument(
            "--use-relu",
            action="store_true",
            help="Use ReLU transition instead of Bezier curve for dynamic threshold",
        )
        parser.add_argument(
            "--relu-activation-point",
            type=float,
            default=0.5,
            help="Point at which ReLU transition begins (0-1), default: 0.5",
        )

        # Cogito model parameters
        parser.add_argument(
            "--enable-thinking",
            action="store_true",
            help="Enable Cogito's deep thinking mode for better reasoning",
        )

        # Default mode option (disables TEMPO)
        parser.add_argument(
            "--default-mode",
            action="store_true",
            help="Run model in default generation mode without TEMPO (still supports thinking mode)",
        )

        # Other parameters
        parser.add_argument(
            "--save-visualization",
            action="store_true",
            default=True,
            help="Save visualization of token sets",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Random seed for reproducibility"
        )
        parser.add_argument(
            "--show-token-ids", action="store_true", help="Show token IDs in the output"
        )
        parser.add_argument(
            "--use-custom-rope",
            action="store_true",
            default=True,
            help="Use custom RoPE modification for parallel token positions",
        )
        parser.add_argument(
            "--debug-mode",
            action="store_true",
            help="Enable debug mode for detailed logging",
        )
        parser.add_argument(
            "--disable-kv-cache-consistency",
            action="store_true",
            help="Disable KV cache consistency checks for RoPE modification",
        )
        parser.add_argument(
            "--disable-kv-cache",
            action="store_true",
            help="Disable KV caching completely for more consistent attention",
        )

        # Parallel token isolation option
        parser.add_argument(
            "--allow-intraset-token-visibility",
            action="store_true",
            help="Allow tokens within the same parallel set to see each other during generation (disabled by default)",
        )

        # Profiling parameters
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable detailed performance profiling",
        )
        parser.add_argument(
            "--use-cprofile",
            action="store_true",
            help="Use cProfile for detailed function-level profiling",
        )
        parser.add_argument(
            "--profile-output",
            type=str,
            default="tempo_profile.prof",
            help="Output file for cProfile results",
        )

        # New parameter
        parser.add_argument(
            "--no-preserve-isolated-tokens",
            action="store_true",
            help="Allow pruning to evaluate isolated tokens (disabled by default in isolated mode)",
        )

        # Parameters for RetroactivePruner
        parser.add_argument(
            "--use-relative-attention",
            action="store_true",
            help="Use relative attention thresholds instead of absolute (default: enabled)",
        )
        parser.add_argument(
            "--no-relative-attention",
            action="store_true",
            help="Disable relative attention thresholds",
        )
        parser.add_argument(
            "--relative-threshold",
            type=float,
            default=0.5,
            help="Threshold for relative attention-based pruning (0-1)",
        )
        parser.add_argument(
            "--use-multi-scale-attention",
            action="store_true",
            help="Use multi-scale attention integration across all layers (default: enabled)",
        )
        parser.add_argument(
            "--no-multi-scale-attention",
            action="store_true",
            help="Disable multi-scale attention integration",
        )
        parser.add_argument(
            "--num-layers-to-use",
            type=int,
            default=None,
            help="Number of last layers to use for attention (None means use all layers)",
        )
        parser.add_argument(
            "--use-lci-dynamic-threshold",
            action="store_true",
            help="Use LCI-based dynamic thresholding (default: enabled)",
        )
        parser.add_argument(
            "--no-lci-dynamic-threshold",
            action="store_true",
            help="Disable LCI-based dynamic thresholding",
        )
        parser.add_argument(
            "--use-sigmoid-threshold",
            action="store_true",
            help="Use sigmoid-based decision boundary (default: enabled)",
        )
        parser.add_argument(
            "--no-sigmoid-threshold",
            action="store_true",
            help="Disable sigmoid-based decision boundary",
        )
        parser.add_argument(
            "--sigmoid-steepness",
            type=float,
            default=10.0,
            help="Controls how sharp the sigmoid transition is",
        )
        parser.add_argument(
            "--complete-pruning-mode",
            type=str,
            default="keep_token",
            choices=["keep_token", "keep_unattended", "remove_position"],
            help="How to handle pruned positions: 'keep_token' (default) - keep the best token, "
            "'keep_unattended' - keep the best token but mark as unattended, "
            "'remove_position' - completely remove the position from generation",
        )

        # Parse arguments
        args = parser.parse_args()

        # Convert arguments to dictionary
        args_dict = vars(args)

        # Combine bezier points
        args_dict["bezier_points"] = [
            args_dict.pop("bezier_p1"),
            args_dict.pop("bezier_p2"),
        ]

        return args_dict

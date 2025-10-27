import argparse
import json
import yaml
from pathlib import Path
from typing import Any


class ArgumentParser:
    """
    Responsible for parsing command line arguments for experiments.
    """

    @staticmethod
    def parse_args() -> dict[str, Any]:
        """
        Parse command line arguments.

        Returns:
            dict[str, Any]: Dictionary of parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Run parallel text generation with TEMPO"
        )

        # Configuration file support
        parser.add_argument(
            "--config",
            type=str,
            help="Path to YAML or JSON configuration file (overrides all other arguments)",
        )
        parser.add_argument(
            "--output-json",
            action="store_true",
            help="Output results in JSON format instead of formatted text",
        )
        parser.add_argument(
            "--json-output-file",
            type=str,
            help="Path to save JSON output (if not specified, prints to stdout)",
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
            "--selection-threshold",
            type=float,
            default=0.1,
            help="Probability threshold for initial token candidate selection",
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

        # Removal parameters
        parser.add_argument(
            "--use-retroactive-removal",
            action="store_true",
            help="Use retroactive removal to refine token sets based on future token attention",
        )
        parser.add_argument(
            "--attention-threshold",
            type=float,
            default=0.01,
            help="Attention threshold for retroactive removal (lower means more tokens kept)",
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
        # Visualization removed - not helpful for ML portfolio
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
            "--isolate",
            action="store_true",
            help="Isolate parallel tokens (prevent them from seeing each other). Default: disabled",
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
            help="Allow removal to evaluate isolated tokens (disabled by default in isolated mode)",
        )

        # Parameters for RetroactiveRemover
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
            help="Threshold for relative attention-based removal (0-1)",
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
            "--complete-removal-mode",
            type=str,
            default="keep_token",
            choices=["keep_token", "keep_unattended", "remove_position"],
            help="How to handle removed positions: 'keep_token' (default) - keep the best token, "
            "'keep_unattended' - keep the best token but mark as unattended, "
            "'remove_position' - completely remove the position from generation",
        )

        # Parse arguments
        args = parser.parse_args()

        # Convert arguments to dictionary
        args_dict = vars(args)

        # Check if config file is provided
        config_file = args_dict.get("config")
        if config_file:
            # Load configuration from YAML or JSON file
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")

            # Determine file format from extension
            suffix = config_path.suffix.lower()
            with open(config_path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif suffix == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")

            # Config file overrides all CLI arguments except output options
            output_json = args_dict.get("output_json", False)
            json_output_file = args_dict.get("json_output_file")

            args_dict = file_config
            args_dict["output_json"] = output_json
            args_dict["json_output_file"] = json_output_file

        # Combine bezier points if they exist as separate values
        if "bezier_p1" in args_dict and "bezier_p2" in args_dict:
            args_dict["bezier_points"] = [
                args_dict.pop("bezier_p1"),
                args_dict.pop("bezier_p2"),
            ]
        elif "bezier_points" not in args_dict:
            # Use defaults if not specified
            args_dict["bezier_points"] = [0.2, 0.8]

        return args_dict

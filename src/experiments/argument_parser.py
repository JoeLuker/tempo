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
        parser = argparse.ArgumentParser(description="Run parallel text generation with TEMPO")
        
        # Basic generation parameters
        parser.add_argument("--prompt", type=str, default="Once upon a time",
                            help="Text prompt to start generation")
        parser.add_argument("--max-tokens", type=int, default=100,
                            help="Maximum number of tokens to generate")
        parser.add_argument("--threshold", type=float, default=0.1,
                            help="Probability threshold for token selection")
        parser.add_argument("--min-steps", type=int, default=0,
                            help="Minimum steps to generate before considering EOS tokens")
        parser.add_argument("--output-dir", type=str, default="./output",
                            help="Directory to save output")
        
        # Pruning parameters
        parser.add_argument("--use-pruning", action="store_true",
                            help="Use pruning to reduce token sets")
        parser.add_argument("--pruning-strategy", type=str, default="coherence",
                            choices=["coherence", "diversity", "hybrid"],
                            help="Pruning strategy to use")
        parser.add_argument("--coherence-threshold", type=float, default=0.7,
                            help="Threshold for coherence pruning")
        parser.add_argument("--diversity-clusters", type=int, default=3,
                            help="Number of clusters for diversity pruning")
        parser.add_argument("--diversity-steps", type=int, default=5,
                            help="Number of steps to use diversity pruning before switching to coherence")
        
        # Dynamic threshold parameters
        parser.add_argument("--dynamic-threshold", action="store_true",
                            help="Use dynamic threshold that increases over steps")
        parser.add_argument("--bezier-p1", type=float, default=0.2,
                            help="First Bezier control point for dynamic threshold")
        parser.add_argument("--bezier-p2", type=float, default=0.8,
                            help="Second Bezier control point for dynamic threshold")
        
        # Cogito model parameters
        parser.add_argument("--enable-thinking", action="store_true",
                            help="Enable Cogito's deep thinking mode for better reasoning")
        
        # Other parameters
        parser.add_argument("--save-visualization", action="store_true", default=True,
                            help="Save visualization of token sets")
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
        parser.add_argument("--show-token-ids", action="store_true",
                            help="Show token IDs in the output")
        parser.add_argument("--use-custom-rope", action="store_true", default=True,
                            help="Use custom RoPE modification for parallel token positions")
        parser.add_argument("--debug-mode", action="store_true",
                            help="Enable debug mode for detailed logging")
        parser.add_argument("--disable-kv-cache-consistency", action="store_true",
                            help="Disable KV cache consistency checks for RoPE modification")
        parser.add_argument("--disable-kv-cache", action="store_true",
                            help="Disable KV caching completely for more consistent attention")
        
        # Parse arguments
        args = parser.parse_args()
        
        # Convert arguments to dictionary
        args_dict = vars(args)
        
        # Combine bezier points
        args_dict["bezier_points"] = [args_dict.pop("bezier_p1"), args_dict.pop("bezier_p2")]
        
        return args_dict 
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from .token_generator import TokenGenerator
from .token_selector import TokenSelector
from .text_formatter import TextFormatter
from .attention_manager import AttentionManager
from .rope_modifier import RoPEModifier
import logging
import os

# Import sequence tracker for monitoring token generation length
try:
    from run_tempo import sequence_tracker
except ImportError:
    # Create a dummy tracker if not available from run_tempo
    print("Warning: sequence_tracker not found, using dummy tracker")
    class DummySequenceTracker:
        def update_length(self, length): pass
        def increment_length(self, step_num=None): pass
        def get_length(self): return 0
    sequence_tracker = DummySequenceTracker()


class ParallelGenerator:
    """
    Main class for parallel token generation with threshold pruning.
    Optimized for performance with batched operations.
    """

    def __init__(
        self,
        model,
        tokenizer,
        pruner=None,
        device: str = "mps",
        has_custom_attention: bool = True,
        use_custom_rope: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize the parallel token generator.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            pruner: Optional Pruner instance
            device: Device to use for computation
            has_custom_attention: Whether the model supports custom attention mask
            use_custom_rope: Whether to use custom RoPE position mapping
            debug_mode: Whether to enable debug mode
        """
        # Validate required arguments
        assert model is not None, "Model cannot be None"
        assert tokenizer is not None, "Tokenizer cannot be None"
        assert device in ["cpu", "cuda", "mps"], f"Unsupported device: {device}"
        assert isinstance(has_custom_attention, bool), "has_custom_attention must be a boolean"
        assert isinstance(use_custom_rope, bool), "use_custom_rope must be a boolean"
        assert isinstance(debug_mode, bool), "debug_mode must be a boolean"
        
        self.model = model
        self.tokenizer = tokenizer
        self.pruner = pruner
        self.device = device
        self.has_custom_attention = has_custom_attention
        self.use_custom_rope = use_custom_rope
        self.debug_mode = debug_mode

        # Check if this is a Qwen-based model
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            self.is_qwen_model = "qwen" in model.config.model_type.lower()
        else:
            self.is_qwen_model = False

        # Set up logging
        self._setup_logger()

        # Initialize RoPE modifier if requested
        self.rope_modifier = None
        if use_custom_rope:
            # Invariant: RoPE modifier must initialize successfully when requested
            self.rope_modifier = RoPEModifier(model, device)
            assert self.rope_modifier is not None, "Failed to initialize RoPE modifier"
            self.rope_modifier.set_debug_mode(debug_mode)

        # Initialize other components
        self.token_generator = TokenGenerator(model, tokenizer, device)
        assert self.token_generator is not None, "Failed to initialize TokenGenerator"
        
        self.token_selector = TokenSelector(tokenizer)
        assert self.token_selector is not None, "Failed to initialize TokenSelector"
        
        self.text_formatter = TextFormatter(tokenizer)
        assert self.text_formatter is not None, "Failed to initialize TextFormatter"

        # Initialize attention manager with RoPE modifier reference
        self.attention_manager = AttentionManager(device, self.rope_modifier, tokenizer)
        assert self.attention_manager is not None, "Failed to initialize AttentionManager"
        self.attention_manager.set_debug_mode(debug_mode)

        # Install RoPE modifier after all components are initialized
        if use_custom_rope and self.rope_modifier is not None:
            # Now install the RoPE modifier
            # Invariant: RoPE modifier must install successfully when requested
            success = self.rope_modifier.install()
            assert success, "Failed to install RoPE modifier"
            self.log("Installed custom RoPE modifier for parallel token positions")

            # Link components for better coordination
            self.attention_manager.set_rope_modifier(self.rope_modifier)

        # Connect TokenGenerator to pruning strategy if using CoherencePruningStrategy
        if self.pruner is not None and hasattr(self.pruner, "strategy"):
            if hasattr(self.pruner.strategy, "set_token_generator"):
                self.pruner.strategy.set_token_generator(self.token_generator)
                self.pruner.strategy.set_debug_mode(debug_mode)
                self.log(
                    "Connected TokenGenerator to pruning strategy for attention reuse"
                )

        # Performance tracking
        self.generation_time = 0
        self.pruning_time = 0

        # Setup tracking for sequence length
        self.sequence_length = 0
        self.initial_prompt_length = 0
        self.step_count = 0
        self.sequence_length_history = []
        
        # Post-initialization check
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have encode method"
        assert hasattr(self.tokenizer, "decode"), "Tokenizer must have decode method"

    def _setup_logger(self):
        """Setup logging to file."""
        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("parallel_generator")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(log_dir, "generation_debug.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Verify logger setup
        assert self.logger.handlers, "Failed to setup logger handlers"

    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        assert message, "Log message cannot be empty"
        assert level in ["info", "debug", "warning", "error"], f"Invalid log level: {level}"
        
        if not self.debug_mode and level != "error":
            return

        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

    def _init_sequence_tracking(self, prompt_length):
        """Initialize sequence length tracking with the prompt length."""
        assert prompt_length >= 0, "Prompt length cannot be negative"
        
        self.sequence_length = 0
        self.initial_prompt_length = prompt_length
        self.step_count = 0
        self.sequence_length_history = []
    
    def get_sequence_length(self):
        """Get the current sequence length (tokens generated beyond prompt)."""
        return self.sequence_length
    
    def get_total_sequence_length(self):
        """Get the total sequence length including prompt."""
        return self.initial_prompt_length + self.sequence_length
    
    def update_sequence_length(self, new_length, callback=None):
        """
        Update sequence length and call any registered callbacks.
        
        Args:
            new_length: The new sequence length to set
            callback: Optional callback function to notify of changes
        """
        assert new_length >= 0, "Sequence length cannot be negative"
        
        if new_length > self.sequence_length:
            old_length = self.sequence_length
            self.sequence_length = new_length
            self.step_count += 1
            self.sequence_length_history.append(new_length)
            
            # Verify state consistency
            assert self.step_count == len(self.sequence_length_history), "Step count and history length mismatch"
            
            # Report changes if debugging
            if self.debug_mode:
                self.log(f"Sequence length updated: {old_length} → {new_length}", "debug")
                
            # Call sequence length callback if provided
            if callback is not None:
                try:
                    callback(new_length, self.step_count, self.initial_prompt_length)
                except Exception as e:
                    self.log(f"Error in sequence length callback: {e}", "error")
                    
            return True
        return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        threshold: Optional[float] = None,
        return_parallel_sets: bool = False,
        use_pruning: bool = False,
        require_custom_attention: bool = False,
        min_steps: int = 0,
        show_token_ids: bool = False,
        debug_mode: bool = False,
        disable_kv_cache: bool = False,
        system_content: Optional[str] = None,
        optimize_pruning: bool = True,
        isolate_parallel_tokens: bool = True,
        preserve_all_isolated_tokens: Optional[bool] = None,
        retroactive_pruner=None,
        sequence_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using parallel token generation with threshold pruning.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Probability threshold for token selection (0.0-1.0)
            return_parallel_sets: Whether to return parallel token sets
            use_pruning: Whether to use pruning to reduce token sets
            require_custom_attention: Whether to require custom attention support
            min_steps: Minimum number of steps to generate
            show_token_ids: Whether to include token IDs in the formatted output
            debug_mode: Whether to enable debug mode
            disable_kv_cache: Whether to disable KV caching
            system_content: Optional system message content for chat models
            optimize_pruning: Whether to optimize pruning operations
            isolate_parallel_tokens: Whether to isolate parallel tokens in the output
            preserve_all_isolated_tokens: Whether to preserve all isolated tokens
            retroactive_pruner: Optional retroactive pruner to use
            sequence_callback: Optional callback for sequence length updates
            
        Returns:
            Dict containing generation results
        """
        # Validate parameters
        assert prompt, "Prompt cannot be empty"
        assert max_tokens > 0, "max_tokens must be positive"
        assert threshold is None or (0.0 <= threshold <= 1.0), "threshold must be between 0.0 and 1.0"
        assert min_steps >= 0, "min_steps cannot be negative"
        assert isinstance(return_parallel_sets, bool), "return_parallel_sets must be a boolean"
        assert isinstance(use_pruning, bool), "use_pruning must be a boolean"
        assert isinstance(show_token_ids, bool), "show_token_ids must be a boolean"
        
        # Set debug mode
        self.debug_mode = debug_mode
        
        # Rest of the generate method...

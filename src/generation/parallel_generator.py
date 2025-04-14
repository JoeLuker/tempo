import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from .token_generator import TokenGenerator
from .token_selector import TokenSelector
from .text_formatter import TextFormatter
from .attention_manager import AttentionManager
from .rope_modifier import RoPEModifier
import time
import traceback
import logging
import os
from tqdm import tqdm
import numpy as np


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
            self.rope_modifier.set_debug_mode(debug_mode)

        # Initialize other components
        self.token_generator = TokenGenerator(model, tokenizer, device)
        self.token_selector = TokenSelector(tokenizer)
        self.text_formatter = TextFormatter(tokenizer)

        # Initialize attention manager with RoPE modifier reference
        self.attention_manager = AttentionManager(device, self.rope_modifier, tokenizer)
        self.attention_manager.set_debug_mode(debug_mode)

        # Install RoPE modifier after all components are initialized
        if use_custom_rope and self.rope_modifier is not None:
            # Now install the RoPE modifier
            # Invariant: RoPE modifier must install successfully when requested
            self.rope_modifier.install()
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

    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
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
    ) -> Dict[str, Any]:
        """
        Generate text using multiple parallel tokens.

        Args:
            prompt: Text prompt to generate from
            max_tokens: Maximum number of tokens to generate
            threshold: Threshold for token selection
            return_parallel_sets: Whether to return parallel token sets
            use_pruning: Whether to use pruning for parallel tokens
            require_custom_attention: Whether to require custom attention for KV-cache support
            min_steps: Minimum steps to generate, even if pruning collapses to single tokens
            show_token_ids: Whether to show token IDs in formatted output
            debug_mode: Whether to show detailed debug information
            disable_kv_cache: Whether to disable KV cache for better attention calculations
            system_content: Optional system content for instruction-following models
            optimize_pruning: Whether to enable pruning optimizations (skip reapply)
            isolate_parallel_tokens: If True, parallel tokens cannot attend to each other
            preserve_all_isolated_tokens: If True, skip pruning when tokens are isolated.
                                         When None (default), automatically set to match isolate_parallel_tokens
            retroactive_pruner: Optional RetroactivePruner instance for retroactive pruning

        Returns:
            Dict[str, Any]: Generated text and related information
        """
        # Initialize pruning_time at the start
        pruning_time = 0.0

        # Set default for preserve_all_isolated_tokens based on isolation mode
        if preserve_all_isolated_tokens is None:
            preserve_all_isolated_tokens = isolate_parallel_tokens

        # Set debug mode if requested
        if debug_mode:
            # Set debug mode for this generator and all components
            self.debug_mode = debug_mode
            if self.rope_modifier is not None:
                self.rope_modifier.set_debug_mode(debug_mode)
            self.attention_manager.set_debug_mode(debug_mode)
            # Enable debug mode for TokenSelector too
            self.token_selector.set_debug_mode(debug_mode)
            # Enable debug mode for TokenGenerator too
            self.token_generator.set_debug_mode(debug_mode)
            # Log to file instead of console
            self.log(
                "Debug mode enabled for generation - logging to files in logs/ directory"
            )
            # Print minimal console message
            print("Debug mode enabled - logging to files in logs/ directory")

        # Performance tracking
        start_time = time.time()

        # Set default threshold if not specified
        if threshold is None:
            threshold = 0.1

        # Validate custom attention requirement
        if require_custom_attention and not self.has_custom_attention:
            raise ValueError(
                "Custom attention is required but model doesn't support it"
            )

        # Reset pruner if using dynamic threshold
        if use_pruning and self.pruner is not None:
            if hasattr(self.pruner, "reset"):
                self.pruner.reset()

            # Set max steps in pruner if using dynamic threshold
            if (
                hasattr(self.pruner, "use_dynamic_threshold")
                and self.pruner.use_dynamic_threshold
            ):
                # Set the maximum steps for the dynamic threshold
                self.pruner.max_steps = max_tokens

        # Reset RoPE modifier position mapping
        if self.rope_modifier is not None:
            self.rope_modifier.reset()

        # Reset attention manager
        self.attention_manager.reset_cache()

        # For Cogito models with thinking mode enabled, we need special handling
        is_thinking_mode = (
            system_content is not None and "thinking" in system_content.lower()
        )
        if is_thinking_mode and self.is_qwen_model:
            if self.debug_mode:
                self.log("Using special handling for Cogito thinking mode")

            # Thinking mode works better with pruning
            if not use_pruning:
                self.log(
                    "Warning: Thinking mode works better with pruning. Consider adding --use-pruning flag."
                )

            # Thinking mode may need a different threshold for stable generation
            if threshold > 0.08:
                self.log(
                    f"Note: Using threshold {threshold} for thinking mode (values below 0.08 often work better)"
                )

        # Prepare input based on whether we're using chat format or raw prompt
        if system_content is not None:
            # Format input as chat for Cogito model
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            # DIAGNOSTIC: Print messages being sent to the model
            self.log("\nDIAGNOSTIC - Chat messages:")
            for msg in messages:
                self.log(f"  {msg['role']}: {msg['content'][:100]}...")

            # Check if the tokenizer supports apply_chat_template with enable_thinking
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    # Try to use the tokenizer's chat template with enable_thinking if available
                    if (
                        "enable_thinking"
                        in self.tokenizer.apply_chat_template.__code__.co_varnames
                    ):
                        # Use the tokenizer's chat template with enable_thinking
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        self.log(
                            "\nDIAGNOSTIC - Using chat template with enable_thinking=True"
                        )
                    else:
                        # Use regular chat template without enable_thinking
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        self.log(
                            "\nDIAGNOSTIC - Using standard chat template (no enable_thinking parameter)"
                        )

                    # DIAGNOSTIC: Show the formatted prompt text
                    self.log(
                        f"\nDIAGNOSTIC - Formatted prompt text (first 100 chars):\n{prompt_text[:200]}..."
                    )

                    input_ids, attention_mask = (
                        self.token_generator.prepare_input_from_prompt(prompt_text)
                    )

                    # DIAGNOSTIC: Show token IDs and decoded tokens
                    self.log("\nDIAGNOSTIC - First 10 input token IDs:")
                    for i in range(min(10, len(input_ids[0]))):
                        token_id = input_ids[0, i].item()
                        token_text = self.tokenizer.decode([token_id])
                        self.log(f"  Token {i}: ID={token_id}, Text='{token_text}'")

                    self.log(
                        f"\nDIAGNOSTIC - Total input length: {len(input_ids[0])} tokens"
                    )

                except Exception as e:
                    self.log(f"\nDIAGNOSTIC - Error in chat template processing: {e}")
                    traceback.print_exc()
                    raise RuntimeError(
                        f"Error applying chat template: {e}. Generation failed."
                    )

            else:
                # Invariant: Chat template must be applied successfully
                raise RuntimeError(
                    "Chat template application failed. The tokenizer doesn't support the required chat template functionality."
                )
        else:
            # Standard input preparation
            input_ids, attention_mask = self.token_generator.prepare_input_from_prompt(
                prompt
            )

        # Pre-allocate storage for token sets (more memory efficient)
        # We'll only store the minimal necessary information and convert formats as needed
        token_sets = []  # List of (position, token_ids, token_probs) tuples

        # Track positions with multiple tokens
        original_parallel_positions = set()

        # More efficient position_to_tokens mapping using direct indices
        position_to_tokens = {}
        prompt_length = len(input_ids[0])

        # Store ALL parallel tokens for retroactive pruning
        all_parallel_tokens = {}  # position -> list of (token_id, prob) pairs

        # Add prompt tokens to the position mapping with efficient batch processing
        # Vectorized with dictionary comprehension
        position_to_tokens = {i: [input_ids[0, i].item()] for i in range(prompt_length)}

        # Use KV cache for faster generation
        past_key_values = None

        # Position mapping for RoPE modification
        rope_position_map = {}

        # Flag to track if we're currently in a repetition loop
        in_repetition_loop = False
        repetition_count = 0
        last_tokens = []

        # Track total tokens across all steps
        running_total_tokens = 0

        # Iteratively generate tokens - optimized for speed
        progress_bar = tqdm(range(max_tokens), desc="Generating tokens", unit="token")
        for i in progress_bar:
            # Show progress while looping
            self.log(f"\nDIAGNOSTIC - Starting token generation step {i}")
            self.log(f"  Input shape: {input_ids.shape}")

            # Determine whether to use the standard or optimized path
            use_optimized_path = (
                isolate_parallel_tokens and i > 0
            )  # Only optimize after first token

            # For non-optimized path or first token, get the logits normally
            if not use_optimized_path:
                # Detailed timing - token logits
                logits_start = time.time()
                # Standard path for first token or non-isolated mode
                next_token_logits, past_key_values = (
                    self.token_generator.get_next_token_logits_cached(
                        input_ids,
                        attention_mask,
                        None if disable_kv_cache else past_key_values,
                        (
                            self.attention_manager.full_attention_mask
                            if self.has_custom_attention
                            else None
                        ),
                    )
                )
                logits_time = time.time() - logits_start

                # Detailed timing - token selection
                select_start = time.time()
                next_token_ids, next_token_probs = (
                    self.token_selector.select_tokens_above_threshold(
                        next_token_logits, threshold
                    )
                )
                select_time = time.time() - select_start
            else:
                # For optimization path, get logits once using optimized method
                # First get token logits using the optimized method
                logits_start = time.time()
                next_token_logits, past_key_values = (
                    self.token_generator.get_next_token_logits_for_isolated_parallel(
                        input_ids,
                        attention_mask,
                        None if disable_kv_cache else past_key_values,
                        (
                            self.attention_manager.full_attention_mask
                            if self.has_custom_attention
                            else None
                        ),
                        num_parallel_tokens=5,  # Initial estimate
                    )
                )
                logits_time = time.time() - logits_start

                # Select tokens
                select_start = time.time()
                next_token_ids, next_token_probs = (
                    self.token_selector.select_tokens_above_threshold(
                        next_token_logits, threshold
                    )
                )
                select_time = time.time() - select_start

                # When using optimized path for isolated tokens, we need to ensure that
                # input_ids and attention_mask remain consistent, since get_next_token_logits_for_isolated_parallel
                # may have modified the input_ids shape
                if past_key_values is not None:
                    # Restore consistent shapes - use only last token with proper attention mask
                    seq_len = 1  # We're using KV cache, so we only need the last token
                    input_ids = input_ids[:, -seq_len:]
                    attention_mask = torch.ones(
                        (input_ids.size(0), seq_len), device=self.device
                    )

                # Update the num_parallel_tokens in the token generator stats to reflect actual count
                if hasattr(self.token_generator, "last_parallel_count"):
                    self.token_generator.last_parallel_count = len(next_token_ids)

            # Update running total of tokens
            running_total_tokens += len(next_token_ids)

            # Show detailed timing in the progress bar
            progress_bar.set_postfix(
                tokens=len(next_token_ids),
                total_tokens=running_total_tokens,
                logits_ms=f"{logits_time*1000:.1f}",
                select_ms=f"{select_time*1000:.1f}",
                prune_ms=f"{pruning_time*1000:.1f}" if use_pruning and self.pruner else "N/A",
                top_prob=(
                    f"{next_token_probs[0]:.4f}" if len(next_token_probs) > 0 else "N/A"
                ),
            )

            # Invariant: Token IDs and probabilities must have same length and proper structure
            if len(next_token_ids) != len(next_token_probs):
                raise ValueError(
                    f"Invariant violation: Mismatch between token IDs ({len(next_token_ids)}) and probabilities ({len(next_token_probs)})"
                )

            # Invariant: Probabilities must be valid values between 0 and 1
            if any(prob <= 0 or prob > 1.0 for prob in next_token_probs):
                raise ValueError(
                    "Invariant violation: Token probabilities must be between 0 and 1"
                )

            # Invariant: If multiple tokens are selected, they must have decreasing probabilities
            if len(next_token_ids) > 1:
                # Vectorized check for descending order using tensor operations
                probs_tensor = torch.tensor(next_token_probs, device=self.device)
                if not torch.all(probs_tensor[:-1] >= probs_tensor[1:]):
                    raise ValueError(
                        "Invariant violation: Token probabilities must be in descending order"
                    )

            # Skip if no tokens above threshold
            if len(next_token_ids) == 0:
                # DEBUG: Show top tokens even when below threshold
                top_token_ids, top_token_probs = self.token_selector.select_top_tokens(
                    next_token_logits, top_k=5
                )
                top_tokens_text = [
                    self.tokenizer.decode([int(tid)]) for tid in top_token_ids
                ]

                self.log(
                    "\nDEBUG: No tokens above threshold. Top 5 tokens and probabilities:"
                )
                for idx, (token_text, token_id, prob) in enumerate(
                    zip(top_tokens_text, top_token_ids, top_token_probs)
                ):
                    self.log(
                        f"  {idx+1}. '{token_text}' (ID: {int(token_id)}): {prob:.6f}"
                    )
                self.log(f"Current threshold: {threshold}")

                # If thinking mode is active, also show special note
                if is_thinking_mode:
                    self.log(
                        "Note: Thinking mode often requires lower thresholds (0.01-0.05)"
                    )
                    self.log(
                        "Try running with --threshold 0.05 or --threshold 0.03 for thinking mode"
                    )

                self.log(
                    f"No tokens above threshold at step {i}. Treating as EOS and finishing generation."
                )
                break

            # Skip single EOS token if this isn't the last step and we haven't reached min_steps
            if (
                len(next_token_ids) == 1
                and self.token_selector.is_eos_token(int(next_token_ids[0]))
                and i < max_tokens - 1
                and i < min_steps
            ):
                continue

            # Check for repetition patterns
            current_token = int(next_token_ids[0]) if len(next_token_ids) > 0 else None
            if current_token is not None:
                # Add to last tokens
                last_tokens.append(current_token)
                if len(last_tokens) > 5:
                    last_tokens.pop(0)

                # Check for repetition
                if len(last_tokens) >= 3:
                    if last_tokens[-1] == last_tokens[-2] == last_tokens[-3]:
                        repetition_count += 1
                        if repetition_count >= 3:
                            in_repetition_loop = True
                            self.log(
                                f"Detected repetition loop at step {i}, applying correction"
                            )
                            # For thinking mode, we need to force a diverse token
                            if is_thinking_mode:
                                # Remove repeated token from options
                                repeated_token = last_tokens[-1]
                                # Filter using NumPy masking
                                mask = next_token_ids != repeated_token
                                next_token_ids = next_token_ids[mask]
                                next_token_probs = next_token_probs[mask]
                                if len(next_token_ids) == 0:
                                    # If no tokens left, get new ones excluding repeated token
                                    next_token_ids, next_token_probs = (
                                        self.token_selector.select_tokens_above_threshold_excluding(
                                            next_token_logits,
                                            threshold * 0.8,
                                            [repeated_token],
                                        )
                                    )
                    else:
                        repetition_count = 0
                        in_repetition_loop = False

            # Store tokens efficiently as NumPy arrays - no conversion needed
            original_token_ids = next_token_ids.copy()
            original_token_probs = next_token_probs.copy()

            # If we have multiple tokens, mark this as a parallel position
            current_position = len(position_to_tokens) - prompt_length
            if len(next_token_ids) > 1:
                original_parallel_positions.add(current_position)

                # Update RoPE position mapping for parallel tokens
                if self.rope_modifier is not None and len(next_token_ids) > 1:
                    # Create position mapping for all tokens in the parallel set
                    position_mapping = {}
                    current_pos = prompt_length + i
                    for j in range(len(next_token_ids)):
                        position_mapping[current_pos + j] = current_pos

                    # Register with RoPE modifier
                    self.rope_modifier.register_parallel_positions(position_mapping)

            # Create copy of original tokens for pruning
            pruned_token_ids = next_token_ids.copy()
            pruned_token_probs = next_token_probs.copy()

            # Apply pruning if requested and available
            pruning_start = time.time()
            if use_pruning and self.pruner is not None and len(pruned_token_ids) > 1:
                # Skip pruning if tokens are isolated and we want to preserve them all
                if isolate_parallel_tokens and preserve_all_isolated_tokens:
                    if self.debug_mode:
                        self.log(
                            f"Skipping pruning for isolated tokens (preserve_all_isolated_tokens=True)"
                        )
                else:
                    # Pass token generator to pruner for attention reuse if not already set
                    if (
                        hasattr(self.pruner.strategy, "token_generator")
                        and self.pruner.strategy.token_generator is None
                    ):
                        self.pruner.strategy.set_token_generator(self.token_generator)

                    # Set the skip_reapply_threshold flag based on the optimize_pruning parameter
                    if hasattr(self.pruner, "skip_reapply_threshold"):
                        self.pruner.skip_reapply_threshold = optimize_pruning

                    # Invariant: Pruning must succeed when requested
                    pruned_result = self.pruner.prune_parallel_tokens(
                        input_ids=input_ids,
                        parallel_tokens=(pruned_token_ids, pruned_token_probs),
                    )

                    # Extract results
                    if (
                        pruned_result
                        and isinstance(pruned_result, tuple)
                        and len(pruned_result) >= 1
                    ):
                        (pruned_token_ids, pruned_token_probs) = pruned_result[0]
            pruning_time = time.time() - pruning_start
            self.pruning_time += pruning_time

            # Store token set info more efficiently using NumPy arrays directly
            token_sets.append(
                (
                    len(position_to_tokens) - prompt_length,  # Position
                    (
                        original_token_ids,
                        original_token_probs,
                    ),  # Original tokens as NumPy arrays
                    (
                        pruned_token_ids,
                        pruned_token_probs,
                    ),  # Pruned tokens as NumPy arrays
                )
            )

            # Add pruned tokens to position_to_tokens mapping
            position_to_tokens[prompt_length + i] = pruned_token_ids.tolist()

            # Store all tokens for this position for retroactive pruning
            current_position = len(position_to_tokens) - prompt_length
            all_parallel_tokens[current_position] = [
                (tid, prob)
                for tid, prob in zip(original_token_ids, original_token_probs)
            ]

            # Create new input representation with the pruned tokens - more efficient approach
            # Invariant: Attention update must succeed
            # Pass disable_kv_cache flag to ensure proper context handling
            attention_start = time.time()

            input_ids, attention_mask, past_key_values = (
                self.attention_manager.update_input_efficiently(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=(
                        None if disable_kv_cache else past_key_values
                    ),  # Pass None explicitly if KV cache is disabled
                    new_token_ids=pruned_token_ids.tolist(),  # Convert to list for attention_manager compatibility
                    is_kv_cache_disabled=disable_kv_cache,  # Pass flag for explicit handling
                )
            )
            attention_time = time.time() - attention_start

            # Log timing information
            if i > 0 and i % 5 == 0:
                self.log(
                    f"Step {i} timing: logits={logits_time*1000:.1f}ms, select={select_time*1000:.1f}ms, prune={pruning_time*1000:.1f}ms, attention={attention_time*1000:.1f}ms"
                )

            # Special handling for thinking mode with Qwen/Cogito models
            if is_thinking_mode and self.is_qwen_model and i > 10 and (i % 20 == 0):
                # Periodically reset KV cache to prevent issues in long thinking chains
                if not disable_kv_cache:
                    self.log(
                        f"Resetting KV cache at step {i} for thinking mode stability"
                    )
                    past_key_values = None

            # Stop generation if all tokens are EOS and we've reached min_steps
            if (
                self.token_selector.all_are_eos_tokens(pruned_token_ids)
                and len(pruned_token_ids) > 0
                and i >= min_steps
            ):
                self.log(
                    f"Stopping because all tokens are EOS after {i+1} steps (min_steps={min_steps})"
                )
                break

            # Apply retroactive pruning if available
            if retroactive_pruner is not None and i > 0:  # Skip first token
                if self.debug_mode:
                    print(f"\nApplying retroactive pruning at step {i}")
                    print(f"Retroactive pruner available: {retroactive_pruner is not None}")
                    print(f"Token generator available: {self.token_generator is not None}")
                    print(f"Number of parallel positions: {len(all_parallel_tokens)}")

                # Set token generator if not already set
                if retroactive_pruner.token_generator is None:
                    if self.debug_mode:
                        print("Setting token generator for retroactive pruner")
                    retroactive_pruner.set_token_generator(self.token_generator)

                # Update step for dynamic thresholding
                if hasattr(retroactive_pruner, 'update_step'):
                    if self.debug_mode:
                        print(f"Updating retroactive pruner step to {i}")
                    retroactive_pruner.update_step(i)

                # Retroactively prune previous positions based on newest token's attention
                if self.debug_mode:
                    print("Calling retroactive_prune...")
                pruned_parallel_tokens = retroactive_pruner.retroactively_prune(
                    prompt_length=prompt_length, all_parallel_tokens=all_parallel_tokens
                )

                # Update all_parallel_tokens with pruned results
                if self.debug_mode:
                    print(f"Pruning complete. Original positions: {len(all_parallel_tokens)}, Pruned positions: {len(pruned_parallel_tokens)}")
                all_parallel_tokens = pruned_parallel_tokens

        # Close progress bar
        progress_bar.close()

        # Print detailed timing summary if available
        if (
            use_pruning
            and self.pruner is not None
            and hasattr(self.pruner, "step_timings")
            and self.pruner.step_timings
        ):
            timings = self.pruner.step_timings
            print("\n=== Timing Analysis ===")
            print(f"Total steps: {len(timings)}")

            # Group timings into bins for analysis
            bins = min(10, len(timings))
            bin_size = max(1, len(timings) // bins)

            # Calculate averages for each bin
            for i in range(bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, len(timings))
                bin_timings = timings[start_idx:end_idx]

                avg_reapply = sum(t["reapply_ms"] for t in bin_timings) / len(
                    bin_timings
                )
                avg_total = sum(t["total_ms"] for t in bin_timings) / len(bin_timings)
                avg_tokens = sum(t["tokens_before"] for t in bin_timings) / len(
                    bin_timings
                )

                print(
                    f"Steps {start_idx}-{end_idx-1}: "
                    + f"reapply={avg_reapply:.1f}ms ({avg_reapply/avg_total*100:.1f}%), "
                    + f"total={avg_total:.1f}ms, avg_tokens={avg_tokens:.1f}"
                )

            # Show change in reapply time from beginning to end
            if len(timings) > 1:
                first_reapply = timings[0]["reapply_ms"]
                last_reapply = timings[-1]["reapply_ms"]
                reapply_increase = (
                    (last_reapply / first_reapply)
                    if first_reapply > 0
                    else float("inf")
                )
                print(
                    f"\nReapply time growth: {first_reapply:.1f}ms → {last_reapply:.1f}ms ({reapply_increase:.1f}x)"
                )

                # Alert if this is the likely bottleneck
                if reapply_increase > 3.0 and last_reapply > 20.0:
                    print(
                        "\n⚠️ Performance bottleneck detected in threshold reapplication!"
                    )
                    print(
                        "   This suggests O(n²) complexity in the dynamic thresholding."
                    )

        # Update position_to_tokens with final pruned sets if using dynamic threshold
        if (
            use_pruning
            and self.pruner is not None
            and hasattr(self.pruner, "use_dynamic_threshold")
            and self.pruner.use_dynamic_threshold
        ):
            # Get final pruned sets - optimize this to avoid recomputing everything
            final_pruned_sets = self.pruner.get_final_pruned_sets()

            # Update position_to_tokens with batch update
            for step, pruned_set in enumerate(final_pruned_sets):
                position = prompt_length + step
                if position in position_to_tokens:
                    position_to_tokens[position] = [t[0] for t in pruned_set]

        # Format the generated text - only decode tokens once
        if show_token_ids:
            formatted_text = self.text_formatter.format_with_token_ids_and_pruning(
                prompt,
                position_to_tokens,
                original_parallel_positions,
                prompt_length,
                all_parallel_tokens,  # Pass pruned parallel sets for improved display
            )
        else:
            formatted_text = self.text_formatter.format_generated_text_with_pruning(
                prompt,
                position_to_tokens,
                original_parallel_positions,
                prompt_length,
                all_parallel_tokens,  # Pass pruned parallel sets for improved display
            )

        # Generate raw text efficiently - single decoding operation
        # Vectorized approach to build token sequence
        token_sequence = []

        # Add prompt tokens using tensor slicing
        if len(input_ids.shape) > 1 and input_ids.shape[1] > 0:
            # Convert the prompt section to a list at once instead of looping
            prompt_tokens = input_ids[
                0, : min(prompt_length, input_ids.shape[1])
            ].tolist()
            token_sequence.extend(prompt_tokens)
        elif len(input_ids.shape) == 1:
            token_sequence.append(input_ids.item())
        else:
            raise ValueError("Unexpected input_ids shape")

        # Add generated tokens
        # Using a list comprehension for generated positions
        generated_tokens = [
            int(token)  # Ensure token IDs are integers
            for pos in sorted(position_to_tokens.keys())
            if pos >= prompt_length
            for token in position_to_tokens[pos]
        ]
        token_sequence.extend(generated_tokens)

        # Batch decode the raw generated text - much faster than token-by-token
        raw_generated_text = self.tokenizer.decode(
            token_sequence, skip_special_tokens=True
        )

        # Total generation time
        self.generation_time = time.time() - start_time

        # Prepare results dictionary - optimize for memory by only including what's needed
        results = {
            "generated_text": formatted_text,
            "raw_generated_text": raw_generated_text,
            "prompt": prompt,
            "threshold": threshold,
            "use_pruning": use_pruning,
            "min_steps": min_steps,
            "generation_time": self.generation_time,
            "pruning_time": self.pruning_time,
            "use_custom_rope": self.use_custom_rope,
            "system_content": system_content,
            "is_qwen_model": self.is_qwen_model,
            "had_repetition_loop": in_repetition_loop,
        }

        # Add isolated tokens mode information
        if isolate_parallel_tokens:
            results["isolate_parallel_tokens"] = True

            # Count how many tokens were generated in parallel mode
            parallel_token_count = 0
            total_parallel_sets = 0
            for pos in sorted(position_to_tokens.keys()):
                if pos >= prompt_length:
                    tokens = position_to_tokens[pos]
                    if len(tokens) > 1:
                        parallel_token_count += len(tokens)
                        total_parallel_sets += 1

            # Calculate efficiency gains
            if total_parallel_sets > 0:
                # Each parallel set with n tokens saved (n-1) forward passes
                model_calls_saved = parallel_token_count - total_parallel_sets

                # Estimate compute savings (1 forward pass per parallel set instead of 1 per token)
                if self.token_generator.perf_stats["model_calls"] > 0:
                    avg_forward_time = (
                        self.token_generator.perf_stats["model_time"]
                        / self.token_generator.perf_stats["model_calls"]
                    )
                    estimated_time_saved = model_calls_saved * avg_forward_time

                    results["isolated_mode_stats"] = {
                        "parallel_token_count": parallel_token_count,
                        "parallel_sets": total_parallel_sets,
                        "model_calls_saved": model_calls_saved,
                        "estimated_time_saved_ms": estimated_time_saved * 1000,
                    }

                    # Print summary if in debug mode
                    if debug_mode:
                        print(f"\nIsolated Parallel Token Optimization:")
                        print(f"  Parallel tokens processed: {parallel_token_count}")
                        print(f"  Parallel token sets: {total_parallel_sets}")
                        print(f"  Model forward passes saved: {model_calls_saved}")
                        print(
                            f"  Estimated compute time saved: {estimated_time_saved*1000:.1f}ms"
                        )

                        # Print efficiency improvement ratio
                        if parallel_token_count > 0:
                            efficiency_ratio = (
                                parallel_token_count / total_parallel_sets
                            )
                            print(
                                f"  Efficiency ratio: {efficiency_ratio:.2f}x (computed {total_parallel_sets} times instead of {parallel_token_count})"
                            )

        # Add parallel sets data only if requested to save memory
        if return_parallel_sets:
            # Efficiently convert to human-readable format only when needed
            token_id_map = {}

            def get_token_text(token_id: int) -> str:
                """
                Get the text representation of a token ID, with caching.

                Args:
                    token_id: Token ID to decode

                Returns:
                    str: Decoded token text
                """
                # Invariant: Token ID must be a numeric type that can be converted to integer
                if not isinstance(token_id, (int, np.integer, np.floating)):
                    raise ValueError(
                        f"Invariant violation: Token ID must be a numeric type, got {type(token_id)}"
                    )

                # Convert to integer to ensure type safety
                token_id = int(token_id)

                # Check cache first
                if token_id in token_id_map:
                    return token_id_map[token_id]

                # Decode and cache
                token_id_map[token_id] = self.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                return token_id_map[token_id]

            # Only include the minimal necessary data for visualization
            if use_pruning and self.pruner is not None:
                # Add pruned information in a memory-efficient way
                position_info = {}
                for pos, tokens in position_to_tokens.items():
                    if pos >= prompt_length:  # Only include generated tokens
                        # Batch decode tokens
                        position_info[str(pos)] = [get_token_text(t) for t in tokens]

                results["position_to_tokens"] = position_info

                # Include raw token sets only if specifically needed
                if (
                    hasattr(self.pruner, "use_dynamic_threshold")
                    and self.pruner.use_dynamic_threshold
                ):
                    pruned_sets = self.pruner.get_final_pruned_sets()
                    results["final_pruned_sets"] = pruned_sets

            # Add position to tokens mapping for visualization
            position_info = {}
            for pos, tokens in position_to_tokens.items():
                if pos >= prompt_length:  # Only include generated tokens
                    decoded_tokens = []
                    for t in tokens:
                        try:
                            if isinstance(t, int):
                                decoded_tokens.append(self.tokenizer.decode([t]))
                            else:
                                # Skip invalid tokens
                                pass
                        except Exception:
                            # Skip on any decoding error
                            pass
                    position_info[str(pos)] = decoded_tokens
            results["position_to_tokens"] = position_info

        # Add model internal diagnostics when in debug mode
        if self.debug_mode and hasattr(self.model, "intermediate_values"):
            # Add keys of captured intermediate values
            results["intermediate_value_keys"] = list(
                self.model.intermediate_values.keys()
            )

        # Generation completed
        results["generation_time"] = time.time() - start_time

        # Print performance statistics if in debug mode
        if debug_mode:
            print("\nPerformance Statistics:")
            # Print token generator stats
            self.token_generator.print_performance_stats()

            # Print attention manager stats
            self.attention_manager.print_performance_stats()

            # Print pruner stats if available
            if (
                use_pruning
                and self.pruner is not None
                and hasattr(self.pruner, "print_performance_stats")
            ):
                self.pruner.print_performance_stats()

            # Print overall generation stats
            print("\nOverall Generation Stats:")
            generation_time = results["generation_time"]
            tokens_generated = (
                len(results["token_sets"]) if "token_sets" in results else 0
            )
            print(f"  Generation time: {generation_time:.2f}s")
            print(f"  Tokens generated: {tokens_generated}")
            if tokens_generated > 0:
                print(f"  Tokens per second: {tokens_generated / generation_time:.2f}")

        return results

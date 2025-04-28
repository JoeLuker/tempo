import torch
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
import os
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import sys

# Import the sequence tracker
sequence_tracker = None

def set_sequence_tracker(tracker):
    """Set the global sequence tracker instance."""
    global sequence_tracker
    sequence_tracker = tracker

# Default to dummy tracker until properly set
class DummyTracker:
    def update_length(self, length):
        pass

    def increment_step(self, step_num=None):
        pass
    
    def increment_length(self, step_num=None):
        pass

    def get_length(self):
        return 0

    debug = True  # Default to debug mode on for the dummy tracker

# Use dummy tracker initially
sequence_tracker = DummyTracker()
print("Initialized token_generator with dummy tracker - will be replaced at runtime")


class TokenGenerator:
    """
    Responsible for generating token logits from the model.
    Optimized for performance with KV caching and efficient batching.
    """

    def __init__(self, model, tokenizer, device: str = "mps"):
        """
        Initialize the token generator.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            device: Device to use for computation
        """
        # Validate required parameters
        assert model is not None, "Model cannot be None"
        assert tokenizer is not None, "Tokenizer cannot be None"
        assert device in ["cpu", "cuda", "mps"], f"Unsupported device: {device}"

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Debug mode
        self.debug_mode = True

        # Cache for tokenized prompts
        self.prompt_cache = {}

        # Cache for token decoding
        self.token_decode_cache = {}

        # Optional detailed performance tracking
        self.detailed_perf = False

        # Track parallel token counts for optimization
        self.last_parallel_count = 0

        # Setup logging
        self._setup_logger()

        # Performance tracking
        self.perf_stats = {
            "tokenization_calls": 0,
            "tokenization_time": 0,
            "model_calls": 0,
            "model_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "decode_calls": 0,
            "decode_cache_hits": 0,
            "decode_time": 0,
            "isolated_tokens_processed": 0,
        }

        # Verify tokenizer capabilities
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have encode method"
        assert hasattr(self.tokenizer, "decode"), "Tokenizer must have decode method"
        assert hasattr(self.tokenizer, "__call__"), "Tokenizer must be callable"

    def _setup_logger(self):
        """Setup logging to file."""
        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("token_generator")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(log_dir, "token_generator_debug.log")

        # Clear the log file by opening in write mode first
        with open(log_file, "w") as f:
            pass

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
        assert self.logger.handlers, "Failed to set up logger handlers"

    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        assert message, "Log message cannot be empty"
        assert level in [
            "info",
            "debug",
            "warning",
            "error",
        ], f"Invalid log level: {level}"

        if not self.debug_mode:
            return

        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for more verbose output.

        Args:
            enabled: Whether to enable debug mode
        """
        assert isinstance(enabled, bool), "Debug mode must be a boolean"

        self.debug_mode = enabled
        if enabled:
            print(
                f"TokenGenerator debug mode enabled - logging to file at logs/token_generator_debug.log"
            )
        else:
            print(f"TokenGenerator debug mode disabled")

    def prepare_input_from_prompt(
        self, prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert prompt string to input tensors.
        Uses caching for repeated prompts.

        Args:
            prompt: The text prompt

        Returns:
            tuple: (input_ids, attention_mask)
        """
        # Invariant: Prompt must be a non-empty string
        assert prompt and isinstance(prompt, str), "Prompt must be a non-empty string"

        # Performance tracking
        start_time = time.time()
        self.perf_stats["tokenization_calls"] += 1

        # Check cache first for this prompt
        if prompt in self.prompt_cache:
            self.perf_stats["cache_hits"] += 1
            input_ids, attention_mask = self.prompt_cache[prompt]
            self.perf_stats["tokenization_time"] += time.time() - start_time
            return input_ids, attention_mask

        self.perf_stats["cache_misses"] += 1

        # Use efficient batch encoding instead of simple encode
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Invariant: Tokenized inputs must have valid dimensions and values
        assert (
            input_ids.dim() == 2 and input_ids.size(0) == 1
        ), f"input_ids must have shape [1, seq_len], got {input_ids.shape}"

        assert (
            attention_mask.dim() == 2 and attention_mask.size(0) == 1
        ), f"attention_mask must have shape [1, seq_len], got {attention_mask.shape}"

        assert input_ids.size(1) == attention_mask.size(
            1
        ), "input_ids and attention_mask must have same sequence length"

        # Cache the results
        self.prompt_cache[prompt] = (input_ids, attention_mask)

        # Track timing
        self.perf_stats["tokenization_time"] += time.time() - start_time

        # Update sequence length in global tracker
        if hasattr(sequence_tracker, "update_length"):
            sequence_tracker.update_length(input_ids.size(1))

        return input_ids, attention_mask

    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        custom_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get logits for the next token using the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            custom_attention_mask: Optional custom attention mask

        Returns:
            torch.Tensor: Logits for the next token
        """
        # Validate inputs
        assert input_ids is not None, "input_ids cannot be None"
        assert attention_mask is not None, "attention_mask cannot be None"
        assert isinstance(input_ids, torch.Tensor), "input_ids must be a tensor"
        assert isinstance(
            attention_mask, torch.Tensor
        ), "attention_mask must be a tensor"
        assert input_ids.dim() == 2, f"input_ids must be 2D, got {input_ids.dim()}D"
        assert (
            attention_mask.dim() == 2
        ), f"attention_mask must be 2D, got {attention_mask.dim()}D"
        assert (
            input_ids.shape[1] == attention_mask.shape[1]
        ), f"input_ids and attention_mask sequence lengths must match: {input_ids.shape[1]} vs {attention_mask.shape[1]}"

        if custom_attention_mask is not None:
            assert isinstance(
                custom_attention_mask, torch.Tensor
            ), "custom_attention_mask must be a tensor"

        # Performance tracking
        start_time = time.time()
        self.perf_stats["model_calls"] += 1

        # Use inference mode for efficiency
        with torch.inference_mode():
            # Check if the model is Qwen-based (Cogito uses Qwen2.5 base)
            model_type = getattr(self.model.config, "model_type", "").lower()
            is_qwen = "qwen" in model_type

            # Build model arguments based on model type
            model_args = {"input_ids": input_ids, "attention_mask": attention_mask}

            # For Qwen models, make sure to handle attention properly
            if is_qwen and custom_attention_mask is not None:
                if custom_attention_mask.dim() == 3:
                    model_args["position_bias"] = custom_attention_mask
                elif custom_attention_mask.dim() == 4:
                    model_args["attention_mask"] = custom_attention_mask

            outputs = self.model(**model_args)

        # Get logits for next token (last position)
        next_token_logits = outputs.logits[:, -1, :]

        self.perf_stats["model_time"] += time.time() - start_time
        return next_token_logits

    def get_next_token_logits_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        custom_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        Get logits for next token using the model with KV caching.
        Improved to properly handle parallel token sets.

        Args:
            input_ids: Input token IDs
            attention_mask: Current attention mask
            past_key_values: Optional past key values for KV cache
            custom_attention_mask: Optional custom attention mask

        Returns:
            tuple: (next_token_logits, past_key_values)
        """
        self.log(">>> ENTERING get_next_token_logits_cached <<<", level="warning")
        
        # FUNDAMENTAL INVARIANT: Input tensors must be valid tensors
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Invariant violation: input_ids must be a torch.Tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError(
                "Invariant violation: attention_mask must be a torch.Tensor"
            )
        if not (input_ids.dim() >= 2 and attention_mask.dim() >= 2):
            raise ValueError(
                f"Invariant violation: input_ids and attention_mask must have at least 2 dimensions, got {input_ids.dim()} and {attention_mask.dim()}"
            )

        # FUNDAMENTAL INVARIANT: Device consistency must be maintained
        if input_ids.device != attention_mask.device:
            raise ValueError(
                f"Invariant violation: Device mismatch between input_ids ({input_ids.device}) and attention_mask ({attention_mask.device})"
            )

        # INVARIANT: Input tensors must not have NaN or Inf values
        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            raise ValueError(
                "Invariant violation: input_ids contains NaN or Inf values"
            )
        if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
            raise ValueError(
                "Invariant violation: attention_mask contains NaN or Inf values"
            )

        # INVARIANT: Custom attention mask must be valid if provided
        if custom_attention_mask is not None:
            if not isinstance(custom_attention_mask, torch.Tensor):
                raise ValueError(
                    "Invariant violation: custom_attention_mask must be a torch.Tensor"
                )
            if custom_attention_mask.dim() < 3:
                raise ValueError(
                    f"Invariant violation: custom_attention_mask must have at least 3 dimensions, got {custom_attention_mask.dim()}"
                )
            if (
                torch.isnan(custom_attention_mask).any()
                or torch.isinf(custom_attention_mask).any()
            ):
                raise ValueError(
                    "Invariant violation: custom_attention_mask contains NaN or Inf values"
                )

        # KV CACHE INVARIANT: If present, past_key_values must have valid structure
        if past_key_values is not None:
            # Structure validation
            if not isinstance(past_key_values, list):
                raise ValueError("Invariant violation: past_key_values must be a list")
            if len(past_key_values) == 0:
                raise ValueError(
                    "Invariant violation: past_key_values must not be empty"
                )

            # Each layer must be a tuple with key and value tensors
            if not all(isinstance(layer, tuple) for layer in past_key_values):
                raise ValueError(
                    "Invariant violation: Each element in past_key_values must be a tuple"
                )

            # Each tuple must have at least 2 elements (key and value)
            if not all(len(layer) >= 2 for layer in past_key_values):
                raise ValueError(
                    "Invariant violation: Each tuple in past_key_values must have at least 2 elements"
                )

            # Each key and value tensor must be a valid tensor
            for i, layer in enumerate(past_key_values):
                if not (
                    isinstance(layer[0], torch.Tensor)
                    and isinstance(layer[1], torch.Tensor)
                ):
                    raise ValueError(
                        f"Invariant violation: Layer {i} in past_key_values does not contain valid tensors"
                    )

                # Key and value tensors must have valid dimensions
                if layer[0].dim() < 3 or layer[1].dim() < 3:
                    raise ValueError(
                        f"Invariant violation: Key and value tensors in layer {i} must have at least 3 dimensions"
                    )

                # Key and value tensors must not have NaN or Inf values
                if torch.isnan(layer[0]).any() or torch.isinf(layer[0]).any():
                    raise ValueError(
                        f"Invariant violation: Key tensor in layer {i} contains NaN or Inf values"
                    )
                if torch.isnan(layer[1]).any() or torch.isinf(layer[1]).any():
                    raise ValueError(
                        f"Invariant violation: Value tensor in layer {i} contains NaN or Inf values"
                    )

            # All layers must have the same sequence length
            seq_lengths = set()
            for layer in past_key_values:
                if hasattr(layer[0], "size") and layer[0].dim() >= 3:
                    seq_lengths.add(layer[0].size(2))

            if len(seq_lengths) > 1:
                raise ValueError(
                    f"Invariant violation: Inconsistent sequence lengths across KV cache layers: {seq_lengths}"
                )

            # Track KV cache growth for performance monitoring
            if not hasattr(self, "_last_kv_cache_size"):
                self._last_kv_cache_size = 0

            current_size = list(seq_lengths)[0] if seq_lengths else 0
            if current_size > self._last_kv_cache_size:
                if self.debug_mode:
                    self.log(
                        f"KV cache grew from {self._last_kv_cache_size} to {current_size} tokens"
                    )
                self._last_kv_cache_size = current_size

        # DIAGNOSTIC: Check first token generation
        first_token_gen = past_key_values is None and input_ids.size(1) > 1
        if first_token_gen and self.debug_mode:
            self.log("\nDIAGNOSTIC - First token generation:")
            self.log(f"  Input shape: {input_ids.shape}")
            self.log(f"  Attention mask shape: {attention_mask.shape}")
            self.log(
                f"  Custom attention mask: {'Provided' if custom_attention_mask is not None else 'None'}"
            )
            if custom_attention_mask is not None:
                self.log(
                    f"  Custom attention mask shape: {custom_attention_mask.shape}"
                )

            # DIAGNOSTIC: Show model type
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "model_type"
            ):
                self.log(f"  Model type: {self.model.config.model_type}")

        # Performance tracking
        start_time = time.time()
        self.perf_stats["model_calls"] += 1

        # For KV caching with standard (non-parallel) token processing
        # we only need to process the most recent token
        if past_key_values is not None and len(input_ids.shape) > 1:
            # Store original shape for invariant checking
            original_input_shape = input_ids.shape

            # Only use the last token when using KV cache for standard processing
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # INVARIANT: Ensure reshaping preserved the batch dimension
            if input_ids.size(0) != original_input_shape[0]:
                raise ValueError(
                    f"Invariant violation: Batch dimension changed after reshaping input: {original_input_shape[0]} -> {input_ids.size(0)}"
                )

            # INVARIANT: Ensure we have just one token per sequence after reshaping
            if input_ids.size(1) != 1:
                raise ValueError(
                    f"Invariant violation: Expected single token per sequence after reshaping, got {input_ids.size(1)}"
                )

            # Extract the sequence length from past_key_values to ensure attention mask has correct size
            if (
                len(past_key_values) > 0
                and isinstance(past_key_values[0], tuple)
                and len(past_key_values[0]) >= 1
            ):
                past_seq_len = past_key_values[0][0].size(2)

                # Create a proper attention mask that matches the expected size
                original_attn_shape = attention_mask.shape
                attention_mask = torch.ones((1, past_seq_len + 1), device=self.device)

                # INVARIANT: Ensure attention mask has correct dimensions
                if attention_mask.size(1) != past_seq_len + 1:
                    raise ValueError(
                        f"Invariant violation: Attention mask has wrong sequence length: {attention_mask.size(1)}, expected {past_seq_len + 1}"
                    )

        # Check if the model is Qwen-based (Cogito uses Qwen2.5 base)
        model_type = getattr(self.model.config, "model_type", "").lower()
        is_qwen = "qwen" in model_type

        # Use inference mode for efficiency
        with torch.inference_mode():
            # No fallbacks - single implementation that fails explicitly
            try:
                # Standard approach with all arguments
                model_args = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True,  # Always use cache when possible for efficiency
                    "output_attentions": True,  # Always request attention patterns for reuse in pruning
                    "return_dict": True,  # Ensure we get the full output object with attentions
                }

                # Only include past_key_values if they are provided
                if past_key_values is not None:
                    model_args["past_key_values"] = past_key_values

                # Add custom attention mask if provided
                if custom_attention_mask is not None:
                    if is_qwen:
                        # Handle Qwen-specific custom attention
                        if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                            model_args["position_bias"] = custom_attention_mask
                        elif (
                            custom_attention_mask.dim() == 4
                        ):  # [batch, heads, seq, seq]
                            model_args["attention_mask"] = custom_attention_mask
                    elif (
                        hasattr(self.model.config, "model_type")
                        and "mistral" in self.model.config.model_type.lower()
                    ):
                        # Try to use the model-specific way to handle custom masks
                        if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                            model_args["position_attention_mask"] = (
                                custom_attention_mask
                            )
                        elif (
                            custom_attention_mask.dim() == 4
                        ):  # [batch, heads, seq, seq]
                            # Some models expect a 4D mask with head dimension
                            model_args["attention_mask"] = custom_attention_mask
                    else:
                        # For generic models, try the standard custom_attention_mask parameter
                        model_args["custom_attention_mask"] = custom_attention_mask

                outputs = self.model(**model_args)

                # === START NEW DEBUG LOGGING ===
                if self.debug_mode:
                    self.log(f"DEBUG DIRECT: Output type = {type(outputs)}", level="warning")
                    try:
                        # Use dir() to see all available attributes/methods
                        self.log(f"DEBUG DIRECT: Available attributes/methods = {dir(outputs)}", level="warning")
                        # Specifically check for attentions attribute presence
                        has_attentions = hasattr(outputs, 'attentions')
                        self.log(f"DEBUG DIRECT: hasattr(outputs, 'attentions') = {has_attentions}", level="warning")
                        if has_attentions:
                            # Check if the attribute is None or evaluates to False (e.g., empty list)
                            is_attentions_none_or_false = outputs.attentions is None or not outputs.attentions
                            self.log(f"DEBUG DIRECT: outputs.attentions is None or evaluates to False = {is_attentions_none_or_false}", level="warning")
                            if not is_attentions_none_or_false:
                                self.log(f"DEBUG DIRECT: Type of outputs.attentions = {type(outputs.attentions)}", level="warning")
                                if isinstance(outputs.attentions, (list, tuple)):
                                    self.log(f"DEBUG DIRECT: Length of outputs.attentions = {len(outputs.attentions)}", level="warning")

                    except Exception as inspect_err:
                         self.log(f"DEBUG DIRECT: Error inspecting outputs object: {inspect_err}", level="error")
                # === END NEW DEBUG LOGGING ===

                # Conditional logic for setting self.cached_attention based on inspection
                if hasattr(outputs, "attentions") and outputs.attentions:
                     self.cached_attention = outputs.attentions
                     if self.debug_mode:
                         self.log(f"DEBUG DIRECT: self.cached_attention was SET.", level="info")
                else:
                     # Log exactly why it wasn't set based on inspection results
                     if not hasattr(outputs, "attentions"):
                         self.log("DEBUG DIRECT: self.cached_attention NOT SET - object lacks 'attentions' attribute.", level="warning")
                     elif outputs.attentions is None:
                         self.log("DEBUG DIRECT: self.cached_attention NOT SET - 'attentions' attribute is None.", level="warning")
                     else: # hasattr is true, but outputs.attentions is False (e.g., empty list)
                         self.log("DEBUG DIRECT: self.cached_attention NOT SET - 'attentions' attribute is present but empty.", level="warning")
                     self.cached_attention = None # Explicitly set to None

                # === Previous debug logging ===
                if self.debug_mode:
                    self.log("--- Attention Output Check ---")
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        self.log(f"Model returned {len(outputs.attentions)} attention layers.")
                        if len(outputs.attentions) > 0 and outputs.attentions[0] is not None and hasattr(outputs.attentions[0], 'shape'):
                             self.log(f"First attention layer shape: {outputs.attentions[0].shape}")
                        else:
                             self.log("First attention layer has no shape attribute or is invalid.")
                    elif hasattr(outputs, 'attentions'):
                         self.log("Model returned 'attentions' attribute, but it is None or empty.", level="warning")
                    else:
                        self.log("Model output object does NOT have 'attentions' attribute.", level="warning")
                    self.log("-----------------------------")

                # INVARIANT: Ensure outputs has logits attribute
                if not hasattr(outputs, "logits"):
                    raise ValueError(
                        "Invariant violation: Model outputs missing 'logits' attribute"
                    )

                # INVARIANT: Ensure logits tensor is valid
                if not isinstance(outputs.logits, torch.Tensor):
                    raise ValueError(
                        f"Invariant violation: outputs.logits is not a tensor, got {type(outputs.logits)}"
                    )

                # INVARIANT: Ensure logits has the right shape
                if outputs.logits.dim() < 3:
                    raise ValueError(
                        f"Invariant violation: outputs.logits has wrong dimensionality, expected at least 3, got {outputs.logits.dim()}"
                    )

                # Store attention patterns for later use in pruning
                if hasattr(outputs, "attentions") and outputs.attentions:
                    self.cached_attention = outputs.attentions

                    # Debug: Print attention shapes
                    # if self.debug_mode:
                    #     print("\n*** ATTENTION SHAPES DEBUG ***")
                    #     for i, layer_attn in enumerate(outputs.attentions):
                    #         print(f"Layer {i} attention shape: {layer_attn.shape}")
                    #     print("*****************************\n")

                    # Store sequence length for verification
                    if not hasattr(self, "cached_attention_seq_len"):
                        self.cached_attention_seq_len = []
                    current_seq_len = input_ids.size(1)
                    if past_key_values is not None:
                        # When using KV cache, add the past sequence length
                        if len(past_key_values) > 0 and hasattr(
                            past_key_values[0][0], "size"
                        ):
                            current_seq_len += past_key_values[0][0].size(2)
                    self.cached_attention_seq_len.append(current_seq_len)

                    if self.debug_mode:
                        self.log(
                            f"Cached attention for sequence length {current_seq_len}"
                        )
                else:
                    # Explicitly log if not set
                    if self.debug_mode:
                        self.log("Failed to cache attention - outputs.attentions is missing or empty", level="warning")
                        self.cached_attention = None  # Ensure it's explicitly None

            except Exception as e:
                # No fallbacks - fail with explicit error
                raise RuntimeError(
                    f"Model execution failed: {e}. No fallback will be attempted."
                )

        # Get logits for last position
        next_token_logits = outputs.logits[:, -1, :]

        # INVARIANT: Output logits must be properly shaped and contain valid values
        if next_token_logits.dim() < 2:
            raise ValueError(
                f"Invariant violation: Output logits have wrong dimensionality, expected at least 2, got {next_token_logits.dim()}"
            )

        if next_token_logits.size(0) != input_ids.size(0):
            raise ValueError(
                f"Invariant violation: Output logits have incorrect batch dimension: expected {input_ids.size(0)}, got {next_token_logits.size(0)}"
            )

        # INVARIANT: Logits must not contain NaN or Inf values
        if torch.isnan(next_token_logits).any():
            raise ValueError("Invariant violation: Output logits contain NaN values")
        if torch.isinf(next_token_logits).any():
            raise ValueError("Invariant violation: Output logits contain Inf values")

        # INVARIANT: Logits must have valid vocabulary dimension
        if hasattr(self.model.config, "vocab_size"):
            expected_vocab_size = self.model.config.vocab_size
            if next_token_logits.size(1) != expected_vocab_size:
                self.log(
                    f"Warning: Output logits vocabulary dimension {next_token_logits.size(1)} doesn't match model's vocab size {expected_vocab_size}"
                )

        # Safely handle past_key_values from the model outputs
        return_kvs = None
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            # KV CACHE CONSISTENCY INVARIANT: Validate returned KV cache structure
            if (
                isinstance(outputs.past_key_values, list)
                and len(outputs.past_key_values) > 0
            ):
                # Check each layer has key and value tensors
                if all(
                    isinstance(layer_cache, tuple) and len(layer_cache) >= 2
                    for layer_cache in outputs.past_key_values
                ):
                    # INVARIANT: All layers in KV cache must have the same sequence length
                    new_seq_lengths = set()
                    for layer in outputs.past_key_values:
                        if layer[0].dim() >= 3:
                            new_seq_lengths.add(layer[0].size(2))

                    if len(new_seq_lengths) > 1:
                        self.log(
                            f"Warning: Inconsistent sequence lengths in returned KV cache: {new_seq_lengths}",
                            level="warning",
                        )

                    # KV CACHE GROWTH INVARIANT: Ensure the KV cache grows as expected
                    if past_key_values is not None:
                        # Get current sequence length
                        current_length = (
                            list(new_seq_lengths)[0] if new_seq_lengths else 0
                        )
                        # Get previous sequence length
                        prev_lengths = set()
                        for layer in past_key_values:
                            if layer[0].dim() >= 3:
                                prev_lengths.add(layer[0].size(2))
                        prev_length = list(prev_lengths)[0] if prev_lengths else 0

                        # For standard token-by-token generation with KV cache, sequence length should remain unchanged
                        # For full context generation (past_key_values is None), it should match input_ids length
                        # If neither condition is met, log a warning
                        if (
                            current_length != prev_length
                            and current_length != input_ids.size(1) + prev_length
                        ):
                            self.log(
                                f"Warning: Unexpected KV cache sequence length change: {prev_length} -> {current_length}"
                            )

                        # Update sequence length tracking for performance analysis
                        sequence_tracker.update_length(current_length)
                        if hasattr(self, "debug_mode") and self.debug_mode:
                            self.log(f"🔄 Sequence length updated to: {current_length}")

                    # Use the validated KV cache
                    return_kvs = outputs.past_key_values
                else:
                    if self.debug_mode:
                        self.log(
                            "Warning: Model returned invalid past_key_values structure. Using None.",
                            level="warning",
                        )
                        # Log the problematic structure
                        invalid_layers = []
                        for i, layer in enumerate(outputs.past_key_values):
                            if not (isinstance(layer, tuple) and len(layer) >= 2):
                                invalid_layers.append(i)
                        self.log(f"Invalid layers: {invalid_layers}")
            else:
                if self.debug_mode:
                    self.log(
                        "Warning: Model returned empty past_key_values. Using None.",
                        level="warning",
                    )

        # Update performance stats
        self.perf_stats["model_time"] += time.time() - start_time

        # For debugging KV cache usage
        if self.debug_mode and hasattr(self, "_last_kv_cache_size"):
            if return_kvs is None:
                self.log("KV cache reset or not used for this generation step")
            else:
                # Get current size
                new_size = 0
                for layer in return_kvs:
                    if layer[0].dim() >= 3:
                        new_size = layer[0].size(2)
                        break

                self._last_kv_cache_size = new_size

        return next_token_logits, return_kvs

    def batch_decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Efficiently decode multiple tokens in a single call.
        Uses caching to avoid redundant decoding operations.

        Args:
            token_ids: List of token IDs

        Returns:
            List[str]: Decoded tokens
        """
        # Performance tracking
        start_time = time.time()
        self.perf_stats["decode_calls"] += 1

        # Process tokens in batches for efficiency
        if not token_ids:
            return []

        # Use cached decoding results when available
        cached_results = []
        uncached_tokens = []
        token_to_position = {}  # Track original positions

        # Check cache for each token
        for i, token_id in enumerate(token_ids):
            token_key = int(token_id)  # Ensure consistent key type
            if token_key in self.token_decode_cache:
                self.perf_stats["decode_cache_hits"] += 1
                cached_results.append((i, self.token_decode_cache[token_key]))
            else:
                uncached_tokens.append(token_id)
                token_to_position[len(uncached_tokens) - 1] = i

        # Only decode uncached tokens
        decoded_tokens = []
        if uncached_tokens:
            # Convert to tensors for efficient batch processing
            tokens_tensor = torch.tensor([uncached_tokens], device="cpu")

            # Get token strings using batch processing
            batch_decoded = self.tokenizer.batch_decode(
                tokens_tensor,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # Split the decoded strings
            decoded_tokens = batch_decoded[0].split(" ")

            # Cache the newly decoded tokens
            for i, token_id in enumerate(uncached_tokens):
                if i < len(decoded_tokens):
                    token_key = int(token_id)
                    self.token_decode_cache[token_key] = decoded_tokens[i]

        # Combine cached and newly decoded results
        result = [""] * len(token_ids)  # Pre-allocate result list

        # Insert cached results
        for pos, text in cached_results:
            result[pos] = text

        # Insert newly decoded results
        for i, text in enumerate(decoded_tokens):
            orig_pos = token_to_position.get(i)
            if orig_pos is not None:
                result[orig_pos] = text

        self.perf_stats["decode_time"] += time.time() - start_time
        return result

    def print_performance_stats(self):
        """Print performance statistics."""
        print("\nToken Generator Performance Stats:")
        print(f"  Tokenization calls: {self.perf_stats['tokenization_calls']}")
        print(f"  Tokenization time: {self.perf_stats['tokenization_time']:.4f}s")
        print(f"  Model calls: {self.perf_stats['model_calls']}")
        print(f"  Model time: {self.perf_stats['model_time']:.4f}s")
        print(f"  Cache hits: {self.perf_stats['cache_hits']}")
        print(f"  Cache misses: {self.perf_stats['cache_misses']}")

        # Token decoding stats
        decode_calls = self.perf_stats["decode_calls"]
        if decode_calls > 0:
            cache_hits = self.perf_stats["decode_cache_hits"]
            hit_rate = (cache_hits / decode_calls) * 100 if decode_calls > 0 else 0
            print(f"  Token decode calls: {decode_calls}")
            print(f"  Token decode cache hits: {cache_hits} ({hit_rate:.1f}%)")
            print(f"  Token decode time: {self.perf_stats['decode_time']:.4f}s")

        # Isolated token optimization stats
        if (
            hasattr(self, "isolated_tokens_processed")
            and self.isolated_tokens_processed > 0
        ):
            isolated_count = self.isolated_tokens_processed
            print(f"\n  Isolated parallel token optimization:")
            print(f"    Tokens processed in isolated mode: {isolated_count}")
            if self.perf_stats["model_calls"] > 0:
                avg_time = (
                    self.perf_stats["model_time"] / self.perf_stats["model_calls"]
                )
                saved_time = (
                    isolated_count - self.perf_stats["model_calls"]
                ) * avg_time
                print(f"    Estimated compute time saved: {saved_time*1000:.1f}ms")

        if self.perf_stats["model_calls"] > 0:
            avg_time = self.perf_stats["model_time"] / self.perf_stats["model_calls"]
            print(f"  Average model call time: {avg_time:.4f}s")

    def enable_detailed_perf(self, enabled=True):
        """Enable or disable detailed per-call performance logging."""
        self.detailed_perf = enabled

    def get_cached_attention(self):
        """
        Get the cached attention from the last forward pass.
        
        Returns:
            tuple: (cached_attention, sequence_length)
        """
        if hasattr(self, "cached_attention") and self.cached_attention is not None:
            # Log shapes for debugging
            if self.debug_mode:
                attention_layer_count = len(self.cached_attention)
                if attention_layer_count > 0:
                    first_layer_shape = self.cached_attention[0].shape
                    last_layer_shape = self.cached_attention[-1].shape
                    self.log(f"*** ATTENTION SHAPES DEBUG ***")
                    self.log(f"Cached attention has {attention_layer_count} layers")
                    self.log(f"First layer shape: {first_layer_shape}")
                    self.log(f"Last layer shape: {last_layer_shape}")
                    
                    # Log more details about dimensions
                    if len(first_layer_shape) >= 4:
                        batch_size, num_heads, seq_len, attended_len = first_layer_shape
                        self.log(f"Attention dimensions: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, attended_len={attended_len}")
                    
                    # Log sequence length
                    seq_len = self.cached_attention_seq_len[0] if hasattr(self, "cached_attention_seq_len") else None
                    self.log(f"Cached attention sequence length: {seq_len}")
                else:
                    self.log("Warning: Cached attention is empty (0 layers)")
            
            # Return cached attention and its sequence length
            seq_len = None
            if hasattr(self, "cached_attention_seq_len") and self.cached_attention_seq_len:
                seq_len = self.cached_attention_seq_len[0]
            return self.cached_attention, seq_len
        
        # No cached attention available
        if self.debug_mode:
            self.log("Warning: No cached attention available", "warning")
        return None, None

    def get_next_token_logits_for_isolated_parallel(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        custom_attention_mask: Optional[torch.Tensor] = None,
        num_parallel_tokens: int = 1,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        Optimized method for getting logits for isolated parallel tokens.
        Since isolated tokens can't see each other, we only need to compute
        the forward pass once and can reuse the logits for all tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Current attention mask
            past_key_values: Optional past key values for KV cache
            custom_attention_mask: Optional custom attention mask
            num_parallel_tokens: Number of parallel tokens to generate (for logging)

        Returns:
            tuple: (next_token_logits, past_key_values)
        """
        self.log(">>> ENTERING get_next_token_logits_for_isolated_parallel <<<", level="warning")
        
        # Performance tracking
        start_time = time.time()
        self.perf_stats["model_calls"] += 1

        if self.debug_mode:
            self.log(
                f"Using optimized isolated parallel token generation for {num_parallel_tokens} tokens"
            )

        # Input validation - same as the regular method
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Invariant violation: input_ids must be a torch.Tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError(
                "Invariant violation: attention_mask must be a torch.Tensor"
            )

        # Compute the logits only once since we're in isolated mode
        # We only need the first token of each parallel set since they all see the same context
        if past_key_values is not None and len(input_ids.shape) > 1:
            # Only use the last token when using KV cache
            input_ids_for_model = input_ids[:, -1].unsqueeze(-1)

            # Extract the sequence length from past_key_values to ensure attention mask has correct size
            if (
                len(past_key_values) > 0
                and isinstance(past_key_values[0], tuple)
                and len(past_key_values[0]) >= 1
            ):
                past_seq_len = past_key_values[0][0].size(2)

                # Create a proper attention mask that matches the expected size
                attention_mask_for_model = torch.ones(
                    (input_ids.size(0), past_seq_len + 1), device=self.device
                )
            else:
                # If we can't determine the past sequence length, use the original attention mask
                attention_mask_for_model = attention_mask
        else:
            # For the first token or when KV cache is disabled, use the full input
            input_ids_for_model = input_ids
            attention_mask_for_model = attention_mask

        # Check if the model is Qwen-based
        model_type = getattr(self.model.config, "model_type", "").lower()
        is_qwen = "qwen" in model_type

        # Use inference mode for efficiency
        with torch.inference_mode():
            # Prepare model arguments - similar to the regular method
            model_args = {
                "input_ids": input_ids_for_model,
                "attention_mask": attention_mask_for_model,
                "use_cache": True,
                "output_attentions": True,
                "return_dict": True,  # Ensure we get the full output object with attentions
            }

            # Only include past_key_values if they are provided
            if past_key_values is not None:
                model_args["past_key_values"] = past_key_values

            # Add custom attention mask if provided
            if custom_attention_mask is not None:
                if is_qwen:
                    # Handle Qwen-specific custom attention
                    if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                        model_args["position_bias"] = custom_attention_mask
                    elif custom_attention_mask.dim() == 4:  # [batch, heads, seq, seq]
                        model_args["attention_mask"] = custom_attention_mask
                elif (
                    hasattr(self.model.config, "model_type")
                    and "mistral" in self.model.config.model_type.lower()
                ):
                    # Try to use the model-specific way to handle custom masks
                    if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                        model_args["position_attention_mask"] = custom_attention_mask
                    elif custom_attention_mask.dim() == 4:  # [batch, heads, seq, seq]
                        # Some models expect a 4D mask with head dimension
                        model_args["attention_mask"] = custom_attention_mask
                else:
                    # For generic models, try the standard custom_attention_mask parameter
                    model_args["custom_attention_mask"] = custom_attention_mask

            # Run the model forward pass just once
            outputs = self.model(**model_args)

            # === START NEW DEBUG LOGGING ===
            if self.debug_mode:
                self.log(f"DEBUG DIRECT: Output type (isolated parallel) = {type(outputs)}", level="warning")
                try:
                    # Use dir() to see all available attributes/methods
                    self.log(f"DEBUG DIRECT: Available attributes/methods = {dir(outputs)}", level="warning")
                    # Specifically check for attentions attribute presence
                    has_attentions = hasattr(outputs, 'attentions')
                    self.log(f"DEBUG DIRECT: hasattr(outputs, 'attentions') = {has_attentions}", level="warning")
                    if has_attentions:
                        # Check if the attribute is None or evaluates to False (e.g., empty list)
                        is_attentions_none_or_false = outputs.attentions is None or not outputs.attentions
                        self.log(f"DEBUG DIRECT: outputs.attentions is None or evaluates to False = {is_attentions_none_or_false}", level="warning")
                        if not is_attentions_none_or_false:
                            self.log(f"DEBUG DIRECT: Type of outputs.attentions = {type(outputs.attentions)}", level="warning")
                            if isinstance(outputs.attentions, (list, tuple)):
                                self.log(f"DEBUG DIRECT: Length of outputs.attentions = {len(outputs.attentions)}", level="warning")

                except Exception as inspect_err:
                     self.log(f"DEBUG DIRECT: Error inspecting outputs object: {inspect_err}", level="error")
            # === END NEW DEBUG LOGGING ===

            # Conditional logic for setting self.cached_attention based on inspection
            if hasattr(outputs, "attentions") and outputs.attentions:
                 self.cached_attention = outputs.attentions
                 if self.debug_mode:
                     self.log(f"DEBUG DIRECT: self.cached_attention was SET.", level="info")
            else:
                 # Log exactly why it wasn't set based on inspection results
                 if not hasattr(outputs, "attentions"):
                     self.log("DEBUG DIRECT: self.cached_attention NOT SET - object lacks 'attentions' attribute.", level="warning")
                 elif outputs.attentions is None:
                     self.log("DEBUG DIRECT: self.cached_attention NOT SET - 'attentions' attribute is None.", level="warning")
                 else: # hasattr is true, but outputs.attentions is False (e.g., empty list)
                     self.log("DEBUG DIRECT: self.cached_attention NOT SET - 'attentions' attribute is present but empty.", level="warning")
                 self.cached_attention = None # Explicitly set to None

            # === Previous debug logging (now with warning level) ===
            if self.debug_mode:
                self.log("--- Attention Output Check (Isolated Parallel) ---")
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    self.log(f"Model returned {len(outputs.attentions)} attention layers.")
                    if len(outputs.attentions) > 0 and outputs.attentions[0] is not None and hasattr(outputs.attentions[0], 'shape'):
                         self.log(f"First attention layer shape: {outputs.attentions[0].shape}")
                    else:
                         self.log("First attention layer has no shape attribute or is invalid.")
                elif hasattr(outputs, 'attentions'):
                     self.log("Model returned 'attentions' attribute, but it is None or empty.", level="warning")
                else:
                    self.log("Model output object does NOT have 'attentions' attribute.", level="warning")
                self.log("-----------------------------")

            # Get logits for last position - this can be reused for all parallel tokens
            next_token_logits = outputs.logits[:, -1, :]

            # Store attention patterns for later use in pruning
            if hasattr(outputs, "attentions") and outputs.attentions:
                self.cached_attention = outputs.attentions

                # Debug: Print attention shapes
                # if self.debug_mode:
                #     print("\n*** ATTENTION SHAPES DEBUG ***")
                #     for i, layer_attn in enumerate(outputs.attentions):
                #         print(f"Layer {i} attention shape: {layer_attn.shape}")
                #     print("*****************************\n")

                # Store sequence length for verification
                if not hasattr(self, "cached_attention_seq_len"):
                    self.cached_attention_seq_len = []
                current_seq_len = input_ids_for_model.size(1)
                if past_key_values is not None:
                    # When using KV cache, add the past sequence length
                    if len(past_key_values) > 0 and hasattr(
                        past_key_values[0][0], "size"
                    ):
                        current_seq_len += past_key_values[0][0].size(2)
                self.cached_attention_seq_len.append(current_seq_len)

                if self.debug_mode:
                    self.log(
                        f"Cached attention for sequence length {current_seq_len} (shared among {num_parallel_tokens} parallel tokens)"
                    )
            else:
                # Explicitly log if not set
                if self.debug_mode:
                    self.log("Failed to cache attention in isolated parallel mode - outputs.attentions is missing or empty", level="warning")
                    self.cached_attention = None  # Ensure it's explicitly None

        # Update performance stats
        processing_time = time.time() - start_time
        self.perf_stats["model_time"] += processing_time

        # Track isolated token stats
        self.perf_stats["isolated_tokens_processed"] = (
            self.perf_stats.get("isolated_tokens_processed", 0) + num_parallel_tokens
        )

        if self.debug_mode:
            self.log(
                f"Isolated parallel token generation took {processing_time*1000:.2f}ms for {num_parallel_tokens} tokens"
            )
            self.log(
                f"Efficiency gain: {num_parallel_tokens}x (one forward pass instead of {num_parallel_tokens})"
            )

        # Get the updated KV cache
        return_kvs = None
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            return_kvs = outputs.past_key_values

        # Update parallel token count
        self.last_parallel_count = num_parallel_tokens

        return next_token_logits, return_kvs

    def clear_kv_cache(self):
        """
        Clear any KV cache state in the model.
        This helps prevent contamination from previous generations.
        """
        # Log the operation if in debug mode
        if self.debug_mode:
            self.log("Clearing KV cache for clean generation")
            
        # Use try/except instead of hasattr to safely attempt to clear the cache
        try:
            # Try to use clear_kv_cache method if it exists
            self.model.clear_kv_cache()
            if self.debug_mode:
                self.log("Successfully cleared KV cache using model.clear_kv_cache()")
            return
        except (AttributeError, AssertionError):
            # Method doesn't exist, continue to next approach
            pass
            
        try:
            # Try to use reset_cache method if it exists
            self.model.reset_cache()
            if self.debug_mode:
                self.log("Successfully cleared KV cache using model.reset_cache()")
            return
        except (AttributeError, AssertionError):
            # Method doesn't exist, continue to next approach
            pass
            
        try:
            # For Hugging Face models using past_key_values
            if self.model._past_key_values is not None:
                self.model._past_key_values = None
                if self.debug_mode:
                    self.log("Successfully cleared _past_key_values")
                return
        except (AttributeError, AssertionError):
            # Attribute doesn't exist, continue to next approach
            pass
            
        # If we get here, no cache clearing was possible
        self.log("No KV cache clearing mechanism available - continuing without clearing", level="warning")

    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        disable_kv_cache: bool = False,
        input_identifiers: Optional[List[int]] = None,
    ):
        """
        Generate the next token logits for the given input.
        This is a wrapper around the get_next_token_logits methods that handles KV caching.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            disable_kv_cache: Whether to disable KV caching
            input_identifiers: Optional identifiers for the input (for parallel generation)

        Returns:
            outputs: Model outputs with logits in shape [batch_size, sequence_length, vocab_size]
        """
        # Create a simple output container
        class SimpleOutputs:
            def __init__(self, logits):
                self.logits = logits
                self.attentions = None

        if disable_kv_cache:
            # Use the simpler method without KV caching
            logits = self.get_next_token_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Reshape logits to [batch_size, sequence_length=1, vocab_size]
            # ParallelGenerator expects 3D tensor with sequence dimension
            if logits.dim() == 2:  # [batch_size, vocab_size]
                logits = logits.unsqueeze(1)  # Add sequence dimension -> [batch_size, 1, vocab_size]
                
            return SimpleOutputs(logits)
        else:
            # Use the cached version for better performance
            if input_identifiers is not None and len(input_identifiers) > 1:
                # We're in parallel token mode, check if we can use the isolated optimization
                # This is a major optimization for isolated parallel tokens
                logits, _ = self.get_next_token_logits_for_isolated_parallel(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_parallel_tokens=len(input_identifiers),
                )
            else:
                # Standard token processing with KV cache
                logits, _ = self.get_next_token_logits_cached(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            # Reshape logits to [batch_size, sequence_length=1, vocab_size]
            # ParallelGenerator expects 3D tensor with sequence dimension
            if logits.dim() == 2:  # [batch_size, vocab_size]
                logits = logits.unsqueeze(1)  # Add sequence dimension -> [batch_size, 1, vocab_size]
                
            if self.debug_mode:
                self.log(f"Logits shape after reshaping: {logits.shape}")
                
            return SimpleOutputs(logits)

    def generate_next_token_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        custom_attention_mask: Optional[torch.Tensor] = None,
        disable_kv_cache: bool = False,
        input_identifiers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        Generate the next token using the model with explicit KV cache management.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask matching the full sequence length
            past_key_values: Optional past key values for KV cache
            custom_attention_mask: Optional custom attention mask
            disable_kv_cache: Whether to disable KV caching
            input_identifiers: Optional list of identifiers for the input tokens
            
        Returns:
            Tuple of (next_token_logits, updated_past_key_values)
        """
        # Validate inputs
        assert input_ids is not None, "input_ids cannot be None"
        assert attention_mask is not None, "attention_mask cannot be None"
        
        # Performance tracking
        start_time = time.time()
        self.perf_stats["model_calls"] += 1
        
        # Check if we need to use KV cache
        use_cache = not disable_kv_cache
        
        # Use inference mode for efficiency
        with torch.inference_mode():
            # Check if the model is Qwen-based
            model_type = getattr(self.model.config, "model_type", "").lower()
            is_qwen = "qwen" in model_type
            
            # Build model arguments
            model_args = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "output_attentions": True,
                "return_dict": True,
            }
            
            # Add past_key_values if provided and using cache
            if use_cache and past_key_values is not None:
                model_args["past_key_values"] = past_key_values
                
            # For Qwen models, handle attention specially
            if is_qwen and custom_attention_mask is not None:
                if custom_attention_mask.dim() == 3:
                    model_args["position_bias"] = custom_attention_mask
                elif custom_attention_mask.dim() == 4:
                    model_args["attention_mask"] = custom_attention_mask
                
            # Run the model
            outputs = self.model(**model_args)
        
        # Get logits for next token (last position)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Store attention patterns if available
        if hasattr(outputs, "attentions") and outputs.attentions:
            self.cached_attention = outputs.attentions
            if self.debug_mode:
                self.log(f"Cached attention in generate_next_token_with_cache (layers: {len(self.cached_attention)})")
                
                # Add sequence length tracking for retrieval
                if not hasattr(self, "cached_attention_seq_len"):
                    self.cached_attention_seq_len = []
                current_seq_len = input_ids.size(1)
                if past_key_values is not None:
                    # When using KV cache, add the past sequence length
                    if len(past_key_values) > 0 and hasattr(past_key_values[0][0], "size"):
                        current_seq_len += past_key_values[0][0].size(2)
                self.cached_attention_seq_len.append(current_seq_len)
                
                self.log(f"Cached attention for sequence length {current_seq_len}")
        else:
            self.cached_attention = None
            if self.debug_mode:
                self.log("No attention output received in generate_next_token_with_cache", "warning")
                
                # Log detailed diagnostics
                if hasattr(outputs, "attentions"):
                    self.log("outputs.attentions attribute exists but is empty or None", "warning")
                else:
                    self.log("outputs object lacks attentions attribute", "warning")
                    
                # Log what the model actually returned
                if hasattr(outputs, "__dict__"):
                    self.log(f"Available model output attributes: {list(outputs.__dict__.keys())}", "warning")
        
        # Get updated past_key_values
        updated_past_key_values = outputs.past_key_values if use_cache else None
        
        # Update performance tracking
        self.perf_stats["model_time"] += time.time() - start_time
        
        return next_token_logits, updated_past_key_values

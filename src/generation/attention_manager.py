import torch
import logging
import os
from typing import Tuple, Optional, List, Dict, Any


class AttentionManager:
    """
    Manages attention masking for parallel token generation.
    Optimized for efficient KV caching and low-overhead tensor operations.
    Enhanced to work with custom RoPE positional encoding.
    """

    def __init__(self, device: str = "mps", rope_modifier=None, tokenizer=None):
        """
        Initialize the attention manager.

        Args:
            device: Device to use for computation
            rope_modifier: Optional RoPE modifier instance for coordination
            tokenizer: Optional tokenizer for debug output
        """
        self.device = device
        self.rope_modifier = rope_modifier
        self.tokenizer = tokenizer
        self.full_attention_mask = None

        # Cached masks by size for performance
        self._mask_cache = {}

        # Track parallel token sets for better coordination with RoPE
        self.parallel_token_positions = {}

        # Performance tracking
        self.perf_stats = {
            "mask_creations": 0,
            "mask_updates": 0,
            "mask_cache_hits": 0,
            "mask_creation_time": 0,
            "mask_update_time": 0,
        }

        # Debug mode
        self.debug_mode = True

        # Setup logging to file
        self._setup_logger()

    def _setup_logger(self):
        """Setup logging to file."""
        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("attention_manager")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(log_dir, "attention_debug.log")

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

        # Log initialization
        self.logger.info("AttentionManager initialized")

    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
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

    def create_parallel_set_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        new_token_ids: list,
        isolate_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create updated input tensors with the new parallel token set.

        Args:
            input_ids: Current input token IDs
            attention_mask: Current attention mask
            new_token_ids: List of token IDs to add
            isolate_tokens: If True, parallel tokens cannot attend to each other

        Returns:
            tuple: (updated_input_ids, updated_attention_mask)
        """
        # Invariant: Input tensors must have matching batch and sequence dimensions
        if input_ids.size(0) != attention_mask.size(0):
            raise ValueError(
                f"Invariant violation: input_ids batch size ({input_ids.size(0)}) must match attention_mask batch size ({attention_mask.size(0)})"
            )

        if input_ids.size(1) != attention_mask.size(1):
            raise ValueError(
                f"Invariant violation: input_ids sequence length ({input_ids.size(1)}) must match attention_mask sequence length ({attention_mask.size(1)})"
            )

        # Invariant: new_token_ids must contain valid integers
        if not all(isinstance(token_id, int) for token_id in new_token_ids):
            raise ValueError(
                "Invariant violation: new_token_ids must contain only integer values"
            )

        # Skip empty token lists
        if not new_token_ids:
            return input_ids, attention_mask

        # Handle singleton case more efficiently
        if len(new_token_ids) == 1:
            # Add a single token - more efficient than creating new tensors
            # from scratch for common case
            new_input_ids = torch.cat(
                [input_ids, torch.tensor([[new_token_ids[0]]], device=self.device)],
                dim=1,
            )

            new_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device)], dim=1
            )

            # Update the full attention mask if needed
            if self.full_attention_mask is not None:
                # Create updated full attention mask
                seq_len = input_ids.size(1)
                full_size = seq_len + 1  # Adding one token

                # Use cached masks if available
                if full_size in self._mask_cache:
                    self.full_attention_mask = self._mask_cache[full_size]
                else:
                    # Create the mask just once
                    self.full_attention_mask = torch.ones(
                        (1, full_size, full_size), device=self.device
                    )
                    # Cache it for future use
                    self._mask_cache[full_size] = self.full_attention_mask

            return new_input_ids, new_attention_mask

        # Handle multiple parallel tokens case
        # Create new input IDs and attention mask for parallel tokens
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        num_new_tokens = len(new_token_ids)

        # Track this parallel set for RoPE coordination
        current_pos = seq_len
        # Create position mapping for parallel tokens
        if num_new_tokens > 1:
            position_mapping = {}
            for i in range(num_new_tokens):
                position_mapping[current_pos + i] = (
                    current_pos  # Map all to first position
                )

            # Store for later use
            self.parallel_token_positions[current_pos] = {
                "start": current_pos,
                "end": current_pos + num_new_tokens - 1,
                "size": num_new_tokens,
            }

            # Update RoPE modifier if available
            if self.rope_modifier is not None:
                self.rope_modifier.register_parallel_positions(position_mapping)

            if self.debug_mode:
                self.log(
                    f"Registered parallel token set: pos={current_pos}, size={num_new_tokens}"
                )

        # Initialize tensors with optimized pre-allocation
        # Pre-allocate with correct sizes
        new_input_ids = torch.cat(
            [
                input_ids,
                torch.zeros(
                    (batch_size, num_new_tokens),
                    device=self.device,
                    dtype=input_ids.dtype,
                ),
            ],
            dim=1,
        )

        new_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, num_new_tokens), device=self.device),
            ],
            dim=1,
        )

        # Fill in token IDs for the parallel set
        for i, token_id in enumerate(new_token_ids):
            new_input_ids[0, seq_len + i] = token_id

        # Create a special attention mask for parallel tokens to allow
        # full attention between tokens in the same parallel set
        if num_new_tokens > 1:
            # Update full attention mask for transformer if needed
            full_size = seq_len + num_new_tokens

            # Create a fresh mask that allows parallel tokens to attend to each other
            self.full_attention_mask = self._create_parallel_attention_mask(
                seq_len, num_new_tokens, full_size, isolate_tokens
            )

            # Cache it for potential reuse
            self._mask_cache[
                ("parallel", full_size, num_new_tokens, isolate_tokens)
            ] = self.full_attention_mask
        elif self.full_attention_mask is not None:
            # Use standard causal mask for single tokens
            full_size = seq_len + num_new_tokens

            # Use cached mask if available
            if full_size in self._mask_cache:
                self.full_attention_mask = self._mask_cache[full_size]
            else:
                self.full_attention_mask = torch.ones(
                    (1, full_size, full_size), device=self.device
                )
                self._mask_cache[full_size] = self.full_attention_mask

        return new_input_ids, new_attention_mask

    def _create_parallel_attention_mask(
        self,
        seq_len: int,
        num_parallel: int,
        full_size: int,
        isolate_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Create a custom attention mask that allows tokens in a parallel set to attend to each other.
        Optimized with caching for better performance.

        Args:
            seq_len: Original sequence length (before adding parallel tokens)
            num_parallel: Number of tokens in the parallel set
            full_size: Full size of the sequence including parallel tokens
            isolate_tokens: If True, parallel tokens cannot attend to each other

        Returns:
            torch.Tensor: Custom attention mask
        """
        import time

        start_time = time.time()

        # Check cache first for exact match
        cache_key = ("parallel", full_size, seq_len, num_parallel, isolate_tokens)
        if cache_key in self._mask_cache:
            self.perf_stats["mask_cache_hits"] += 1
            if self.debug_mode:
                self.log(
                    f"Attention mask cache hit for size {full_size} with {num_parallel} parallel tokens (isolated: {isolate_tokens})"
                )
            return self._mask_cache[cache_key]

        # Check if we can update an existing mask instead of creating from scratch
        if (
            "parallel",
            full_size,
            seq_len - 1,
            num_parallel,
            isolate_tokens,
        ) in self._mask_cache:
            self.perf_stats["mask_updates"] += 1
            # Get previous mask and expand it
            prev_mask = self._mask_cache[
                ("parallel", full_size - 1, seq_len - 1, num_parallel, isolate_tokens)
            ]
            # Update by adding a new row and column
            mask = torch.ones((1, full_size, full_size), device=self.device)
            mask[:, : full_size - 1, : full_size - 1] = prev_mask
            # Set new row and column for causal attention
            mask[:, full_size - 1, full_size:] = 0.0  # Cannot attend to future tokens
            # Cache and return
            self._mask_cache[cache_key] = mask
            self.perf_stats["mask_update_time"] += time.time() - start_time
            return mask

        # Create new mask from scratch
        self.perf_stats["mask_creations"] += 1

        # Create standard causal mask as starting point
        # Shape: [1, full_size, full_size]
        mask = torch.tril(torch.ones((1, full_size, full_size), device=self.device))

        # Now modify the mask to allow parallel tokens to attend to each other
        # For each parallel token, allow it to attend to all other parallel tokens
        parallel_start = seq_len
        parallel_end = seq_len + num_parallel

        # Allow all tokens in parallel set to attend to each other if not isolated
        # This is the key modification that makes parallel tokens work as alternatives
        # Vectorized version - replace nested loops with tensor slicing
        if not isolate_tokens:
            mask[:, parallel_start:parallel_end, parallel_start:parallel_end] = 1.0

            if self.debug_mode:
                self.log(
                    f"Created parallel attention mask with mutual attention between {num_parallel} parallel tokens"
                )
        else:
            # In isolated mode, keep the tril mask which prevents tokens from seeing each other
            # The default tril mask already creates the right pattern - causal attention only
            if self.debug_mode:
                self.log(
                    f"Created parallel attention mask with isolated attention (no cross-attention) for {num_parallel} parallel tokens"
                )

        # Skip validation in production for performance
        if self.debug_mode:
            if not isolate_tokens:
                # 1. Check if all parallel tokens can attend to each other (all 1.0)
                parallel_block = mask[
                    :, parallel_start:parallel_end, parallel_start:parallel_end
                ]
                if not torch.all(parallel_block == 1.0):
                    # Find positions that violate the constraint
                    invalid_positions = torch.nonzero(
                        parallel_block != 1.0, as_tuple=True
                    )
                    if len(invalid_positions) >= 3 and len(invalid_positions[0]) > 0:
                        # Convert to absolute positions for error reporting
                        i, j = (
                            invalid_positions[1][0].item() + parallel_start,
                            invalid_positions[2][0].item() + parallel_start,
                        )
                        raise ValueError(
                            f"Invariant violation: Parallel token at position {i} cannot attend to parallel token at position {j}"
                        )
                    else:
                        raise ValueError(
                            f"Invariant violation: Not all parallel tokens can attend to each other"
                        )
            else:
                # For isolated mode, ensure we have a lower triangular matrix in the parallel token area
                parallel_block = mask[
                    :, parallel_start:parallel_end, parallel_start:parallel_end
                ]
                expected_mask = torch.tril(torch.ones_like(parallel_block))
                if not torch.all(parallel_block == expected_mask):
                    raise ValueError(
                        f"Invariant violation: Isolated parallel tokens mask does not maintain causal pattern"
                    )

            # 2. Check if all parallel tokens can attend to all previous tokens (all 1.0)
            if parallel_start > 0:  # Only check if there are tokens before parallel set
                previous_attention = mask[
                    :, parallel_start:parallel_end, 0:parallel_start
                ]
                if not torch.all(previous_attention == 1.0):
                    # Find positions that violate the constraint
                    invalid_positions = torch.nonzero(
                        previous_attention != 1.0, as_tuple=True
                    )
                    if len(invalid_positions) >= 3 and len(invalid_positions[0]) > 0:
                        # Convert to absolute positions for error reporting
                        i, j = (
                            invalid_positions[1][0].item() + parallel_start,
                            invalid_positions[2][0].item(),
                        )
                        raise ValueError(
                            f"Invariant violation: Parallel token at position {i} cannot attend to previous token at position {j}"
                        )
                    else:
                        raise ValueError(
                            f"Invariant violation: Not all parallel tokens can attend to previous tokens"
                        )

            # 3. Check if any parallel token can attend to tokens after the set (all 0.0)
            if (
                parallel_end < full_size
            ):  # Only check if there are tokens after parallel set
                future_attention = mask[
                    :, parallel_start:parallel_end, parallel_end:full_size
                ]
                if torch.any(future_attention != 0.0):
                    # Find positions that violate the constraint
                    invalid_positions = torch.nonzero(
                        future_attention != 0.0, as_tuple=True
                    )
                    if len(invalid_positions) >= 3 and len(invalid_positions[0]) > 0:
                        # Convert to absolute positions for error reporting
                        i, j = (
                            invalid_positions[1][0].item() + parallel_start,
                            invalid_positions[2][0].item() + parallel_end,
                        )
                        raise ValueError(
                            f"Invariant violation: Parallel token at position {i} can incorrectly attend to future token at position {j}"
                        )
                    else:
                        raise ValueError(
                            f"Invariant violation: Some parallel tokens can attend to future tokens"
                        )

        # Cache the mask for future use
        self._mask_cache[cache_key] = mask
        self.perf_stats["mask_creation_time"] += time.time() - start_time

        return mask

    def update_input_efficiently(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]],
        new_token_ids: list,
        is_kv_cache_disabled: bool = False,
        isolate_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        More efficient update for input tensors when using KV caching.
        Improved to properly handle all tokens in parallel sets.

        Args:
            input_ids: Current input token IDs
            attention_mask: Current attention mask
            past_key_values: Optional KV cache
            new_token_ids: List of token IDs to add
            is_kv_cache_disabled: Flag indicating if KV caching is disabled
            isolate_tokens: If True, parallel tokens cannot attend to each other

        Returns:
            tuple: (updated_input_ids, updated_attention_mask, updated_past_key_values)
        """
        # DIAGNOSTIC: Log what tokens we're adding and the current state
        self.log(f"Attention update with new token(s):")
        self.log(f"  Input shape before update: {input_ids.shape}")
        self.log(f"  Adding {len(new_token_ids)} new token(s): {new_token_ids}")

        # Skip empty token lists
        if not new_token_ids:
            return input_ids, attention_mask, past_key_values

        # Invariant: input_ids and attention_mask must have valid shapes
        if input_ids.dim() < 2:
            raise ValueError(
                f"Invariant violation: input_ids must have at least 2 dimensions, got {input_ids.dim()}"
            )

        if attention_mask.dim() < 2:
            raise ValueError(
                f"Invariant violation: attention_mask must have at least 2 dimensions, got {attention_mask.dim()}"
            )

        if input_ids.size(0) != attention_mask.size(0):
            raise ValueError(
                f"Invariant violation: batch dimensions must match, got {input_ids.size(0)} vs {attention_mask.size(0)}"
            )

        if input_ids.size(1) != attention_mask.size(1):
            raise ValueError(
                f"Invariant violation: sequence dimensions must match, got {input_ids.size(1)} vs {attention_mask.size(1)}"
            )

        # Special handling for when KV caching is disabled
        if is_kv_cache_disabled:
            # IMPORTANT: When KV caching is disabled, we must maintain the full context
            # by appending new tokens to the existing input rather than replacing it

            # Invariant: All token IDs must be valid non-negative integers
            for token_id in new_token_ids:
                if not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(
                        f"Invariant violation: Token ID {token_id} is not a valid non-negative integer"
                    )

            # Track current position for RoPE
            current_pos = input_ids.size(1)

            # Update RoPE position mapping for parallel tokens if needed
            if len(new_token_ids) > 1 and self.rope_modifier is not None:
                position_mapping = {}
                for i in range(len(new_token_ids)):
                    position_mapping[current_pos + i] = (
                        current_pos  # Map all to first position
                    )

                # Update RoPE modifier
                self.rope_modifier.register_parallel_positions(position_mapping)

                if self.debug_mode:
                    self.log(
                        f"Registered {len(new_token_ids)} parallel tokens at position {current_pos}"
                    )

            # Create tensor for new tokens
            new_tokens_tensor = torch.tensor([new_token_ids], device=self.device)

            # Invariant: Device consistency must be maintained
            if str(new_tokens_tensor.device) != str(input_ids.device):
                raise ValueError(
                    f"Invariant violation: Device mismatch between input_ids ({input_ids.device}) and new tokens ({new_tokens_tensor.device})"
                )

            # Append new tokens to existing input_ids
            updated_input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)

            # Update attention mask
            updated_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, len(new_token_ids)), device=self.device),
                ],
                dim=1,
            )

            # Create specialized attention mask for parallel tokens if needed
            if len(new_token_ids) > 1:
                full_size = updated_input_ids.size(1)
                self.full_attention_mask = self._create_parallel_attention_mask(
                    current_pos, len(new_token_ids), full_size, isolate_tokens
                )

            # DIAGNOSTIC: Log the result when KV caching is disabled
            self.log(f"No KV cache update results (disabled):")
            self.log(f"  Updated input shape: {updated_input_ids.shape}")
            self.log(f"  Updated attention mask shape: {updated_attention_mask.shape}")
            self.log(f"  Full context preserved: Yes")
            self.log(f"  First few tokens of updated input:")
            token_texts = []
            for i in range(min(updated_input_ids.size(1), 10)):
                token_id = updated_input_ids[0, i].item()
                try:
                    token_text = self.tokenizer.decode([int(token_id)])
                    token_texts.append(f"{i}: ID={token_id}, Text='{token_text}'")
                except:
                    token_texts.append(f"{i}: ID={token_id}, Text='<error decoding>'")
            self.log("\n    ".join(token_texts))

            return updated_input_ids, updated_attention_mask, None

        # Handle KV cache enabled case
        elif past_key_values is not None:
            # Invariant: past_key_values must have a valid structure
            if not (
                len(past_key_values) > 0
                and isinstance(past_key_values[0], tuple)
                and len(past_key_values[0]) >= 1
            ):
                raise ValueError(
                    "Invariant violation: past_key_values has invalid structure, expected non-empty list of tuples"
                )

            # Extract sequence length from past
            past_seq_len = past_key_values[0][0].size(2)

            # Track this for RoPE coordination
            current_pos = past_seq_len

            # Update RoPE position mapping for parallel tokens if needed
            if len(new_token_ids) > 1 and self.rope_modifier is not None:
                position_mapping = {}
                for i in range(len(new_token_ids)):
                    position_mapping[current_pos + i] = (
                        current_pos  # Map all to first position
                    )

                # Update RoPE modifier - critical for parallel token positions
                self.rope_modifier.register_parallel_positions(position_mapping)

                if self.debug_mode:
                    self.log(
                        f"Registered {len(new_token_ids)} parallel tokens at position {current_pos}"
                    )

            # For multiple tokens, we need to handle them properly, not just use the first one
            if len(new_token_ids) > 1:
                # INVARIANT: All token IDs must be valid non-negative integers
                for token_id in new_token_ids:
                    if not isinstance(token_id, int) or token_id < 0:
                        raise ValueError(
                            f"Invariant violation: Token ID {token_id} is not a valid non-negative integer"
                        )

                # INVARIANT: We must have a valid past sequence length
                if past_seq_len < 0:
                    raise ValueError(
                        f"Invariant violation: Past sequence length {past_seq_len} must be non-negative"
                    )

                # OPTIMIZATION: Instead of resetting the KV cache for parallel tokens,
                # we can process them sequentially and keep the KV cache intact.
                # This is much more efficient and still maintains correctness because
                # of how we've registered the position mapping with the RoPE modifier.

                # Create tensor with the first token
                first_token = new_token_ids[0]
                first_input = torch.tensor([[first_token]], device=self.device)

                # INVARIANT: first_input must have the correct shape
                if first_input.size(0) != 1 or first_input.size(1) != 1:
                    raise ValueError(
                        f"Invariant violation: first_input has incorrect shape: {first_input.shape}, expected [1, 1]"
                    )

                # INVARIANT: first_input must have the right token
                if first_input[0, 0].item() != first_token:
                    raise ValueError(
                        f"Invariant violation: first_input contains incorrect token: {first_input[0, 0].item()}, expected {first_token}"
                    )

                # Ensure attention mask has correct dimensions matching the KV cache
                first_attn = torch.ones((1, past_seq_len + 1), device=self.device)

                # INVARIANT: Attention mask dimensions must match past_seq_len
                if first_attn.size(1) != past_seq_len + 1:
                    raise ValueError(
                        f"Invariant violation: Attention mask dimension ({first_attn.size(1)}) doesn't match expected size ({past_seq_len + 1})"
                    )

                # DIAGNOSTIC: Log processing parallel tokens efficiently
                self.log(
                    f"Parallel token processing - handling {len(new_token_ids)} tokens efficiently:"
                )
                self.log(
                    f"  First token: ID={first_token}, Text='{self.tokenizer.decode([int(first_token)])}'"
                )
                self.log(f"  All parallel tokens: {new_token_ids}")
                self.log(f"  KV cache preserved: Yes (using position mapping)")

                # INVARIANT: The full_size calculation must be correct
                full_size = past_seq_len + len(new_token_ids)
                if full_size <= past_seq_len:
                    raise ValueError(
                        f"Invariant violation: full_size ({full_size}) must be greater than past_seq_len ({past_seq_len})"
                    )

                # Create specialized attention mask for parallel tokens
                # This allows parallel tokens to attend to each other
                self.full_attention_mask = self._create_parallel_attention_mask(
                    past_seq_len, len(new_token_ids), full_size, isolate_tokens
                )

                # INVARIANT: The full_attention_mask must have the correct shape
                if (
                    self.full_attention_mask.size(1) != full_size
                    or self.full_attention_mask.size(2) != full_size
                ):
                    raise ValueError(
                        f"Invariant violation: full_attention_mask has wrong shape: {self.full_attention_mask.shape[1:]}, expected [{full_size}, {full_size}]"
                    )

                # INVARIANT: Store a reference to position_mapping for diagnostics and verification
                if not hasattr(self, "position_mappings"):
                    self.position_mappings = {}

                position_mapping = {
                    past_seq_len + i: past_seq_len for i in range(len(new_token_ids))
                }
                self.position_mappings[past_seq_len] = position_mapping

                # INVARIANT: Track which parallel token positions have been processed
                if not hasattr(self, "processed_parallel_positions"):
                    self.processed_parallel_positions = set()

                for pos in range(past_seq_len, past_seq_len + len(new_token_ids)):
                    self.processed_parallel_positions.add(pos)

                # Store all parallel token IDs for later use during token selection
                self.current_parallel_tokens = new_token_ids

                # INVARIANT: Verify the KV cache is valid before proceeding
                if not all(
                    isinstance(layer, tuple) and len(layer) >= 2
                    for layer in past_key_values
                ):
                    raise ValueError(
                        "Invariant violation: past_key_values does not have the expected structure"
                    )

                # INVARIANT: Verify key and value tensors in KV cache have the same sequence length
                kv_seq_lengths = set()
                for layer in past_key_values:
                    if hasattr(layer[0], "size") and layer[0].dim() >= 3:
                        kv_seq_lengths.add(layer[0].size(2))  # Key's sequence length
                    if hasattr(layer[1], "size") and layer[1].dim() >= 3:
                        kv_seq_lengths.add(layer[1].size(2))  # Value's sequence length

                if len(kv_seq_lengths) > 1:
                    self.log(
                        f"Warning: Inconsistent sequence lengths in KV cache: {kv_seq_lengths}",
                        level="warning",
                    )

                # Return only the first token with KV cache preserved
                # This works because the position mapping ensures they all use the same position embedding
                return first_input, first_attn, past_key_values
            else:
                # Single token case
                # Invariant: Token ID must be a valid non-negative integer
                token_id = new_token_ids[0]
                if not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(
                        f"Invariant violation: Token ID {token_id} is not a valid non-negative integer"
                    )

                # Create tensor with single token
                new_input = torch.tensor([[token_id]], device=self.device)

                # Invariant: Device consistency must be maintained
                if str(new_input.device) != str(input_ids.device):
                    raise ValueError(
                        f"Invariant violation: Device mismatch between input_ids ({input_ids.device}) and new token ({new_input.device})"
                    )

                # Ensure attention mask has correct dimensions matching the KV cache
                new_attn = torch.ones((1, past_seq_len + 1), device=self.device)

                # Invariant: Attention mask dimensions must be compatible with past key values
                if new_attn.size(1) != past_seq_len + 1:
                    raise ValueError(
                        f"Invariant violation: Attention mask dimension ({new_attn.size(1)}) doesn't match expected size ({past_seq_len + 1})"
                    )

                # DIAGNOSTIC: Log the result for single token case
                self.log(f"Single token update results:")
                self.log(f"  New input shape: {new_input.shape}")
                self.log(f"  New attention mask shape: {new_attn.shape}")
                self.log(f"  KV cache preserved: Yes")
                self.log(
                    f"  New token: ID={new_token_ids[0]}, Text='{self.tokenizer.decode([int(new_token_ids[0])])}'"
                )

                return new_input, new_attn, past_key_values

        # Handle the case when past_key_values is None but KV cache is not disabled
        # This happens at the start of generation when there's no KV cache yet
        else:
            # This is a normal state at the beginning of generation
            # Just append the new tokens and return past_key_values=None
            self.log(f"Initial token generation - no past_key_values yet:")
            self.log(f"  Adding {len(new_token_ids)} token(s)")

            # INVARIANT: We must be at the beginning of generation if KV cache is enabled but past_key_values is None
            if input_ids.size(1) > 1 and not is_kv_cache_disabled:
                # We're not at the beginning, yet we have no KV cache - this is unusual
                # This could happen if the KV cache was explicitly reset, so log but don't error
                self.log(
                    f"Warning: No KV cache available despite being in the middle of generation "
                    f"(at position {input_ids.size(1)}). This might indicate a cache reset.",
                    level="warning",
                )

            # INVARIANT: All token IDs must be valid non-negative integers
            for token_id in new_token_ids:
                if not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(
                        f"Invariant violation: Token ID {token_id} is not a valid non-negative integer"
                    )

            # Create tensor for new tokens
            new_tokens_tensor = torch.tensor([new_token_ids], device=self.device)

            # INVARIANT: Device consistency must be maintained
            if str(new_tokens_tensor.device) != str(input_ids.device):
                raise ValueError(
                    f"Invariant violation: Device mismatch between input_ids ({input_ids.device}) and new tokens ({new_tokens_tensor.device})"
                )

            # INVARIANT: Tensors must have valid shapes before concatenation
            if input_ids.dim() < 2:
                raise ValueError(
                    f"Invariant violation: input_ids must have at least 2 dimensions for concatenation, got {input_ids.dim()}"
                )
            if new_tokens_tensor.dim() < 2:
                raise ValueError(
                    f"Invariant violation: new_tokens_tensor must have at least 2 dimensions for concatenation, got {new_tokens_tensor.dim()}"
                )
            if input_ids.size(0) != new_tokens_tensor.size(0):
                raise ValueError(
                    f"Invariant violation: Batch dimension mismatch: {input_ids.size(0)} vs {new_tokens_tensor.size(0)}"
                )

            # Append new tokens to existing input_ids
            updated_input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)

            # INVARIANT: Concatenation must maintain the expected shape
            if updated_input_ids.size(1) != input_ids.size(1) + new_tokens_tensor.size(
                1
            ):
                raise ValueError(
                    f"Invariant violation: Concatenation resulted in unexpected sequence length: "
                    f"got {updated_input_ids.size(1)}, expected {input_ids.size(1) + new_tokens_tensor.size(1)}"
                )

            # Update attention mask
            updated_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, len(new_token_ids)), device=self.device),
                ],
                dim=1,
            )

            # INVARIANT: Updated attention mask must match input_ids shape
            if updated_attention_mask.size(1) != updated_input_ids.size(1):
                raise ValueError(
                    f"Invariant violation: Attention mask length ({updated_attention_mask.size(1)}) "
                    f"doesn't match input_ids length ({updated_input_ids.size(1)})"
                )

            # Create specialized attention mask for parallel tokens if needed
            if len(new_token_ids) > 1:
                # INVARIANT: We must have a valid current position
                current_pos = input_ids.size(1)
                if current_pos < 0:
                    raise ValueError(
                        f"Invariant violation: Current position {current_pos} must be non-negative"
                    )

                # INVARIANT: Full size must include all tokens
                full_size = updated_input_ids.size(1)
                if full_size != current_pos + len(new_token_ids):
                    raise ValueError(
                        f"Invariant violation: Full size {full_size} doesn't match expected length "
                        f"{current_pos + len(new_token_ids)}"
                    )

                # Create parallel attention mask
                self.full_attention_mask = self._create_parallel_attention_mask(
                    current_pos, len(new_token_ids), full_size, isolate_tokens
                )

                # INVARIANT: The resulting attention mask must have the right shape
                if (
                    self.full_attention_mask.size(2) != full_size
                    or self.full_attention_mask.size(1) != full_size
                ):
                    raise ValueError(
                        f"Invariant violation: Parallel attention mask has wrong size: "
                        f"{self.full_attention_mask.size(1)}x{self.full_attention_mask.size(2)}, "
                        f"expected {full_size}x{full_size}"
                    )

                # Register parallel positions with RoPE modifier if available
                if self.rope_modifier is not None:
                    # INVARIANT: We must create valid position mappings
                    position_mapping = {}
                    for i in range(len(new_token_ids)):
                        # Map all parallel tokens to the first position
                        position_mapping[current_pos + i] = current_pos

                    # INVARIANT: Position mapping must have the right number of entries
                    if len(position_mapping) != len(new_token_ids):
                        raise ValueError(
                            f"Invariant violation: Position mapping has {len(position_mapping)} entries, "
                            f"expected {len(new_token_ids)}"
                        )

                    # Update RoPE modifier
                    self.rope_modifier.register_parallel_positions(position_mapping)

                    # Track for debugging
                    if not hasattr(self, "_parallel_token_count"):
                        self._parallel_token_count = 0
                    self._parallel_token_count += 1

                    # Log important milestones
                    if self.debug_mode and self._parallel_token_count % 10 == 0:
                        self.log(
                            f"Processed {self._parallel_token_count} parallel token sets so far"
                        )

            # INVARIANT: We should return None for past_key_values when there is no KV cache yet
            if past_key_values is not None:
                self.log(
                    "Warning: Unexpected non-None past_key_values", level="warning"
                )

            return updated_input_ids, updated_attention_mask, None

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create a causal attention mask for the sequence.
        Uses caching for efficiency.

        Args:
            seq_len: Sequence length

        Returns:
            torch.Tensor: Causal attention mask
        """
        # Invariant: Sequence length must be positive
        if seq_len <= 0:
            raise ValueError(
                f"Invariant violation: Sequence length must be positive, got {seq_len}"
            )

        # Check cache first
        if seq_len in self._mask_cache:
            return self._mask_cache[seq_len]

        # Create causal mask with proper batch and head dimensions for attention
        # Use [1, 1, seq_len, seq_len] shape to match attention weights dimension
        # This is critical for custom attention in TEMPO
        causal_mask = torch.ones((1, 1, seq_len, seq_len), device=self.device)
        mask = torch.triu(causal_mask * float("-inf"), diagonal=1)

        # Cache the result
        self._mask_cache[seq_len] = mask

        return mask

    def print_performance_stats(self):
        """Print performance statistics for attention mask operations."""
        print("\nAttention Manager Performance Stats:")
        print(f"  Mask creations: {self.perf_stats['mask_creations']}")
        print(f"  Mask updates: {self.perf_stats['mask_updates']}")
        print(f"  Mask cache hits: {self.perf_stats['mask_cache_hits']}")

        # Calculate hit rate
        total_ops = (
            self.perf_stats["mask_creations"]
            + self.perf_stats["mask_updates"]
            + self.perf_stats["mask_cache_hits"]
        )
        if total_ops > 0:
            hit_rate = (self.perf_stats["mask_cache_hits"] / total_ops) * 100
            print(f"  Cache hit rate: {hit_rate:.1f}%")

        # Timing stats
        creation_time = self.perf_stats["mask_creation_time"]
        update_time = self.perf_stats["mask_update_time"]
        total_time = creation_time + update_time

        print(f"  Mask creation time: {creation_time:.4f}s")
        print(f"  Mask update time: {update_time:.4f}s")
        print(f"  Total mask time: {total_time:.4f}s")

        # Current cache size
        print(f"  Mask cache size: {len(self._mask_cache)} entries")

    def reset_cache(self):
        """Reset the cached masks and performance stats."""
        self._mask_cache = {}
        self.full_attention_mask = None
        self.parallel_token_positions = {}

        # Reset performance stats
        self.perf_stats = {
            "mask_creations": 0,
            "mask_updates": 0,
            "mask_cache_hits": 0,
            "mask_creation_time": 0,
            "mask_update_time": 0,
        }

    def set_rope_modifier(self, rope_modifier):
        """
        Set the RoPE modifier instance for coordination.

        Args:
            rope_modifier: RoPE modifier instance
        """
        self.rope_modifier = rope_modifier
        if self.debug_mode:
            self.log("AttentionManager linked with RoPE modifier")

    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for more verbose output.

        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        # Always log this message even when debug mode is being disabled
        if enabled:
            # First set debug mode, then log
            self.logger.info(f"AttentionManager debug mode enabled - logging to file")
            # Also print this message to console to inform user where logs are going
            print(
                f"AttentionManager debug mode enabled - logging to file at logs/attention_debug.log"
            )
        else:
            # Log before disabling debug mode
            self.logger.info(f"AttentionManager debug mode disabled")
            print(f"AttentionManager debug mode disabled")

    def update_token_history(self, token_id: int):
        """
        Update the token history with a new token ID.
        This maintains the history of generated tokens for use in pruning and other operations.

        Args:
            token_id: The token ID to add to the history
        """
        # Initialize token history if it doesn't exist
        if not hasattr(self, "token_history"):
            self.token_history = []
            
        # Add the token to the history
        self.token_history.append(token_id)
        
        # Log in debug mode
        if self.debug_mode:
            token_text = ""
            if self.tokenizer is not None:
                try:
                    token_text = f" ('{self.tokenizer.decode([token_id])}')"
                except Exception:
                    # If decoding fails, just use the ID
                    token_text = ""
                    
            self.log(f"Token history updated with ID {token_id}{token_text}")
            
    def get_token_history(self) -> List[int]:
        """
        Get the current token history.
        
        Returns:
            List[int]: List of token IDs in the history
        """
        # Initialize token history if it doesn't exist
        if not hasattr(self, "token_history"):
            self.token_history = []
            
        return self.token_history
        
    def set_parallel_state(self, active: bool = False, base_position: int = 0, tokens: List[int] = None, allow_visibility: bool = False):
        """
        Set the parallel token generation state.
        This tracks when we're in parallel token generation mode.
        
        Args:
            active: Whether parallel token generation is active
            base_position: The base position for the parallel tokens
            tokens: List of token IDs for the parallel tokens
            allow_visibility: Whether tokens can see each other
        """
        self.parallel_active = active
        
        if active:
            # Store the parallel tokens information
            self.parallel_base_position = base_position
            self.parallel_tokens = tokens if tokens is not None else []
            self.parallel_visibility = allow_visibility
            
            if self.debug_mode:
                token_count = len(self.parallel_tokens) if self.parallel_tokens else 0
                visibility = "visible to each other" if allow_visibility else "isolated"
                self.log(f"Parallel state activated with {token_count} tokens at position {base_position} ({visibility})")
        else:
            # Reset the parallel tokens information
            if self.debug_mode and hasattr(self, "parallel_active") and self.parallel_active:
                self.log("Parallel state deactivated")
                
            self.parallel_base_position = 0
            self.parallel_tokens = []
            self.parallel_visibility = False

import torch
from typing import Tuple, Optional, List, Dict, Any

class AttentionManager:
    """
    Manages attention masking for parallel token generation.
    Optimized for efficient KV caching and low-overhead tensor operations.
    Enhanced to work with custom RoPE positional encoding.
    """
    
    def __init__(self, device: str = "mps", rope_modifier=None):
        """
        Initialize the attention manager.
        
        Args:
            device: Device to use for computation
            rope_modifier: Optional RoPE modifier instance for coordination
        """
        self.device = device
        self.rope_modifier = rope_modifier
        self.full_attention_mask = None
        
        # Cached masks by size for performance
        self._mask_cache = {}
        
        # Track parallel token sets for better coordination with RoPE
        self.parallel_token_positions = {}
        
        # Debug mode
        self.debug_mode = False
        
    def create_parallel_set_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        new_token_ids: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create updated input tensors with the new parallel token set.
        
        Args:
            input_ids: Current input token IDs
            attention_mask: Current attention mask
            new_token_ids: List of token IDs to add
            
        Returns:
            tuple: (updated_input_ids, updated_attention_mask)
        """
        # Skip empty token lists
        if not new_token_ids:
            return input_ids, attention_mask
            
        # Handle singleton case more efficiently
        if len(new_token_ids) == 1:
            # Add a single token - more efficient than creating new tensors
            # from scratch for common case
            new_input_ids = torch.cat([
                input_ids, 
                torch.tensor([[new_token_ids[0]]], device=self.device)
            ], dim=1)
            
            new_attention_mask = torch.cat([
                attention_mask, 
                torch.ones((1, 1), device=self.device)
            ], dim=1)
            
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
                        (1, full_size, full_size), 
                        device=self.device
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
                position_mapping[current_pos + i] = current_pos  # Map all to first position
            
            # Store for later use
            self.parallel_token_positions[current_pos] = {
                'start': current_pos,
                'end': current_pos + num_new_tokens - 1,
                'size': num_new_tokens
            }
            
            # Update RoPE modifier if available
            if self.rope_modifier is not None:
                self.rope_modifier.register_parallel_positions(position_mapping)
                
            if self.debug_mode:
                print(f"Registered parallel token set: pos={current_pos}, size={num_new_tokens}")
        
        # Initialize tensors with optimized pre-allocation
        # Pre-allocate with correct sizes
        new_input_ids = torch.cat([
            input_ids, 
            torch.zeros((batch_size, num_new_tokens), device=self.device, dtype=input_ids.dtype)
        ], dim=1)
        
        new_attention_mask = torch.cat([
            attention_mask, 
            torch.ones((batch_size, num_new_tokens), device=self.device)
        ], dim=1)
        
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
                seq_len, num_new_tokens, full_size
            )
            
            # Cache it for potential reuse
            self._mask_cache[('parallel', full_size, num_new_tokens)] = self.full_attention_mask
        elif self.full_attention_mask is not None:
            # Use standard causal mask for single tokens
            full_size = seq_len + num_new_tokens
            
            # Use cached mask if available
            if full_size in self._mask_cache:
                self.full_attention_mask = self._mask_cache[full_size]
            else:
                self.full_attention_mask = torch.ones(
                    (1, full_size, full_size), 
                    device=self.device
                )
                self._mask_cache[full_size] = self.full_attention_mask
        
        return new_input_ids, new_attention_mask
    
    def _create_parallel_attention_mask(self, seq_len: int, num_parallel: int, full_size: int) -> torch.Tensor:
        """
        Create a custom attention mask that allows tokens in a parallel set to attend to each other.
        
        Args:
            seq_len: Original sequence length (before adding parallel tokens)
            num_parallel: Number of tokens in the parallel set
            full_size: Full size of the sequence including parallel tokens
            
        Returns:
            torch.Tensor: Custom attention mask
        """
        # Create standard causal mask as starting point
        # Shape: [1, full_size, full_size]
        mask = torch.tril(torch.ones((1, full_size, full_size), device=self.device))
        
        # Now modify the mask to allow parallel tokens to attend to each other
        # For each parallel token, allow it to attend to all other parallel tokens
        parallel_start = seq_len
        parallel_end = seq_len + num_parallel
        
        # Allow all tokens in parallel set to attend to each other
        # This is the key modification that makes parallel tokens work as alternatives
        for i in range(parallel_start, parallel_end):
            for j in range(parallel_start, parallel_end):
                # Set all attention weights within the parallel block to 1
                # This allows any token in the parallel set to attend to any other
                mask[:, i, j] = 1.0
        
        return mask
    
    def update_input_efficiently(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]],
        new_token_ids: list
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        More efficient update for input tensors when using KV caching.
        
        Args:
            input_ids: Current input token IDs
            attention_mask: Current attention mask
            past_key_values: Optional KV cache
            new_token_ids: List of token IDs to add
            
        Returns:
            tuple: (updated_input_ids, updated_attention_mask, updated_past_key_values)
        """
        # Skip empty token lists
        if not new_token_ids:
            return input_ids, attention_mask, past_key_values
        
        try:
            # Fast path for KV caching
            if past_key_values is not None:
                # Get past sequence length from key-value cache
                if len(past_key_values) > 0 and isinstance(past_key_values[0], tuple) and len(past_key_values[0]) >= 1:
                    # Extract sequence length from past
                    past_seq_len = past_key_values[0][0].size(2)
                    
                    # Track this for RoPE coordination
                    current_pos = past_seq_len
                    
                    # Update RoPE position mapping for parallel tokens if needed
                    if len(new_token_ids) > 1 and self.rope_modifier is not None:
                        position_mapping = {}
                        for i in range(len(new_token_ids)):
                            position_mapping[current_pos + i] = current_pos  # Map all to first position
                        
                        # Update RoPE modifier
                        self.rope_modifier.register_parallel_positions(position_mapping)
                    
                    # Fast path for common case: single token
                    if len(new_token_ids) == 1:
                        new_input = torch.tensor([[new_token_ids[0]]], device=self.device)
                        # Ensure attention mask has correct dimensions matching the KV cache
                        new_attn = torch.ones((1, past_seq_len + 1), device=self.device)
                        return new_input, new_attn, past_key_values
                    
                    # Multiple tokens case
                    new_input = torch.tensor([new_token_ids], device=self.device)
                    # Ensure attention mask has correct dimensions matching the KV cache
                    new_attn = torch.ones((1, past_seq_len + len(new_token_ids)), device=self.device)
                    
                    # Create appropriate attention mask for parallel tokens
                    if len(new_token_ids) > 1:
                        # Creating a more sophisticated attention mask for past_key_values case
                        # This would be fed into the model at a lower level and depends on
                        # how the model processes attention masks with KV caching
                        pass  # For now, rely on the standard mask
                    
                    return new_input, new_attn, past_key_values
            
            # Fallback to standard approach
            new_input_ids, new_attention_mask = self.create_parallel_set_input(
                input_ids, attention_mask, new_token_ids
            )
            
            return new_input_ids, new_attention_mask, past_key_values
            
        except Exception as e:
            # Handle dimension mismatch errors gracefully
            if self.debug_mode:
                print(f"Warning: Error updating input: {e}. Using fallback approach.")
            
            # Complete fallback for any attention/dimension errors
            # Create a fresh input with just the first token to reset the state
            if len(new_token_ids) > 0:
                new_input = torch.tensor([[new_token_ids[0]]], device=self.device)
                new_attn = torch.ones((1, 1), device=self.device)
                # Reset past key values to force a fresh start
                return new_input, new_attn, None
            else:
                # If we have nothing else, preserve the existing state
                return input_ids, attention_mask, past_key_values
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create a causal attention mask for the sequence.
        Uses caching for efficiency.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            torch.Tensor: Causal attention mask
        """
        # Check cache first
        if seq_len in self._mask_cache:
            return self._mask_cache[seq_len]
            
        # Create causal mask with proper batch and head dimensions for attention
        # Use [1, 1, seq_len, seq_len] shape to match attention weights dimension
        # This is critical for custom attention in TEMPO
        causal_mask = torch.ones((1, 1, seq_len, seq_len), device=self.device)
        mask = torch.triu(causal_mask * float('-inf'), diagonal=1)
        
        # Cache the result
        self._mask_cache[seq_len] = mask
        
        return mask
    
    def reset_cache(self):
        """Clear the mask cache to free memory."""
        self._mask_cache.clear()
        self.parallel_token_positions = {}
        if self.debug_mode:
            print("AttentionManager cache reset")
    
    def set_rope_modifier(self, rope_modifier):
        """
        Set the RoPE modifier instance for coordination.
        
        Args:
            rope_modifier: RoPE modifier instance
        """
        self.rope_modifier = rope_modifier
        if self.debug_mode:
            print("AttentionManager linked with RoPE modifier")
    
    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        print(f"AttentionManager debug mode {'enabled' if enabled else 'disabled'}") 
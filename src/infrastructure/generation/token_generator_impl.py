"""Token generator implementation for the TEMPO system.

This module implements the TokenGeneratorInterface, coordinating
between the model, tokenizer, caches, and other infrastructure components.
"""

from typing import Optional, Any
import torch
import time
from src.utils.logging_utils import LoggingMixin
from src.domain.interfaces.token_generation import TokenGeneratorInterface
from src.domain.entities.logits import TokenLogits
from src.domain.entities.generation_state import GenerationState, AttentionPattern
from src.infrastructure.model import ModelAdapter
from src.infrastructure.cache import CacheManager
from src.infrastructure.performance import PerformanceTracker


class TokenGeneratorImpl(LoggingMixin, TokenGeneratorInterface):
    """Implementation of the token generator interface."""
    
    def __init__(self,
                 model_adapter: ModelAdapter,
                 cache_manager: Optional[CacheManager] = None,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 debug_mode: bool = False):
        """Initialize the token generator.
        
        Args:
            model_adapter: Adapter for the language model
            cache_manager: Optional cache manager
            performance_tracker: Optional performance tracker
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        
        self.model = model_adapter
        self.cache_manager = cache_manager
        self.performance_tracker = performance_tracker
        self.debug_mode = debug_mode
        
        # Setup logging
        self.setup_logging("token_generator_impl", "token_generator_impl_debug.log", debug_mode)
    
    def generate_logits(self, 
                        state: GenerationState, 
                        custom_attention_mask: Optional[torch.Tensor] = None) -> TokenLogits:
        """Generate logits for the next token given the current state.
        
        Args:
            state: Current generation state
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            TokenLogits containing the raw logits from the model
        """
        start_time = time.time()
        
        # Run model forward pass without KV cache
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                use_cache=False,
                output_attentions=True,
                custom_attention_mask=custom_attention_mask
            )
        
        # Extract logits for the last position
        logits = outputs.logits[:, -1, :]
        
        # Cache attention if available
        # Cache attention if available
        if hasattr(outputs, "attentions") and outputs.attentions and self.cache_manager:
            self.cache_manager.cache_attention(outputs.attentions, state.get_current_sequence_length())
        
        # Track performance
        if self.performance_tracker:
            duration = time.time() - start_time
            self.performance_tracker.track_model_call(duration, 1)
        
        # Create and return TokenLogits
        return TokenLogits(
            tensor=logits,
            sequence_position=state.get_current_sequence_length() - 1,
            batch_index=0
        )
    
    def generate_logits_with_cache(self,
                                   state: GenerationState,
                                   custom_attention_mask: Optional[torch.Tensor] = None) -> tuple[TokenLogits, GenerationState]:
        """Generate logits and update the generation state with new KV cache.
        
        Args:
            state: Current generation state
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            Tuple of (TokenLogits, updated GenerationState with new KV cache)
        """
        start_time = time.time()
        
        # Prepare inputs for KV cached generation
        input_ids = state.input_ids
        attention_mask = state.attention_mask
        
        # If we have KV cache, only process the last token
        if state.past_key_values is not None:
            # Store original shape for validation
            original_shape = input_ids.shape
            
            # Only use the last token
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Adjust attention mask for KV cache
            if hasattr(state.past_key_values, 'get_seq_length'):
                # DynamicCache format
                past_seq_len = state.past_key_values.get_seq_length() or 0
            elif isinstance(state.past_key_values, list) and len(state.past_key_values) > 0:
                # Legacy format
                past_seq_len = state.past_key_values[0][0].size(2) if state.past_key_values[0][0].dim() >= 3 else 0
            else:
                past_seq_len = 0
            
            # Create attention mask for the full sequence length
            attention_mask = torch.ones((1, past_seq_len + 1), device=state.input_ids.device)
        
        # Run model forward pass with KV cache
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_attentions=True,
                custom_attention_mask=custom_attention_mask
            )
        
        # Extract logits for the last position
        logits = outputs.logits[:, -1, :]
        
        # Cache attention if available
        if hasattr(outputs, "attentions") and outputs.attentions and self.cache_manager:
            current_seq_len = state.get_current_sequence_length()
            self.cache_manager.cache_attention(outputs.attentions, current_seq_len)
        
        # Track performance
        if self.performance_tracker:
            duration = time.time() - start_time
            self.performance_tracker.track_model_call(duration, 1)
        
        # Create TokenLogits
        token_logits = TokenLogits(
            tensor=logits,
            sequence_position=state.get_current_sequence_length() - 1,
            batch_index=0
        )
        
        # Create updated state with new KV cache
        updated_state = GenerationState(
            input_ids=state.input_ids,  # Keep original full sequence
            attention_mask=state.attention_mask,  # Keep original attention mask
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            sequence_length=state.sequence_length,
            generated_tokens=state.generated_tokens,
            timestamp=state.timestamp
        )
        
        return token_logits, updated_state
    
    def generate_logits_for_isolated_parallel(self,
                                              state: GenerationState,
                                              num_parallel_tokens: int,
                                              custom_attention_mask: Optional[torch.Tensor] = None) -> tuple[TokenLogits, GenerationState]:
        """Optimized logit generation for isolated parallel tokens.
        
        Since isolated tokens can't see each other, we only need to compute
        the forward pass once and can reuse the logits for all tokens.
        
        Args:
            state: Current generation state
            num_parallel_tokens: Number of parallel tokens to generate
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            Tuple of (TokenLogits that can be reused, updated GenerationState)
        """
        start_time = time.time()
        
        if self.debug_mode:
            self.log(f"Generating logits for {num_parallel_tokens} isolated parallel tokens")
        
        # Since all parallel tokens see the same context in isolated mode,
        # we can run the forward pass just once
        input_ids = state.input_ids
        attention_mask = state.attention_mask
        
        # If we have KV cache, only process the last token
        if state.past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Adjust attention mask for KV cache
            if hasattr(state.past_key_values, 'get_seq_length'):
                past_seq_len = state.past_key_values.get_seq_length() or 0
            elif isinstance(state.past_key_values, list) and len(state.past_key_values) > 0:
                past_seq_len = state.past_key_values[0][0].size(2) if state.past_key_values[0][0].dim() >= 3 else 0
            else:
                past_seq_len = 0
            
            attention_mask = torch.ones((1, past_seq_len + 1), device=state.input_ids.device)
        
        # Run model forward pass just once
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_attentions=True,
                custom_attention_mask=custom_attention_mask
            )
        
        # Extract logits for the last position
        logits = outputs.logits[:, -1, :]
        
        # Cache attention if available
        if hasattr(outputs, "attentions") and outputs.attentions and self.cache_manager:
            current_seq_len = state.get_current_sequence_length()
            self.cache_manager.cache_attention(outputs.attentions, current_seq_len)
            if self.debug_mode:
                self.log(f"Cached attention for isolated parallel generation (shared among {num_parallel_tokens} tokens)")
        
        # Track performance - count this as processing multiple tokens
        if self.performance_tracker:
            duration = time.time() - start_time
            self.performance_tracker.track_model_call(duration, num_parallel_tokens)
        
        if self.debug_mode:
            self.log(f"Isolated parallel generation took {duration*1000:.2f}ms for {num_parallel_tokens} tokens")
            self.log(f"Efficiency gain: {num_parallel_tokens}x (one forward pass instead of {num_parallel_tokens})")
        
        # Create TokenLogits that can be reused for all parallel tokens
        token_logits = TokenLogits(
            tensor=logits,
            sequence_position=state.get_current_sequence_length() - 1,
            batch_index=0
        )
        
        # Create updated state with new KV cache
        updated_state = GenerationState(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            sequence_length=state.sequence_length,
            generated_tokens=state.generated_tokens,
            timestamp=state.timestamp
        )
        
        return token_logits, updated_state
    
    def get_cached_attention(self) -> Optional[tuple[AttentionPattern, int]]:
        """Get the most recently cached attention patterns.
        
        Returns:
            Tuple of (AttentionPattern, sequence_length) or None if not cached
        """
        if self.cache_manager:
            return self.cache_manager.get_cached_attention()
        return None
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all_caches()
        
        # Also clear model KV cache
        self.model.clear_kv_cache()
        
        if self.debug_mode:
            self.log("Cleared all caches including model KV cache")

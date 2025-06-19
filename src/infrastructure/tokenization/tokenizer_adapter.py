"""Tokenizer adapter implementation for the TEMPO system.

This module provides an adapter for HuggingFace tokenizers,
implementing the TokenizerInterface from the domain layer.
"""

from typing import Optional
import torch
import time
from src.utils.logging_utils import LoggingMixin
from src.domain.interfaces.tokenizer import TokenizerInterface
from src.domain.entities.generation_state import TokenizationResult
from src.infrastructure.cache import CacheManager
from src.infrastructure.performance import PerformanceTracker


class TokenizerAdapter(LoggingMixin, TokenizerInterface):
    """Adapter for HuggingFace tokenizers."""
    
    def __init__(self, 
                 tokenizer,
                 device: str = "mps",
                 cache_manager: Optional[CacheManager] = None,
                 performance_tracker: Optional[PerformanceTracker] = None):
        """Initialize the tokenizer adapter.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            device: Device to use for tensors
            cache_manager: Optional cache manager for caching operations
            performance_tracker: Optional performance tracker
        """
        super().__init__()
        
        # Validate inputs
        assert tokenizer is not None, "Tokenizer cannot be None"
        assert device in ["cpu", "cuda", "mps"], f"Unsupported device: {device}"
        
        self.tokenizer = tokenizer
        self.device = device
        self.cache_manager = cache_manager
        self.performance_tracker = performance_tracker
        
        # Setup logging
        self.setup_logging("tokenizer_adapter", "tokenizer_adapter_debug.log")
        
        # Verify tokenizer capabilities
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have encode method"
        assert hasattr(self.tokenizer, "decode"), "Tokenizer must have decode method"
        assert hasattr(self.tokenizer, "batch_decode"), "Tokenizer must have batch_decode method"
        assert callable(self.tokenizer), "Tokenizer must be callable"
    
    def tokenize_prompt(self, prompt: str) -> TokenizationResult:
        """Tokenize a text prompt into input tensors.
        
        Args:
            prompt: Text prompt to tokenize
            
        Returns:
            TokenizationResult with input_ids and attention_mask
        """
        # Validate input
        assert prompt and isinstance(prompt, str), "Prompt must be a non-empty string"
        
        start_time = time.time()
        cache_hit = False
        
        # Check cache first if available
        if self.cache_manager:
            cached_result = self.cache_manager.get_tokenized_prompt(prompt)
            if cached_result is not None:
                cache_hit = True
                duration = time.time() - start_time
                
                # Track performance
                if self.performance_tracker:
                    self.performance_tracker.track_tokenization(duration, cache_hit)
                
                if self.debug_mode:
                    self.log(f"Tokenization cache hit for prompt: {prompt[:50]}...")
                
                return cached_result
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            add_special_tokens=True
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Validate outputs
        assert input_ids.dim() == 2 and input_ids.size(0) == 1, f"input_ids must have shape [1, seq_len], got {input_ids.shape}"
        assert attention_mask.dim() == 2 and attention_mask.size(0) == 1, f"attention_mask must have shape [1, seq_len], got {attention_mask.shape}"
        assert input_ids.size(1) == attention_mask.size(1), "input_ids and attention_mask must have same sequence length"
        
        # Create result
        result = TokenizationResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt=prompt,
            token_count=input_ids.size(1)
        )
        
        # Cache the result if cache manager is available
        if self.cache_manager:
            self.cache_manager.cache_tokenized_prompt(prompt, result)
        
        duration = time.time() - start_time
        
        # Track performance
        if self.performance_tracker:
            self.performance_tracker.track_tokenization(duration, cache_hit)
        
        if self.debug_mode:
            self.log(f"Tokenized prompt to {result.token_count} tokens (took {duration*1000:.2f}ms)")
        
        return result
    
    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        """Decode a list of token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            List of decoded token strings
        """
        if not token_ids:
            return []
        
        start_time = time.time()
        cache_hits = 0
        
        # Use cache if available
        if self.cache_manager:
            # Get cached results and identify uncached tokens
            cached_results, uncached_ids = self.cache_manager.get_decoded_tokens_batch(token_ids)
            
            # Count cache hits
            cache_hits = len(token_ids) - len(uncached_ids)
            
            # If all tokens are cached, return immediately
            if not uncached_ids:
                duration = time.time() - start_time
                
                # Track performance
                if self.performance_tracker:
                    self.performance_tracker.track_decode(duration, len(token_ids), cache_hits)
                
                # Filter out None values and return
                return [text for text in cached_results if text is not None]
            
            # Decode only uncached tokens
            if uncached_ids:
                # Convert to tensor for batch processing
                tokens_tensor = torch.tensor([uncached_ids], device="cpu")
                
                # Batch decode
                batch_decoded = self.tokenizer.batch_decode(
                    tokens_tensor,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                
                # Split the decoded string (assuming space separation)
                decoded_texts = batch_decoded[0].split(" ") if batch_decoded else []
                
                # Cache the newly decoded tokens
                if len(decoded_texts) == len(uncached_ids):
                    self.cache_manager.cache_decoded_tokens_batch(uncached_ids, decoded_texts)
                
                # Merge cached and newly decoded results
                result = []
                uncached_idx = 0
                for i, cached_text in enumerate(cached_results):
                    if cached_text is not None:
                        result.append(cached_text)
                    else:
                        if uncached_idx < len(decoded_texts):
                            result.append(decoded_texts[uncached_idx])
                            uncached_idx += 1
                        else:
                            # Fallback to individual decoding if batch decode failed
                            token_id = token_ids[i]
                            text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            result.append(text)
                            self.cache_manager.cache_decoded_token(token_id, text)
                
                duration = time.time() - start_time
                
                # Track performance
                if self.performance_tracker:
                    self.performance_tracker.track_decode(duration, len(token_ids), cache_hits)
                
                return result
        else:
            # No cache available, decode all tokens
            tokens_tensor = torch.tensor([token_ids], device="cpu")
            
            # Batch decode
            batch_decoded = self.tokenizer.batch_decode(
                tokens_tensor,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            
            # Split the decoded string
            result = batch_decoded[0].split(" ") if batch_decoded else []
            
            # Fallback to individual decoding if batch decode gives wrong number of results
            if len(result) != len(token_ids):
                result = []
                for token_id in token_ids:
                    text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    result.append(text)
            
            duration = time.time() - start_time
            
            # Track performance
            if self.performance_tracker:
                self.performance_tracker.track_decode(duration, len(token_ids), cache_hits)
            
            return result
    
    def decode_single_token(self, token_id: int) -> str:
        """Decode a single token ID to text.
        
        Args:
            token_id: Token ID to decode
            
        Returns:
            Decoded token string
        """
        # Use cache if available
        if self.cache_manager:
            cached_text = self.cache_manager.get_decoded_token(token_id)
            if cached_text is not None:
                return cached_text
        
        # Decode the token
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        
        # Cache the result
        if self.cache_manager:
            self.cache_manager.cache_decoded_token(token_id, text)
        
        return text
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

import torch
from typing import Dict, List, Tuple, Optional, Any
import time

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
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Cache for tokenized prompts
        self.prompt_cache = {}
        
        # Performance tracking
        self.perf_stats = {
            "tokenization_calls": 0,
            "tokenization_time": 0,
            "model_calls": 0,
            "model_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def prepare_input_from_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert prompt string to input tensors.
        Uses caching for repeated prompts.
        
        Args:
            prompt: The text prompt
            
        Returns:
            tuple: (input_ids, attention_mask)
        """
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
            add_special_tokens=True
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Store in cache
        self.prompt_cache[prompt] = (input_ids, attention_mask)
        
        self.perf_stats["tokenization_time"] += time.time() - start_time
        return input_ids, attention_mask
    
    def get_next_token_logits(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        custom_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits for next token using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            torch.Tensor: Next token logits
        """
        # Performance tracking
        start_time = time.time()
        self.perf_stats["model_calls"] += 1
        
        # Use inference mode for efficiency
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Get logits for next token (last position)
        next_token_logits = outputs.logits[:, -1, :]
        
        self.perf_stats["model_time"] += time.time() - start_time
        return next_token_logits
    
    def get_next_token_logits_cached(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        custom_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        """
        Get logits for next token using the model with KV caching.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Optional past key values for KV cache
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            tuple: (next_token_logits, past_key_values)
        """
        # For KV caching, we only need to process the most recent token(s)
        # if past_key_values are provided
        if past_key_values is not None and len(input_ids.shape) > 1:
            # Only use the last token when using KV cache
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Extract the sequence length from past_key_values to ensure attention mask has correct size
            if len(past_key_values) > 0 and isinstance(past_key_values[0], tuple) and len(past_key_values[0]) >= 1:
                past_seq_len = past_key_values[0][0].size(2)
                # Create a proper attention mask that matches the expected size
                attention_mask = torch.ones((1, past_seq_len + 1), device=self.device)
        
        # Use inference mode for efficiency
        with torch.inference_mode():
            for attempt in range(3):  # Try up to 3 times with different fallbacks
                try:
                    if attempt == 0:
                        # First attempt: standard approach with all arguments
                        model_args = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "past_key_values": past_key_values,
                            "use_cache": True
                        }
                        
                        # Add custom attention mask if provided
                        if custom_attention_mask is not None:
                            # For Mistral and similar models, pass the custom mask as position_attention_mask
                            # or direct as attention_mask if it's the right shape
                            if hasattr(self.model.config, "model_type") and "mistral" in self.model.config.model_type.lower():
                                # Try to use the model-specific way to handle custom masks
                                if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                                    model_args["position_attention_mask"] = custom_attention_mask
                                elif custom_attention_mask.dim() == 4:  # [batch, heads, seq, seq]
                                    # Some models expect a 4D mask with head dimension
                                    model_args["attention_mask"] = custom_attention_mask
                            else:
                                # For generic models, try the standard custom_attention_mask parameter
                                model_args["custom_attention_mask"] = custom_attention_mask
                                
                        outputs = self.model(**model_args)
                            
                    elif attempt == 1:
                        # Second attempt: reset KV cache
                        print(f"Trying with reset KV cache")
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,  # Reset KV cache
                            use_cache=True
                        )
                    else:
                        # Third attempt: use a simplified approach
                        print(f"Using simplified model call")
                        # Create a new single-token input
                        if len(input_ids.shape) > 1 and input_ids.shape[1] > 0:
                            last_token = input_ids[:, -1].unsqueeze(-1)
                        else:
                            last_token = input_ids
                        
                        # Reset everything and just process one token
                        simple_attn = torch.ones((1, 1), device=self.device)
                        outputs = self.model(
                            input_ids=last_token,
                            attention_mask=simple_attn,
                            past_key_values=None,
                            use_cache=True
                        )
                    
                    # If we reached here, it succeeded
                    break
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if "must match the size of tensor" in error_msg or "got multiple values for argument" in error_msg:
                        # Dimension mismatch error or argument error - try the next fallback
                        print(f"Error in attempt {attempt}: {error_msg}")
                        if attempt == 2:  # Last attempt failed
                            raise RuntimeError(f"All fallback attempts failed: {error_msg}")
                    else:
                        # Non-dimension error - re-raise immediately
                        raise
        
        # Get logits for next token (last position)
        next_token_logits = outputs.logits[:, -1, :]
        
        return next_token_logits, outputs.past_key_values
    
    def batch_decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Efficiently decode multiple tokens in a single call.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List[str]: Decoded tokens
        """
        # Performance tracking
        start_time = time.time()
        
        # Process tokens in batches for efficiency
        if not token_ids:
            return []
            
        # Convert to tensors for efficient batch processing
        tokens_tensor = torch.tensor([token_ids], device="cpu")
        
        # Get token strings using batch processing
        token_strings = self.tokenizer.batch_decode(
            tokens_tensor, 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        
        self.perf_stats["tokenization_time"] += time.time() - start_time
        # Return the processed strings
        return token_strings[0].split(' ')
    
    def print_performance_stats(self):
        """Print performance statistics."""
        print("\nToken Generator Performance Stats:")
        print(f"  Tokenization calls: {self.perf_stats['tokenization_calls']}")
        print(f"  Tokenization time: {self.perf_stats['tokenization_time']:.4f}s")
        print(f"  Model calls: {self.perf_stats['model_calls']}")
        print(f"  Model time: {self.perf_stats['model_time']:.4f}s")
        print(f"  Cache hits: {self.perf_stats['cache_hits']}")
        print(f"  Cache misses: {self.perf_stats['cache_misses']}")
        
        if self.perf_stats['model_calls'] > 0:
            avg_time = self.perf_stats['model_time'] / self.perf_stats['model_calls']
            print(f"  Average model call time: {avg_time:.4f}s")
    
    def enable_detailed_perf(self, enabled=True):
        """Enable or disable detailed per-call performance logging."""
        self.detailed_perf = enabled 
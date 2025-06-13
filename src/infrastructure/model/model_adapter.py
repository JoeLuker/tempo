"""Model adapter implementation for the TEMPO system.

This module provides an adapter layer for interacting with the underlying
language model, implementing the ModelInterface from the domain layer.
"""

from typing import Any, Optional
import torch
from src.utils.logging_utils import LoggingMixin
from src.domain.interfaces.model import ModelInterface


class ModelAdapter(LoggingMixin, ModelInterface):
    """Adapter for the underlying language model."""
    
    def __init__(self, model: Any, device: str = "mps"):
        """Initialize the model adapter.
        
        Args:
            model: The underlying language model
            device: Device to use for computation
        """
        super().__init__()
        
        # Validate inputs
        assert model is not None, "Model cannot be None"
        assert device in ["cpu", "cuda", "mps"], f"Unsupported device: {device}"
        
        self.model = model
        self.device = device
        
        # Setup logging
        self.setup_logging("model_adapter", "model_adapter_debug.log")
        
        # Verify model has required methods
        assert hasattr(self.model, "forward") or callable(self.model), "Model must be callable or have forward method"
        assert hasattr(self.model, "config"), "Model must have config attribute"
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values: Optional[Any] = None,
                use_cache: bool = True,
                output_attentions: bool = True,
                custom_attention_mask: Optional[torch.Tensor] = None) -> Any:
        """Run forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Optional KV cache
            use_cache: Whether to use/return KV cache
            output_attentions: Whether to output attention patterns
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            Model outputs
        """
        # Check if the model is Qwen-based
        model_type = getattr(self.model.config, "model_type", "").lower()
        is_qwen = "qwen" in model_type
        
        # Build model arguments
        model_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "return_dict": True  # Always use dict format for consistency
        }
        
        # Add past_key_values if provided
        if past_key_values is not None:
            model_args["past_key_values"] = past_key_values
        
        # Handle custom attention mask based on model type
        if custom_attention_mask is not None:
            if is_qwen:
                # Handle Qwen-specific custom attention
                if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                    model_args["position_bias"] = custom_attention_mask
                elif custom_attention_mask.dim() == 4:  # [batch, heads, seq, seq]
                    model_args["attention_mask"] = custom_attention_mask
            elif hasattr(self.model.config, "model_type") and "mistral" in self.model.config.model_type.lower():
                # Handle Mistral-specific custom attention
                if custom_attention_mask.dim() == 3:  # [batch, seq, seq]
                    model_args["position_attention_mask"] = custom_attention_mask
                elif custom_attention_mask.dim() == 4:  # [batch, heads, seq, seq]
                    model_args["attention_mask"] = custom_attention_mask
            else:
                # For generic models, try the standard custom_attention_mask parameter
                model_args["custom_attention_mask"] = custom_attention_mask
        
        # Run the model
        outputs = self.model(**model_args)
        
        # Log debug information if enabled
        if self.debug_mode:
            self.log(f"Model forward pass completed")
            if hasattr(outputs, "logits"):
                self.log(f"Output logits shape: {outputs.logits.shape}")
            if hasattr(outputs, "attentions") and outputs.attentions:
                self.log(f"Attention output: {len(outputs.attentions)} layers")
            if hasattr(outputs, "past_key_values") and outputs.past_key_values:
                self.log(f"KV cache returned")
        
        return outputs
    
    def clear_kv_cache(self) -> None:
        """Clear the model's KV cache."""
        # Try different methods to clear KV cache
        cleared = False
        
        # Method 1: clear_kv_cache method
        if hasattr(self.model, "clear_kv_cache"):
            try:
                self.model.clear_kv_cache()
                cleared = True
                if self.debug_mode:
                    self.log("Cleared KV cache using model.clear_kv_cache()")
            except Exception as e:
                if self.debug_mode:
                    self.log(f"Failed to clear cache with clear_kv_cache: {e}", level="warning")
        
        # Method 2: reset_cache method
        if not cleared and hasattr(self.model, "reset_cache"):
            try:
                self.model.reset_cache()
                cleared = True
                if self.debug_mode:
                    self.log("Cleared KV cache using model.reset_cache()")
            except Exception as e:
                if self.debug_mode:
                    self.log(f"Failed to clear cache with reset_cache: {e}", level="warning")
        
        # Method 3: _past_key_values attribute
        if not cleared and hasattr(self.model, "_past_key_values"):
            try:
                self.model._past_key_values = None
                cleared = True
                if self.debug_mode:
                    self.log("Cleared KV cache by setting _past_key_values to None")
            except Exception as e:
                if self.debug_mode:
                    self.log(f"Failed to clear _past_key_values: {e}", level="warning")
        
        if not cleared and self.debug_mode:
            self.log("No KV cache clearing mechanism available", level="warning")
    
    @property
    def config(self) -> Any:
        """Get model configuration."""
        return self.model.config
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model config."""
        if hasattr(self.model.config, "vocab_size"):
            return self.model.config.vocab_size
        else:
            raise AttributeError("Model config does not have vocab_size attribute")
    
    @property
    def model_type(self) -> str:
        """Get model type from config."""
        return getattr(self.model.config, "model_type", "unknown").lower()
    
    def to(self, device: str) -> 'ModelAdapter':
        """Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.model = self.model.to(device)
        self.device = device
        return self

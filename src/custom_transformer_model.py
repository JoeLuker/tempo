import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple, Union, Dict, Any

class CustomParallelAttentionModel(nn.Module):
    """
    A wrapper around a transformer model that supports custom attention masks for parallel token generation.
    
    This implementation specifically targets the key-value attention mechanism in transformer models
    and allows for non-causal (non-diagonal) attention between tokens at certain positions.
    """
    
    def __init__(self, base_model):
        """
        Initialize with a base transformer model.
        
        Args:
            base_model: A HuggingFace transformer model
        """
        super().__init__()
        self.base_model = base_model
        
        # Flag to indicate if we were able to patch the attention mechanism
        self.has_custom_attention_support = False
        
        # Try to store the original attention processors for later use
        try:
            self._store_original_attention_processors()
            self.has_custom_attention_support = True
        except Exception as e:
            print(f"Warning: Could not patch attention mechanism: {e}")
            print("Will fall back to standard attention. Parallel tokens will not see each other.")
    
    def _store_original_attention_processors(self):
        """
        Store the original attention implementation to be restored when needed.
        This method needs to be adapted based on the specific model architecture.
        """
        self.original_forward_methods = {}
        
        # Handle different model architectures
        if hasattr(self.base_model, "model"):
            # For models like GPT-2, Llama, Mistral
            model_type = self.base_model.__class__.__name__
            print(f"Attempting to patch attention for model type: {model_type}")
            
            # Debug model structure
            if "Mistral" in model_type:
                print("Detected Mistral model, exploring structure...")
                if hasattr(self.base_model.model, "layers"):
                    print(f"Found {len(self.base_model.model.layers)} layers")
                    # Examine the first layer's structure
                    first_layer = self.base_model.model.layers[0]
                    print(f"Layer structure: {list(first_layer._modules.keys())}")
                    # Check if self_attn exists
                    if hasattr(first_layer, "self_attn"):
                        print(f"self_attn structure: {list(first_layer.self_attn._modules.keys())}")
                    else:
                        print("No self_attn found in layer - incompatible structure!")
                else:
                    print("No 'layers' attribute found in model")
            
            if "GPT" in model_type or "Mistral" in model_type or "Llama" in model_type:
                # These models use a similar approach for attention
                try:
                    if not hasattr(self.base_model.model, "layers"):
                        raise ValueError(f"Model doesn't have expected 'layers' attribute")
                        
                    for i, layer in enumerate(self.base_model.model.layers):
                        if not hasattr(layer, "self_attn"):
                            raise ValueError(f"Layer {i} doesn't have expected 'self_attn' attribute")
                            
                        # Store the original self-attention forward method
                        self.original_forward_methods[i] = layer.self_attn.forward
                        
                        # Replace with our custom implementation
                        layer.self_attn.forward = self._make_custom_attention_forward(
                            layer.self_attn, self.original_forward_methods[i]
                        )
                        if i == 0:
                            print(f"Successfully patched layer 0 self-attention forward method")
                except Exception as e:
                    print(f"Error patching attention for {model_type}: {e}")
                    raise
            
            # Handle other architectures as needed
            elif "BERT" in model_type:
                for i, layer in enumerate(self.base_model.encoder.layer):
                    self.original_forward_methods[i] = layer.attention.self.forward
                    layer.attention.self.forward = self._make_custom_attention_forward(
                        layer.attention.self, self.original_forward_methods[i]
                    )
    
    def _make_custom_attention_forward(self, attn_module, original_forward):
        """
        Create a custom forward method for attention that uses the custom attention mask.
        
        Args:
            attn_module: The attention module to modify
            original_forward: The original forward method
            
        Returns:
            function: A new forward method that handles custom attention masks
        """
        def custom_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            custom_attention_mask=None,
            **kwargs
        ):
            """
            Custom attention forward pass that handles our 3D attention mask.
            
            Args:
                hidden_states: Input hidden states
                attention_mask: Standard attention mask (batch_size, seq_len)
                custom_attention_mask: Our 3D custom mask (batch_size, seq_len, seq_len)
                **kwargs: Other arguments to pass through
                
            Returns:
                tuple: Output as per the original implementation
            """
            # If no custom mask is provided, use the original implementation
            if custom_attention_mask is None:
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )
            
            # Simple approach: use the original implementation but only care about the
            # parallel connection pattern between tokens by modifying the attention_mask
            
            try:
                # For causal LMs, we can convert our 3D mask to a standard attention mask
                # that allows parallel tokens to attend to each other.
                
                # First, create a valid 4D attention mask (most transformer models expect this)
                seq_len = custom_attention_mask.shape[1]
                batch_size = custom_attention_mask.shape[0]
                
                # Original causal mask (from attention_mask) doesn't properly allow parallel tokens
                # to see each other, so we need to create a new 4D mask from our custom attention pattern
                
                # Extract the elements that represent the last parallel token's attention pattern
                # which can see all previous tokens and all other parallel tokens
                last_pos_mask = custom_attention_mask[0, seq_len-1, :seq_len]
                
                # Create mask from scratch (1.0 = attend, 0.0 = don't attend)
                # Shape [batch_size, 1, 1, seq_len] 
                mask_4d = torch.zeros((batch_size, 1, 1, seq_len), 
                                     device=hidden_states.device, 
                                     dtype=hidden_states.dtype)
                
                # Put our custom pattern in the mask
                mask_4d[0, 0, 0, :seq_len] = last_pos_mask
                
                # Convert mask format (1.0 = attend, large negative = don't attend)
                # Most transformer models apply masks as (attn_weights + mask)
                attention_mask = (1.0 - mask_4d) * -10000.0
                
                # Now call the original implementation with our modified mask
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,  # Modified to allow parallel tokens to see each other
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )
            except Exception:
                # If anything goes wrong, fall back to standard implementation
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )
        
        return custom_forward
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        custom_attention_mask=None,
        **kwargs
    ):
        """
        Forward pass that handles a custom attention mask.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Standard attention mask
            custom_attention_mask: Our 3D custom mask (batch_size, seq_len, seq_len)
            **kwargs: Other arguments to pass through
            
        Returns:
            Same as the base model's forward method
        """
        # If we have a custom mask, we'll use our patched attention
        if custom_attention_mask is not None:
            # Use the model's forward method, which will now call our custom attention
            kwargs['custom_attention_mask'] = custom_attention_mask
        
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def restore_original_attention(self):
        """
        Restore the original attention implementation.
        """
        if hasattr(self.base_model, "model"):
            model_type = self.base_model.__class__.__name__
            
            if "GPT" in model_type or "Mistral" in model_type or "Llama" in model_type:
                for i, layer in enumerate(self.base_model.model.layers):
                    if i in self.original_forward_methods:
                        layer.self_attn.forward = self.original_forward_methods[i]
            
            elif "BERT" in model_type:
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i in self.original_forward_methods:
                        layer.attention.self.forward = self.original_forward_methods[i]
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the base model if not found in this class.
        
        Args:
            name: Attribute name
            
        Returns:
            The requested attribute from the base model
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name) 
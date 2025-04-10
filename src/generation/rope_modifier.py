import torch
import math
from typing import List, Tuple, Optional, Dict, Any, Set

class RoPEModifier:
    """
    Custom implementation to modify RoPE (Rotary Position Embeddings) for parallel token generation.
    This implementation allows multiple tokens at the same position to share identical positional embeddings.
    Addresses potential issues with KV cache consistency and patching all appropriate layers.
    """
    
    def __init__(self, model, device: str = "mps"):
        """
        Initialize the RoPE modifier.
        
        Args:
            model: The transformer model
            device: The device for computation
        """
        self.model = model
        self.device = device
        self.original_forward_fns = {}  # Store original methods by module path
        self.position_map = {}  # Maps token positions to their effective positions
        self.sequence_position_map = {}  # Tracks absolute sequence positions for KV cache consistency
        self.patched_modules = set()  # Keep track of which modules have been patched
        self.is_installed = False
        
        # Cache for position embeddings to avoid recomputation
        self.position_embedding_cache = {}
        
        # Configuration
        self.enable_kv_cache_consistency = True
        self.debug_mode = False
        
    def install(self):
        """
        Install the custom RoPE implementation by patching all model's attention layers.
        """
        if self.is_installed:
            print("RoPE modifier already installed")
            return
            
        patched_count = 0
        mistral_specific_patches = 0
        
        # First find modules that are directly responsible for RoPE
        # For Mistral, we want to patch the MistralRotaryEmbedding class
        for name, module in self.model.named_modules():
            # Direct rotary embedding classes
            if any(rotary_name in name.lower() for rotary_name in ['rotaryembedding', 'mistralrotary', 'rotary_emb']):
                if hasattr(module, 'forward') and name not in self.patched_modules:
                    print(f"Patching rotary embedding module directly: {name}")
                    self._patch_module(name, module)
                    patched_count += 1
                    self.patched_modules.add(name)
                    mistral_specific_patches += 1
                
        # Then look for attention modules that might apply RoPE internally
        if mistral_specific_patches == 0:
            # If we didn't find any Mistral-specific modules, fall back to general approach
            for name, module in self.model.named_modules():
                # Look for attention modules that have rotary position embeddings
                if hasattr(module, 'rotary_emb') and hasattr(module, 'forward') and name not in self.patched_modules:
                    print(f"Patching attention module with rotary_emb attribute: {name}")
                    self._patch_module(name, module)
                    patched_count += 1
                    self.patched_modules.add(name)
                    
            # Also look for modules that might apply rotary embeddings differently
            for name, module in self.model.named_modules():
                # Various ways RoPE might be implemented
                if (hasattr(module, '_apply_rotary_pos_emb') or 
                    hasattr(module, 'apply_rotary_pos_emb') or
                    hasattr(module, '_apply_rope') or 
                    'rotary' in name.lower() or 
                    'rope' in name.lower()) and name not in self.patched_modules:
                    
                    if hasattr(module, 'forward'):
                        print(f"Patching module with rotary-related attribute or name: {name}")
                        self._patch_module(name, module)
                        patched_count += 1
                        self.patched_modules.add(name)
        
        print(f"RoPE modifier installed: patched {patched_count} modules")
        self.is_installed = True
    
    def _patch_module(self, name: str, module: Any):
        """
        Patch a specific module with our custom forward implementation.
        
        Args:
            name: Module name/path
            module: The module to patch
        """
        print(f"Patching rotary embedding in module: {name}")
        self.original_forward_fns[name] = module.forward
        
        # Create a closure that captures the current instance and module
        module.forward = self._make_custom_forward(module.forward, module)
    
    def _make_custom_forward(self, original_forward, module_instance):
        """
        Create a custom forward function that applies position mapping.
        
        Args:
            original_forward: The original forward function
            module_instance: The module instance
            
        Returns:
            function: Custom forward function
        """
        rope_modifier = self  # Capture the current instance
        
        def custom_forward(*args, **kwargs):
            # Debug information if enabled
            if rope_modifier.debug_mode:
                print(f"Custom forward called with {len(args)} args and {len(kwargs)} kwargs")
                if len(args) > 0:
                    print(f"First arg shape: {args[0].shape if hasattr(args[0], 'shape') else 'N/A'}")
                for k, v in list(kwargs.items())[:3]:  # Show first 3 kwargs for brevity
                    if hasattr(v, 'shape'):
                        print(f"Kwarg {k} shape: {v.shape}")
            
            # Deep copy args and kwargs to avoid modifying the originals
            new_args = list(args)
            new_kwargs = kwargs.copy()
            
            # Handle Mistral's rotary embedding which expects position_ids as the second positional arg
            position_ids_in_args = False
            position_ids_index = -1
            
            # Check if this is the rotary embedding forward call (has exactly 2 args with position_ids as the second)
            if len(args) == 2 and hasattr(args[1], 'shape'):
                # This is likely hidden_states, position_ids
                position_ids_in_args = True
                position_ids_index = 1
                
                # Apply position mapping to the positional argument
                original_position_ids = args[position_ids_index]
                mapped_position_ids = rope_modifier.apply_position_mapping(original_position_ids)
                new_args[position_ids_index] = mapped_position_ids
                
                # Remove position_ids from kwargs if present to avoid duplication
                if 'position_ids' in new_kwargs:
                    del new_kwargs['position_ids']
            
            # If position_ids is in kwargs and not already handled above
            elif 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                position_ids = kwargs['position_ids']
                # Apply position mapping to ensure parallel tokens use the same position
                mapped_position_ids = rope_modifier.apply_position_mapping(position_ids)
                new_kwargs['position_ids'] = mapped_position_ids
                
                # Handle KV cache consistency if enabled
                if rope_modifier.enable_kv_cache_consistency and 'past_key_value' in kwargs and kwargs['past_key_value'] is not None:
                    new_kwargs = rope_modifier._ensure_kv_cache_consistency(new_kwargs)
            
            # If no explicit position_ids are provided but we have seq_len in hidden_states
            elif len(args) >= 1 and args[0] is not None and hasattr(args[0], 'size') and len(args[0].size()) >= 2 and 'position_ids' not in kwargs:
                seq_len = args[0].size(1)
                batch_size = args[0].size(0)
                device = args[0].device
                
                # Generate position IDs and apply our mapping
                position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
                mapped_position_ids = rope_modifier.apply_position_mapping(position_ids)
                
                # Add position_ids to kwargs if not present and not expected as positional arg
                if not position_ids_in_args:
                    new_kwargs['position_ids'] = mapped_position_ids
                
                # Handle KV cache consistency if enabled
                if rope_modifier.enable_kv_cache_consistency and 'past_key_value' in kwargs and kwargs['past_key_value'] is not None:
                    new_kwargs = rope_modifier._ensure_kv_cache_consistency(new_kwargs)
            
            # Call original method with potentially modified position_ids
            try:
                return original_forward(*new_args, **new_kwargs)
            except Exception as e:
                # Provide more helpful error messages in debug mode
                if rope_modifier.debug_mode:
                    print(f"Error in custom_forward: {e}")
                    print(f"Args shapes: {[a.shape if hasattr(a, 'shape') else 'N/A' for a in new_args[:3]]}")
                    print(f"Kwargs keys: {list(new_kwargs.keys())}")
                    
                    # Try a more conservative approach if we have an argument error
                    if "got multiple values for argument" in str(e) and position_ids_in_args:
                        print("Trying fallback approach with original args")
                        return original_forward(*args, **kwargs)
                        
                # Re-raise to not silently fail
                raise
        
        return custom_forward
    
    def register_parallel_positions(self, input_position_mapping: dict):
        """
        Register position mapping for parallel tokens.
        
        Args:
            input_position_mapping: Dictionary mapping token positions to their effective positions
        """
        if not self.is_installed:
            print("Warning: RoPE modifier not installed yet. Call install() first.")
            
        # Reset the sequence_position_map when updating position_map to avoid stale mappings
        self.sequence_position_map = {}
        self.position_map = input_position_mapping.copy()  # Create a copy to avoid external modification
        
        # Clear position embedding cache when mapping changes
        self.position_embedding_cache = {}
        
        if self.debug_mode:
            print(f"Registered {len(input_position_mapping)} parallel position mappings")
    
    def apply_position_mapping(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply custom position mapping to the position_ids tensor with improved robustness.
        
        Args:
            position_ids: Original position IDs tensor
        
        Returns:
            torch.Tensor: Modified position IDs tensor
        """
        # Fast path for empty mapping
        if not self.position_map:
            return position_ids
            
        # Create a copy to avoid modifying the original
        mapped_position_ids = position_ids.clone()
        
        # Cache key for this tensor to avoid redundant computation
        cache_key = (position_ids.shape, position_ids.sum().item(), len(self.position_map))
        if cache_key in self.position_embedding_cache:
            return self.position_embedding_cache[cache_key]
        
        # For each position in the sequence, apply mapping more carefully
        for batch_idx in range(position_ids.size(0)):
            for seq_idx in range(position_ids.size(1)):
                absolute_idx = position_ids[batch_idx, seq_idx].item()
                
                # Apply our parallel position mapping if available
                if seq_idx in self.position_map:
                    mapped_position = self.position_map[seq_idx]
                    mapped_position_ids[batch_idx, seq_idx] = mapped_position
                    # Track this mapping for KV cache consistency
                    self.sequence_position_map[absolute_idx] = mapped_position
                
                # Also check if this absolute position has a mapping
                elif absolute_idx in self.position_map:
                    mapped_position = self.position_map[absolute_idx]
                    mapped_position_ids[batch_idx, seq_idx] = mapped_position
                    # Track this mapping for KV cache consistency
                    self.sequence_position_map[absolute_idx] = mapped_position
        
        # Cache the result to avoid redundant computation
        self.position_embedding_cache[cache_key] = mapped_position_ids
        return mapped_position_ids
    
    def _ensure_kv_cache_consistency(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure KV cache is consistent with the current position mapping.
        This is critical for ensuring that queries with modified positions
        interact correctly with keys in the cache that used different positions.
        
        Args:
            kwargs: Forward function kwargs
            
        Returns:
            Dict[str, Any]: Updated kwargs
        """
        # If position mapping has changed significantly, we might need to reset the KV cache
        past_key_value = kwargs.get('past_key_value')
        position_ids = kwargs.get('position_ids')
        
        if past_key_value is None or not self.enable_kv_cache_consistency:
            return kwargs
            
        # Track current position mapping for this pass
        if not hasattr(self, 'kv_cache_position_maps'):
            self.kv_cache_position_maps = []
        
        # Check if we have a significant change in position mappings
        # This happens when tokens that should be at different positions are mapped to the same position
        # or when tokens that were previously mapped together are now mapped separately
        mapping_changed = self._has_significant_position_changes()
        
        if mapping_changed and self.debug_mode:
            print("Significant position mapping change detected!")
            
        # Track the position mapping used for this KV cache entry
        if position_ids is not None:
            self.kv_cache_position_maps.append(self.position_map.copy())
            
        # Option 1: Reset the KV cache if position mappings have significantly changed
        # This forces recalculation with consistent position mappings
        if mapping_changed:
            if self.debug_mode:
                print("Resetting KV cache due to significant position mapping changes")
            kwargs['past_key_value'] = None
            self.kv_cache_position_maps = []  # Reset tracking
        
        # Option 2: For more sophisticated approaches, we could:
        # 1. Apply compensating offsets to attention scores based on position delta
        # 2. Selectively invalidate only the affected parts of the cache
        # 3. Transform the cached keys to match the new effective positions
        # These would require more complex modifications to the attention mechanism
            
        return kwargs
        
    def _has_significant_position_changes(self) -> bool:
        """
        Determine if there has been a significant change in position mappings
        that would affect KV cache consistency.
        
        Returns:
            bool: True if significant changes detected, False otherwise
        """
        # No previous mappings tracked, no change
        if not hasattr(self, 'previous_position_map'):
            self.previous_position_map = self.position_map.copy()
            return False
            
        # If we had mappings before but now don't, or vice versa
        if bool(self.previous_position_map) != bool(self.position_map):
            self.previous_position_map = self.position_map.copy()
            return True
            
        # If the set of mapped positions has changed
        if set(self.previous_position_map.keys()) != set(self.position_map.keys()):
            self.previous_position_map = self.position_map.copy()
            return True
            
        # If the mapping values have changed for any position
        for pos, mapped_pos in self.position_map.items():
            if pos in self.previous_position_map and self.previous_position_map[pos] != mapped_pos:
                self.previous_position_map = self.position_map.copy()
                return True
                
        # No significant changes
        return False
    
    def reset(self):
        """Reset the position mapping and related caches."""
        self.position_map = {}
        self.sequence_position_map = {}
        self.position_embedding_cache = {}
        
        if self.debug_mode:
            print("RoPE modifier state reset")

    def uninstall(self):
        """Restore the original forward methods for all patched modules."""
        if not self.is_installed:
            print("RoPE modifier not installed, nothing to uninstall")
            return
            
        for name, original_forward in self.original_forward_fns.items():
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module.forward = original_forward
                    if self.debug_mode:
                        print(f"Restored original forward method for module: {name}")
        
        self.original_forward_fns = {}
        self.patched_modules = set()
        self.is_installed = False
        print("RoPE modifier uninstalled")
        
    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        print(f"RoPE modifier debug mode {'enabled' if enabled else 'disabled'}")
        
    def enable_kv_cache_consistency(self, enabled: bool = True):
        """
        Enable or disable KV cache consistency checks.
        
        Args:
            enabled: Whether to enable KV cache consistency
        """
        self.enable_kv_cache_consistency = enabled
        print(f"RoPE modifier KV cache consistency {'enabled' if enabled else 'disabled'}") 
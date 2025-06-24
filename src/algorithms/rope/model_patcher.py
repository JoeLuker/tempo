"""Model patching utilities for RoPE modification."""

import torch
import torch.nn as nn
from typing import Dict, Set, Callable, Any, Optional
from functools import wraps


class ModelPatcher:
    """Patches model methods to use modified RoPE embeddings."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.original_methods: Dict[str, Callable] = {}
        self.patched_modules: Set[str] = set()
        
    def patch_module(
        self, 
        module_name: str,
        module: nn.Module,
        forward_modifier: Callable
    ) -> bool:
        """
        Patch a module's forward method.
        
        Args:
            module_name: Name/path of the module
            module: The module to patch
            forward_modifier: Function to modify forward behavior
            
        Returns:
            True if patched successfully
        """
        if module_name in self.patched_modules:
            return False
            
        if not hasattr(module, 'forward'):
            return False
            
        # Store original method
        self.original_methods[module_name] = module.forward
        
        # Create patched forward
        @wraps(module.forward)
        def patched_forward(*args, **kwargs):
            return forward_modifier(module, *args, **kwargs)
        
        # Replace method
        module.forward = patched_forward
        self.patched_modules.add(module_name)
        
        return True
    
    def unpatch_module(self, module_name: str, module: nn.Module) -> bool:
        """Restore original forward method."""
        if module_name not in self.patched_modules:
            return False
            
        if module_name in self.original_methods:
            module.forward = self.original_methods[module_name]
            self.patched_modules.remove(module_name)
            del self.original_methods[module_name]
            return True
            
        return False
    
    def unpatch_all(self, model: nn.Module):
        """Unpatch all modules in the model."""
        for name, module in model.named_modules():
            if name in self.patched_modules:
                self.unpatch_module(name, module)
    
    def find_rope_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all RoPE-related modules in the model."""
        rope_modules = {}
        
        # Common RoPE module names across different architectures
        rope_keywords = [
            'rotary', 'rope', 'position', 'embedding',
            'mistral_rotary', 'llama_rotary', 'qwen_rotary'
        ]
        
        for name, module in model.named_modules():
            module_type = type(module).__name__.lower()
            module_name = name.lower()
            
            # Check if this might be a RoPE module
            if any(keyword in module_type or keyword in module_name 
                   for keyword in rope_keywords):
                rope_modules[name] = module
                
        return rope_modules
    
    def create_attention_patcher(
        self, 
        position_mapper,
        embedding_cache
    ) -> Callable:
        """
        Create a forward modifier for attention modules.
        
        Args:
            position_mapper: PositionMapper instance
            embedding_cache: RoPECache instance
            
        Returns:
            Forward modifier function
        """
        def attention_forward_modifier(module, *args, **kwargs):
            # Extract position_ids if present
            position_ids = kwargs.get('position_ids', None)
            
            if position_ids is not None and position_mapper.position_map:
                # Modify positions for parallel tokens
                logical_positions = position_mapper.get_logical_positions(position_ids)
                kwargs['position_ids'] = logical_positions
            
            # Call original forward
            if hasattr(module, '__wrapped_forward__'):
                return module.__wrapped_forward__(*args, **kwargs)
            else:
                # Fallback to original method from storage
                module_name = next(
                    (name for name, mod in self.patched_modules.items() if mod is module),
                    None
                )
                if module_name and module_name in self.original_methods:
                    return self.original_methods[module_name](*args, **kwargs)
                    
            raise RuntimeError("Could not find original forward method")
        
        return attention_forward_modifier
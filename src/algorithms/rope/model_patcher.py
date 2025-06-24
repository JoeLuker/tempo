"""Model patching utilities for RoPE modification."""

import torch
import torch.nn as nn
from typing import Dict, Set, Callable, Any, Optional
from functools import wraps
from contextlib import contextmanager


class ModelPatcher:
    """Patches model methods to use modified RoPE embeddings.
    
    This class provides a cleaner interface for temporarily modifying
    model behavior during TEMPO generation.
    """
    
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
        """
        Restore a module's original forward method.
        
        Args:
            module_name: Name/path of the module
            module: The module to unpatch
            
        Returns:
            True if unpatched successfully
        """
        if module_name not in self.patched_modules:
            return False
            
        if module_name not in self.original_methods:
            return False
            
        # Restore original method
        module.forward = self.original_methods[module_name]
        
        # Clean up tracking
        del self.original_methods[module_name]
        self.patched_modules.remove(module_name)
        
        return True
    
    def unpatch_all(self, model: nn.Module) -> int:
        """
        Restore all patched methods in the model.
        
        Args:
            model: The model to restore
            
        Returns:
            Number of modules unpatched
        """
        unpatched_count = 0
        modules_to_unpatch = list(self.patched_modules)  # Copy to avoid modification during iteration
        
        for module_name in modules_to_unpatch:
            # Navigate to module
            parts = module_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            
            if module is not None and self.unpatch_module(module_name, module):
                unpatched_count += 1
                
        return unpatched_count
    
    @contextmanager
    def temporary_patch(self, model: nn.Module, patches: Dict[str, Callable]):
        """
        Context manager for temporary model patching.
        
        Args:
            model: The model to patch
            patches: Dictionary mapping module paths to forward modifiers
            
        Example:
            with patcher.temporary_patch(model, {
                'model.layers.0.self_attn': modified_attention_forward,
                'model.layers.1.self_attn': modified_attention_forward,
            }):
                # Model is patched here
                output = model.generate(...)
            # Model is automatically restored here
        """
        # Apply patches
        for module_path, forward_modifier in patches.items():
            parts = module_path.split('.')
            module = model
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            
            if module is not None:
                self.patch_module(module_path, module, forward_modifier)
        
        try:
            yield
        finally:
            # Always restore, even if an exception occurs
            self.unpatch_all(model)
    
    def get_patched_modules(self) -> Set[str]:
        """Get set of currently patched module names."""
        return self.patched_modules.copy()
    
    def is_patched(self, module_name: str) -> bool:
        """Check if a module is currently patched."""
        return module_name in self.patched_modules
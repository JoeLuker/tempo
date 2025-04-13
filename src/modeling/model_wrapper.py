import torch
from typing import Dict, List, Tuple, Optional, Any
from torch import nn


class TEMPOModelWrapper(nn.Module):
    """
    TEMPO Model Wrapper for language models.
    Captures intermediate values during generation and provides hooks for debugging.
    """

    def __init__(self, model):
        """
        Initialize the model wrapper.

        Args:
            model: The language model to wrap
        """
        super().__init__()
        self.model = model
        self.intermediate_values = {}
        self.activation_hooks = []
        self._register_hooks()

        # Flag for debug mode
        self.debug_mode = False

        # Original model attributes need to be accessible through the wrapper
        self.config = model.config
        if hasattr(model, "generation_config"):
            self.generation_config = model.generation_config

        # Store device information
        self.device = next(model.parameters()).device

    def _register_hooks(self):
        """
        Register hooks to capture intermediate values from key model components.
        """
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            # Hook into attention modules
            if any(attn_name in name.lower() for attn_name in ["attention", "attn"]):
                hook = module.register_forward_hook(self._create_hook(f"{name}_output"))
                self.activation_hooks.append(hook)

            # Hook into rotary embedding modules
            if any(rope_name in name.lower() for rope_name in ["rotary", "rope"]):
                hook = module.register_forward_hook(self._create_hook(f"{name}_output"))
                self.activation_hooks.append(hook)

            # Hook into output processing
            if "lm_head" in name.lower() or "output" in name.lower():
                hook = module.register_forward_hook(self._create_hook(f"{name}_output"))
                self.activation_hooks.append(hook)

    def _create_hook(self, name):
        """
        Create a hook function for a specific module.

        Args:
            name: Name of the module for storing intermediate values

        Returns:
            function: Hook function
        """

        def hook(module, inputs, outputs):
            # Store the output in intermediate_values
            self.intermediate_values[name] = outputs

            # Print debug info if in debug mode
            if self.debug_mode:
                if isinstance(outputs, tuple):
                    shapes = [
                        o.shape if hasattr(o, "shape") else "N/A" for o in outputs
                    ]
                    print(f"Module {name} output shapes: {shapes}")
                elif hasattr(outputs, "shape"):
                    print(f"Module {name} output shape: {outputs.shape}")

            return outputs

        return hook

    def _ensure_device(self, tensor):
        """
        Ensure a tensor is on the correct device.
        """
        if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model with device management.
        """
        # Ensure inputs are on the correct device
        args = tuple(self._ensure_device(arg) for arg in args)
        kwargs = {k: self._ensure_device(v) for k, v in kwargs.items()}

        # Clear intermediate values
        self.intermediate_values.clear()

        # Forward pass
        outputs = self.model(*args, **kwargs)

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generation method to capture values during generation.

        Returns:
            The same output as the wrapped model's generate method
        """
        # Reset intermediate values
        self.intermediate_values = {}

        # Call the model's generate method
        outputs = self.model.generate(*args, **kwargs)

        return outputs

    def unwrap(self):
        """
        Return the original unwrapped model.
        Useful for operations that need direct access to the model.

        Returns:
            The unwrapped model
        """
        # Remove all hooks before unwrapping
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []

        return self.model

    def set_debug_mode(self, enabled=False):
        """
        Enable or disable debug mode for more verbose output.

        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        print(f"TEMPO Model Wrapper debug mode {'enabled' if enabled else 'disabled'}")

    # Forward attribute access to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

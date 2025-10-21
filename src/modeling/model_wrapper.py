import torch
from typing import Optional, Any
from torch import nn
from src.utils.logging_utils import LoggingMixin


class TEMPOModelWrapper(nn.Module, LoggingMixin):
    """
    TEMPO Model Wrapper for language models.
    Captures intermediate values during generation and provides hooks for debugging.
    """

    def __init__(self, model, tokenizer=None, device=None):
        """
        Initialize the model wrapper.

        Args:
            model: The language model to wrap
            tokenizer: The tokenizer to use with the model
            device: The device to use for the model
        """
        nn.Module.__init__(self)
        LoggingMixin.__init__(self)

        # Assert model is not None and has the required attributes
        assert model is not None, "Model cannot be None"
        assert hasattr(model, "forward"), "Model must have a forward method"
        assert hasattr(model, "config"), "Model must have a config attribute"

        self.model = model
        self.tokenizer = tokenizer
        self.intermediate_values = {}
        self.activation_hooks = []

        # Original model attributes need to be accessible through the wrapper
        self.config = model.config
        if hasattr(model, "generation_config"):
            self.generation_config = model.generation_config

        # Store device information
        self.device = device if device is not None else next(model.parameters()).device
        # Assert device is valid
        assert self.device is not None, "Model device could not be determined"

        # Setup logging using the mixin with centralized config
        self.setup_logging("model_wrapper", "model_wrapper_debug.log")

        # Register hooks after logger is set up
        self._register_hooks()

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

        # Assert that at least one hook was registered
        assert (
            len(self.activation_hooks) > 0
        ), "No hooks were registered. Check model architecture."

    def _create_hook(self, name):
        """
        Create a hook function for a specific module.

        Args:
            name: Name of the module for storing intermediate values

        Returns:
            function: Hook function
        """
        assert name, "Hook name cannot be empty"

        def hook(module, inputs, outputs):
            # Store the output in intermediate_values
            self.intermediate_values[name] = outputs

            # Log debug info if in debug mode
            if self.debug_mode:
                if isinstance(outputs, tuple):
                    shapes = [
                        o.shape if hasattr(o, "shape") else "N/A" for o in outputs
                    ]
                    self.log(f"Module {name} output shapes: {shapes}")
                elif hasattr(outputs, "shape"):
                    self.log(f"Module {name} output shape: {outputs.shape}")
                else:
                    self.log(
                        f"Module {name} output has no shape attribute, type: {type(outputs)}"
                    )

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
        # Handle custom_attention_mask if provided
        if 'custom_attention_mask' in kwargs:
            custom_mask = kwargs.pop('custom_attention_mask')
            if custom_mask is not None:
                # Convert 2D custom mask [seq_len, seq_len] to 4D [batch, 1, seq_len, seq_len]
                # HuggingFace adds this as bias to attention scores
                if custom_mask.dim() == 2:
                    custom_mask = custom_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

                # Replace the attention_mask with our custom 4D mask
                # This gets added directly to attention scores before softmax
                kwargs['attention_mask'] = custom_mask

        # Ensure inputs are on the correct device
        args = tuple(self._ensure_device(arg) for arg in args)
        kwargs = {k: self._ensure_device(v) for k, v in kwargs.items()}

        # Clear intermediate values
        self.intermediate_values.clear()

        # Forward pass
        outputs = self.model(*args, **kwargs)

        # Assert outputs are not None
        assert outputs is not None, "Model forward pass returned None"

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generation method to capture values during generation.

        Returns:
            The same output as the wrapped model's generate method
        """
        # Reset intermediate values
        self.intermediate_values = {}

        # Assert model has generate method
        assert hasattr(self.model, "generate"), "Model does not have a generate method"

        # Call the model's generate method
        outputs = self.model.generate(*args, **kwargs)

        # Assert outputs are not None and have the expected shape
        assert outputs is not None, "Model generation returned None"
        assert isinstance(
            outputs, torch.Tensor
        ), "Model generation must return a tensor"

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

        # Assert model is still valid
        assert self.model is not None, "Model was lost during unwrapping"

        return self.model

    def set_debug_mode(self, enabled=False):
        """
        Enable or disable debug mode for more verbose output.

        Args:
            enabled: Whether to enable debug mode
        """
        assert isinstance(enabled, bool), "Debug mode must be a boolean"
        self.debug_mode = enabled
        self.log(
            f"TEMPO Model Wrapper debug mode {'enabled' if enabled else 'disabled'}"
        )

    # Forward attribute access to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            assert hasattr(
                self.model, name
            ), f"Neither wrapper nor model has attribute '{name}'"
            return getattr(self.model, name)

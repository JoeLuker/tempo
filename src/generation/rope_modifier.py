import torch
import math
import warnings
from typing import List, Tuple, Optional, Dict, Any, Set
from src.utils.logging_utils import LoggingMixin


class RoPEModifier(LoggingMixin):
    """
    Custom implementation to modify RoPE (Rotary Position Embeddings) for parallel token generation.
    This implementation allows multiple tokens at the same position to share identical positional embeddings.
    """

    def __init__(self, model, device: str = "mps"):
        """
        Initialize the RoPE modifier.

        Args:
            model: The transformer model
            device: The device for computation
        """
        super().__init__()
        self.model = model
        self.device = device
        self.original_forward_fns = {}  # Store original methods by module path
        self.position_map = {}  # Maps token positions to their effective positions
        self.patched_modules = set()  # Keep track of which modules have been patched
        self.is_installed = False

        # Cache for position embeddings to avoid recomputation
        self.position_embedding_cache = {}

        # Track parallel token sets
        self.parallel_token_sets = {}

        # Setup logging with debug mode disabled by default
        self.setup_logging("rope_modifier", "rope_modifier_debug.log")

    def install(self):
        """
        Install the custom RoPE implementation by patching all model's attention layers.
        """
        if self.is_installed:
            print("RoPE modifier already installed")
            return True  # Return True when already installed

        patched_count = 0
        mistral_specific_patches = 0
        qwen_specific_patches = 0

        # Get model type for debugging
        model_type = getattr(self.model.config, "model_type", "unknown").lower()
        print(f"DEBUG: Model type detected: {model_type}")
        print(f"DEBUG: Installing RoPE modifier on device: {self.device}")

        # First check if this is a Qwen model, which has a different RoPE implementation
        is_qwen = "qwen" in model_type

        # Try to identify all modules related to RoPE
        print("\nDEBUG: Searching for RoPE-related modules...")
        rope_related_modules = {}
        for name, module in self.model.named_modules():
            if any(
                rope_name in str(type(module).__name__).lower()
                or rope_name in name.lower()
                for rope_name in ["ropeembedding", "rotary", "rope"]
            ):
                rope_related_modules[name] = str(type(module).__name__)

        if rope_related_modules:
            print(
                f"DEBUG: Found {len(rope_related_modules)} potential RoPE-related modules:"
            )
            for name, module_type in rope_related_modules.items():
                print(f"  - {name} ({module_type})")
        else:
            print(
                "DEBUG: No RoPE-related modules found by name or type. This may indicate an unsupported model architecture."
            )

        if is_qwen:
            # For Qwen models, look for specific RoPE implementations
            print("\nDEBUG: Attempting to patch Qwen-specific RoPE modules...")
            for name, module in self.model.named_modules():
                if any(
                    rope_name in str(type(module).__name__).lower()
                    for rope_name in ["ropeembedding", "rotary", "rope"]
                ):
                    if hasattr(module, "forward") and name not in self.patched_modules:
                        print(f"Patching Qwen rotary embedding module: {name}")
                        self._patch_module(name, module)
                        patched_count += 1
                        self.patched_modules.add(name)
                        qwen_specific_patches += 1
                    else:
                        if not hasattr(module, "forward"):
                            print(f"DEBUG: Module {name} skipped - no forward method")
                        else:
                            print(f"DEBUG: Module {name} skipped - already patched")

        # First find modules that are directly responsible for RoPE
        # For Mistral, we want to patch the MistralRotaryEmbedding class
        if qwen_specific_patches == 0:
            print("\nDEBUG: Attempting to patch direct rotary embedding modules...")
            for name, module in self.model.named_modules():
                # Direct rotary embedding classes
                if any(
                    rotary_name in name.lower()
                    for rotary_name in [
                        "rotaryembedding",
                        "mistralrotary",
                        "rotary_emb",
                    ]
                ):
                    if hasattr(module, "forward") and name not in self.patched_modules:
                        print(f"Patching rotary embedding module directly: {name}")
                        self._patch_module(name, module)
                        patched_count += 1
                        self.patched_modules.add(name)
                        mistral_specific_patches += 1
                    else:
                        if not hasattr(module, "forward"):
                            print(f"DEBUG: Module {name} skipped - no forward method")
                        else:
                            print(f"DEBUG: Module {name} skipped - already patched")

        # Then look for attention modules that might apply RoPE internally
        if mistral_specific_patches == 0 and qwen_specific_patches == 0:
            # If we didn't find any specific modules, fall back to general approach
            print(
                "\nDEBUG: No specific modules found, falling back to general approach..."
            )

            # First check for modules with rotary_emb attribute
            print("DEBUG: Looking for modules with rotary_emb attribute...")
            for name, module in self.model.named_modules():
                # Look for attention modules that have rotary position embeddings
                if (
                    hasattr(module, "rotary_emb")
                    and hasattr(module, "forward")
                    and name not in self.patched_modules
                ):
                    print(
                        f"Patching attention module with rotary_emb attribute: {name}"
                    )
                    self._patch_module(name, module)
                    patched_count += 1
                    self.patched_modules.add(name)
                elif hasattr(module, "rotary_emb") and not hasattr(module, "forward"):
                    print(f"DEBUG: Module {name} has rotary_emb but no forward method")

            # Also look for modules that might apply rotary embeddings differently
            print("DEBUG: Looking for modules with rotary-related methods or names...")
            for name, module in self.model.named_modules():
                # Various ways RoPE might be implemented
                if (
                    hasattr(module, "_apply_rotary_pos_emb")
                    or hasattr(module, "apply_rotary_pos_emb")
                    or hasattr(module, "_apply_rope")
                    or "rotary" in name.lower()
                    or "rope" in name.lower()
                ) and name not in self.patched_modules:

                    if hasattr(module, "forward"):
                        print(
                            f"Patching module with rotary-related attribute or name: {name}"
                        )
                        self._patch_module(name, module)
                        patched_count += 1
                        self.patched_modules.add(name)
                    else:
                        print(f"DEBUG: Module {name} skipped - no forward method")

        # Invariant: At least one module must be patched
        if patched_count == 0:
            print("\nDEBUG: CRITICAL ERROR - Failed to patch any RoPE modules")
            print("DEBUG: Dumping model structure to help diagnose...")
            model_structure = {}
            for name, module in self.model.named_modules():
                module_attrs = [
                    attr for attr in dir(module) if not attr.startswith("__")
                ]
                model_structure[name] = {
                    "type": str(type(module).__name__),
                    "has_forward": hasattr(module, "forward"),
                    "rope_related_attributes": [
                        attr
                        for attr in module_attrs
                        if "rotary" in attr.lower() or "rope" in attr.lower()
                    ],
                }

            # Print a summary of the model structure
            print("\nDEBUG: Model module summary:")
            for name, info in list(model_structure.items())[
                :20
            ]:  # Show first 20 to avoid overwhelming output
                if (
                    info["rope_related_attributes"]
                    or "rotary" in name.lower()
                    or "rope" in name.lower()
                ):
                    print(
                        f"  {name} ({info['type']}): has_forward={info['has_forward']}, rope_attrs={info['rope_related_attributes']}"
                    )

            return False  # Return False to indicate failure

        print(f"RoPE modifier installed: patched {patched_count} modules")
        self.is_installed = True
        return True  # Return True to indicate success

    def _patch_module(self, name: str, module: Any):
        """
        Patch a module's forward method to use custom RoPE handling.
        """
        if name in self.patched_modules:
            return

        # Store original forward method
        self.original_forward_fns[name] = module.forward
        self.patched_modules.add(name)

        # Create custom forward method
        rope_modifier = self  # Capture self for closure
        original_forward = module.forward  # Capture original forward method

        def custom_forward(*args, **kwargs):
            # Debug information if enabled
            if rope_modifier.debug_mode:
                print(f"Custom forward called for {name}")
                print(
                    f"Args: {[a.shape if hasattr(a, 'shape') else a for a in args[:3]]}"
                )
                print(f"Kwargs keys: {list(kwargs.keys())}")

            # Create new args and kwargs to avoid modifying originals
            new_args = list(args)
            new_kwargs = kwargs.copy()
            position_ids_in_args = False

            # Check if this is the rotary embedding forward call (has exactly 2 args with position_ids as the second)
            if len(args) == 2 and hasattr(args[1], "shape"):
                # This is likely hidden_states, position_ids
                position_ids_in_args = True
                position_ids_index = 1

                # Apply position mapping to the positional argument
                original_position_ids = args[position_ids_index]
                mapped_position_ids = rope_modifier.apply_position_mapping(
                    original_position_ids
                )
                new_args[position_ids_index] = mapped_position_ids

                # Remove position_ids from kwargs if present to avoid duplication
                if "position_ids" in new_kwargs:
                    del new_kwargs["position_ids"]

            # If position_ids is in kwargs and not already handled above
            elif "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids = kwargs["position_ids"]
                # Apply position mapping to ensure parallel tokens use the same position
                mapped_position_ids = rope_modifier.apply_position_mapping(position_ids)
                new_kwargs["position_ids"] = mapped_position_ids

            # If no explicit position_ids are provided but we have seq_len in hidden_states
            elif (
                len(args) >= 1
                and args[0] is not None
                and hasattr(args[0], "size")
                and len(args[0].size()) >= 2
                and "position_ids" not in kwargs
            ):
                seq_len = args[0].size(1)
                batch_size = args[0].size(0)
                device = args[0].device

                # Generate position IDs and apply our mapping
                position_ids = (
                    torch.arange(0, seq_len, device=device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
                mapped_position_ids = rope_modifier.apply_position_mapping(position_ids)

                # Add position_ids to kwargs if not present and not expected as positional arg
                if not position_ids_in_args:
                    new_kwargs["position_ids"] = mapped_position_ids

            # Call original method with potentially modified position_ids
            try:
                return original_forward(*new_args, **new_kwargs)
            except Exception as e:
                # Provide more helpful error messages in debug mode
                if rope_modifier.debug_mode:
                    print(f"Error in custom_forward: {e}")
                    print(
                        f"Args shapes: {[a.shape if hasattr(a, 'shape') else 'N/A' for a in new_args[:3]]}"
                    )
                    print(f"Kwargs keys: {list(new_kwargs.keys())}")

                    # No fallbacks with invariant programming
                    if (
                        "got multiple values for argument" in str(e)
                        and position_ids_in_args
                    ):
                        raise ValueError(
                            "Position ID conflict in arguments. Invariant: Arguments must be correctly structured."
                        )

        return custom_forward

    def register_parallel_positions(self, position_mapping: Dict[int, int]):
        """
        Register position mapping for parallel tokens.
        Maps physical token positions to their logical positions.

        Args:
            position_mapping: Dictionary mapping physical token positions to logical positions
        """
        if not self.is_installed:
            print("Warning: RoPE modifier not installed yet. Call install() first.")
            return

        if self.debug_mode:
            print(f"Registering parallel position mapping: {position_mapping}")

        # Reset the sequence_position_map when updating position_map to avoid stale mappings
        self.sequence_position_map = {}

        # Update the main position map
        self.position_map.update(position_mapping)

        # Clear the embedding cache to force regeneration
        self.position_embedding_cache = {}

        # Find all parallel token sets by grouping by target position
        parallel_sets = {}
        for physical_pos, logical_pos in position_mapping.items():
            if logical_pos not in parallel_sets:
                parallel_sets[logical_pos] = []
            parallel_sets[logical_pos].append(physical_pos)

        # Update parallel token sets
        for logical_pos, physical_positions in parallel_sets.items():
            self.parallel_token_sets[logical_pos] = physical_positions

        # Sanity checks in debug mode
        if self.debug_mode:
            for logical_pos, physical_positions in parallel_sets.items():
                if len(physical_positions) > 1:
                    print(
                        f"Set up parallel tokens at logical position {logical_pos} with {len(physical_positions)} physical positions: {physical_positions}"
                    )

    def apply_position_mapping(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply position mapping to the position IDs tensor.
        For positions that are mapped, replace the position ID with the mapped ID.

        Args:
            position_ids: Position IDs tensor [batch_size, seq_len]

        Returns:
            torch.Tensor: Modified position IDs
        """
        # Quick return if no position mapping is registered
        if not self.position_map and not self.parallel_token_sets:
            return position_ids

        # Copy the input tensor to avoid modifying the original
        mapped_position_ids = position_ids.clone()

        # Create a unique cache key for this position_ids tensor
        # This helps avoid redundant computation for repeated forward passes
        cache_key = f"{position_ids.shape}_{position_ids.sum().item()}"
        if cache_key in self.position_embedding_cache:
            return self.position_embedding_cache[cache_key]

        # Apply simple position remapping first
        modified_positions = set()
        for orig_pos, mapped_pos in self.position_map.items():
            if orig_pos < position_ids.size(1):
                mapped_position_ids[:, orig_pos] = mapped_pos
                modified_positions.add(orig_pos)

        # Then handle the parallel token sets
        if self.parallel_token_sets:
            for target_pos, positions in self.parallel_token_sets.items():
                if isinstance(positions, list) and len(positions) > 1:
                    # Check if any position in this set was modified
                    if any(pos in modified_positions for pos in positions):
                        # Check that all tokens in this set have the same position ID
                        for batch_idx in range(position_ids.size(0)):
                            # Validate positions are within bounds of the tensor
                            valid_positions = [
                                pos for pos in positions if pos < position_ids.size(1)
                            ]

                            if len(valid_positions) > 1:
                                pos_values = mapped_position_ids[
                                    batch_idx, valid_positions
                                ].tolist()
                                if len(set(pos_values)) > 1:
                                    if self.debug_mode:
                                        print(
                                            f"Ensuring position consistency in parallel set: {positions}"
                                        )
                                    # Set all positions in the set to the same value (target_pos)
                                    for pos in valid_positions:
                                        mapped_position_ids[batch_idx, pos] = target_pos

        # Cache the result to avoid redundant computation
        self.position_embedding_cache[cache_key] = mapped_position_ids
        return mapped_position_ids

    def _has_significant_position_changes(self) -> bool:
        """
        Determine if there has been a significant change in position mappings
        that would affect KV cache consistency.

        This function is no longer needed for KV cache handling in TEMPO.
        The function is kept for backward compatibility but doesn't affect the KV cache reset logic.

        Deprecated since v0.2.0: This method will be removed in a future version.
        Position mapping for parallel tokens no longer requires KV cache resets in TEMPO.

        Returns:
            bool: True if significant changes detected, False otherwise
        """
        warnings.warn(
            "_has_significant_position_changes is deprecated and will be removed in a future version. "
            "Position mapping for parallel tokens no longer requires KV cache resets in TEMPO.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For TEMPO's usage pattern, the position mapping for parallel tokens
        # doesn't require a complete KV cache reset, so we always return False
        return False

    def reset(self):
        """Reset all position mappings and caches."""
        self.position_map = {}
        self.sequence_position_map = {}
        self.parallel_token_sets = {}
        self.position_embedding_cache = {}
        if self.debug_mode:
            print("RoPE modifier reset - all position mappings cleared")

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
        if enabled:
            print("RoPE modifier debug mode enabled")
        else:
            print("RoPE modifier debug mode disabled")

    def set_attention_manager(self, attention_manager):
        """
        Set the attention manager for coordination.

        Args:
            attention_manager: Attention manager instance
        """
        self.attention_manager = attention_manager
        if self.debug_mode:
            print("RoPE modifier linked with attention manager")

    def set_token_generator(self, token_generator):
        """
        Set the token generator for coordination.

        Args:
            token_generator: Token generator instance
        """
        self.token_generator = token_generator
        if self.debug_mode:
            print("RoPE modifier linked with token generator")

    def add_parallel_position_mappings(
        self, position_idx: int, virtual_positions: List[int]
    ):
        """
        Add mappings for parallel token positions at a specific index.

        Args:
            position_idx: The base position index for the parallel tokens
            virtual_positions: List of virtual positions to map to the base position
        """
        if self.debug_mode:
            print(
                f"Adding parallel position mappings: base={position_idx}, virtual={virtual_positions}"
            )

        # Create position mappings
        for virtual_pos in virtual_positions:
            self.position_map[virtual_pos] = position_idx

        # Track this parallel set
        # Store as a list of positions instead of a dictionary to match how it's used in apply_position_mapping
        self.parallel_token_sets[position_idx] = virtual_positions

        # Clear the embedding cache for these positions
        self.position_embedding_cache = {}

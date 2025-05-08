import torch
import random
import time
import numpy as np
import traceback
from typing import List, Dict, Tuple, Any, Optional, Set, Callable
from tqdm import tqdm

# Import necessary components
from src.generation.rope_modifier import RoPEModifier
from src.generation.attention_manager import AttentionManager
from src.generation.token_selector import TokenSelector
from src.generation.token_generator import TokenGenerator
from src.generation.text_formatter import TextFormatter
from src.utils.logging_utils import LoggingMixin

class ParallelGenerator(LoggingMixin):
    """
    Main class for parallel token generation with threshold pruning.
    Optimized for performance with batched operations.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "mps",
        has_custom_attention: bool = True,
        use_custom_rope: bool = True,
        debug_mode: bool = False,
        token_generator: Optional[TokenGenerator] = None,
    ):
        """
        Initialize the parallel generator.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            device: Device to use for computation
            has_custom_attention: Whether to use custom attention handling
            use_custom_rope: Whether to use custom RoPE modifications
            debug_mode: Enable debug mode for detailed logging
            token_generator: Optional external TokenGenerator instance to use
        """
        super().__init__()
        # Store parameters
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.has_custom_attention = has_custom_attention
        self.use_custom_rope = use_custom_rope

        # Setup logging using the mixin with explicitly provided debug_mode if passed
        if debug_mode is not None:
            self.setup_logging("parallel_generator", "generation_debug.log", debug_mode)
        else:
            self.setup_logging("parallel_generator", "generation_debug.log")

        # Track logical layout of parallel tokens
        self.logical_layout = []  # [(logical_pos, start_idx, end_idx)]

        # When debug mode is enabled, start with a clear log message
        if self.debug_mode:
            self.log("Initializing ParallelGenerator with debug mode ENABLED")
            # Also print to stdout for immediate visibility
            print("ParallelGenerator starting with debug mode ENABLED")

        # Validate required arguments
        assert model is not None, "Model must be provided"
        assert tokenizer is not None, "Tokenizer must be provided"

        # Check if this is a Qwen-based model
        self.is_qwen = hasattr(model.config, "architectures") and any(
            "Qwen" in arch for arch in model.config.architectures
        )

        # Initialize RoPE modifier if requested
        self.rope_modifier = None
        if use_custom_rope:
            try:
                # Create and initialize RoPE modifier
                self.rope_modifier = RoPEModifier(model=model, device=device)
                self.rope_modifier.set_debug_mode(self.debug_mode)

                # Install the modifier
                rope_install_success = self.rope_modifier.install()
                if rope_install_success:
                    print("Using custom RoPE modifications for parallel token positioning")
                    if self.debug_mode:
                        self.log(f"RoPE modifier initialized with debug mode {'ENABLED' if self.debug_mode else 'DISABLED'}")
                else:
                    self.log("RoPE modifier installation failed, disabling custom RoPE", "warning")
                    self.rope_modifier = None
                    self.use_custom_rope = False
            except Exception as e:
                self.log(f"Warning: Failed to initialize RoPE modifier: {e}", "warning")
                print(f"Warning: Could not initialize RoPE modifier: {e}")
                self.rope_modifier = None
                self.use_custom_rope = False  # Disable since initialization failed

        # Initialize components (only once!) and set debug mode immediately
        if token_generator is not None:
            self.token_generator = token_generator
            self.log("Using externally provided TokenGenerator instance")
        else:
            self.token_generator = TokenGenerator(model, tokenizer, device)
            self.log("Created internal TokenGenerator instance")

        # Explicitly set debug mode after creation
        self.token_generator.set_debug_mode(self.debug_mode)
        assert self.token_generator is not None, "Failed to initialize TokenGenerator"

        # Initialize TokenSelector with debug mode
        self.token_selector = TokenSelector(tokenizer)
        self.token_selector.set_debug_mode(self.debug_mode)
        assert self.token_selector is not None, "Failed to initialize TokenSelector"

        self.text_formatter = TextFormatter(tokenizer)
        assert self.text_formatter is not None, "Failed to initialize TextFormatter"

        # Create attention manager with debug mode
        self.attention_manager = AttentionManager(
            device=device, rope_modifier=self.rope_modifier, tokenizer=tokenizer
        )
        self.attention_manager.set_debug_mode(self.debug_mode)

        # Log initialization messages only when debug mode is enabled
        if self.debug_mode:
            self.log(f"AttentionManager initialized with debug mode {'ENABLED' if self.debug_mode else 'DISABLED'}")
            self.log(f"TokenSelector initialized with debug mode {'ENABLED' if self.debug_mode else 'DISABLED'}")
            self.log(f"TokenGenerator initialized with debug mode {'ENABLED' if self.debug_mode else 'DISABLED'}")

        # Link components for better coordination if RoPE modifier is available
        if self.rope_modifier is not None:
            self.rope_modifier.set_attention_manager(self.attention_manager)
            self.rope_modifier.set_token_generator(self.token_generator)

        # Performance tracking
        self.generation_time = 0
        self.pruning_time = 0

        # Setup tracking for sequence length
        self.sequence_length = 0
        self.initial_prompt_length = 0
        self.step_count = 0
        self.sequence_length_history = []
        
        # Post-initialization check
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have encode method"
        assert hasattr(self.tokenizer, "decode"), "Tokenizer must have decode method"

    def _init_sequence_tracking(self, prompt_length):
        """Initialize sequence length tracking with the prompt length."""
        assert prompt_length >= 0, "Prompt length cannot be negative"
        
        self.sequence_length = 0
        self.initial_prompt_length = prompt_length
        self.step_count = 0
        self.sequence_length_history = []
        
        self.sequence_length = 0
        self.initial_prompt_length = prompt_length
        self.step_count = 0
        self.sequence_length_history = []
    
    def get_sequence_length(self):
        """Get the current sequence length (tokens generated beyond prompt)."""
        return self.sequence_length
    
    def get_total_sequence_length(self):
        """Get the total sequence length including prompt."""
        return self.initial_prompt_length + self.sequence_length
    
    def update_sequence_length(self, new_length, callback=None):
        """
        Update sequence length and call any registered callbacks.
        
        Args:
            new_length: The new sequence length to set
            callback: Optional callback function to notify of changes
        """
        assert new_length >= 0, "Sequence length cannot be negative"
        
        if new_length > self.sequence_length:
            old_length = self.sequence_length
            self.sequence_length = new_length
            self.step_count += 1
            self.sequence_length_history.append(new_length)
            
            # Verify state consistency
            assert self.step_count == len(self.sequence_length_history), "Step count and history length mismatch"
            
            # Report changes if debugging
            if self.debug_mode:
                self.log(f"Sequence length updated: {old_length} → {new_length}", "debug")
                
            # Call sequence length callback if provided
            if callback is not None:
                try:
                    callback(new_length, self.step_count, self.initial_prompt_length)
                except Exception as e:
                    self.log(f"Error in sequence length callback: {e}", "error")
                    
            return True
        return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        selection_threshold: Optional[float] = 0.1,
        return_parallel_sets: bool = False,
        use_retroactive_pruning: bool = False,
        min_steps: int = 0,
        show_token_ids: bool = False,
        debug_mode: Optional[bool] = None,
        disable_kv_cache: bool = False,
        system_content: Optional[str] = None,
        isolate_parallel_tokens: bool = True,
        preserve_all_isolated_tokens: Optional[bool] = None,
        retroactive_pruner: Optional[object] = None,
        sequence_callback: Optional[Callable[[int, int, int], None]] = None,
        # MCTS parameters
        use_mcts: bool = False,
        mcts_simulations: int = 10,
        mcts_c_puct: float = 1.0,
        mcts_depth: int = 5,
        # Dynamic threshold parameters
        dynamic_threshold: bool = False,
        final_threshold: float = 1.0,
        bezier_p1: float = 0.2,
        bezier_p2: float = 0.8,
        use_relu: bool = False,
        relu_activation_point: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate text from prompt with parallel token generation using Sequential Layout approach.
        
        This implementation uses a single sequence with logical positions mapping to potentially
        multiple physical positions in the sequence.

        Args:
            prompt: The text prompt
            max_tokens: Maximum number of tokens to generate
            selection_threshold: Probability threshold for INITIAL token selection (used by TokenSelector)
            return_parallel_sets: Whether to return parallel token sets for visualization
            use_retroactive_pruning: Whether to use retroactive pruning
            min_steps: Minimum number of steps before stopping
            show_token_ids: Include token IDs in return for debugging
            debug_mode: Override the instance debug_mode setting
            disable_kv_cache: Whether to disable KV caching
            system_content: System content for chat prompts
            isolate_parallel_tokens: Whether tokens at the same position cannot attend to each other
            preserve_all_isolated_tokens: Override default pruning behavior for isolated tokens
            retroactive_pruner: Retroactive pruner to use for token selection
            sequence_callback: Callback function for sequence length updates
            use_mcts: Whether to use Monte Carlo Tree Search for token selection
            mcts_simulations: Number of MCTS simulations per step
            mcts_c_puct: Exploration constant for MCTS
            mcts_depth: Maximum depth of MCTS simulations
            dynamic_threshold: Whether to use dynamic thresholding
            final_threshold: Final threshold value for dynamic thresholding
            bezier_p1: First Bezier control point for dynamic thresholding
            bezier_p2: Second Bezier control point for dynamic thresholding
            use_relu: Whether to use ReLU transition instead of Bezier curve
            relu_activation_point: Point at which ReLU transition begins

        Returns:
            Dict with generated text and metadata
        """
        # Update debug mode if override is provided
        original_debug_mode = self.debug_mode
        if debug_mode is not None:
            self.set_debug_mode(debug_mode)

        # Initialize timing variables
        generation_start = time.time()
        pruning_time = 0.0
        pruned_tokens_count = 0
        pruning_steps = 0

        # Validate parameters
        assert prompt, "Prompt cannot be empty"
        assert max_tokens > 0, "max_tokens must be positive"
        assert selection_threshold is None or (0.0 <= selection_threshold <= 1.0), "selection_threshold must be between 0.0 and 1.0"
        if selection_threshold is None:
            selection_threshold = 0.1  # Default threshold for token selection
        assert min_steps >= 0, "min_steps cannot be negative"
        
        # Check RoPE isolation availability
        if isolate_parallel_tokens and (not self.use_custom_rope or self.rope_modifier is None):
            self.log("Warning: Disabling isolate_parallel_tokens because RoPE modifier is not available", "warning")
            isolate_parallel_tokens = False
            print("Warning: Parallel token isolation disabled - RoPE modifier not available")
            
        # Set default for preserve_all_isolated_tokens based on isolation mode
        if preserve_all_isolated_tokens is None:
            preserve_all_isolated_tokens = isolate_parallel_tokens
            
        # If debug mode is enabled, log this information
        if self.debug_mode:
            print("Debug mode enabled for this generation run - logs will be written to files in logs/ directory")

        # Reset state
        if self.rope_modifier is not None:
            self.rope_modifier.reset()
        self.attention_manager.reset_cache()
        self.token_generator.clear_kv_cache()
        self.logical_layout = []  # Reset logical layout tracking

        # --- Initial Prompt Processing ---
        if system_content:
            # First check if the tokenizer has a chat template
            if hasattr(self.tokenizer, 'apply_chat_template') and callable(getattr(self.tokenizer, 'apply_chat_template')):
                # Use the tokenizer's built-in chat template
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                self.log(f"Applied tokenizer's chat template with system content")
            else:
                # Fallback to a generic template based on model type
                model_type = getattr(self.model.config, "model_type", "").lower() if hasattr(self.model, "config") else ""
                
                if "llama" in model_type:
                    # Llama-style template
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{prompt} [/INST]"
                elif "mistral" in model_type:
                    # Mistral-style template
                    formatted_prompt = f"<s>[INST] {system_content}\n\n{prompt} [/INST]"
                elif "qwen" in model_type or self.is_qwen:
                    # Qwen-style template
                    formatted_prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    # Generic ChatML template as fallback
                    formatted_prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                self.log(f"Applied generic chat template for {model_type if model_type else 'unknown'} model with system content")
            
            # Store original prompt and update with formatted version
            original_prompt = prompt
            prompt = formatted_prompt
            
        # Tokenize the prompt (now potentially with system content)
        input_ids, attention_mask = self.token_generator.prepare_input_from_prompt(prompt)
        prompt_tokens = input_ids.size(1)
        self._init_sequence_tracking(prompt_tokens)
        
        if sequence_callback:
            sequence_callback(0, 0, prompt_tokens)  # Initial callback
            
        # Initialize logical layout for the prompt
        self.logical_layout.append((0, 0, prompt_tokens - 1))  # Prompt is logical step 0

        if self.debug_mode:
            self.log(f"Tokenized prompt length: {prompt_tokens} tokens")
            if system_content:
                self.log(f"Prompt includes system content")
            self.log(f"Initial input shape: {input_ids.shape}")

        # --- Initialize KV Cache (Crucial!) ---
        past_key_values = None
        if not disable_kv_cache:
            try:
                self.log("Priming KV cache with initial prompt...")
                # Pass prompt through model once to get initial KV state
                with torch.inference_mode():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                past_key_values = outputs.past_key_values
                self.log(f"KV cache primed. Num layers: {len(past_key_values) if past_key_values else 'None'}")
            except Exception as e:
                self.log(f"Error priming KV cache: {e}. Will proceed without.", "error")
                disable_kv_cache = True  # Force disable if priming failed
                past_key_values = None

        # --- Tracking Variables ---
        all_original_token_sets = {}  # {logical_pos: [(id, prob), ...]} for originals
        all_pruned_token_sets = {}    # {logical_pos: [(id, prob), ...]} for pruned
        had_repetition_loop = False

        # --- Main Generation Loop ---
        for logical_step in tqdm(range(max_tokens), desc="Generating tokens"):
            current_physical_len = input_ids.size(1)
            self.log(f"\n--- Logical Step {logical_step} (Physical Length: {current_physical_len}) ---")

            # 1. Determine input for logit generation
            if disable_kv_cache:
                # Use full history
                input_ids_for_logits = input_ids
                attention_mask_for_logits = attention_mask
                kv_cache_for_logits = None
                self.log("Logit Gen: KV cache disabled, using full input.")
            elif past_key_values is None:
                # First step after prompt (or if cache disabled earlier)
                input_ids_for_logits = input_ids
                attention_mask_for_logits = attention_mask
                kv_cache_for_logits = None
                self.log("Logit Gen: No KV cache yet, using full input.")
            else:
                # Use KV cache: Input is only the *last* token(s) added
                last_logical_pos, last_start_idx, last_end_idx = self.logical_layout[-1]
                input_ids_for_logits = input_ids[:, last_start_idx:last_end_idx+1]
                # Attention mask needs correct length for KV cache
                past_len = past_key_values[0][0].shape[2]  # Length stored in cache
                num_new = input_ids_for_logits.shape[1]
                # Create attention mask for the *new* tokens plus the past length
                attention_mask_for_logits = torch.ones((1, past_len + num_new), device=self.device)
                kv_cache_for_logits = past_key_values
                self.log(f"Logit Gen: Using KV cache. Input shape: {input_ids_for_logits.shape}. Attention mask shape: {attention_mask_for_logits.shape}. Past len: {past_len}")

            # 2. Get Logits for the *next physical position*
            try:
                # Get next token logits with cache
                next_token_logits, new_past_key_values = self.token_generator.generate_next_token_with_cache(
                    input_ids=input_ids_for_logits,
                    attention_mask=attention_mask_for_logits,
                    past_key_values=kv_cache_for_logits,
                    disable_kv_cache=disable_kv_cache
                )
                
                # Update the main KV cache for next iteration
                if not disable_kv_cache:
                    past_key_values = new_past_key_values
                    if past_key_values:
                        self.log(f"KV Cache updated. New cache len: {past_key_values[0][0].shape[2]}")
                    else:
                        self.log("KV Cache is None after logit generation.", "warning")
                        
            except Exception as e:
                self.log(f"Error during token generation at logical step {logical_step}: {e}", "error")
                self.log(traceback.format_exc(), "error")
                break

            # 3. Select Parallel Candidates T using the SELECTION threshold
            try:
                # Use the threshold parameter ONLY for initial token selection
                selection_threshold = selection_threshold  # Explicitly name it for clarity
                
                # Apply dynamic threshold if enabled
                if dynamic_threshold:
                    # Calculate current progress through generation
                    progress = logical_step / max_tokens
                    
                    if use_relu:
                        # Use ReLU transition
                        if progress < relu_activation_point:
                            current_threshold = selection_threshold
                        else:
                            # Linear transition from selection_threshold to final_threshold
                            transition_progress = (progress - relu_activation_point) / (1.0 - relu_activation_point)
                            current_threshold = selection_threshold + (final_threshold - selection_threshold) * transition_progress
                    else:
                        # Use Bezier curve for smooth transition
                        # Calculate Bezier curve point
                        t = progress
                        p0 = selection_threshold
                        p3 = final_threshold
                        current_threshold = (1-t)**3 * p0 + 3*(1-t)**2*t * bezier_p1 + 3*(1-t)*t**2 * bezier_p2 + t**3 * p3
                    
                    self.log(f"Dynamic threshold at step {logical_step}: {current_threshold:.4f}")
                    selection_threshold = current_threshold
                
                # Use MCTS for token selection if enabled
                if use_mcts:
                    # Initialize MCTS state if first step
                    if logical_step == 0:
                        self.log("Initializing MCTS for token selection")
                        # Create MCTS state with current context
                        mcts_state = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'past_key_values': past_key_values,
                            'logical_step': logical_step
                        }
                    
                    # Run MCTS simulations
                    best_tokens = []
                    for _ in range(mcts_simulations):
                        # Simulate MCTS rollout
                        simulation_result = self._mcts_simulation(
                            mcts_state,
                            depth=mcts_depth,
                            c_puct=mcts_c_puct
                        )
                        best_tokens.extend(simulation_result)
                    
                    # Select top tokens based on MCTS results
                    token_distribution = self._select_from_mcts_results(best_tokens, selection_threshold)
                else:
                    # Use standard token selection
                    token_distribution, subset_size = self.token_selector.select_tokens(
                        next_token_logits, threshold=selection_threshold
                    )
                
                if not token_distribution:
                    self.log(f"Warning: No tokens above selection threshold {selection_threshold} at step {logical_step}. Falling back.", "warning")
                    token_distribution, _ = self.token_selector.select_tokens(next_token_logits, threshold=0.0, max_tokens=1)
                    if not token_distribution:
                        self.log("Critical error: Still no tokens selected. Stopping.", "error")
                        break
                        
                self.log(f"Selected {len(token_distribution)} candidate tokens for logical step {logical_step} using threshold {selection_threshold}")
                
                # Store original candidates
                original_candidates = [(tid.item(), float(prob)) for tid, prob in token_distribution]
                all_original_token_sets[logical_step] = original_candidates
                
            except ValueError as e:
                self.log(f"Error selecting tokens at step {logical_step}: {e}. Logits shape: {next_token_logits.shape}", "error")
                break

            # 4. Apply Pruning to get T' (pruners use their own thresholds)
            pruned_distribution = token_distribution  # Start with original candidates
            if use_retroactive_pruning and logical_step > 0:
                pruning_start = time.time()
                try:
                    # Update retroactive pruner with current step (updates its internal threshold)
                    if hasattr(retroactive_pruner, 'update_step'):
                        retroactive_pruner.update_step(logical_step)

                    # Prune history before adding current tokens (uses its own threshold)
                    if hasattr(retroactive_pruner, 'retroactively_prune'):
                        pruned_history = retroactive_pruner.retroactively_prune(
                            prompt_length=prompt_tokens,
                            all_parallel_tokens=all_original_token_sets,
                            step=logical_step
                        )
                        # Update the pruned sets history
                        all_pruned_token_sets.update(pruned_history)
                        self.log(f"Retroactive pruning applied up to step {logical_step-1}.")
                
                except Exception as e:
                    self.log(f"Error during retroactive pruning at step {logical_step}: {e}", "error")
                    self.log(traceback.format_exc(), "error")
                    
                pruning_time += time.time() - pruning_start
                pruning_steps += 1
                
                # Store pruned results for this step
                pruned_candidates = [(tid.item(), float(prob)) for tid, prob in pruned_distribution]
                all_pruned_token_sets[logical_step] = pruned_candidates

            # 5. Extract token IDs from final pruned distribution
            T_prime_ids = [tid.item() for tid, _ in pruned_distribution]
            T_prime_probs = [float(prob) for _, prob in pruned_distribution]
            
            # If no tokens left after pruning (shouldn't happen due to fallbacks, but just in case)
            if not T_prime_ids:
                self.log("Warning: No tokens left after pruning, using top original token", "warning")
                T_prime_ids = [token_distribution[0][0].item()]
                T_prime_probs = [float(token_distribution[0][1])]

            # 6. Update Canonical State (input_ids stream)
            physical_start_idx = input_ids.size(1)
            new_tokens_tensor = torch.tensor([T_prime_ids], device=self.device)
            
            input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, len(T_prime_ids)), device=self.device)], dim=1)
            
            physical_end_idx = input_ids.size(1) - 1
            
            # Update logical layout
            self.logical_layout.append((logical_step, physical_start_idx, physical_end_idx))
            self.log(f"Appended {len(T_prime_ids)} tokens. New physical length: {input_ids.size(1)}")
            self.log(f"Logical layout updated: ({logical_step}, {physical_start_idx}, {physical_end_idx})")
            
            # Update sequence length tracking
            self.update_sequence_length(input_ids.size(1) - prompt_tokens, sequence_callback)
            
            # 7. Configure RoPE/Attention for Next Step
            if self.rope_modifier is not None and len(T_prime_ids) > 1:
                # Create position mapping from physical positions to logical step
                position_mapping = {idx: logical_step for idx in range(physical_start_idx, physical_end_idx + 1)}
                self.rope_modifier.register_parallel_positions(position_mapping)
                self.log(f"RoPE configured for parallel set at logical position {logical_step} (physical indices {physical_start_idx}-{physical_end_idx})")
            
            # Also update the attention manager with the token history
            if hasattr(self.attention_manager, 'update_token_history'):
                for token_id in T_prime_ids:
                    self.attention_manager.update_token_history(token_id)
                    
            # 8. Termination Check
            if logical_step >= min_steps:
                if any(tid == self.tokenizer.eos_token_id for tid in T_prime_ids):
                    self.log(f"EOS token generated at logical step {logical_step}, ending generation.")
                    # Truncate input_ids at first EOS token
                    eos_indices = [i for i, tid in enumerate(T_prime_ids) if tid == self.tokenizer.eos_token_id]
                    if eos_indices:
                        first_eos_physical_idx = physical_start_idx + min(eos_indices)
                        input_ids = input_ids[:, :first_eos_physical_idx + 1]  # Include the EOS token
                        attention_mask = attention_mask[:, :first_eos_physical_idx + 1]
                        # Adjust logical layout
                        if self.logical_layout[-1][0] == logical_step:
                            self.logical_layout[-1] = (logical_step, physical_start_idx, first_eos_physical_idx)
                        self.log(f"Truncated sequence at first EOS token (physical index {first_eos_physical_idx}).")
                    break
                    
        # --- Post-Generation Cleanup ---
        if self.rope_modifier is not None:
            self.rope_modifier.reset()
            
        # Decode the generated text
        generated_token_ids = input_ids[0][prompt_tokens:].tolist()
        raw_generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        full_text = prompt + raw_generated_text
        
        # Build visualization data
        token_sets_for_vis = []
        if return_parallel_sets:
            for logical_step in range(len(all_original_token_sets)):
                original_set = all_original_token_sets.get(logical_step, [])
                pruned_set = all_pruned_token_sets.get(logical_step, original_set)  # Use original if no pruned entry
                
                # Extract ids and probs
                original_ids = [tid for tid, _ in original_set]
                original_probs = [prob for _, prob in original_set]
                
                # Extract only the tokens that were pruned
                pruned_ids = []
                pruned_probs = []
                for tid, prob in original_set:
                    if tid not in [p_tid for p_tid, _ in pruned_set]:
                        pruned_ids.append(tid)
                        pruned_probs.append(prob)
                        
                token_sets_for_vis.append(
                    (
                        logical_step,  # Logical step
                        (original_ids, original_probs),
                        (pruned_ids, pruned_probs)
                    )
                )
                
        # Format output using the logical layout
        formatted_output = self.text_formatter.format_using_layout(
            prompt=prompt,
            input_ids=input_ids[0].tolist(),
            logical_layout=self.logical_layout,
            prompt_length=prompt_tokens,
            all_original_token_sets=all_original_token_sets,
            tokenizer=self.tokenizer,
            show_token_ids=show_token_ids
        )
        
        # Check for repetition patterns
        had_repetition_loop = False
        
        # Simple repetition detection
        def check_for_repetition(text, min_length=5, max_length=20, min_repeats=3):
            if len(text) < min_length * min_repeats:
                return None
                
            for seq_len in range(min_length, min(max_length, len(text) // min_repeats)):
                for i in range(len(text) - seq_len * min_repeats):
                    seq = text[i:i+seq_len]
                    # Skip sequences that are just spaces or very simple patterns
                    if seq.isspace() or seq == seq[0] * seq_len:
                        continue
                    # Count non-overlapping occurrences
                    count = 0
                    pos = i
                    while pos < len(text):
                        found_pos = text.find(seq, pos)
                        if found_pos == -1:
                            break
                        count += 1
                        pos = found_pos + seq_len
                    
                    if count >= min_repeats:
                        return f"'{seq}' repeats {count} times"
            return None
            
        repetition_check = check_for_repetition(raw_generated_text)
        if repetition_check:
            had_repetition_loop = True
            self.log(f"Repetition detected: {repetition_check}")
            
        # Calculate generation time
        generation_time = time.time() - generation_start
        self.generation_time = generation_time
        self.pruning_time = pruning_time
        
        # Prepare result
        result = {
            "generated_text": formatted_output,
            "raw_generated_text": raw_generated_text,
            "generation_time": generation_time,
            "pruning_time": pruning_time,
            "is_qwen_model": self.is_qwen,
            "had_repetition_loop": had_repetition_loop,
            "prompt": prompt,
            "selection_threshold": selection_threshold,
            "use_retroactive_pruning": use_retroactive_pruning,
            "min_steps": min_steps,
            "disable_kv_cache": disable_kv_cache,
            "isolate_parallel_tokens": isolate_parallel_tokens,
            "logical_layout": self.logical_layout,
        }
        
        # Add visualization data if requested
        if return_parallel_sets:
            result["token_sets"] = token_sets_for_vis
            result["all_original_token_sets"] = all_original_token_sets
            result["all_pruned_token_sets"] = all_pruned_token_sets
            
        # Add pruning statistics
        if use_retroactive_pruning:
            result["pruning_time"] = pruning_time
            result["pruning_steps"] = pruning_steps
            
        # Restore original debug mode if needed
        if debug_mode is not None:
            self.set_debug_mode(original_debug_mode)
            
        return result

    def set_debug_mode(self, enabled: bool = True):
        """Enable or disable debug mode for detailed logging."""
        # Only change if the value is changing
        if self.debug_mode != enabled:
            self.debug_mode = enabled
            if enabled:
                print(f"ParallelGenerator debug mode ENABLED")
            else:
                print(f"ParallelGenerator debug mode disabled")
            
            # Propagate debug mode to components
            if self.rope_modifier is not None:
                self.rope_modifier.set_debug_mode(enabled)
            
            if hasattr(self, "attention_manager") and self.attention_manager is not None:
                self.attention_manager.set_debug_mode(enabled)
                
            if hasattr(self, "token_selector") and self.token_selector is not None:
                self.token_selector.set_debug_mode(enabled)
                
            if hasattr(self, "token_generator") and self.token_generator is not None:
                self.token_generator.set_debug_mode(enabled)

    def _mcts_simulation(self, state, depth, c_puct):
        """
        Perform a single MCTS simulation for token selection.
        
        Args:
            state: Current MCTS state
            depth: Maximum simulation depth
            c_puct: Exploration constant
            
        Returns:
            List of (token_id, probability) tuples from the simulation
        """
        current_state = state.copy()
        simulation_tokens = []
        
        for _ in range(depth):
            # Get next token logits
            next_token_logits, new_past_key_values = self.token_generator.generate_next_token_with_cache(
                input_ids=current_state['input_ids'],
                attention_mask=current_state['attention_mask'],
                past_key_values=current_state['past_key_values']
            )
            
            # Update state
            current_state['past_key_values'] = new_past_key_values
            
            # Select token using UCB1 formula
            token_probs = torch.softmax(next_token_logits, dim=-1)
            token_values = torch.zeros_like(token_probs)
            
            # Calculate UCB1 values
            for i in range(token_probs.size(-1)):
                if token_probs[0, i] > 0:
                    # UCB1 formula: Q + c * sqrt(ln(N)/n)
                    # For simplicity, we use the probability as Q
                    token_values[0, i] = token_probs[0, i] + c_puct * torch.sqrt(
                        torch.log(torch.tensor(current_state['logical_step'] + 1)) / 
                        (token_probs[0, i] + 1e-8)
                    )
            
            # Select token with highest UCB1 value
            selected_token = torch.argmax(token_values).item()
            selected_prob = token_probs[0, selected_token].item()
            
            simulation_tokens.append((selected_token, selected_prob))
            
            # Update input_ids and attention_mask
            new_token = torch.tensor([[selected_token]], device=self.device)
            current_state['input_ids'] = torch.cat([current_state['input_ids'], new_token], dim=1)
            current_state['attention_mask'] = torch.cat([
                current_state['attention_mask'],
                torch.ones((1, 1), device=self.device)
            ], dim=1)
            current_state['logical_step'] += 1
            
        return simulation_tokens

    def _select_from_mcts_results(self, mcts_tokens, threshold):
        """
        Select tokens from MCTS simulation results based on threshold.
        
        Args:
            mcts_tokens: List of (token_id, probability) tuples from MCTS simulations
            threshold: Probability threshold for selection
            
        Returns:
            List of (token_id, probability) tuples above threshold
        """
        # Count occurrences of each token
        token_counts = {}
        for token_id, prob in mcts_tokens:
            if token_id not in token_counts:
                token_counts[token_id] = []
            token_counts[token_id].append(prob)
        
        # Calculate average probability for each token
        token_avg_probs = {
            token_id: sum(probs) / len(probs)
            for token_id, probs in token_counts.items()
        }
        
        # Filter tokens above threshold
        selected_tokens = [
            (token_id, avg_prob)
            for token_id, avg_prob in token_avg_probs.items()
            if avg_prob >= threshold
        ]
        
        # Sort by probability
        selected_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return selected_tokens

import torch
import random
import logging
import os
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

class ParallelGenerator:
    """
    Main class for parallel token generation with threshold pruning.
    Optimized for performance with batched operations.
    """

    def __init__(
        self,
        model,
        tokenizer,
        pruner=None,
        device: str = "mps",
        has_custom_attention: bool = True,
        use_custom_rope: bool = True,
        debug_mode: bool = True,
        token_generator: Optional[TokenGenerator] = None,
    ):
        """
        Initialize the parallel generator.

        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for the model
            pruner: Optional pruner for generations
            device: Device to use for computation
            has_custom_attention: Whether the model supports custom attention masks
            use_custom_rope: Whether to use custom RoPE modifications for parallel tokens
            debug_mode: Enable debug mode for detailed logging
            token_generator: Optional external TokenGenerator instance to use
        """
        # Store parameters
        self.model = model
        self.tokenizer = tokenizer
        self.pruner = pruner
        self.device = device
        self.has_custom_attention = has_custom_attention
        self.use_custom_rope = use_custom_rope
        
        # Set debug mode FIRST so everything initialized after gets this value
        self.debug_mode = debug_mode
        
        # Setup logging before other components
        self._setup_logger()
        
        # Track logical layout of parallel tokens
        self.logical_layout = []  # [(logical_pos, start_idx, end_idx)]
        
        # When debug mode is enabled, start with a clear log message
        if self.debug_mode:
            self.log("Initializing ParallelGenerator with debug mode ENABLED")
            # Also print to stdout for immediate visibility
            print("ParallelGenerator starting with debug mode ENABLED")

        # Validate required arguments
        assert model is not None, "Model cannot be None"
        assert tokenizer is not None, "Tokenizer cannot be None"
        assert device in ["cpu", "cuda", "mps"], f"Unsupported device: {device}"
        assert isinstance(has_custom_attention, bool), "has_custom_attention must be a boolean"
        assert isinstance(use_custom_rope, bool), "use_custom_rope must be a boolean"
        assert isinstance(debug_mode, bool), "debug_mode must be a boolean"
        
        # Check if this is a Qwen-based model
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            self.is_qwen_model = "qwen" in model.config.model_type.lower()
        else:
            self.is_qwen_model = False

        # Initialize RoPE modifier if requested
        self.rope_modifier = None
        if use_custom_rope:
            try:
                # Create and initialize RoPE modifier with appropriate debug mode
                self.rope_modifier = RoPEModifier(model=model, device=device)
                # Explicitly set debug mode based on our debug_mode value
                self.rope_modifier.set_debug_mode(self.debug_mode)
                
                # Install the modifier
                rope_install_success = self.rope_modifier.install()
                if rope_install_success:
                    print("Using custom RoPE modifications for parallel token positioning")
                    if self.debug_mode:
                        self.log("RoPE modifier initialized with debug mode ENABLED")
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
        
        self.token_generator.set_debug_mode(self.debug_mode)
        assert self.token_generator is not None, "Failed to initialize TokenGenerator"
        
        self.token_selector = TokenSelector(tokenizer)
        self.token_selector.set_debug_mode(self.debug_mode)
        assert self.token_selector is not None, "Failed to initialize TokenSelector"
        
        self.text_formatter = TextFormatter(tokenizer)
        assert self.text_formatter is not None, "Failed to initialize TextFormatter"

        # Create attention manager
        self.attention_manager = AttentionManager(
            device=device, rope_modifier=self.rope_modifier, tokenizer=tokenizer
        )
        self.attention_manager.set_debug_mode(self.debug_mode)
        
        # Log initialization messages only when debug mode is enabled
        if self.debug_mode:
            self.log("AttentionManager initialized with debug mode ENABLED")
            self.log("TokenSelector initialized with debug mode ENABLED")
            self.log("TokenGenerator initialized with debug mode ENABLED")

        # Link components for better coordination if RoPE modifier is available
        if self.use_custom_rope and self.rope_modifier is not None:
            # Link RoPE modifier to attention manager for enhanced capabilities
            self.attention_manager.set_rope_modifier(self.rope_modifier)

        # Connect TokenGenerator to pruning strategy if using CoherencePruningStrategy
        if self.pruner is not None and hasattr(self.pruner, "strategy"):
            if hasattr(self.pruner.strategy, "set_token_generator"):
                self.pruner.strategy.set_token_generator(self.token_generator)
                self.pruner.strategy.set_debug_mode(debug_mode)
                self.log(
                    "Connected TokenGenerator to pruning strategy for attention reuse"
                )

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

    def _setup_logger(self):
        """Setup logging to file."""
        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("parallel_generator")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(log_dir, "generation_debug.log")
        
        # Clear the log file by opening in write mode first
        with open(log_file, "w") as f:
            pass
            
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Verify logger setup
        assert self.logger.handlers, "Failed to setup logger handlers"

    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        assert message, "Log message cannot be empty"
        assert level in ["info", "debug", "warning", "error"], f"Invalid log level: {level}"
        
        if not self.debug_mode and level != "error":
            return

        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

    def _init_sequence_tracking(self, prompt_length):
        """Initialize sequence length tracking with the prompt length."""
        assert prompt_length >= 0, "Prompt length cannot be negative"
        
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
        threshold: Optional[float] = 0.1,
        return_parallel_sets: bool = False,
        use_pruning: bool = False,
        min_steps: int = 0,
        show_token_ids: bool = False,
        debug_mode: Optional[bool] = None,
        disable_kv_cache: bool = False,
        system_content: Optional[str] = None,
        isolate_parallel_tokens: bool = True,
        preserve_all_isolated_tokens: Optional[bool] = None,
        pruner: Optional[object] = None,
        retroactive_pruner: Optional[object] = None,
        sequence_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from prompt with parallel token generation using Sequential Layout approach.
        
        This implementation uses a single sequence with logical positions mapping to potentially
        multiple physical positions in the sequence.

        Args:
            prompt: The text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Probability threshold for selecting tokens (default: 0.1)
            return_parallel_sets: Whether to return parallel token sets for visualization
            use_pruning: Whether to use pruning on parallel groups
            min_steps: Minimum number of steps before stopping
            show_token_ids: Include token IDs in return for debugging
            debug_mode: Override the instance debug_mode setting
            disable_kv_cache: Whether to disable KV caching
            system_content: System content for chat prompts
            isolate_parallel_tokens: Whether tokens at the same position cannot attend to each other
            preserve_all_isolated_tokens: Override default pruning behavior for isolated tokens
            pruner: Pruner to use for token selection (overrides self.pruner if provided)
            retroactive_pruner: Retroactive pruner to use for token selection
            sequence_callback: Callback function for sequence length updates

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
        assert threshold is None or (0.0 <= threshold <= 1.0), "threshold must be between 0.0 and 1.0"
        if threshold is None:
            threshold = 0.1  # Default threshold
        assert min_steps >= 0, "min_steps cannot be negative"
        
        # Which pruner to use - instance pruner or parameter pruner
        active_pruner = pruner if pruner is not None else self.pruner
        
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
                elif "qwen" in model_type or getattr(self, 'is_qwen_model', False):
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

            # 3. Select Parallel Candidates T
            try:
                # Get top token distribution
                token_distribution, subset_size = self.token_selector.select_tokens(
                    next_token_logits, threshold=threshold
                )
                
                if not token_distribution:
                    self.log(f"Warning: No tokens above threshold {threshold} at step {logical_step}. Falling back.", "warning")
                    token_distribution, _ = self.token_selector.select_tokens(next_token_logits, threshold=0.0, max_tokens=1)
                    if not token_distribution:
                        self.log("Critical error: Still no tokens selected. Stopping.", "error")
                        break
                        
                self.log(f"Selected {len(token_distribution)} candidate tokens for logical step {logical_step}.")
                
                # Store original candidates
                original_candidates = [(tid.item(), float(prob)) for tid, prob in token_distribution]
                all_original_token_sets[logical_step] = original_candidates
                
            except ValueError as e:
                self.log(f"Error selecting tokens at step {logical_step}: {e}. Logits shape: {next_token_logits.shape}", "error")
                break

            # 4. Apply Pruning to get T'
            pruned_distribution = token_distribution  # Start with original candidates
            if use_pruning and active_pruner is not None and len(token_distribution) > 1:
                pruning_start = time.time()
                try:
                    # Extract token ids and probs for pruner
                    token_ids = [tid.item() for tid, _ in token_distribution]
                    token_probs = [float(prob) for _, prob in token_distribution]
                    
                    # Get context from full input_ids
                    context_ids = input_ids[0].tolist()
                    
                    # Apply pruning
                    if hasattr(active_pruner, 'prune'):
                        # Use the prune method if available
                        prune_results = active_pruner.prune(
                            token_ids=token_ids,
                            token_probs=token_probs,
                            token_logits=[next_token_logits[0]],
                            step=logical_step,
                            position_history=self.attention_manager.get_token_history() if hasattr(self.attention_manager, 'get_token_history') else []
                        )
                        
                        # Extract pruned indices
                        if isinstance(prune_results, tuple):
                            pruned_indices = prune_results[0]
                        else:
                            pruned_indices = prune_results
                            
                        # If all tokens were pruned, this is a problem
                        if len(pruned_indices) == len(token_distribution):
                            self.log("Warning: All tokens were pruned, selecting the highest probability token", "warning")
                            pruned_indices = []  # Don't prune anything
                            
                        # Keep only non-pruned tokens
                        pruned_distribution = [token_distribution[i] for i in range(len(token_distribution)) if i not in pruned_indices]
                        
                    elif hasattr(active_pruner, 'strategy') and hasattr(active_pruner.strategy, 'prune_tokens'):
                        # Use the strategy's prune_tokens method - IMPORTANT: Pass tensor not list
                        pruned_tuples = active_pruner.strategy.prune_tokens(
                            input_ids,  # Pass the tensor directly, not the list
                            [(tid.item(), float(prob)) for tid, prob in token_distribution]
                        )
                        
                        # If pruning returned some tokens
                        if pruned_tuples:
                            pruned_distribution = [(torch.tensor(tid, device=self.device), prob) for tid, prob in pruned_tuples]
                            self.log(f"Pruning kept {len(pruned_distribution)} tokens out of {len(token_distribution)}.")
                        else:
                            self.log("Warning: Pruning removed all tokens, keeping highest prob original.", "warning")
                            pruned_distribution = [token_distribution[0]]  # Keep best original
                    
                except Exception as e:
                    self.log(f"Error during pruning at step {logical_step}: {e}", "error")
                    self.log(traceback.format_exc(), "error")
                    # Fallback to original distribution
                    pruned_distribution = token_distribution
                    
                pruning_time += time.time() - pruning_start
                pruning_steps += 1
                
                # Store pruned results for this step
                pruned_candidates = [(tid.item(), float(prob)) for tid, prob in pruned_distribution]
                all_pruned_token_sets[logical_step] = pruned_candidates

            # Apply retroactive pruning if available
            if retroactive_pruner is not None and logical_step > 0:  # Can only prune previous steps
                retroactive_pruning_start = time.time()
                try:
                    # Update retroactive pruner with current step
                    if hasattr(retroactive_pruner, 'update_step'):
                        retroactive_pruner.update_step(logical_step)

                    # Prune history before adding current tokens
                    if hasattr(retroactive_pruner, 'retroactively_prune'):
                        pruned_history = retroactive_pruner.retroactively_prune(
                            prompt_length=prompt_tokens,
                            all_parallel_tokens=all_original_token_sets,  # Pass history of originals
                            step=logical_step  # Prune based on attention from predicting this step
                        )
                        # Update the pruned sets history (used for formatting later)
                        all_pruned_token_sets.update(pruned_history)
                        self.log(f"Retroactive pruning applied up to step {logical_step-1}.")
                
                except Exception as e:
                    self.log(f"Error during retroactive pruning at step {logical_step}: {e}", "error")
                    self.log(traceback.format_exc(), "error")
                    
                pruning_time += time.time() - retroactive_pruning_start

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
            "is_qwen_model": getattr(self, 'is_qwen_model', False),
            "had_repetition_loop": had_repetition_loop,
            "prompt": prompt,
            "threshold": threshold,
            "use_pruning": use_pruning,
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
        if use_pruning:
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

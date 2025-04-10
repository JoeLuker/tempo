import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from .token_generator import TokenGenerator
from .token_selector import TokenSelector
from .text_formatter import TextFormatter
from .attention_manager import AttentionManager
from .rope_modifier import RoPEModifier
import time
import traceback
import logging
import os

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
        debug_mode: bool = False
    ):
        """
        Initialize the parallel generator.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            pruner: Optional pruner for reducing token sets
            device: Device to use for computation
            has_custom_attention: Whether the model supports custom attention masks
            use_custom_rope: Whether to use custom RoPE modifications
            debug_mode: Enable debug mode for detailed logging
        """
        # Invariant: Model and tokenizer must be provided
        if model is None:
            raise ValueError("Invariant violation: Model cannot be None")
        if tokenizer is None:
            raise ValueError("Invariant violation: Tokenizer cannot be None")
            
        self.model = model
        self.tokenizer = tokenizer
        self.pruner = pruner
        self.device = device
        self.has_custom_attention = has_custom_attention
        self.use_custom_rope = use_custom_rope
        self.debug_mode = debug_mode
        
        # Setup logging
        self._setup_logger()
        
        # Determine if model is Qwen-based
        self.is_qwen_model = False
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            self.is_qwen_model = "qwen" in model.config.model_type.lower()
            if self.debug_mode:
                self.log(f"Detected model type: {model.config.model_type}, is_qwen={self.is_qwen_model}")
        
        # Initialize component classes - order matters here!
        # First create the RoPE modifier if requested
        self.rope_modifier = None
        if use_custom_rope:
            # Invariant: RoPE modifier must initialize successfully when requested
            self.rope_modifier = RoPEModifier(model, device)
            self.rope_modifier.set_debug_mode(debug_mode)
        
        # Initialize other components
        self.token_generator = TokenGenerator(model, tokenizer, device)
        self.token_selector = TokenSelector(tokenizer)
        self.text_formatter = TextFormatter(tokenizer)
        
        # Initialize attention manager with RoPE modifier reference
        self.attention_manager = AttentionManager(device, self.rope_modifier, tokenizer)
        self.attention_manager.set_debug_mode(debug_mode)
        
        # Install RoPE modifier after all components are initialized
        if use_custom_rope and self.rope_modifier is not None:
            # Now install the RoPE modifier
            # Invariant: RoPE modifier must install successfully when requested
            self.rope_modifier.install()
            self.log("Installed custom RoPE modifier for parallel token positions")
            
            # Link components for better coordination
            self.attention_manager.set_rope_modifier(self.rope_modifier)
        
        # Performance tracking
        self.generation_time = 0
        self.pruning_time = 0
    
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
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.
        
        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
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
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        threshold: Optional[float] = None,
        return_parallel_sets: bool = False,
        use_pruning: bool = False,
        require_custom_attention: bool = False,
        min_steps: int = 0,
        show_token_ids: bool = False,
        debug_mode: bool = False,
        disable_kv_cache: bool = False,
        system_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using parallel threshold decoding.
        Optimized for batched operations and memory efficiency.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Probability threshold for token selection
            return_parallel_sets: Whether to return parallel token sets in result
            use_pruning: Whether to use pruning to reduce token sets
            require_custom_attention: Whether to require custom attention masks
            min_steps: Minimum number of steps to generate before considering EOS
            show_token_ids: Whether to show token IDs in the output
            debug_mode: Enable debug mode for detailed logging
            disable_kv_cache: Disable KV caching for more consistent attention
            system_content: Optional system message content for chat models
            
        Returns:
            Dict[str, Any]: Results dictionary with generated text and metadata
        """
        # Set debug mode if requested
        if debug_mode:
            self.debug_mode = True
            if self.rope_modifier is not None:
                self.rope_modifier.set_debug_mode(True)
            self.attention_manager.set_debug_mode(True)
            # Enable debug mode for TokenSelector too
            self.token_selector.set_debug_mode(True)
            # Enable debug mode for TokenGenerator too
            self.token_generator.set_debug_mode(True)
            # Log to file instead of console
            self.log("Debug mode enabled for generation - logging to files in logs/ directory")
            # Print minimal console message
            print("Debug mode enabled - logging to files in logs/ directory")
            
        # Performance tracking
        start_time = time.time()
        
        # Set default threshold if not specified
        if threshold is None:
            threshold = 0.1
            
        # Validate custom attention requirement
        if require_custom_attention and not self.has_custom_attention:
            raise ValueError("Custom attention is required but model doesn't support it")
            
        # Reset pruner if using dynamic threshold
        if use_pruning and self.pruner is not None:
            if hasattr(self.pruner, 'reset'):
                self.pruner.reset()
            
            # Set max steps in pruner if using dynamic threshold
            if hasattr(self.pruner, 'use_dynamic_threshold') and self.pruner.use_dynamic_threshold:
                # Set the maximum steps for the dynamic threshold
                self.pruner.max_steps = max_tokens
        
        # Reset RoPE modifier position mapping
        if self.rope_modifier is not None:
            self.rope_modifier.reset()
            
        # Reset attention manager
        self.attention_manager.reset_cache()
        
        # For Cogito models with thinking mode enabled, we need special handling
        is_thinking_mode = system_content is not None and "thinking" in system_content.lower()
        if is_thinking_mode and self.is_qwen_model:
            if self.debug_mode:
                self.log("Using special handling for Cogito thinking mode")
            
            # Thinking mode works better with pruning
            if not use_pruning:
                self.log("Warning: Thinking mode works better with pruning. Consider adding --use-pruning flag.")
                
            # Thinking mode may need a different threshold for stable generation
            if threshold > 0.08:
                self.log(f"Note: Using threshold {threshold} for thinking mode (values below 0.08 often work better)")
        
        # Prepare input based on whether we're using chat format or raw prompt
        if system_content is not None:
            # Format input as chat for Cogito model
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # DIAGNOSTIC: Print messages being sent to the model
            self.log("\nDIAGNOSTIC - Chat messages:")
            for msg in messages:
                self.log(f"  {msg['role']}: {msg['content'][:100]}...")
                
            # Check if the tokenizer supports apply_chat_template with enable_thinking
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    # Try to use the tokenizer's chat template with enable_thinking if available
                    if "enable_thinking" in self.tokenizer.apply_chat_template.__code__.co_varnames:
                        # Use the tokenizer's chat template with enable_thinking
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True
                        )
                        self.log("\nDIAGNOSTIC - Using chat template with enable_thinking=True")
                    else:
                        # Use regular chat template without enable_thinking
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        self.log("\nDIAGNOSTIC - Using standard chat template (no enable_thinking parameter)")
                        
                    # DIAGNOSTIC: Show the formatted prompt text
                    self.log(f"\nDIAGNOSTIC - Formatted prompt text (first 100 chars):\n{prompt_text[:200]}...")
                    
                    input_ids, attention_mask = self.token_generator.prepare_input_from_prompt(prompt_text)
                    
                    # DIAGNOSTIC: Show token IDs and decoded tokens
                    self.log("\nDIAGNOSTIC - First 10 input token IDs:")
                    for i in range(min(10, len(input_ids[0]))):
                        token_id = input_ids[0, i].item()
                        token_text = self.tokenizer.decode([token_id])
                        self.log(f"  Token {i}: ID={token_id}, Text='{token_text}'")
                        
                    self.log(f"\nDIAGNOSTIC - Total input length: {len(input_ids[0])} tokens")
                    
                except Exception as e:
                    self.log(f"\nDIAGNOSTIC - Error in chat template processing: {e}")
                    traceback.print_exc()
                    raise RuntimeError(f"Error applying chat template: {e}. Generation failed.")

            else:
                # Invariant: Chat template must be applied successfully
                raise RuntimeError("Chat template application failed. The tokenizer doesn't support the required chat template functionality.")
        else:
            # Standard input preparation
            input_ids, attention_mask = self.token_generator.prepare_input_from_prompt(prompt)
        
        # Pre-allocate storage for token sets (more memory efficient)
        # We'll only store the minimal necessary information and convert formats as needed
        token_sets = []  # List of (position, token_ids, token_probs) tuples
        
        # Track positions with multiple tokens
        original_parallel_positions = set()
        
        # More efficient position_to_tokens mapping using direct indices
        position_to_tokens = {}
        prompt_length = len(input_ids[0])
        
        # Add prompt tokens to the position mapping with efficient batch processing
        for i in range(prompt_length):
            position_to_tokens[i] = [input_ids[0, i].item()]
        
        # Use KV cache for faster generation
        past_key_values = None
        
        # Position mapping for RoPE modification
        rope_position_map = {}
        
        # Flag to track if we're currently in a repetition loop
        in_repetition_loop = False
        repetition_count = 0
        last_tokens = []
        
        # Iteratively generate tokens - optimized for speed
        for i in range(max_tokens):
            # Get next token logits from the model with KV caching - simplified for performance
            # DIAGNOSTIC: Track step number and input shape for each token generation
            self.log(f"\nDIAGNOSTIC - Starting token generation step {i}")
            self.log(f"  Input shape: {input_ids.shape}")
            if i > 0:  # First generated token after prompt
                self.log(f"  Input tokens at step {i}:")
                for j in range(min(input_ids.shape[1], 10)):
                    token_id = input_ids[0, j].item()
                    token_text = self.tokenizer.decode([token_id])
                    self.log(f"    {j}: ID={token_id}, Text='{token_text}'")
                
            next_token_logits, past_key_values = self.token_generator.get_next_token_logits_cached(
                input_ids, 
                attention_mask,
                None if disable_kv_cache else past_key_values,
                self.attention_manager.full_attention_mask if self.has_custom_attention else None
            )
            
            # Get tokens above threshold using optimized tensor operations
            next_token_ids, next_token_probs = self.token_selector.select_tokens_above_threshold(
                next_token_logits, threshold
            )
            
            # Invariant: Token IDs and probabilities must have same length and proper structure
            if len(next_token_ids) != len(next_token_probs):
                raise ValueError(f"Invariant violation: Mismatch between token IDs ({len(next_token_ids)}) and probabilities ({len(next_token_probs)})")
                
            # Invariant: Probabilities must be valid values between 0 and 1
            if any(prob <= 0 or prob > 1.0 for prob in next_token_probs):
                raise ValueError("Invariant violation: Token probabilities must be between 0 and 1")
            
            # Invariant: If multiple tokens are selected, they must have decreasing probabilities
            if len(next_token_ids) > 1:
                # Vectorized check for descending order using tensor operations
                probs_tensor = torch.tensor(next_token_probs, device=self.device)
                if not torch.all(probs_tensor[:-1] >= probs_tensor[1:]):
                    raise ValueError("Invariant violation: Token probabilities must be in descending order")
            
            # Skip if no tokens above threshold
            if not next_token_ids:
                # DEBUG: Show top tokens even when below threshold
                top_token_ids, top_token_probs = self.token_selector.select_top_tokens(next_token_logits, top_k=5)
                top_tokens_text = [self.tokenizer.decode([tid]) for tid in top_token_ids]
                
                self.log("\nDEBUG: No tokens above threshold. Top 5 tokens and probabilities:")
                for idx, (token_text, token_id, prob) in enumerate(zip(top_tokens_text, top_token_ids, top_token_probs)):
                    self.log(f"  {idx+1}. '{token_text}' (ID: {token_id}): {prob:.6f}")
                self.log(f"Current threshold: {threshold}")
                
                # If thinking mode is active, also show special note
                if is_thinking_mode:
                    self.log("Note: Thinking mode often requires lower thresholds (0.01-0.05)")
                    self.log("Try running with --threshold 0.05 or --threshold 0.03 for thinking mode")
                
                raise RuntimeError(f"No tokens above threshold at step {i}. Generation failed.")
            
            # Skip single EOS token if this isn't the last step and we haven't reached min_steps
            if (len(next_token_ids) == 1 and 
                self.token_selector.is_eos_token(next_token_ids[0]) and
                i < max_tokens - 1 and
                i < min_steps):
                continue
                
            # Check for repetition patterns
            current_token = next_token_ids[0] if next_token_ids else None
            if current_token is not None:
                # Add to last tokens
                last_tokens.append(current_token)
                if len(last_tokens) > 5:
                    last_tokens.pop(0)
                
                # Check for repetition
                if len(last_tokens) >= 3:
                    if last_tokens[-1] == last_tokens[-2] == last_tokens[-3]:
                        repetition_count += 1
                        if repetition_count >= 3:
                            in_repetition_loop = True
                            self.log(f"Detected repetition loop at step {i}, applying correction")
                            # For thinking mode, we need to force a diverse token
                            if is_thinking_mode:
                                # Remove repeated token from options
                                repeated_token = last_tokens[-1]
                                next_token_ids = [t for t in next_token_ids if t != repeated_token]
                                if not next_token_ids:
                                    # If no tokens left, get new ones excluding repeated token
                                    next_token_ids, next_token_probs = self.token_selector.select_tokens_above_threshold_excluding(
                                        next_token_logits, threshold * 0.8, [repeated_token]
                                    )
                    else:
                        repetition_count = 0
                        in_repetition_loop = False
            
            # Store tokens efficiently - only store the raw data
            # We'll decode only when needed to save computation
            original_tokens = list(zip(next_token_ids, next_token_probs))
                
            # If we have multiple tokens, mark this as a parallel position
            current_position = len(position_to_tokens) - prompt_length
            if len(next_token_ids) > 1:
                original_parallel_positions.add(current_position)
                
                # Update RoPE position mapping for parallel tokens
                if self.rope_modifier is not None and len(next_token_ids) > 1:
                    # Create position mapping for all tokens in the parallel set
                    position_mapping = {}
                    current_pos = prompt_length + i
                    for j in range(len(next_token_ids)):
                        position_mapping[current_pos + j] = current_pos
                    
                    # Register with RoPE modifier
                    self.rope_modifier.register_parallel_positions(position_mapping)
                
            # Create copy of original tokens for pruning
            pruned_tokens = original_tokens.copy()
                
            # Apply pruning if requested and available
            pruning_start = time.time()
            if use_pruning and self.pruner is not None and len(pruned_tokens) > 1:
                # Invariant: Pruning must succeed when requested
                pruned_result = self.pruner.prune_parallel_tokens(
                    input_ids=input_ids,
                    parallel_tokens=pruned_tokens
                )
                
                # Extract results
                if pruned_result and isinstance(pruned_result, tuple) and len(pruned_result) >= 1:
                    pruned_tokens = pruned_result[0]
                else:
                    raise RuntimeError("Pruning returned invalid result format")
            self.pruning_time += time.time() - pruning_start
            
            # Store token set info more efficiently
            token_sets.append((
                len(position_to_tokens) - prompt_length,  # Position
                original_tokens,  # Original tokens
                pruned_tokens  # Pruned tokens
            ))
            
            # Add pruned tokens to position_to_tokens mapping
            pruned_token_ids = [t for t, _ in pruned_tokens]
            position_to_tokens[prompt_length + i] = pruned_token_ids
            
            # Create new input representation with the pruned tokens - more efficient approach
            # Invariant: Attention update must succeed
            # Pass disable_kv_cache flag to ensure proper context handling
            input_ids, attention_mask, past_key_values = self.attention_manager.update_input_efficiently(
                input_ids, attention_mask, 
                None if disable_kv_cache else past_key_values,  # Pass None explicitly if KV cache is disabled
                pruned_token_ids,
                is_kv_cache_disabled=disable_kv_cache  # Pass flag for explicit handling
            )
            
            # Special handling for thinking mode with Qwen/Cogito models
            if is_thinking_mode and self.is_qwen_model and i > 10 and (i % 20 == 0):
                # Periodically reset KV cache to prevent issues in long thinking chains
                if not disable_kv_cache:
                    self.log(f"Resetting KV cache at step {i} for thinking mode stability")
                    past_key_values = None
            
            # Stop generation if all tokens are EOS and we've reached min_steps
            if (all(self.token_selector.is_eos_token(t) for t in pruned_token_ids) and
                len(pruned_token_ids) > 0 and
                i >= min_steps):
                self.log(f"Stopping because all tokens are EOS after {i+1} steps (min_steps={min_steps})")
                break
        
        # Update position_to_tokens with final pruned sets if using dynamic threshold
        if (use_pruning and self.pruner is not None and 
            hasattr(self.pruner, 'use_dynamic_threshold') and self.pruner.use_dynamic_threshold):
            # Get final pruned sets - optimize this to avoid recomputing everything
            final_pruned_sets = self.pruner.get_final_pruned_sets()
            
            # Update position_to_tokens with batch update
            for step, pruned_set in enumerate(final_pruned_sets):
                position = prompt_length + step
                if position in position_to_tokens:
                    position_to_tokens[position] = [t[0] for t in pruned_set]
        
        # Format the generated text - only decode tokens once
        if show_token_ids:
            formatted_text = self.text_formatter.format_with_token_ids(
                prompt,
                position_to_tokens, 
                original_parallel_positions, 
                prompt_length,
                {}  # Add empty token_indices if not tracking them
            )
        else:
            formatted_text = self.text_formatter.format_generated_text(
                prompt,
                position_to_tokens, 
                original_parallel_positions, 
                prompt_length,
                {}  # Add empty token_indices if not tracking them
            )
        
        # Generate raw text efficiently - single decoding operation
        token_sequence = []
        # Invariant: input_ids must have a valid shape and be processable
        if len(input_ids.shape) > 1 and input_ids.shape[1] > 0:
            # Standard case: batch x sequence
            for i in range(min(prompt_length, input_ids.shape[1])):
                token_sequence.append(input_ids[0, i].item())
        else:
            # Single token case or reshaped tensor
            if len(input_ids.shape) == 1:
                token_sequence.append(input_ids.item())
            else:
                raise ValueError("Unexpected input_ids shape")
        
        # Add generated tokens
        for pos in sorted(position_to_tokens.keys()):
            if pos >= prompt_length:  # Only add tokens after the prompt
                token_sequence.extend(position_to_tokens[pos])
        
        # Batch decode the raw generated text - much faster than token-by-token
        raw_generated_text = self.tokenizer.decode(token_sequence, skip_special_tokens=True)
        
        # Total generation time
        self.generation_time = time.time() - start_time
        
        # Prepare results dictionary - optimize for memory by only including what's needed
        results = {
            "generated_text": formatted_text,
            "raw_generated_text": raw_generated_text,
            "prompt": prompt,
            "threshold": threshold,
            "use_pruning": use_pruning,
            "min_steps": min_steps,
            "generation_time": self.generation_time,
            "pruning_time": self.pruning_time,
            "use_custom_rope": self.use_custom_rope,
            "system_content": system_content,
            "is_qwen_model": self.is_qwen_model,
            "had_repetition_loop": in_repetition_loop
        }
        
        # Add parallel sets data only if requested to save memory
        if return_parallel_sets:
            # Efficiently convert to human-readable format only when needed
            token_id_map = {}
            
            def get_token_text(token_id):
                # Cache token decoding to avoid repeated calls
                if token_id not in token_id_map:
                    token_id_map[token_id] = self.tokenizer.decode([token_id], skip_special_tokens=False)
                return token_id_map[token_id]
            
            # Only include the minimal necessary data for visualization
            if use_pruning and self.pruner is not None:
                # Add pruned information in a memory-efficient way
                position_info = {}
                for pos, tokens in position_to_tokens.items():
                    if pos >= prompt_length:  # Only include generated tokens
                        # Batch decode tokens
                        position_info[str(pos)] = [get_token_text(t) for t in tokens]
                
                results["position_to_tokens"] = position_info
            
                # Include raw token sets only if specifically needed
                if hasattr(self.pruner, 'use_dynamic_threshold') and self.pruner.use_dynamic_threshold:
                    pruned_sets = self.pruner.get_final_pruned_sets()
                    results["final_pruned_sets"] = pruned_sets
                
            # Add position to tokens mapping for visualization
            position_info = {}
            for pos, tokens in position_to_tokens.items():
                if pos >= prompt_length:  # Only include generated tokens
                    decoded_tokens = []
                    for t in tokens:
                        try:
                            if isinstance(t, int):
                                decoded_tokens.append(self.tokenizer.decode([t]))
                            else:
                                # Skip invalid tokens
                                pass
                        except Exception:
                            # Skip on any decoding error
                            pass
                    position_info[str(pos)] = decoded_tokens
            results["position_to_tokens"] = position_info
        
        # Add model internal diagnostics when in debug mode
        if self.debug_mode and hasattr(self.model, "intermediate_values"):
            # Add keys of captured intermediate values
            results["intermediate_value_keys"] = list(self.model.intermediate_values.keys())
        
        return results 
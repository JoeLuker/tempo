import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from .token_generator import TokenGenerator
from .token_selector import TokenSelector
from .text_formatter import TextFormatter
from .attention_manager import AttentionManager
from .rope_modifier import RoPEModifier
import time

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
        use_custom_rope: bool = True
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
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pruner = pruner
        self.device = device
        self.has_custom_attention = has_custom_attention
        self.use_custom_rope = use_custom_rope
        
        # Initialize component classes - order matters here!
        # First create the RoPE modifier if requested
        self.rope_modifier = None
        if use_custom_rope:
            try:
                self.rope_modifier = RoPEModifier(model, device)
            except Exception as e:
                print(f"Warning: Failed to initialize RoPE modifier: {e}")
                print("Continuing without RoPE modification")
                self.use_custom_rope = False
        
        # Initialize other components
        self.token_generator = TokenGenerator(model, tokenizer, device)
        self.token_selector = TokenSelector(tokenizer)
        self.text_formatter = TextFormatter(tokenizer)
        
        # Initialize attention manager with RoPE modifier reference
        self.attention_manager = AttentionManager(device, self.rope_modifier)
        
        # Install RoPE modifier after all components are initialized
        if use_custom_rope and self.rope_modifier is not None:
            # Now install the RoPE modifier
            try:
                self.rope_modifier.install()
                print("Installed custom RoPE modifier for parallel token positions")
                
                # Link components for better coordination
                self.attention_manager.set_rope_modifier(self.rope_modifier)
            except Exception as e:
                print(f"Warning: Failed to install RoPE modifier: {e}")
                print("Continuing without RoPE modification")
                self.rope_modifier = None
                self.use_custom_rope = False
    
        # Performance tracking
        self.generation_time = 0
        self.pruning_time = 0
        
        # Debug setting
        self.debug_mode = False
    
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
        disable_kv_cache: bool = False
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
            
        Returns:
            Dict[str, Any]: Results dictionary with generated text and metadata
        """
        # Set debug mode if requested
        if debug_mode:
            self.debug_mode = True
            if self.rope_modifier is not None:
                self.rope_modifier.set_debug_mode(True)
            self.attention_manager.set_debug_mode(True)
            print("Debug mode enabled for generation")
            
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
        
        # Prepare input from prompt - make this more efficient with faster tokenization
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
        
        # Iteratively generate tokens - optimized for speed
        for i in range(max_tokens):
            # Get next token logits from the model with KV caching - simplified for performance
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
            
            # Skip if no tokens above threshold
            if not next_token_ids:
                continue
            
            # Skip single EOS token if this isn't the last step and we haven't reached min_steps
            if (len(next_token_ids) == 1 and 
                self.token_selector.is_eos_token(next_token_ids[0]) and
                i < max_tokens - 1 and
                i < min_steps):
                continue
                
            # Store tokens efficiently - only store the raw data
            # We'll decode only when needed to save computation
            original_tokens = list(zip(next_token_ids, next_token_probs))
                
            # If we have multiple tokens, mark this as a parallel position
            current_position = len(position_to_tokens) - prompt_length
            if len(next_token_ids) > 1:
                original_parallel_positions.add(current_position)
                
                # Update RoPE position mapping for parallel tokens
                if self.rope_modifier is not None and len(next_token_ids) > 1:
                    # Let the attention manager handle this, as it now coordinates with RoPE
                    # This is now handled within create_parallel_set_input and update_input_efficiently
                    pass
                
            # Create copy of original tokens for pruning
            pruned_tokens = original_tokens.copy()
                
            # Apply pruning if requested and available
            pruning_start = time.time()
            if use_pruning and self.pruner is not None and len(pruned_tokens) > 1:
                try:
                    # Call the pruner with proper error handling
                    pruned_result = self.pruner.prune_parallel_tokens(
                        input_ids=input_ids,
                        parallel_tokens=pruned_tokens
                    )
                    
                    # Extract results
                    if pruned_result and isinstance(pruned_result, tuple) and len(pruned_result) >= 1:
                        pruned_tokens = pruned_result[0]
                except Exception as e:
                    print(f"Pruning failed: {e}")
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
            try:
                # Only update the necessary parts of the input tensor to avoid reallocation
                input_ids, attention_mask, past_key_values = self.attention_manager.update_input_efficiently(
                    input_ids, attention_mask, past_key_values, pruned_token_ids
                )
            except Exception as e:
                # Fall back to simpler approach if anything goes wrong
                print(f"Attention update failed: {e}. Using fallback.")
                if pruned_token_ids and len(pruned_token_ids) > 0:
                    # Create a clean input with the first token only
                    new_input_ids = torch.tensor([[pruned_token_ids[0]]], device=self.device)
                    input_ids = new_input_ids
                    
                    # Reset KV cache in fallback case
                    past_key_values = None
                    # Use a simple attention mask
                    attention_mask = torch.ones((1, 1), device=self.device)
                    print(f"Reset KV cache and restarted with single token.")
                else:
                    # If no tokens are available, just keep the existing input_ids
                    print(f"Warning: No tokens available for fallback: {e}")
            
            # Stop generation if all tokens are EOS and we've reached min_steps
            if (all(self.token_selector.is_eos_token(t) for t in pruned_token_ids) and
                len(pruned_token_ids) > 0 and
                i >= min_steps):
                print(f"Stopping because all tokens are EOS after {i+1} steps (min_steps={min_steps})")
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
        try:
            # Handle different tensor shapes safely
            if len(input_ids.shape) > 1 and input_ids.shape[1] > 0:
                # Standard case: batch x sequence
                for i in range(min(prompt_length, input_ids.shape[1])):
                    token_sequence.append(input_ids[0, i].item())
            else:
                # Single token case or reshaped tensor
                if len(input_ids.shape) == 1:
                    token_sequence.append(input_ids.item())
                else:
                    # Fall back to position_to_tokens for prompt
                    for i in range(prompt_length):
                        if i in position_to_tokens:
                            token_sequence.extend(position_to_tokens[i])
        except Exception as e:
            print(f"Warning: Error processing input_ids: {e}. Using position_to_tokens fallback.")
            # Fall back to position_to_tokens completely
            for i in range(prompt_length):
                if i in position_to_tokens:
                    token_sequence.extend(position_to_tokens[i])
            
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
            "use_custom_rope": self.use_custom_rope
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
            
        return results 
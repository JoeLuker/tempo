# Parallel Token Generation with Custom Attention Masks
# 
# IMPORTANT NOTE:
# This implementation requires a modified transformer model that accepts a 'custom_attention_mask' parameter
# in its forward method. The custom_attention_mask should be a 3D tensor of shape (batch_size, seq_len, seq_len)
# where a value of 1.0 means the token can attend to another token, and 0.0 means it cannot.
#
# To modify a huggingface transformer model to accept this parameter, you'll need to:
# 1. Subclass the model and override the forward method
# 2. Modify the forward method to use the custom_attention_mask when provided
# 3. Pass the custom_attention_mask to each layer's attention mechanism
#
# Example modification for a transformer model:
#
# ```python
# class CustomTransformerModel(AutoModelForCausalLM):
#     def forward(self, input_ids, attention_mask=None, custom_attention_mask=None, **kwargs):
#         if custom_attention_mask is not None:
#             # Use the custom attention mask
#             # This would require modifying how attention is computed in each layer
#             return super().forward(input_ids, attention_mask=attention_mask, 
#                                   custom_attention_pattern=custom_attention_mask, **kwargs)
#         else:
#             return super().forward(input_ids, attention_mask=attention_mask, **kwargs)
# ```
#
# The specific implementation details will vary depending on the transformer architecture.

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from colorama import Fore, Style, just_fix_windows_console
import os
import platform

# Initialize colorama with the safer, newer approach
# This makes Windows act like Unix with respect to ANSI escape handling
just_fix_windows_console()

# Set terminal capabilities for better color support
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'
if platform.system() == 'Darwin':  # macOS
    print("Terminal should support colors with just_fix_windows_console()")

# Define a list of colors to cycle through for parallel tokens
COLORS = [
    Fore.LIGHTYELLOW_EX,  # Replacing red with light yellow as first color
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTMAGENTA_EX,  # Light purple/pink
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTGREEN_EX,
]

class ParallelThresholdGenerator:
    """
    Text generator that produces multiple tokens at each position based on a probability threshold.
    
    This approach outputs all tokens above a certain probability threshold at each step,
    providing a more nuanced view of the model's predictions and handling ambiguity better.
    
    Features:
    - Outputs tokens with probabilities above a threshold at each generation step
    - Tracks parallel tokens at the same position for analysis or visualization
    - Optional retroactive pruning to prune parallel tokens based on additional criteria
    - Uses custom attention masking to allow all tokens at position N to attend to each other
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device="cpu", 
        threshold=0.005,
        pruner=None
    ):
        """
        Initialize the parallel text generator.
        
        Args:
            model: Hugging Face transformer model (should be a CustomParallelAttentionModel for full functionality)
            tokenizer: Hugging Face tokenizer
            device: Device to use (cuda, cpu, etc.)
            threshold: Probability threshold for token selection (default: 0.005)
            pruner: Optional RetroactivePruner for pruning parallel tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.pruner = pruner
        
        # Check if we have a custom model that supports custom attention masks
        # Look for the has_custom_attention_support attribute from our CustomParallelAttentionModel
        self.has_custom_attention = (
            hasattr(model, 'has_custom_attention_support') and 
            model.has_custom_attention_support
        )
        
        if self.has_custom_attention:
            print("Using custom parallel attention masking")
        else:
            print("Custom attention masking not available - parallel tokens will be independent")
        
    def _get_parallel_tokens(
        self, 
        logits: torch.Tensor, 
        threshold: float
    ) -> Tuple[List[int], List[float]]:
        """
        Get tokens that exceed the probability threshold.
        
        Args:
            logits: Raw logits from the model (batch_size, vocab_size)
            threshold: Probability threshold for selection
            
        Returns:
            tuple: (token_ids, probabilities)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get tokens above threshold
        tokens_above_threshold = torch.where(probs > threshold)[1]
        selected_probs = probs[0, tokens_above_threshold]
        
        # Sort by probability (highest first)
        sorted_indices = torch.argsort(selected_probs, descending=True)
        
        # Get token IDs and probabilities as integers and floats
        tokens = [int(token_id.item()) for token_id in tokens_above_threshold[sorted_indices]]
        probabilities = [float(prob.item()) for prob in selected_probs[sorted_indices]]
        
        # If no tokens above threshold, get the single highest probability token
        if not tokens:
            # Get the token with highest probability
            max_prob_token = torch.argmax(probs, dim=-1).item()
            max_prob = probs[0, max_prob_token].item()
            tokens = [int(max_prob_token)]
            probabilities = [float(max_prob)]
        
        return tokens, probabilities
    
    def _create_parallel_set_input(
        self,
        base_input_ids: torch.Tensor,
        base_attention_mask: torch.Tensor,
        parallel_tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input tensors for the next forward pass where all tokens in the
        parallel set are represented with their own position but can attend to each other.
        
        Args:
            base_input_ids: Current input token IDs
            base_attention_mask: Current attention mask
            parallel_tokens: List of token IDs in the parallel set
            
        Returns:
            tuple: (new_input_ids, new_attention_mask)
        """
        # Verify all tokens are integers
        valid_tokens = []
        for token in parallel_tokens:
            if isinstance(token, int):
                valid_tokens.append(token)
            else:
                try:
                    # Try to convert to int if possible
                    valid_tokens.append(int(token))
                except (ValueError, TypeError):
                    pass
        
        # Update parallel_tokens with valid tokens only
        parallel_tokens = valid_tokens
        
        # If we have no valid tokens, return the base inputs unchanged
        if not parallel_tokens:
            return base_input_ids, base_attention_mask
            
        batch_size, seq_len = base_input_ids.shape
        num_parallel_tokens = len(parallel_tokens)
        
        
        # Create a new tensor that will hold all parallel tokens
        new_seq_len = seq_len + num_parallel_tokens  # Add each parallel token as its own position
        new_input_ids = torch.zeros((batch_size, new_seq_len), dtype=base_input_ids.dtype, device=self.device)
        
        # Copy existing tokens
        new_input_ids[:, :seq_len] = base_input_ids
        
        # Add all parallel tokens at subsequent positions
        for i, token_id in enumerate(parallel_tokens):
            new_input_ids[:, seq_len + i] = token_id
        
        # Create standard token-level attention mask (1=use token, 0=mask token)
        # This is the default causal mask used by most transformer models
        simple_attention_mask = torch.ones((batch_size, new_seq_len), 
                                          dtype=base_attention_mask.dtype, 
                                          device=self.device)
        
        # For our 3D custom attention mask (used by the custom model)
        # We'll store this separately and pass it to the forward pass
        custom_attention_mask = torch.zeros((batch_size, new_seq_len, new_seq_len), 
                                           dtype=torch.float, 
                                           device=self.device)
        
        # Build the custom 3D attention mask:
        # 1. Standard causal mask for original tokens (lower triangular)
        for i in range(seq_len):
            for j in range(i+1):  # j <= i
                custom_attention_mask[:, i, j] = 1.0
                
        # 2. Parallel tokens can attend to all previous tokens
        for i in range(num_parallel_tokens):
            pos = seq_len + i
            # Allow this parallel token to attend to all original tokens
            custom_attention_mask[:, pos, :seq_len] = 1.0
            
            # 3. All parallel tokens can attend to each other (full connectivity)
            for j in range(num_parallel_tokens):
                other_pos = seq_len + j
                custom_attention_mask[:, pos, other_pos] = 1.0
        
        # First, simplify to a binary mask where 1=can attend, 0=cannot attend
        self.full_attention_mask = custom_attention_mask
        # Ensure the mask has the right shape and dtype for the model
        if self.full_attention_mask is not None:
            # Should be float tensor with shape [batch_size, seq_len, seq_len]
            self.full_attention_mask = self.full_attention_mask.to(dtype=torch.float)
        
        return new_input_ids, simple_attention_mask
    
    def _format_text(self, prompt_text: str, position_to_tokens: Dict[int, List[int]], original_parallel_positions: Set[int], prompt_length: int, token_original_indices: Dict[Tuple[int, int], int]) -> str:
        """
        Format generated text using colored brackets for parallel tokens.
        
        Args:
            prompt_text: The initial prompt text
            position_to_tokens: Mapping of positions to parallel token IDs
            original_parallel_positions: Set of positions that originally had multiple tokens
            prompt_length: Length of the prompt in tokens
            token_original_indices: Mapping of (position, token_id) to original index in the set
            
        Returns:
            str: Formatted text with colored brackets notation
        """
        # Start with prompt text (we'll use the raw prompt, not reconstruct it)
        formatted_text = prompt_text
        
        # Define fixed colors using direct ANSI codes for better compatibility
        # Using direct ANSI codes since they work in the terminal
        RED = "\033[93m"          # Light Yellow (replacing Red)
        BLUE = "\033[94m"         # Bright Blue
        GREEN = "\033[92m"        # Bright Green
        YELLOW = "\033[95m"       # Light Magenta (swapping with Yellow)
        MAGENTA = "\033[95m"      # Bright Magenta
        CYAN = "\033[96m"         # Bright Cyan
        RESET = "\033[0m"
        BOLD = "\033[1m"  # Make brackets bold for better visibility
        
        # Process only generated tokens (after prompt)
        generated_positions = sorted([p for p in position_to_tokens.keys() if p >= prompt_length])
        
        # First, create the base sequence with just one token per position
        # This gives us proper spacing and context
        base_tokens = []
        for pos in generated_positions:
            base_tokens.append(position_to_tokens[pos][0])  # Just take first token from each position
        
        # Decode the base sequence to get spacing right
        base_text = self.tokenizer.decode(base_tokens, skip_special_tokens=True)
        
        # Create a text representation for each position
        position_texts = {}
        
        for pos in generated_positions:
            tokens = position_to_tokens[pos]
            
            # Check if this position originally had multiple tokens (before pruning)
            was_parallel = pos in original_parallel_positions
            
            # Format tokens based on whether they were originally part of a parallel set
            if len(tokens) > 1:
                # Get all token texts first
                token_texts = []
                for token_id in tokens:
                    text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    token_texts.append(text)
                    
                # Now color them in order with direct ANSI codes
                colored_tokens = []
                
                # First token - RED
                colored_tokens.append(f"{RED}{token_texts[0]}{RESET}")
                
                # Second token - BLUE
                if len(token_texts) > 1:
                    colored_tokens.append(f"{BLUE}{token_texts[1]}{RESET}")
                
                # Third token - GREEN
                if len(token_texts) > 2:
                    colored_tokens.append(f"{GREEN}{token_texts[2]}{RESET}")
                
                # Fourth token - YELLOW
                if len(token_texts) > 3:
                    colored_tokens.append(f"{YELLOW}{token_texts[3]}{RESET}")
                
                # Fifth token - MAGENTA
                if len(token_texts) > 4:
                    colored_tokens.append(f"{MAGENTA}{token_texts[4]}{RESET}")
                
                # Any remaining tokens - CYAN
                if len(token_texts) > 5:
                    for i in range(5, len(token_texts)):
                        colored_tokens.append(f"{CYAN}{token_texts[i]}{RESET}")
                
                # Join with explicit slash character
                joined_tokens = "/".join(colored_tokens)
                
                # Add BOLD brackets with RESET codes to make the bracket notation more visible
                position_texts[pos] = f"{BOLD}[{RESET}{joined_tokens}{BOLD}]{RESET}"
            elif was_parallel:
                # This position originally had multiple tokens but was pruned to one
                token_id = tokens[0]
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                position_texts[pos] = f"{RED}{token_text}{RESET}"  # Always RED for single token
            else:
                # Single token that was never part of a parallel set
                token_text = self.tokenizer.decode([tokens[0]], skip_special_tokens=False)
                position_texts[pos] = token_text
        
        # Reconstruct the text with formatting
        result = ""
        remaining_text = base_text
        
        # For each position, find its token in the base text and replace with formatted version
        for pos_idx, pos in enumerate(generated_positions):
            # Get the token we want to find in the base text
            token = position_to_tokens[pos][0]
            token_text = self.tokenizer.decode([token], skip_special_tokens=True)
            
            # Single tokens might be subwords that are hard to find, so we need to be smarter
            # If this is not the first token, use the preceding text as context
            if pos_idx > 0:
                # Get a chunk of text to search within
                search_idx = remaining_text.find(token_text)
                if search_idx != -1:
                    # Add text up to this token
                    result += remaining_text[:search_idx]
                    # Add our formatted token
                    result += position_texts[pos]
                    # Update remaining text
                    remaining_text = remaining_text[search_idx + len(token_text):]
                else:
                    # Fallback - just append the formatted token
                    result += position_texts[pos]
            else:
                # First token - simpler case
                if remaining_text.startswith(token_text):
                    result += position_texts[pos]
                    remaining_text = remaining_text[len(token_text):]
                else:
                    # Fallback
                    result += position_texts[pos]
        
        # Add any remaining text
        result += remaining_text
            
        # Combine prompt with formatted generated text
        return prompt_text + " " + result
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        threshold: Optional[float] = None,
        return_parallel_sets: bool = False,
        use_pruning: bool = False,
        require_custom_attention: bool = False
    ) -> Dict:
        """
        Generate text using the Parallel Threshold Output mechanism.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Override default threshold
            return_parallel_sets: If True, return the sets of parallel tokens
            use_pruning: Whether to use retroactive pruning (if pruner is available)
            require_custom_attention: If True, will raise an error if custom attention is not available
            
        Returns:
            dict: Results including the generated text
        """
        if threshold is None:
            threshold = self.threshold
        
        # Check if we need custom attention
        if require_custom_attention and not self.has_custom_attention:
            raise ValueError(
                "Custom attention is required but not available. "
                "Please use a model with CustomParallelAttentionModel wrapper."
            )
        
        # If we don't have custom attention support, warn the user
        if not self.has_custom_attention:
            print(
                "Warning: Custom attention is not available. "
                "Parallel tokens will be added to the sequence but won't see each other. "
                "This may produce less coherent results."
            )
        
        # Encode prompt
        input_data = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input_data.input_ids.to(self.device)
        attention_mask = input_data.attention_mask.to(self.device)
        
        # Initialize our full attention mask property
        self.full_attention_mask = None
        
        # If using dynamic threshold, set the max steps in the pruner
        if use_pruning and self.pruner is not None and hasattr(self.pruner, 'use_dynamic_threshold') and self.pruner.use_dynamic_threshold:
            # Set the maximum steps in the pruner
            # We set it to exactly max_tokens since we want the final step to definitely use threshold 1.0
            self.pruner.max_steps = max_tokens
            self.pruner.current_step = 0
            # Clear any existing token sets
            if hasattr(self.pruner, 'all_token_sets'):
                self.pruner.all_token_sets = []
        
        # Track sets of parallel tokens for analysis
        original_token_sets = []  # Before pruning
        pruned_token_sets = []    # After pruning
        original_token_sets_raw = []  # Raw token IDs and probs before pruning
        pruned_token_sets_raw = []    # Raw token IDs and probs after pruning
        generated_ids = []
        
        # Track positions that originally had multiple tokens (before pruning)
        original_parallel_positions = set()
        
        # Track token indices within their original sets
        # This is used for consistent coloring in the final output
        token_original_indices = {}
        
        # Initialize position_to_tokens mapping
        position_to_tokens = {}
        prompt_length = len(input_data.input_ids[0])
        
        # Add prompt tokens to the position mapping
        for i in range(prompt_length):
            position_to_tokens[i] = [input_data.input_ids[0, i].item()]
        
        # Iteratively generate tokens
        for i in range(max_tokens):
            # Get model prediction
            with torch.no_grad():
                if self.has_custom_attention and self.full_attention_mask is not None:
                    # Use custom attention mask if available
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        custom_attention_mask=self.full_attention_mask
                    )
                else:
                    # Standard forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            
            # Get logits for the next token (last position in sequence)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Get tokens above threshold
            next_token_ids, next_token_probs = self._get_parallel_tokens(
                next_token_logits, threshold
            )
            
            # Skip if no tokens above threshold
            if not next_token_ids:
                continue
            
            # Only filter out EOS tokens if they are the only token in the set
            # and this isn't the last token generation
            if (len(next_token_ids) == 1 and 
                hasattr(self.tokenizer, 'eos_token_id') and 
                next_token_ids[0] == self.tokenizer.eos_token_id and
                i < max_tokens - 1):
                # Skip this step - get a different token
                continue
                
            # Create a list of (token_id, probability) tuples for the original set
            original_raw_token_set = list(zip(next_token_ids, next_token_probs))
            
            # Store raw token IDs and probabilities of the original set
            original_token_sets_raw.append(original_raw_token_set)
            
            # Decode tokens to strings for human-readable output
            original_token_texts = []
            for token_id, prob in original_raw_token_set:
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    original_token_texts.append((token_text, prob))
                except Exception:
                    pass
            
            # Store decoded token texts for the original set
            original_token_sets.append(original_token_texts)
            
            # Track original token indices in the set
            for idx, (token_id, _) in enumerate(original_raw_token_set):
                token_original_indices[(len(generated_ids), token_id)] = idx
                
            # If we have multiple tokens, mark this as a parallel position
            if len(next_token_ids) > 1:
                original_parallel_positions.add(len(generated_ids))
                
            # Create a copy of the original tokens for pruning
            pruned_token_ids = next_token_ids.copy()
            pruned_token_probs = next_token_probs.copy()
                
            # Apply pruning if requested and available
            if (use_pruning and self.pruner is not None and 
                hasattr(self.pruner, 'prune_parallel_tokens')):
                
                # Apply retroactive pruning to potentially reduce the token set
                try:
                    if len(pruned_token_ids) > 1:
                        # Only bother pruning if we have multiple tokens
                        pruned_result = self.pruner.prune_parallel_tokens(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            parallel_tokens=pruned_token_ids,
                            position_idx=i,
                            token_probs=pruned_token_probs
                        )
                        
                        # Replace with the pruned set if we got results
                        if pruned_result and isinstance(pruned_result, tuple) and len(pruned_result) >= 1:
                            pruned_token_ids = pruned_result[0]
                            
                            # Extract probabilities from the pruned result if available
                            pruned_token_probs = [p for _, p in pruned_token_ids] if hasattr(pruned_token_ids[0], '__iter__') else pruned_token_probs[:len(pruned_token_ids)]
                            
                            # Ensure pruned_token_ids is just a list of IDs (not tuples)
                            if hasattr(pruned_token_ids[0], '__iter__'):
                                pruned_token_ids = [t for t, _ in pruned_token_ids]
                except Exception as e:
                    # If pruning fails, just continue with the original tokens
                    pass
            
            # Create a list of (token_id, probability) tuples for the pruned set
            pruned_raw_token_set = list(zip(pruned_token_ids, pruned_token_probs))
            
            # Store raw token IDs and probabilities of the pruned set
            pruned_token_sets_raw.append(pruned_raw_token_set)
            
            # Decode tokens to strings for human-readable output of the pruned set
            pruned_token_texts = []
            for token_id, prob in pruned_raw_token_set:
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    pruned_token_texts.append((token_text, prob))
                except Exception:
                    pass
            
            # Store decoded token texts for the pruned set
            pruned_token_sets.append(pruned_token_texts)
            
            # Store pruned tokens at this position
            generated_ids.append(pruned_token_ids)
            
            # Add pruned tokens to position_to_tokens mapping
            position_to_tokens[prompt_length + len(generated_ids) - 1] = pruned_token_ids
            
            # Create new input representation for next step with all tokens in the pruned set
            try:
                input_ids, attention_mask = self._create_parallel_set_input(
                    input_ids, attention_mask, pruned_token_ids
                )
            except Exception as e:
                # Fall back to simpler approach if anything goes wrong - just add the first token
                if pruned_token_ids:
                    # Append the first token to input_ids
                    new_input_ids = torch.cat([input_ids, torch.tensor([[pruned_token_ids[0]]], device=self.device)], dim=1)
                    new_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                    input_ids = new_input_ids
                    attention_mask = new_attention_mask
            
            # Only stop generation if ALL tokens in the set are EOS tokens
            if (hasattr(self.tokenizer, 'eos_token_id') and 
                len(pruned_token_ids) > 0 and 
                all(t == self.tokenizer.eos_token_id for t in pruned_token_ids)):
                print(f"DEBUG: Stopping because all tokens are EOS")
                break
        
        # If using dynamic threshold, we need to update position_to_tokens with final pruned states
        if use_pruning and self.pruner is not None and hasattr(self.pruner, 'use_dynamic_threshold') and self.pruner.use_dynamic_threshold:
            # Check if we have final pruned sets
            if hasattr(self.pruner, 'get_final_pruned_sets'):
                final_pruned_sets = self.pruner.get_final_pruned_sets()
                
                # Update position_to_tokens with the final pruned state
                for step, pruned_set in enumerate(final_pruned_sets):
                    position = prompt_length + step
                    if position in position_to_tokens:
                        # Update with the final pruned tokens
                        position_to_tokens[position] = [t[0] for t in pruned_set]
        
        # Format generated text
        formatted_text = self._format_text(
            prompt,
            position_to_tokens, 
            original_parallel_positions, 
            prompt_length, 
            token_original_indices
        )
        
        # Also generate raw text for analysis purposes
        full_token_sequence = []
        for i in range(prompt_length):
            full_token_sequence.append(input_data.input_ids[0, i].item())
            
        # Add generated tokens
        for pos in sorted(position_to_tokens.keys()):
            if pos >= prompt_length:  # Only add tokens after the prompt
                full_token_sequence.extend(position_to_tokens[pos])
        
        # Decode the raw generated text
        raw_generated_text = self.tokenizer.decode(full_token_sequence, skip_special_tokens=True)
        
        # Return results
        results = {
            "generated_text": formatted_text,
            "raw_generated_text": raw_generated_text,
            "prompt": prompt,
            "threshold": threshold,
            "use_pruning": use_pruning
        }
        
        if return_parallel_sets:
            # Include both raw token IDs and human-readable texts
            results["parallel_sets_raw"] = original_token_sets_raw
            
            # Store both original and pruned sets
            results["parallel_sets"] = original_token_sets
            
            if use_pruning and self.pruner is not None:
                results["pruned_sets"] = pruned_token_sets
                results["pruned_sets_raw"] = pruned_token_sets_raw
                
            # Also add positional information
            position_info = {}
            for pos, tokens in position_to_tokens.items():
                if pos >= prompt_length:  # Only include generated tokens
                    # Safely decode each token
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
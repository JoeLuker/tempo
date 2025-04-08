import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for colored terminal output
colorama_init()

# Define a list of colors to cycle through for parallel tokens
COLORS = [
    Fore.RED,
    Fore.GREEN,
    Fore.BLUE,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.CYAN,
]

class ParallelThresholdGenerator:
    """
    Implements the Parallel Threshold Output generation mechanism for Mistral-7B.
    
    This mechanism generates multiple tokens in parallel at each step based on a
    probability threshold, instead of the standard autoregressive approach.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        threshold: float = 0.1,
        max_length: int = 512,
        device: str = "mps",
        pruner = None
    ):
        """
        Initialize the generator.
        
        Args:
            model: The Mistral-7B model
            tokenizer: HuggingFace tokenizer
            threshold: Probability threshold for token selection
            max_length: Maximum sequence length
            device: Device to use for computation
            pruner: Optional RetroactivePruner instance for coherence-based pruning
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self.pruner = pruner
        
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
        
        tokens = tokens_above_threshold[sorted_indices].tolist()
        probabilities = selected_probs[sorted_indices].tolist()
        
        return tokens, probabilities
    
    def _create_parallel_set_input(
        self,
        base_input_ids: torch.Tensor,
        base_attention_mask: torch.Tensor,
        parallel_tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input tensors for the next forward pass where all tokens in the
        parallel set are represented with the same positional encoding.
        
        Args:
            base_input_ids: Current input token IDs
            base_attention_mask: Current attention mask
            parallel_tokens: List of token IDs in the parallel set
            
        Returns:
            tuple: (new_input_ids, new_attention_mask)
        """
        batch_size, seq_len = base_input_ids.shape
        num_parallel_tokens = len(parallel_tokens)
        
        # Create a new tensor that will hold all parallel tokens at the next position
        new_seq_len = seq_len + 1  # Only add one position for all parallel tokens
        new_input_ids = torch.zeros((batch_size, new_seq_len), dtype=base_input_ids.dtype, device=self.device)
        
        # Copy existing tokens
        new_input_ids[:, :seq_len] = base_input_ids
        
        # Set the first token of the parallel set at the next position
        # Other tokens in the set are tracked separately but not added to the input sequence
        # This implements Option A: all tokens share the exact same position
        if parallel_tokens:
            new_input_ids[:, seq_len] = parallel_tokens[0]
        
        # Create new attention mask for expanded sequence
        new_attention_mask = torch.ones((batch_size, new_seq_len), dtype=base_attention_mask.dtype, device=self.device)
        new_attention_mask[:, :seq_len] = base_attention_mask
        
        return new_input_ids, new_attention_mask, parallel_tokens
    
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
                # Multiple tokens - use colored bracket notation
                colored_tokens = []
                
                for i, token_id in enumerate(tokens):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    
                    # Get color based on token's original index in the set
                    orig_idx = token_original_indices.get((pos, token_id), i)
                    color_idx = orig_idx % len(COLORS)
                    color = COLORS[color_idx]
                    
                    colored_tokens.append(f"{color}{token_text}{Style.RESET_ALL}")
                    
                position_texts[pos] = f"[{'/'.join(colored_tokens)}]"
            elif was_parallel:
                # This position originally had multiple tokens but was pruned to one
                token_id = tokens[0]
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                
                # Get color based on token's original index in the set
                orig_idx = token_original_indices.get((pos, token_id), 0)
                color_idx = orig_idx % len(COLORS)
                color = COLORS[color_idx]
                
                position_texts[pos] = f"{color}{token_text}{Style.RESET_ALL}"
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
        use_pruning: bool = False
    ) -> Dict:
        """
        Generate text using the Parallel Threshold Output mechanism.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Override default threshold
            return_parallel_sets: If True, return the sets of parallel tokens
            use_pruning: Whether to use retroactive pruning (if pruner is available)
            
        Returns:
            dict: Results including the generated text
        """
        if threshold is None:
            threshold = self.threshold
            
        # Encode prompt
        input_data = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input_data.input_ids.to(self.device)
        attention_mask = input_data.attention_mask.to(self.device)
        
        # Track sets of parallel tokens for analysis
        parallel_token_sets = []
        pruned_token_sets = []
        
        # For properly tracking token positions
        position_to_tokens = {}  # Maps position -> list of parallel tokens
        # Also track which positions originally had multiple tokens before pruning
        original_parallel_positions = set()  # Set of positions that had multiple tokens before pruning
        
        # Track the original token indices to maintain colors
        token_original_indices = {}  # Maps (position, token_id) -> original index in the set
        
        # Store prompt token positions
        prompt_length = len(input_data.input_ids[0])
        for i in range(prompt_length):
            position_to_tokens[i] = [input_data.input_ids[0, i].item()]
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass to get logits
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get last token's logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get parallel tokens above threshold
                tokens, probs = self._get_parallel_tokens(next_token_logits, threshold)
                
                # If no tokens above threshold, fall back to standard sampling
                if not tokens:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    tokens = [next_token.item()]
                    probs = [F.softmax(next_token_logits, dim=-1)[0, next_token].item()]
                
                # Create token set for this generation step
                token_set = [(t, p) for t, p in zip(tokens, probs)]
                parallel_token_sets.append(token_set)
                
                # Track the current position
                position = input_ids.size(1)
                
                # Store original token indices for color tracking
                for i, token_id in enumerate(tokens):
                    token_original_indices[(position, token_id)] = i
                
                # Check if this is originally a parallel position (has multiple tokens)
                if len(tokens) > 1:
                    original_parallel_positions.add(position)
                
                # Apply retroactive pruning if enabled and available
                if use_pruning and self.pruner is not None:
                    pruned_token_set = self.pruner.prune_parallel_tokens(input_ids, token_set)
                    tokens = [t[0] for t in pruned_token_set]
                    probs = [t[1] for t in pruned_token_set]
                    pruned_token_sets.append(pruned_token_set)
                else:
                    pruned_token_sets.append(token_set)
                
                # Store parallel tokens at this position
                position_to_tokens[position] = tokens
                
                # Create new input representation for next step
                # This properly implements Option A by treating all tokens at the same position
                input_ids, attention_mask, _ = self._create_parallel_set_input(
                    input_ids, attention_mask, tokens
                )
                
                # Check if any token is EOS
                if self.tokenizer.eos_token_id in tokens:
                    break
        
        # Format output text
        prompt_text = self.tokenizer.decode(input_data.input_ids[0], skip_special_tokens=True)
        formatted_text = self._format_text(
            prompt_text, 
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
            # Convert tokens to text for easier analysis
            readable_sets = []
            readable_pruned_sets = []
            
            for token_set, pruned_set in zip(parallel_token_sets, pruned_token_sets):
                # Original parallel set
                readable_set = []
                for token, prob in token_set:
                    token_text = self.tokenizer.decode([token])
                    readable_set.append((token_text, prob))
                readable_sets.append(readable_set)
                
                # Pruned set (if different)
                if use_pruning and self.pruner is not None:
                    readable_pruned = []
                    for token, prob in pruned_set:
                        token_text = self.tokenizer.decode([token])
                        readable_pruned.append((token_text, prob))
                    readable_pruned_sets.append(readable_pruned)
                
            results["parallel_sets"] = readable_sets
            if use_pruning and self.pruner is not None:
                results["pruned_sets"] = readable_pruned_sets
                
            # Also add positional information
            position_info = {}
            for pos, tokens in position_to_tokens.items():
                if pos >= prompt_length:  # Only include generated tokens
                    position_info[str(pos)] = [self.tokenizer.decode([t]) for t in tokens]
            results["position_to_tokens"] = position_info
            
        return results 
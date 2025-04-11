import re
from typing import Dict, List, Set, Tuple, Any, Optional

class TextFormatter:
    """
    Responsible for formatting generated text with colors and annotations.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the text formatter.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
        """
        self.tokenizer = tokenizer
    
    def format_generated_text(
        self,
        prompt: str,
        position_to_tokens: Dict[int, List[int]],
        original_parallel_positions: Set[int],
        prompt_length: int,
        token_indices: Dict[Tuple[int, int], int],
        show_token_ids: bool = False
    ) -> str:
        """
        Format the generated text with colored annotations for parallel tokens.
        
        Args:
            prompt: Original prompt text
            position_to_tokens: Mapping from positions to token IDs
            original_parallel_positions: Set of positions that originally had multiple tokens
            prompt_length: Length of the prompt in tokens
            token_indices: Mapping from (position, token_id) to original index
            show_token_ids: Whether to show token IDs in the output
            
        Returns:
            str: Formatted text with colored annotations
        """
        if show_token_ids:
            return self._format_text_with_ids(prompt, position_to_tokens, original_parallel_positions, prompt_length, token_indices)
        else:
            return self._format_text(prompt, position_to_tokens, original_parallel_positions, prompt_length, token_indices)
    
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
                    # Invariant: Token must be found in remaining text
                    raise ValueError(f"Token '{token_text}' not found in remaining text. Text formatting invariant violated.")
            else:
                # First token - simpler case
                if remaining_text.startswith(token_text):
                    result += position_texts[pos]
                    remaining_text = remaining_text[len(token_text):]
                else:
                    # Invariant: First token must be at the start of remaining text
                    raise ValueError(f"First token '{token_text}' not at start of remaining text. Text formatting invariant violated.")
        
        # Add any remaining text
        result += remaining_text
            
        # Combine prompt with formatted generated text
        return prompt_text + " " + result
        
    def _format_text_with_ids(self, prompt_text: str, position_to_tokens: Dict[int, List[int]], original_parallel_positions: Set[int], prompt_length: int, token_original_indices: Dict[Tuple[int, int], int]) -> str:
        """
        Format generated text with token IDs using colored brackets for parallel tokens.
        
        Args:
            prompt_text: The initial prompt text
            position_to_tokens: Mapping of positions to parallel token IDs
            original_parallel_positions: Set of positions that originally had multiple tokens
            prompt_length: Length of the prompt in tokens
            token_original_indices: Mapping of (position, token_id) to original index in the set
            
        Returns:
            str: Formatted text with colored brackets notation and token IDs
        """
        # Get the base formatted text
        formatted_text = self._format_text(prompt_text, position_to_tokens, original_parallel_positions, prompt_length, token_original_indices)
        
        # Add token IDs to the formatted text
        # For simplicity, we'll just show the token IDs after the generated text
        token_id_info = "\n\nToken IDs:\n"
        
        # Process only generated tokens (after prompt)
        generated_positions = sorted([p for p in position_to_tokens.keys() if p >= prompt_length])
        
        for pos in generated_positions:
            tokens = position_to_tokens[pos]
            tokens_info = ", ".join([f"{t}" for t in tokens])
            token_id_info += f"Position {pos - prompt_length}: [{tokens_info}]\n"
        
        return formatted_text + token_id_info
    
    def format_with_token_ids(
        self,
        prompt: str,
        position_to_tokens: Dict[int, List[int]],
        parallel_positions: Set[int],
        prompt_length: int,
        token_indices: Dict[Tuple[int, int], int]
    ) -> str:
        """
        Format the generated text with token IDs.
        
        Args:
            prompt: Original prompt text
            position_to_tokens: Mapping from positions to token IDs
            parallel_positions: Set of positions with multiple tokens
            prompt_length: Length of the prompt in tokens
            token_indices: Mapping from (position, token_id) to original index
            
        Returns:
            str: Formatted text with token IDs
        """
        return self.format_generated_text(
            prompt, 
            position_to_tokens, 
            parallel_positions, 
            prompt_length, 
            token_indices, 
            show_token_ids=True
        )
    
    def _clean_formatted_text(self, text: str) -> str:
        """
        Clean up the formatted text.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove any double spaces that might have been introduced
        cleaned = re.sub(r' {2,}', ' ', text)
        
        # Ensure proper spacing around punctuation
        cleaned = re.sub(r' ([.,;:!?])', r'\1', cleaned)
        
        return cleaned 
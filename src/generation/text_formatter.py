import re
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np


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
        show_token_ids: bool = False,
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
            return self._format_text_with_ids(
                prompt,
                position_to_tokens,
                original_parallel_positions,
                prompt_length,
                token_indices,
            )
        else:
            return self._format_text(
                prompt,
                position_to_tokens,
                original_parallel_positions,
                prompt_length,
                token_indices,
            )

    def _format_text(
        self,
        prompt_text: str,
        position_to_tokens: Dict[int, List[int]],
        original_parallel_positions: Set[int],
        prompt_length: int,
        token_original_indices: Dict[Tuple[int, int], int],
    ) -> str:
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
        RED = "\033[93m"  # Light Yellow (replacing Red)
        BLUE = "\033[94m"  # Bright Blue
        GREEN = "\033[92m"  # Bright Green
        YELLOW = "\033[95m"  # Light Magenta (swapping with Yellow)
        MAGENTA = "\033[95m"  # Bright Magenta
        CYAN = "\033[96m"  # Bright Cyan
        RESET = "\033[0m"
        BOLD = "\033[1m"  # Make brackets bold for better visibility

        # Process only generated tokens (after prompt)
        generated_positions = sorted(
            [p for p in position_to_tokens.keys() if p >= prompt_length]
        )

        # First, create the base sequence with just one token per position
        # This gives us proper spacing and context
        base_tokens = []
        for pos in generated_positions:
            base_tokens.append(
                position_to_tokens[pos][0]
            )  # Just take first token from each position

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
                    text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
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
                token_text = self.tokenizer.decode(
                    [int(token_id)], skip_special_tokens=False
                )
                position_texts[pos] = (
                    f"{RED}{token_text}{RESET}"  # Always RED for single token
                )
            else:
                # Single token that was never part of a parallel set
                token_text = self.tokenizer.decode(
                    [int(tokens[0])], skip_special_tokens=False
                )
                position_texts[pos] = token_text

        # Reconstruct the text with formatting
        result = ""
        remaining_text = base_text

        # For each position, find its token in the base text and replace with formatted version
        for pos_idx, pos in enumerate(generated_positions):
            # Get the token we want to find in the base text
            token = position_to_tokens[pos][0]
            token_text = self.tokenizer.decode([int(token)], skip_special_tokens=True)

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
                    remaining_text = remaining_text[search_idx + len(token_text) :]
                else:
                    # Handle the case where token is not found in text
                    # This might happen with Unicode replacement characters or special tokens
                    try:
                        # Try to decode with different settings in case of encoding issues
                        alt_token_text = self.tokenizer.decode([int(token)], skip_special_tokens=False)
                        search_idx = remaining_text.find(alt_token_text)
                        
                        if search_idx != -1:
                            # Found with alternative decoding
                            result += remaining_text[:search_idx]
                            result += position_texts[pos]
                            remaining_text = remaining_text[search_idx + len(alt_token_text) :]
                        else:
                            # If we still can't find it, just append the formatted token
                            # This is a fallback to keep generation going
                            result += position_texts[pos]
                            # Attempt to advance remaining_text by best guess of token length
                            # Skip at most the first character of remaining text
                            if len(remaining_text) > 0:
                                remaining_text = remaining_text[1:]
                            
                            print(f"Warning: Token '{token_text}' (ID: {token}) not found in text. Using fallback method.")
                    except Exception as e:
                        print(f"Error processing token: {e}")
                        # Emergency fallback: just add the token and continue
                        result += position_texts[pos]
                        # Skip first character to avoid infinite loops
                        if len(remaining_text) > 0:
                            remaining_text = remaining_text[1:]
            else:
                # First token - simpler case
                if remaining_text.startswith(token_text):
                    result += position_texts[pos]
                    remaining_text = remaining_text[len(token_text) :]
                else:
                    # Try alternative decoding for first token
                    try:
                        alt_token_text = self.tokenizer.decode([int(token)], skip_special_tokens=False)
                        if remaining_text.startswith(alt_token_text):
                            result += position_texts[pos]
                            remaining_text = remaining_text[len(alt_token_text) :]
                        else:
                            # Fallback: just add the token and skip a character
                            result += position_texts[pos]
                            if len(remaining_text) > 0:
                                # Try to determine a reasonable amount to skip
                                # For safety, skip at most one character
                                remaining_text = remaining_text[1:]
                            
                            print(f"Warning: First token '{token_text}' (ID: {token}) not at start of text. Using fallback method.")
                    except Exception as e:
                        print(f"Error processing first token: {e}")
                        # Emergency fallback
                        result += position_texts[pos]
                        if len(remaining_text) > 0:
                            remaining_text = remaining_text[1:]

        # Add any remaining text
        result += remaining_text

        # Combine prompt with formatted generated text
        return prompt_text + " " + result

    def _format_text_with_ids(
        self,
        prompt_text: str,
        position_to_tokens: Dict[int, List[int]],
        original_parallel_positions: Set[int],
        prompt_length: int,
        token_original_indices: Dict[Tuple[int, int], int],
    ) -> str:
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
        formatted_text = self._format_text(
            prompt_text,
            position_to_tokens,
            original_parallel_positions,
            prompt_length,
            token_original_indices,
        )

        # Add token IDs to the formatted text
        # For simplicity, we'll just show the token IDs after the generated text
        token_id_info = "\n\nToken IDs:\n"

        # Process only generated tokens (after prompt)
        generated_positions = sorted(
            [p for p in position_to_tokens.keys() if p >= prompt_length]
        )

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
        token_indices: Dict[Tuple[int, int], int],
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
            show_token_ids=True,
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
        cleaned = re.sub(r" {2,}", " ", text)

        # Ensure proper spacing around punctuation
        cleaned = re.sub(r" ([.,;:!?])", r"\1", cleaned)

        return cleaned

    def format_generated_text_with_pruning(
        self,
        prompt: str,
        position_to_tokens: Dict[int, List[int]],
        original_parallel_positions: Set[int],
        prompt_length: int,
        all_parallel_tokens: Dict[int, List[Tuple[int, float]]],
        token_indices: Dict[Tuple[int, int], int] = None,
    ) -> str:
        """
        Format the generated text with colored annotations for parallel tokens,
        taking into account retroactively pruned tokens.

        Args:
            prompt: Original prompt text
            position_to_tokens: Mapping from positions to token IDs
            original_parallel_positions: Set of positions that originally had multiple tokens
            prompt_length: Length of the prompt in tokens
            all_parallel_tokens: Mapping from positions to lists of (token_id, prob) tuples,
                                including retroactively pruned tokens
            token_indices: Optional mapping from (position, token_id) to original index

        Returns:
            str: Formatted text with colored annotations
        """
        # Create new position_to_tokens that incorporates retroactively pruned tokens
        enhanced_position_to_tokens = {}
        for pos in position_to_tokens:
            if pos < prompt_length:
                # Keep prompt tokens as is
                enhanced_position_to_tokens[pos] = position_to_tokens[pos]
            else:
                # For generated tokens, use the pruned list if available
                rel_pos = pos - prompt_length
                if rel_pos in all_parallel_tokens:
                    # Extract token IDs from (token_id, prob) tuples and ensure they are integers
                    enhanced_position_to_tokens[pos] = [
                        int(tid) for tid, _ in all_parallel_tokens[rel_pos]
                    ]
                else:
                    # Fallback to original list, ensuring integers
                    enhanced_position_to_tokens[pos] = [int(tid) for tid in position_to_tokens[pos]]

        # Invariant: All token IDs must be integers
        for pos, tokens in enhanced_position_to_tokens.items():
            if not all(isinstance(tid, (int, np.integer)) for tid in tokens):
                raise ValueError(
                    f"Invariant violation: Token IDs at position {pos} must be integers, got {[type(tid) for tid in tokens]}"
                )

        # Call the standard formatter with the enhanced tokens
        if token_indices is None:
            token_indices = {}

        return self._format_text(
            prompt,
            enhanced_position_to_tokens,
            original_parallel_positions,
            prompt_length,
            token_indices,
        )

    def format_with_token_ids_and_pruning(
        self,
        prompt: str,
        position_to_tokens: Dict[int, List[int]],
        parallel_positions: Set[int],
        prompt_length: int,
        all_parallel_tokens: Dict[int, List[Tuple[int, float]]],
        token_indices: Dict[Tuple[int, int], int] = None,
    ) -> str:
        """
        Format the generated text with token IDs, taking into account retroactively pruned tokens.

        Args:
            prompt: Original prompt text
            position_to_tokens: Mapping from positions to token IDs
            parallel_positions: Set of positions with multiple tokens
            prompt_length: Length of the prompt in tokens
            all_parallel_tokens: Mapping from positions to lists of (token_id, prob) tuples,
                                including retroactively pruned tokens
            token_indices: Optional mapping from (position, token_id) to original index

        Returns:
            str: Formatted text with token IDs
        """
        # Get the formatted text
        formatted_text = self.format_generated_text_with_pruning(
            prompt,
            position_to_tokens,
            parallel_positions,
            prompt_length,
            all_parallel_tokens,
            token_indices,
        )

        # Add token IDs and pruning information
        token_id_info = "\n\nToken IDs (with pruning):\n"

        # Process only generated tokens (after prompt)
        generated_positions = sorted(
            [p for p in position_to_tokens.keys() if p >= prompt_length]
        )

        for pos in generated_positions:
            # Get tokens that survived pruning
            rel_pos = pos - prompt_length
            if rel_pos in all_parallel_tokens:
                tokens_info = ", ".join(
                    [f"{int(t[0])}" for t in all_parallel_tokens[rel_pos]]
                )
                pruned_tokens = len(all_parallel_tokens.get(rel_pos, []))
                original_tokens = len(position_to_tokens.get(pos, []))

                # Show pruning indicator if tokens were pruned
                if pruned_tokens < original_tokens:
                    token_id_info += f"Position {rel_pos}: [{tokens_info}] (pruned from {original_tokens} to {pruned_tokens})\n"
                else:
                    token_id_info += f"Position {rel_pos}: [{tokens_info}]\n"
            else:
                tokens = position_to_tokens[pos]
                tokens_info = ", ".join([f"{int(t)}" for t in tokens])
                token_id_info += f"Position {rel_pos}: [{tokens_info}]\n"

        return formatted_text + token_id_info

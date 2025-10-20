"""Text formatting service for TEMPO generation output.

This service formats generation output to properly display parallel tokens
and provide clean text extraction.
"""

from typing import List, Dict, Optional
from ...domain.entities.parallel_generation import LogicalPosition
from ...domain.interfaces.tokenizer import TokenizerInterface
from ...utils.logging_utils import LoggingMixin


class TextFormatter(LoggingMixin):
    """Formats TEMPO generation output for display."""

    def __init__(self, tokenizer: TokenizerInterface, debug_mode: bool = False):
        """Initialize the text formatter.

        Args:
            tokenizer: Tokenizer interface for decoding tokens
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("text_formatter", "text_formatter.log", debug_mode)
        self.tokenizer = tokenizer

    def format_with_parallel_indicators(
        self,
        token_ids: List[int],
        logical_layout: List[LogicalPosition],
        prompt_length: int
    ) -> str:
        """Format text showing parallel token alternatives with brackets.

        Args:
            token_ids: All generated token IDs
            logical_layout: Layout showing parallel token groups
            prompt_length: Number of tokens in the prompt

        Returns:
            Formatted text with parallel tokens indicated
        """
        # Decode all tokens first
        all_tokens = self.tokenizer.decode_tokens(token_ids)

        if isinstance(all_tokens, str):
            # Already decoded as one string
            return all_tokens

        # Build formatted output
        formatted = []
        prompt_text = "".join(all_tokens[:prompt_length])
        formatted.append(prompt_text)

        # Process each logical step
        for logical_pos in logical_layout:
            start = logical_pos.physical_start_idx
            end = logical_pos.physical_end_idx + 1

            if start >= len(all_tokens):
                break

            parallel_tokens = all_tokens[start:end]

            if len(parallel_tokens) == 1:
                # Single token, no brackets
                formatted.append(parallel_tokens[0])
            else:
                # Multiple parallel tokens - show alternatives
                formatted.append("[" + "/".join(parallel_tokens) + "]")

        return "".join(formatted)

    def extract_clean_text(
        self,
        token_ids: List[int],
        logical_layout: List[LogicalPosition],
        prompt_length: int
    ) -> str:
        """Extract clean text using only the first token from each parallel set.

        Args:
            token_ids: All generated token IDs
            logical_layout: Layout showing parallel token groups
            prompt_length: Number of tokens in the prompt

        Returns:
            Clean text without parallel alternatives
        """
        # Decode all tokens
        all_tokens = self.tokenizer.decode_tokens(token_ids)

        if isinstance(all_tokens, str):
            # Can't extract from already-joined string, return as-is
            return all_tokens

        # Build clean output using first token of each parallel set
        clean_tokens = list(all_tokens[:prompt_length])  # Keep all prompt tokens

        for logical_pos in logical_layout:
            start = logical_pos.physical_start_idx

            if start >= len(all_tokens):
                break

            # Take only the first token from each parallel set
            clean_tokens.append(all_tokens[start])

        return "".join(clean_tokens)

    def format_with_probabilities(
        self,
        token_sets: Dict[int, List[tuple[int, float]]],
        show_all: bool = False
    ) -> str:
        """Format output showing token probabilities.

        Args:
            token_sets: Dictionary mapping step -> [(token_id, prob), ...]
            show_all: Whether to show all parallel options

        Returns:
            Formatted string with probability information
        """
        formatted = []

        for step, tokens in sorted(token_sets.items()):
            if not tokens:
                continue

            # Decode tokens
            token_ids = [tid for tid, _ in tokens]
            decoded = self.tokenizer.decode_tokens(token_ids)

            if not show_all:
                # Show only the first (highest probability) token
                formatted.append(f"{decoded[0]}")
            else:
                # Show all with probabilities
                token_strs = [
                    f"{decoded[i]}({prob:.3f})"
                    for i, (_, prob) in enumerate(tokens)
                ]
                formatted.append(f"[{', '.join(token_strs)}]")

        return "".join(formatted)

    def get_generation_stats(
        self,
        logical_layout: List[LogicalPosition],
        total_tokens: int
    ) -> Dict:
        """Get statistics about the generation.

        Args:
            logical_layout: Layout of logical positions
            total_tokens: Total number of tokens generated

        Returns:
            Dictionary with generation statistics
        """
        num_steps = len(logical_layout)
        num_parallel_steps = sum(
            1 for pos in logical_layout
            if (pos.physical_end_idx - pos.physical_start_idx) > 0
        )

        total_parallel_tokens = sum(
            (pos.physical_end_idx - pos.physical_start_idx + 1)
            for pos in logical_layout
        )

        avg_parallel_width = (
            total_parallel_tokens / num_steps if num_steps > 0 else 0
        )

        return {
            "logical_steps": num_steps,
            "parallel_steps": num_parallel_steps,
            "total_physical_tokens": total_tokens,
            "average_parallel_width": avg_parallel_width,
            "parallelization_ratio": total_tokens / num_steps if num_steps > 0 else 1.0
        }

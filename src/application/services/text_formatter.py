"""Text formatting service for TEMPO generation output.

This service formats generation output to properly display parallel tokens
and provide clean text extraction.
"""

from typing import List, Dict, Optional, Set
from ...domain.entities.parallel_generation import LogicalPosition
from ...domain.interfaces.tokenizer import TokenizerInterface
from ...utils.logging_utils import LoggingMixin


class TextFormatter(LoggingMixin):
    """Formats TEMPO generation output for display."""

    # ANSI color codes for terminal output
    RED = "\033[93m"          # Light Yellow
    BLUE = "\033[94m"         # Bright Blue
    GREEN = "\033[92m"        # Bright Green
    YELLOW = "\033[95m"       # Light Magenta
    MAGENTA = "\033[95m"      # Bright Magenta
    CYAN = "\033[96m"         # Bright Cyan
    RESET = "\033[0m"
    BOLD = "\033[1m"

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
        prompt_length: int,
        all_original_token_sets: Optional[Dict[int, List[tuple[int, float]]]] = None,
        all_surviving_token_sets: Optional[Dict[int, List[tuple[int, float]]]] = None
    ) -> str:
        """Format text showing parallel token alternatives with colored brackets.

        Args:
            token_ids: All generated token IDs
            logical_layout: Layout showing parallel token groups
            prompt_length: Number of tokens in the prompt
            all_original_token_sets: Optional dict of original token sets before pruning
            all_surviving_token_sets: Optional dict of surviving tokens after pruning

        Returns:
            Formatted text with colored brackets showing parallel tokens
        """
        # Decode prompt tokens
        prompt_tokens = self.tokenizer.decode_tokens(token_ids[:prompt_length])
        prompt_text = "".join(prompt_tokens) if isinstance(prompt_tokens, list) else prompt_tokens

        # Build formatted output starting with prompt
        result = prompt_text

        # Track original parallel positions (positions that had multiple tokens)
        original_parallel_positions = set()
        if all_original_token_sets:
            for step, tokens in all_original_token_sets.items():
                if len(tokens) > 1:
                    original_parallel_positions.add(step)

        # Process generated tokens by logical step
        for logical_idx, logical_pos in enumerate(logical_layout):
            step = logical_pos.logical_step
            start = logical_pos.physical_start_idx
            end = logical_pos.physical_end_idx + 1

            # Skip prompt tokens
            if end <= prompt_length:
                continue

            # Adjust indices if they overlap with prompt
            if start < prompt_length:
                start = prompt_length

            if start >= len(token_ids):
                break

            # Get tokens for this position
            position_tokens = token_ids[start:end]

            # Check if using surviving tokens (after pruning)
            tokens_to_display = position_tokens
            was_parallel = step in original_parallel_positions

            if len(tokens_to_display) > 1:
                # Multiple tokens - create colored bracket notation
                token_texts = []
                for token_id in tokens_to_display:
                    decoded = self.tokenizer.decode_tokens([token_id])
                    text = decoded[0] if isinstance(decoded, list) else decoded
                    token_texts.append(text)

                # Color each token
                colored_tokens = []
                colors = [self.RED, self.BLUE, self.GREEN, self.YELLOW, self.MAGENTA]

                for i, text in enumerate(token_texts):
                    color = colors[i] if i < len(colors) else self.CYAN
                    colored_tokens.append(f"{color}{text}{self.RESET}")

                # Join with slashes and add bold brackets
                joined = "/".join(colored_tokens)
                result += f"{self.BOLD}[{self.RESET}{joined}{self.BOLD}]{self.RESET}"

            elif was_parallel and len(tokens_to_display) == 1:
                # Was parallel but pruned to one token - color it red
                token_id = tokens_to_display[0]
                decoded = self.tokenizer.decode_tokens([token_id])
                token_text = decoded[0] if isinstance(decoded, list) else decoded
                result += f"{self.RED}{token_text}{self.RESET}"
            else:
                # Single token, never parallel
                token_id = tokens_to_display[0]
                decoded = self.tokenizer.decode_tokens([token_id])
                token_text = decoded[0] if isinstance(decoded, list) else decoded
                result += token_text

        return result

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

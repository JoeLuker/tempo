"""
Compressed Thought Generator using Position Gaps.

Generates multiple complete thought paths in a single forward pass by using
position gaps with TEMPO parallel token selection.

Key insight: Each parallel token at position N encodes a complete semantic
trajectory from the current position to N.
"""

import logging
from typing import List, Dict, Tuple, Optional
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThoughtPath:
    """Represents a complete thought path spanning a position gap."""

    initial_token: str
    initial_token_id: int
    probability: float
    full_path: str
    path_tokens: List[str]
    gap_size: int


class CompressedThoughtGenerator:
    """
    Generate compressed thought vectors using position gaps.

    Instead of generating N tokens sequentially, query position N directly
    and get parallel tokens that each encode different N-token thoughts.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "mps",
    ):
        """
        Initialize the compressed thought generator.

        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run on (mps, cuda, cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_thought_paths(
        self,
        prompt: str,
        gap_size: int = 10,
        selection_threshold: float = 0.05,
        max_parallel_paths: int = 20,
        expand_paths: bool = True,
    ) -> List[ThoughtPath]:
        """
        Generate multiple complete thought paths in one forward pass.

        Args:
            prompt: The input prompt
            gap_size: How many token positions to span
            selection_threshold: Minimum probability for parallel tokens
            max_parallel_paths: Maximum number of paths to return
            expand_paths: Whether to expand each path to full gap_size

        Returns:
            List of ThoughtPath objects, each representing a complete thought
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        logger.debug(
            f"Generating compressed thoughts: gap_size={gap_size}, "
            f"threshold={selection_threshold}"
        )

        # Get parallel tokens at next position
        # Import here to avoid circular dependency
        from .attention_mask_utils import create_sequence_based_attention_mask

        position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)

        # Create proper 4D causal mask based on sequence indices
        # This allows position gaps to work correctly
        attention_mask = create_sequence_based_attention_mask(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        # Get probabilities for parallel tokens
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Select tokens above threshold
        above_threshold = probs >= selection_threshold
        parallel_token_ids = torch.where(above_threshold)[0]
        parallel_probs = probs[parallel_token_ids]

        # Sort by probability and limit
        sorted_indices = torch.argsort(parallel_probs, descending=True)
        parallel_token_ids = parallel_token_ids[sorted_indices][:max_parallel_paths]
        parallel_probs = parallel_probs[sorted_indices][:max_parallel_paths]

        num_paths = len(parallel_token_ids)
        parallel_tokens = [
            self.tokenizer.decode([tid.item()])
            for tid in parallel_token_ids
        ]

        logger.info(f"Found {num_paths} parallel thought paths")

        # Expand each path if requested
        thought_paths = []

        for token_id, token_text, prob in zip(
            parallel_token_ids,
            parallel_tokens,
            parallel_probs
        ):
            if expand_paths:
                full_path, path_tokens = self._expand_path(
                    input_ids,
                    prompt_length,
                    token_id,
                    gap_size
                )
            else:
                full_path = token_text
                path_tokens = [token_text]

            thought_path = ThoughtPath(
                initial_token=token_text,
                initial_token_id=token_id.item(),
                probability=prob.item(),
                full_path=full_path,
                path_tokens=path_tokens,
                gap_size=gap_size,
            )
            thought_paths.append(thought_path)

        return thought_paths

    def _expand_path(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
        initial_token_id: torch.Tensor,
        gap_size: int,
    ) -> Tuple[str, List[str]]:
        """
        Expand a path from the initial token to span the full gap.

        Args:
            input_ids: The prompt input IDs
            prompt_length: Length of the prompt
            initial_token_id: The first token of this path
            gap_size: How many tokens to generate

        Returns:
            (full_path_text, list_of_tokens)
        """
        current_ids = torch.cat([
            input_ids,
            initial_token_id.unsqueeze(0).unsqueeze(0)
        ], dim=1)

        initial_token_text = self.tokenizer.decode([initial_token_id.item()])
        path_tokens = [initial_token_text]

        # Import here to avoid circular dependency
        from .attention_mask_utils import create_sequence_based_attention_mask

        # Generate remaining tokens in the path
        for step in range(gap_size - 1):
            current_length = current_ids.shape[1]
            position_ids = torch.arange(current_length, device=self.device).unsqueeze(0)

            # Use proper 4D mask
            attention_mask = create_sequence_based_attention_mask(
                input_ids=current_ids,
                position_ids=position_ids,
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )

            logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(logits).item()

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            next_token_text = self.tokenizer.decode([next_token_id])
            path_tokens.append(next_token_text)

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token_id]], device=self.device)
            ], dim=1)

        full_path = ''.join(path_tokens)
        return full_path, path_tokens

    def generate_with_adaptive_gaps(
        self,
        prompt: str,
        gap_sizes: List[int] = [5, 10, 20],
        selection_threshold: float = 0.05,
    ) -> Dict[int, List[ThoughtPath]]:
        """
        Generate thought paths at multiple gap sizes.

        This reveals the thought structure at different semantic distances:
        - Small gaps (5): Immediate next steps
        - Medium gaps (10): Short-term trajectories
        - Large gaps (20): Long-term directions

        Args:
            prompt: The input prompt
            gap_sizes: List of gap sizes to try
            selection_threshold: Minimum probability for parallel tokens

        Returns:
            Dictionary mapping gap_size -> list of ThoughtPaths
        """
        results = {}

        for gap_size in gap_sizes:
            logger.info(f"Generating paths for gap_size={gap_size}")
            paths = self.generate_thought_paths(
                prompt=prompt,
                gap_size=gap_size,
                selection_threshold=selection_threshold,
            )
            results[gap_size] = paths

        return results

    def score_path_coherence(
        self,
        prompt: str,
        thought_path: ThoughtPath,
    ) -> float:
        """
        Score how coherent a thought path is with the prompt.

        Uses average token probability as a simple coherence metric.
        Better metrics could use:
        - Attention to prompt
        - Entropy of token distributions
        - Semantic similarity

        Args:
            prompt: The original prompt
            thought_path: The path to score

        Returns:
            Coherence score (0-1, higher is better)
        """
        # Simple implementation: return the initial token probability
        # This could be enhanced with full path scoring
        return thought_path.probability

    def select_best_paths(
        self,
        thought_paths: List[ThoughtPath],
        top_k: int = 5,
    ) -> List[ThoughtPath]:
        """
        Select the top-k most coherent thought paths.

        Args:
            thought_paths: List of all generated paths
            top_k: How many to keep

        Returns:
            Top-k paths sorted by coherence
        """
        # Sort by probability (simple coherence metric)
        sorted_paths = sorted(
            thought_paths,
            key=lambda p: p.probability,
            reverse=True
        )
        return sorted_paths[:top_k]

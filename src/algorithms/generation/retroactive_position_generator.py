"""
Retroactive Position Generator.

Uses minimal computation by:
1. Generate token at normal position N
2. Re-run same token at position N+gap to explore different semantic contexts
3. Avoid full regeneration - just change position_ids

This is much more efficient than traditional compressed thought generation
for exploring multiple gap sizes.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch

from .attention_mask_utils import create_sequence_based_attention_mask

logger = logging.getLogger(__name__)


@dataclass
class RetroactiveExploration:
    """Result of exploring a token at a retroactive position."""

    original_token: str
    original_token_id: int
    original_position: int
    retroactive_position: int
    gap: int

    # Parallel tokens discovered at retroactive position
    parallel_tokens: List[str]
    parallel_token_ids: List[int]
    parallel_probs: List[float]

    # Top predicted next token
    top_next_token: str
    top_next_token_id: int
    top_next_prob: float


@dataclass
class MultiPositionExploration:
    """Result of exploring same token at multiple positions."""

    original_token: str
    original_token_id: int
    base_position: int

    # Explorations at different positions
    explorations: Dict[int, RetroactiveExploration]

    def get_exploration(self, gap: int) -> Optional[RetroactiveExploration]:
        """Get exploration for a specific gap size."""
        position = self.base_position + gap
        return self.explorations.get(position)

    def gaps(self) -> List[int]:
        """Get all explored gap sizes."""
        return sorted([pos - self.base_position for pos in self.explorations.keys()])


class RetroactivePositionGenerator:
    """
    Generator that uses retroactive position assignment for efficient exploration.

    Key insight: Generate a token once, then explore it at multiple positions
    by just changing position_ids (much cheaper than full regeneration).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "mps",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_and_explore(
        self,
        prompt: str,
        gaps: List[int],
        selection_threshold: float = 0.05,
        max_parallel_tokens: int = 10,
    ) -> MultiPositionExploration:
        """
        Generate next token, then explore it at multiple positions.

        Args:
            prompt: Input prompt
            gaps: List of gap sizes to explore (e.g., [5, 10, 20])
            selection_threshold: Probability threshold for parallel tokens
            max_parallel_tokens: Maximum parallel tokens to return

        Returns:
            MultiPositionExploration with results for all gaps

        Example:
            >>> explorer = RetroactivePositionGenerator(model, tokenizer)
            >>> result = explorer.generate_and_explore(
            ...     "The answer is",
            ...     gaps=[5, 10, 20]
            ... )
            >>> # Now result.explorations[10] has the exploration at gap=10
        """

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        logger.debug(f"Prompt: {prompt!r} (length={prompt_length})")

        # Step 1: Generate next token at normal position
        position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)
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

        # Get generated token
        logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(logits).item()
        next_token = self.tokenizer.decode([next_token_id])

        logger.debug(f"Generated token: {next_token!r} (ID: {next_token_id})")

        # Create extended sequence
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=self.device)
        ], dim=1)

        # Step 2: Explore at each gap size
        explorations = {}

        for gap in gaps:
            retroactive_position = prompt_length + gap

            logger.debug(f"Exploring gap={gap} (position={retroactive_position})")

            exploration = self._explore_at_position(
                extended_ids=extended_ids,
                prompt_length=prompt_length,
                retroactive_position=retroactive_position,
                original_token=next_token,
                original_token_id=next_token_id,
                selection_threshold=selection_threshold,
                max_parallel_tokens=max_parallel_tokens,
            )

            explorations[retroactive_position] = exploration

        return MultiPositionExploration(
            original_token=next_token,
            original_token_id=next_token_id,
            base_position=prompt_length,
            explorations=explorations,
        )

    def _explore_at_position(
        self,
        extended_ids: torch.Tensor,
        prompt_length: int,
        retroactive_position: int,
        original_token: str,
        original_token_id: int,
        selection_threshold: float,
        max_parallel_tokens: int,
    ) -> RetroactiveExploration:
        """Explore the generated token at a specific retroactive position."""

        # Create position IDs with gap
        position_ids = torch.cat([
            torch.arange(prompt_length, device=self.device),
            torch.tensor([retroactive_position], device=self.device)
        ]).unsqueeze(0)

        # Create attention mask
        attention_mask = create_sequence_based_attention_mask(
            input_ids=extended_ids,
            position_ids=position_ids,
        )

        # Forward pass with retroactive position
        with torch.no_grad():
            outputs = self.model(
                input_ids=extended_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Get parallel tokens above threshold
        mask = probs >= selection_threshold
        parallel_probs_tensor = probs[mask]
        parallel_token_ids_tensor = torch.nonzero(mask).squeeze(-1)

        # Limit to max_parallel_tokens
        if len(parallel_token_ids_tensor) > max_parallel_tokens:
            top_k = torch.topk(parallel_probs_tensor, k=max_parallel_tokens)
            parallel_probs_tensor = top_k.values
            parallel_token_ids_tensor = parallel_token_ids_tensor[top_k.indices]

        # Convert to lists
        parallel_probs_list = parallel_probs_tensor.cpu().tolist()
        parallel_token_ids_list = parallel_token_ids_tensor.cpu().tolist()
        parallel_tokens_list = [
            self.tokenizer.decode([tid]) for tid in parallel_token_ids_list
        ]

        # Get top predicted token
        top_token_id = torch.argmax(probs).item()
        top_token = self.tokenizer.decode([top_token_id])
        top_prob = probs[top_token_id].item()

        gap = retroactive_position - prompt_length

        logger.debug(
            f"Gap={gap}: Found {len(parallel_tokens_list)} parallel tokens, "
            f"top={top_token!r} ({top_prob:.4f})"
        )

        return RetroactiveExploration(
            original_token=original_token,
            original_token_id=original_token_id,
            original_position=prompt_length,
            retroactive_position=retroactive_position,
            gap=gap,
            parallel_tokens=parallel_tokens_list,
            parallel_token_ids=parallel_token_ids_list,
            parallel_probs=parallel_probs_list,
            top_next_token=top_token,
            top_next_token_id=top_token_id,
            top_next_prob=top_prob,
        )

    def adaptive_exploration(
        self,
        prompt: str,
        token_analyzer: Optional[callable] = None,
        default_gaps: List[int] = [5, 10, 20],
        selection_threshold: float = 0.05,
    ) -> MultiPositionExploration:
        """
        Generate token, analyze it, then adaptively choose gaps to explore.

        Args:
            prompt: Input prompt
            token_analyzer: Function that takes token and returns list of gaps
                           If None, uses default_gaps
            default_gaps: Default gaps if no analyzer provided
            selection_threshold: Threshold for parallel tokens

        Returns:
            MultiPositionExploration

        Example:
            >>> def my_analyzer(token: str) -> List[int]:
            ...     if len(token.strip()) <= 2:
            ...         return [3, 5]  # Small gaps for short tokens
            ...     else:
            ...         return [10, 20]  # Large gaps for long tokens
            >>>
            >>> result = explorer.adaptive_exploration(
            ...     "The answer is",
            ...     token_analyzer=my_analyzer
            ... )
        """

        # Encode and generate token
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        next_token = self.tokenizer.decode([next_token_id])

        # Analyze token to choose gaps
        if token_analyzer is not None:
            gaps = token_analyzer(next_token)
        else:
            gaps = default_gaps

        logger.info(
            f"Generated {next_token!r}, adaptively exploring gaps: {gaps}"
        )

        # Explore at chosen gaps
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=self.device)
        ], dim=1)

        explorations = {}
        for gap in gaps:
            retroactive_position = prompt_length + gap
            exploration = self._explore_at_position(
                extended_ids=extended_ids,
                prompt_length=prompt_length,
                retroactive_position=retroactive_position,
                original_token=next_token,
                original_token_id=next_token_id,
                selection_threshold=selection_threshold,
                max_parallel_tokens=10,
            )
            explorations[retroactive_position] = exploration

        return MultiPositionExploration(
            original_token=next_token,
            original_token_id=next_token_id,
            base_position=prompt_length,
            explorations=explorations,
        )

    def compare_positions(
        self,
        prompt: str,
        positions: List[int],
        selection_threshold: float = 0.05,
    ) -> Tuple[str, Dict[int, RetroactiveExploration]]:
        """
        Generate token once, compare it at multiple specific positions.

        Args:
            prompt: Input prompt
            positions: Absolute positions to test (e.g., [4, 10, 20, 50])
            selection_threshold: Threshold for parallel tokens

        Returns:
            (generated_token, explorations_dict)
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        # Generate token
        position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        next_token = self.tokenizer.decode([next_token_id])

        # Explore at each position
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=self.device)
        ], dim=1)

        explorations = {}
        for pos in positions:
            exploration = self._explore_at_position(
                extended_ids=extended_ids,
                prompt_length=prompt_length,
                retroactive_position=pos,
                original_token=next_token,
                original_token_id=next_token_id,
                selection_threshold=selection_threshold,
                max_parallel_tokens=10,
            )
            explorations[pos] = exploration

        return next_token, explorations

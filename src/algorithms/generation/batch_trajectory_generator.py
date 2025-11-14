#!/usr/bin/env python3
"""
Batch Trajectory Generator using TEMPO + Position Gaps.

Combines TEMPO parallel token selection at BOTH starting and endpoint positions
to generate batches of complete multi-token trajectories efficiently.

Key innovation:
- TEMPO at position N → K starting tokens
- For each starting token, apply position gap to N+gap
- TEMPO at position N+gap → M endpoint tokens per start
- Result: K × M complete trajectories in ~K forward passes!
"""

import logging
from typing import List, Tuple
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompleteTrajectory:
    """Represents a complete trajectory from start to end across a position gap."""

    start_token: str
    start_token_id: int
    start_probability: float
    end_token: str
    end_token_id: int
    end_probability: float
    gap_size: int
    full_sequence: str  # If expanded


class BatchTrajectoryGenerator:
    """
    Generate batches of complete trajectories using TEMPO + Position Gaps.

    This combines:
    1. TEMPO parallel tokens at starting position
    2. Position gaps to compress multi-token paths
    3. TEMPO parallel tokens at gap endpoint
    4. Batching for efficiency

    Result: Explore K×M complete trajectories in ~K forward passes!
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "mps",
    ):
        """
        Initialize the batch trajectory generator.

        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_batch_trajectories(
        self,
        prompt: str,
        gap_size: int = 5,
        selection_threshold: float = 0.05,
        max_start_tokens: int = 10,
        max_end_tokens_per_start: int = 10,
    ) -> List[List[CompleteTrajectory]]:
        """
        Generate batches of complete trajectories.

        Args:
            prompt: Input prompt
            gap_size: How many token positions to span
            selection_threshold: TEMPO threshold for parallel selection
            max_start_tokens: Max parallel tokens at starting position
            max_end_tokens_per_start: Max endpoints per starting token

        Returns:
            For each starting token, a list of complete trajectories.
            Result[i][j] = trajectory from start_token[i] to end_token[j]
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        logger.info(
            f"Generating batch trajectories: gap={gap_size}, threshold={selection_threshold}"
        )

        # Step 1: Generate parallel tokens at next position (TEMPO)
        starting_tokens = self._generate_starting_tokens(
            input_ids,
            prompt_length,
            selection_threshold,
            max_start_tokens,
        )

        logger.info(f"Found {len(starting_tokens)} starting parallel tokens")

        # Step 2: For each starting token, explore with position gap
        all_trajectories = []

        for start_info in starting_tokens:
            start_token, start_id, start_prob = start_info

            # Generate endpoints for this starting token
            endpoints = self._generate_endpoints_with_gap(
                input_ids,
                prompt_length,
                start_id,
                gap_size,
                selection_threshold,
                max_end_tokens_per_start,
            )

            # Create trajectories
            trajectories_from_this_start = []
            for end_token, end_id, end_prob in endpoints:
                trajectory = CompleteTrajectory(
                    start_token=start_token,
                    start_token_id=start_id,
                    start_probability=start_prob,
                    end_token=end_token,
                    end_token_id=end_id,
                    end_probability=end_prob,
                    gap_size=gap_size,
                    full_sequence=f"{start_token} → [{gap_size-1} tokens] → {end_token}",
                )
                trajectories_from_this_start.append(trajectory)

            all_trajectories.append(trajectories_from_this_start)

            logger.info(
                f"Start token '{start_token}' → {len(endpoints)} endpoints"
            )

        total_trajectories = sum(len(t) for t in all_trajectories)
        logger.info(f"Generated {total_trajectories} complete trajectories")

        return all_trajectories

    def _generate_starting_tokens(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
        threshold: float,
        max_tokens: int,
    ) -> List[Tuple[str, int, float]]:
        """
        Generate parallel starting tokens using TEMPO threshold selection.

        Args:
            input_ids: Prompt token IDs
            prompt_length: Length of prompt
            threshold: Probability threshold
            max_tokens: Maximum number of tokens to return

        Returns:
            List of (token_text, token_id, probability)
        """
        from .attention_mask_utils import create_sequence_based_attention_mask

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

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # TEMPO threshold selection
        above_threshold = probs >= threshold
        token_ids = torch.where(above_threshold)[0]
        token_probs = probs[token_ids]

        # Sort and limit
        sorted_indices = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_indices][:max_tokens]
        token_probs = token_probs[sorted_indices][:max_tokens]

        # Convert to tuples
        results = []
        for tid, prob in zip(token_ids, token_probs):
            token_text = self.tokenizer.decode([tid.item()])
            results.append((token_text, tid.item(), prob.item()))

        return results

    def _generate_endpoints_with_gap(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
        start_token_id: int,
        gap_size: int,
        threshold: float,
        max_tokens: int,
    ) -> List[Tuple[str, int, float]]:
        """
        Generate endpoint tokens using position gap.

        Args:
            input_ids: Prompt token IDs
            prompt_length: Length of prompt
            start_token_id: The starting token
            gap_size: Position gap to apply
            threshold: TEMPO threshold
            max_tokens: Max endpoints to return

        Returns:
            List of (end_token_text, end_token_id, probability)
        """
        from .attention_mask_utils import create_sequence_based_attention_mask

        # Create sequence: prompt + start_token
        sequence = torch.cat([
            input_ids[0],
            torch.tensor([start_token_id], device=self.device)
        ]).unsqueeze(0)

        # Create position IDs with gap: [0,1,2,3,N,N+gap]
        gap_positions = torch.cat([
            torch.arange(prompt_length + 1, device=self.device),
            torch.tensor([prompt_length + gap_size], device=self.device)
        ]).unsqueeze(0)

        # NOTE: We're creating a sequence with just the start token,
        # then asking "what would be at position N+gap?"
        # This is the COMPRESSED THOUGHT approach!

        attention_mask = create_sequence_based_attention_mask(
            input_ids=sequence,
            position_ids=gap_positions,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=sequence,
                position_ids=gap_positions,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # TEMPO threshold selection at gap endpoint
        above_threshold = probs >= threshold
        token_ids = torch.where(above_threshold)[0]
        token_probs = probs[token_ids]

        # Sort and limit
        sorted_indices = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_indices][:max_tokens]
        token_probs = token_probs[sorted_indices][:max_tokens]

        # Convert to tuples
        results = []
        for tid, prob in zip(token_ids, token_probs):
            token_text = self.tokenizer.decode([tid.item()])
            results.append((token_text, tid.item(), prob.item()))

        return results

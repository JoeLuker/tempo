#!/usr/bin/env python3
"""
Batch Gap Generator - Actually uses position gaps with TEMPO.

This is the REAL implementation of batch token creation via position gaps.
"""

import logging
from typing import List, Tuple
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GapTrajectory:
    """A complete trajectory across a position gap."""
    start_token: str
    start_token_id: int
    start_probability: float
    end_token: str
    end_token_id: int
    end_probability: float
    gap_size: int


class BatchGapGenerator:
    """
    Generate batches using ACTUAL position gaps.

    The key: Don't generate sequentially! Jump directly to future positions.
    """

    def __init__(self, model, tokenizer, device: str = "mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_with_gap(
        self,
        prompt: str,
        gap_size: int = 5,
        threshold: float = 0.05,
        max_starts: int = 5,
        max_ends: int = 5,
    ) -> List[List[GapTrajectory]]:
        """
        Generate K starting tokens, then M endpoints for each using position gaps.

        Returns:
            List[List[GapTrajectory]] - for each start, list of trajectories to ends
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        logger.info(f"Generating batch with gap={gap_size}")

        # Step 1: Get starting tokens (TEMPO at next position)
        starts = self._get_parallel_tokens(
            input_ids, prompt_length, threshold, max_starts
        )

        logger.info(f"Found {len(starts)} starting tokens")

        # Step 2: For each start, get endpoints using position gap
        all_trajectories = []

        for start_text, start_id, start_prob in starts:
            # Create sequence with start token
            seq_with_start = torch.cat([
                input_ids[0],
                torch.tensor([start_id], device=self.device)
            ])

            # Get endpoints at gap position
            # This is the KEY: We use position gap here!
            ends = self._get_endpoints_at_gap(
                seq_with_start,
                prompt_length,
                gap_size,
                threshold,
                max_ends,
            )

            # Create trajectories
            trajectories = []
            for end_text, end_id, end_prob in ends:
                traj = GapTrajectory(
                    start_token=start_text,
                    start_token_id=start_id,
                    start_probability=start_prob,
                    end_token=end_text,
                    end_token_id=end_id,
                    end_probability=end_prob,
                    gap_size=gap_size,
                )
                trajectories.append(traj)

            all_trajectories.append(trajectories)
            logger.info(f"Start '{start_text}' → {len(ends)} endpoints")

        return all_trajectories

    def _get_parallel_tokens(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
        threshold: float,
        max_tokens: int,
    ) -> List[Tuple[str, int, float]]:
        """Get parallel tokens at next position using TEMPO."""
        position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # TEMPO threshold selection
        mask = probs >= threshold
        token_ids = torch.where(mask)[0]
        token_probs = probs[token_ids]

        # Sort and limit
        sorted_idx = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_idx][:max_tokens]
        token_probs = token_probs[sorted_idx][:max_tokens]

        results = []
        for tid, prob in zip(token_ids, token_probs):
            text = self.tokenizer.decode([tid.item()])
            results.append((text, tid.item(), prob.item()))

        return results

    def _get_endpoints_at_gap(
        self,
        sequence: torch.Tensor,
        prompt_length: int,
        gap_size: int,
        threshold: float,
        max_tokens: int,
    ) -> List[Tuple[str, int, float]]:
        """
        Get endpoints using POSITION GAP.

        This is the critical part - we DON'T generate sequentially!
        We jump directly to position (prompt_length + gap_size).
        """
        seq_length = len(sequence)
        sequence = sequence.unsqueeze(0)

        # KEY: Position gap!
        # Sequence has seq_length tokens, but last one is at position (prompt_length + gap_size)
        position_ids = torch.cat([
            torch.arange(seq_length - 1, device=self.device),  # [0, 1, 2, ..., prompt_length]
            torch.tensor([prompt_length + gap_size], device=self.device)  # Jump to future!
        ]).unsqueeze(0)

        # Debug: Verify we're actually using position gaps
        max_pos = position_ids.max().item()
        assert max_pos > seq_length - 1, f"NO GAP! max_pos={max_pos}, seq_len={seq_length}"

        logger.info(f"USING POSITION GAP: seq_len={seq_length}, positions={position_ids[0].tolist()}, gap detected at position {max_pos}")

        # Position gaps ARE supported by the model's RoPE implementation!
        with torch.no_grad():
            outputs = self.model(
                input_ids=sequence,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )

        logger.info(f"Position gap forward pass completed successfully")

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # TEMPO threshold selection
        mask = probs >= threshold
        token_ids = torch.where(mask)[0]
        token_probs = probs[token_ids]

        # Sort and limit
        sorted_idx = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_idx][:max_tokens]
        token_probs = token_probs[sorted_idx][:max_tokens]

        results = []
        for tid, prob in zip(token_ids, token_probs):
            text = self.tokenizer.decode([tid.item()])
            results.append((text, tid.item(), prob.item()))

        return results

    def _fallback_sequential(
        self,
        sequence: torch.Tensor,
        gap_size: int,
        threshold: float,
        max_tokens: int,
    ) -> List[Tuple[str, int, float]]:
        """
        Fallback: Generate sequentially if position gap fails.

        This proves the concept even if we can't use gaps yet.
        """
        logger.info(f"Using sequential fallback for gap_size={gap_size}")

        current = sequence.unsqueeze(0)

        # Generate gap_size tokens sequentially
        for step in range(gap_size - 1):
            seq_len = current.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current,
                    position_ids=position_ids,
                    return_dict=True,
                    use_cache=False,
                )

            logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(logits).item()

            current = torch.cat([
                current,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)

        # Now get parallel tokens at final position
        seq_len = current.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=current,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # TEMPO threshold selection
        mask = probs >= threshold
        token_ids = torch.where(mask)[0]
        token_probs = probs[token_ids]

        # Sort and limit
        sorted_idx = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_idx][:max_tokens]
        token_probs = token_probs[sorted_idx][:max_tokens]

        results = []
        for tid, prob in zip(token_ids, token_probs):
            text = self.tokenizer.decode([tid.item()])
            results.append((text, tid.item(), prob.item()))

        return results

"""Parallel token processing for TEMPO generation."""

import torch
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParallelTokenSet:
    """Represents a set of parallel tokens at the same position."""
    position: int
    token_ids: List[int]
    probabilities: List[float]
    parent_idx: int  # Index in sequence where these branch from


class ParallelProcessor:
    """Processes multiple tokens in parallel at same positions with memory controls."""

    def __init__(
        self,
        device: str = "cuda",
        max_parallel_tokens: Optional[int] = None
    ):
        """Initialize parallel processor.

        Args:
            device: Device for processing
            max_parallel_tokens: Maximum parallel tokens per step (optional)
        """
        self.device = device
        self.active_sets: List[ParallelTokenSet] = []
        self.sequence_map: Dict[int, int] = {}  # Maps sequence position to set index
        self.max_parallel_tokens = max_parallel_tokens
        
    def create_parallel_set(
        self,
        position: int,
        token_ids: List[int],
        probabilities: List[float],
        parent_idx: int
    ) -> ParallelTokenSet:
        """Create a new parallel token set with memory limits.

        Args:
            position: Logical position
            token_ids: List of token IDs
            probabilities: List of probabilities
            parent_idx: Parent sequence index

        Returns:
            ParallelTokenSet, possibly trimmed to max_parallel_tokens
        """
        # Enforce maximum parallel tokens if set
        if self.max_parallel_tokens is not None and len(token_ids) > self.max_parallel_tokens:
            logger.debug(
                f"Trimming parallel set from {len(token_ids)} to {self.max_parallel_tokens} tokens"
            )

            # Sort by probability and keep top-k
            sorted_pairs = sorted(
                zip(probabilities, token_ids),
                key=lambda x: x[0],
                reverse=True
            )[:self.max_parallel_tokens]

            probabilities, token_ids = zip(*sorted_pairs)
            token_ids = list(token_ids)
            probabilities = list(probabilities)

        return ParallelTokenSet(
            position=position,
            token_ids=token_ids,
            probabilities=probabilities,
            parent_idx=parent_idx
        )
    
    def batch_tokens_for_processing(
        self,
        token_sets: List[ParallelTokenSet],
        base_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Batch parallel tokens for efficient processing.
        
        Args:
            token_sets: List of parallel token sets
            base_sequence: Base sequence up to branching point
            
        Returns:
            Batched sequences and position mapping
        """
        sequences = []
        position_map = []
        
        for set_idx, token_set in enumerate(token_sets):
            base = base_sequence[:token_set.parent_idx + 1]
            
            for token_id in token_set.token_ids:
                # Create sequence with this token
                seq = torch.cat([
                    base,
                    torch.tensor([token_id], device=self.device)
                ])
                sequences.append(seq)
                position_map.append(set_idx)
                
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.zeros(
                    max_len - seq.size(0),
                    dtype=seq.dtype,
                    device=seq.device
                )
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
            
        return torch.stack(padded_sequences), position_map
    
    def merge_parallel_results(
        self,
        results: torch.Tensor,
        position_map: List[int],
        token_sets: List[ParallelTokenSet]
    ) -> Dict[int, torch.Tensor]:
        """
        Merge results from parallel processing.
        
        Args:
            results: Results tensor [batch, ...]
            position_map: Mapping from batch to set index
            token_sets: Original token sets
            
        Returns:
            Dictionary mapping set index to results
        """
        merged = {}
        
        for batch_idx, set_idx in enumerate(position_map):
            if set_idx not in merged:
                merged[set_idx] = []
            merged[set_idx].append(results[batch_idx])
            
        # Stack results for each set
        for set_idx in merged:
            merged[set_idx] = torch.stack(merged[set_idx])
            
        return merged
    
    def prune_token_set(
        self,
        token_set: ParallelTokenSet,
        keep_indices: List[int]
    ) -> ParallelTokenSet:
        """Prune tokens from a parallel set."""
        new_tokens = [token_set.token_ids[i] for i in keep_indices]
        new_probs = [token_set.probabilities[i] for i in keep_indices]
        
        return ParallelTokenSet(
            position=token_set.position,
            token_ids=new_tokens,
            probabilities=new_probs,
            parent_idx=token_set.parent_idx
        )
    
    def get_sequence_with_token(
        self,
        base_sequence: List[int],
        token_set: ParallelTokenSet,
        token_idx: int
    ) -> List[int]:
        """Get full sequence with specific token from set."""
        sequence = base_sequence[:token_set.parent_idx + 1]
        sequence.append(token_set.token_ids[token_idx])
        return sequence
    
    def compute_set_entropy(
        self,
        token_set: ParallelTokenSet
    ) -> float:
        """Compute entropy of probability distribution in set."""
        probs = torch.tensor(token_set.probabilities)
        probs = probs / probs.sum()  # Normalize
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()
    
    def should_expand_set(
        self,
        token_set: ParallelTokenSet,
        min_entropy: float = 0.5,
        max_size: int = 10
    ) -> bool:
        """Determine if a parallel set should be expanded further.

        Args:
            token_set: Token set to evaluate
            min_entropy: Minimum entropy threshold
            max_size: Maximum set size

        Returns:
            True if set should be expanded
        """
        # Check against configured limit
        effective_max = max_size
        if self.max_parallel_tokens is not None:
            effective_max = min(max_size, self.max_parallel_tokens)

        if len(token_set.token_ids) >= effective_max:
            return False

        entropy = self.compute_set_entropy(token_set)
        return entropy > min_entropy

    def set_max_parallel_tokens(self, max_tokens: int) -> None:
        """Set maximum parallel tokens per step.

        Args:
            max_tokens: Maximum number of parallel tokens
        """
        assert max_tokens > 0, "Max parallel tokens must be positive"
        self.max_parallel_tokens = max_tokens
        logger.info(f"Set max parallel tokens to {max_tokens}")
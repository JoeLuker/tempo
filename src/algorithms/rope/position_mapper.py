"""Position mapping for parallel tokens in TEMPO."""

import torch
from typing import Dict, List, Tuple, Optional


class PositionMapper:
    """Maps physical token positions to logical positions for parallel processing."""
    
    def __init__(self):
        self.position_map: Dict[int, int] = {}
        self.parallel_sets: List[Tuple[int, int]] = []
        
    def build_position_map(
        self, 
        token_indices: List[int], 
        position_offset: int = 0
    ) -> Dict[int, int]:
        """
        Build mapping from physical positions to logical positions.
        
        Args:
            token_indices: List of token IDs (-1 indicates set boundary)
            position_offset: Starting position offset
            
        Returns:
            Dictionary mapping physical to logical positions
        """
        position_map = {}
        logical_position = position_offset
        physical_position = position_offset
        
        for token_idx in token_indices:
            if token_idx == -1:  # Separator between parallel sets
                logical_position += 1
            else:
                position_map[physical_position] = logical_position
                physical_position += 1
                
        return position_map
    
    def get_logical_positions(
        self, 
        physical_positions: torch.Tensor
    ) -> torch.Tensor:
        """Convert physical positions to logical positions."""
        logical_positions = physical_positions.clone()
        
        for i, pos in enumerate(physical_positions.tolist()):
            if pos in self.position_map:
                logical_positions[i] = self.position_map[pos]
                
        return logical_positions
    
    def identify_parallel_sets(
        self, 
        token_indices: List[int],
        start_offset: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Identify start and end positions of parallel token sets.
        
        Returns:
            List of (start, end) tuples for each parallel set
        """
        parallel_sets = []
        set_start = start_offset
        
        for i, token_idx in enumerate(token_indices):
            if token_idx == -1:  # End of parallel set
                if i > set_start:
                    parallel_sets.append((set_start, start_offset + i))
                set_start = start_offset + i + 1
                
        # Handle final set
        if set_start < start_offset + len(token_indices):
            parallel_sets.append((set_start, start_offset + len(token_indices)))
            
        return parallel_sets
    
    def clear(self):
        """Clear all position mappings."""
        self.position_map.clear()
        self.parallel_sets.clear()
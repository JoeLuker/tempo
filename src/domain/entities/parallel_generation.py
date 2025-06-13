"""Parallel generation entities for the TEMPO system.

This module defines entities specific to parallel token generation,
including logical layouts, token sets, and generation configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
import torch
from datetime import datetime


@dataclass
class LogicalPosition:
    """Represents a mapping between logical and physical positions in the sequence."""
    logical_step: int
    physical_start_idx: int
    physical_end_idx: int
    
    def __post_init__(self):
        """Validate logical position."""
        if self.logical_step < 0:
            raise ValueError(f"logical_step must be non-negative, got {self.logical_step}")
        if self.physical_start_idx < 0:
            raise ValueError(f"physical_start_idx must be non-negative, got {self.physical_start_idx}")
        if self.physical_end_idx < self.physical_start_idx:
            raise ValueError(f"physical_end_idx {self.physical_end_idx} must be >= physical_start_idx {self.physical_start_idx}")
    
    @property
    def num_tokens(self) -> int:
        """Get the number of tokens at this logical position."""
        return self.physical_end_idx - self.physical_start_idx + 1


@dataclass
class ParallelTokenSet:
    """Represents a set of parallel tokens at a logical position."""
    logical_step: int
    tokens: List[Tuple[int, float]]  # List of (token_id, probability)
    original_tokens: Optional[List[Tuple[int, float]]] = None  # Before pruning
    removed_tokens: Optional[List[Tuple[int, float]]] = None  # Tokens that were pruned
    
    def __post_init__(self):
        """Validate parallel token set."""
        if self.logical_step < 0:
            raise ValueError(f"logical_step must be non-negative, got {self.logical_step}")
        if not self.tokens:
            raise ValueError("tokens list cannot be empty")
        
        # Validate token format
        for token_id, prob in self.tokens:
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"Invalid token_id: {token_id}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"Token probability must be in [0, 1], got {prob}")
    
    @property
    def token_ids(self) -> List[int]:
        """Get just the token IDs."""
        return [tid for tid, _ in self.tokens]
    
    @property
    def probabilities(self) -> List[float]:
        """Get just the probabilities."""
        return [prob for _, prob in self.tokens]
    
    @property
    def num_tokens(self) -> int:
        """Get the number of tokens in this set."""
        return len(self.tokens)
    
    @property
    def was_pruned(self) -> bool:
        """Check if this set was pruned."""
        return self.removed_tokens is not None and len(self.removed_tokens) > 0


@dataclass
class GenerationConfig:
    """Configuration for parallel text generation."""
    max_tokens: int = 100
    selection_threshold: float = 0.1
    min_steps: int = 0
    
    # Retroactive pruning
    use_retroactive_removal: bool = False
    attention_threshold: Optional[float] = None
    
    # Dynamic threshold
    dynamic_threshold: bool = False
    final_threshold: float = 1.0
    bezier_p1: float = 0.2
    bezier_p2: float = 0.8
    use_relu: bool = False
    relu_activation_point: float = 0.5
    
    # MCTS parameters
    use_mcts: bool = False
    mcts_simulations: int = 10
    mcts_c_puct: float = 1.0
    mcts_depth: int = 5
    
    # Other options
    disable_kv_cache: bool = False
    isolate_parallel_tokens: bool = True
    preserve_all_isolated_tokens: Optional[bool] = None
    show_token_ids: bool = False
    return_parallel_sets: bool = False
    debug_mode: Optional[bool] = None
    system_content: Optional[str] = None
    
    # Callbacks
    sequence_callback: Optional[Callable[[int, int, int], None]] = None
    
    def __post_init__(self):
        """Validate generation configuration."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError(f"selection_threshold must be in [0, 1], got {self.selection_threshold}")
        if self.min_steps < 0:
            raise ValueError(f"min_steps must be non-negative, got {self.min_steps}")
        
        # Validate dynamic threshold parameters
        if self.dynamic_threshold:
            if not 0.0 <= self.final_threshold <= 1.0:
                raise ValueError(f"final_threshold must be in [0, 1], got {self.final_threshold}")
            if not 0.0 <= self.bezier_p1 <= 1.0:
                raise ValueError(f"bezier_p1 must be in [0, 1], got {self.bezier_p1}")
            if not 0.0 <= self.bezier_p2 <= 1.0:
                raise ValueError(f"bezier_p2 must be in [0, 1], got {self.bezier_p2}")
            if not 0.0 <= self.relu_activation_point <= 1.0:
                raise ValueError(f"relu_activation_point must be in [0, 1], got {self.relu_activation_point}")
        
        # Validate MCTS parameters
        if self.use_mcts:
            if self.mcts_simulations <= 0:
                raise ValueError(f"mcts_simulations must be positive, got {self.mcts_simulations}")
            if self.mcts_c_puct <= 0:
                raise ValueError(f"mcts_c_puct must be positive, got {self.mcts_c_puct}")
            if self.mcts_depth <= 0:
                raise ValueError(f"mcts_depth must be positive, got {self.mcts_depth}")


@dataclass 
class GenerationResult:
    """Result of parallel text generation."""
    generated_text: str  # Formatted output text
    raw_generated_text: str  # Raw decoded text
    clean_text: Optional[str] = None  # Clean text without formatting
    
    # Timing information
    generation_time: float = 0.0
    removal_time: float = 0.0
    removal_steps: int = 0
    
    # Generation metadata
    prompt: str = ""
    selection_threshold: float = 0.1
    use_retroactive_removal: bool = False
    min_steps: int = 0
    disable_kv_cache: bool = False
    isolate_parallel_tokens: bool = True
    is_qwen_model: bool = False
    had_repetition_loop: bool = False
    
    # Layout information
    logical_layout: List[LogicalPosition] = field(default_factory=list)
    
    # Token sets for visualization
    token_sets: Optional[List[Tuple[int, Tuple[List[int], List[float]], Tuple[List[int], List[float]]]]] = None
    all_original_token_sets: Optional[Dict[int, List[Tuple[int, float]]]] = None
    all_surviving_token_sets: Optional[Dict[int, List[Tuple[int, float]]]] = None
    
    def __post_init__(self):
        """Validate generation result."""
        if self.generation_time < 0:
            raise ValueError(f"generation_time must be non-negative, got {self.generation_time}")
        if self.removal_time < 0:
            raise ValueError(f"removal_time must be non-negative, got {self.removal_time}")
        if self.removal_steps < 0:
            raise ValueError(f"removal_steps must be non-negative, got {self.removal_steps}")


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    token_id: int
    probability: float
    value: float = 0.0
    visits: int = 0
    children: List['MCTSNode'] = field(default_factory=list)
    parent: Optional['MCTSNode'] = None
    
    def __post_init__(self):
        """Validate MCTS node."""
        if self.token_id < 0:
            raise ValueError(f"token_id must be non-negative, got {self.token_id}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.visits < 0:
            raise ValueError(f"visits must be non-negative, got {self.visits}")
    
    @property
    def ucb1_value(self, c_puct: float = 1.0) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c_puct * torch.sqrt(torch.log(torch.tensor(self.parent.visits)) / self.visits).item()
        return exploitation + exploration
    
    def add_child(self, child: 'MCTSNode') -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def update(self, value: float) -> None:
        """Update node statistics after a simulation."""
        self.visits += 1
        self.value += value


@dataclass
class MCTSState:
    """State for MCTS simulation."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Any]
    logical_step: int
    depth: int = 0
    
    def __post_init__(self):
        """Validate MCTS state."""
        if self.logical_step < 0:
            raise ValueError(f"logical_step must be non-negative, got {self.logical_step}")
        if self.depth < 0:
            raise ValueError(f"depth must be non-negative, got {self.depth}")
    
    def copy(self) -> 'MCTSState':
        """Create a copy of this state."""
        return MCTSState(
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            past_key_values=self.past_key_values,  # Note: KV cache may be shared
            logical_step=self.logical_step,
            depth=self.depth
        )
